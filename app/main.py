from typing import Any
from time import perf_counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.orchestrator import RunRequest, approve_workflow, restore_workflow_run, run_workflow
from app.pipeline import SUPPORTED_DATA_EXTENSIONS, _is_missing, profile_dataset
from app.metrics import (
    API_ERRORS,
    PREDICTIONS,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    REQUEST_LATENCY,
    WORKFLOW_RUNS,
    metrics_payload,
)
from app.monitoring import (
    initialize_monitoring,
    monitoring_status,
    record_feedback,
    record_prediction,
    refresh_monitoring_metrics,
)

app = FastAPI(title="AI Engineering Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
if Path("app/static/assets").is_dir():
    app.mount("/assets", StaticFiles(directory="app/static/assets"), name="assets")
RUNS: dict[str, dict[str, Any]] = {}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-engineering-agent"}


@app.post("/api/profile")
async def profile_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A dataset filename is required")
    if not file.filename.lower().endswith(SUPPORTED_DATA_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail="Supported formats: CSV, JSON, JSONL/NDJSON, XML, Excel, and Parquet.",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty")
    try:
        profile = profile_dataset(file.filename, content)
    except (ValueError, UnicodeError, OSError) as error:
        API_ERRORS.labels("profile").inc()
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "filename": profile.filename,
        "format": profile.format,
        "rows": profile.rows,
        "columns": profile.columns,
        "candidate_target": profile.target,
        "candidate_task": profile.task,
        "warnings": profile.warnings,
        "column_profiles": [item.__dict__ for item in profile.column_profiles],
    }


@app.post("/api/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
    goal: str = Form(default="Build the best predictive model"),
    metric: str = Form(default="f1"),
    minimum_score: str | None = Form(default=None),
    validation_strategy: str = Form(default="holdout"),
    max_retries: int = Form(default=2, ge=0, le=5),
    target_confirmed: bool = Form(default=False),
) -> dict:
    started_at = perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail="A dataset filename is required")
    if not file.filename.lower().endswith(SUPPORTED_DATA_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail="Supported formats: CSV, JSON, JSONL/NDJSON, XML, Excel, and Parquet.",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty")
    try:
        if not target and not target_confirmed:
            suggested = profile_dataset(file.filename, content).target
            if suggested:
                raise HTTPException(
                    status_code=409,
                    detail=f"'{suggested}' looks like the target column. Confirm it in the target field, or explicitly choose profile-only analysis.",
                )
        threshold = float(minimum_score) if minimum_score else None
        request = RunRequest(
            goal=goal,
            metric=metric,
            minimum_score=threshold,
            target=target or None,
            validation_strategy=validation_strategy,
            max_retries=max_retries,
            target_confirmed=target_confirmed or bool(target),
        )
        run_id, result, model, feature_names = run_workflow(file.filename, content, request)
        RUNS[run_id] = {
            "result": result,
            "model": model,
            "feature_names": feature_names,
            "profile": result.get("profile", {}),
        }
        evaluation = result.get("evaluation", {})
        selected_model = result.get("decision", {}).get("best_model")
        selected_metrics = next(
            (item for item in evaluation.get("models", []) if item.get("model") == selected_model),
            {},
        )
        if feature_names and evaluation.get("status") == "complete":
            initialize_monitoring(
                run_id,
                file.filename,
                content,
                feature_names,
                result["profile"]["task"],
                evaluation.get("selection_metric"),
                selected_metrics.get(evaluation.get("selection_metric")),
            )
        WORKFLOW_RUNS.labels(result["decision"]["status"]).inc()
        return result
    except HTTPException:
        raise
    except (ValueError, UnicodeError, OSError, KeyError) as error:
        API_ERRORS.labels("analyze").inc()
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        REQUEST_LATENCY.labels("analyze").observe(perf_counter() - started_at)


@app.post("/api/runs/{run_id}/approve")
def approve_run(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        run = restore_workflow_run(run_id)
        if run:
            RUNS[run_id] = run
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["result"].get("deployment", {}).get("status") != "awaiting_approval":
        raise HTTPException(status_code=409, detail="This run is not waiting for deployment approval")
    approved = payload.get("approved")
    if not isinstance(approved, bool):
        raise HTTPException(status_code=422, detail="'approved' must be a boolean")
    try:
        result, model, feature_names = approve_workflow(run_id, approved, payload.get("reviewer"))
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        API_ERRORS.labels("approve").inc()
        raise HTTPException(status_code=422, detail=str(error)) from error
    run.update({
        "result": result,
        "model": model,
        "feature_names": feature_names,
        "profile": result.get("profile", {}),
    })
    WORKFLOW_RUNS.labels(result["decision"]["status"]).inc()
    return result


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        run = restore_workflow_run(run_id)
        if run:
            RUNS[run_id] = run
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["result"]


@app.post("/api/runs/{run_id}/predict")
def predict(run_id: str, features: dict[str, Any]) -> dict[str, Any]:
    started_at = perf_counter()
    task = "unknown"
    try:
        run = RUNS.get(run_id)
        if not run:
            run = restore_workflow_run(run_id)
            if run:
                RUNS[run_id] = run
        if not run or run["model"] is None:
            raise HTTPException(status_code=409, detail="This run has no approved deployable model")
        task = run["profile"]["task"]
        numeric_names = {
            item["name"] for item in run["profile"]["column_profiles"]
            if item["data_type"] == "numeric"
        }
        ordered = []
        normalized_features = {}
        for name in run["feature_names"]:
            value = features.get(name)
            if _is_missing(value):
                normalized_features[name] = None
                ordered.append(None)
            elif name in numeric_names:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise HTTPException(status_code=422, detail=f"Feature '{name}' must be numeric") from error
                normalized_features[name] = numeric_value
                ordered.append(numeric_value)
            else:
                normalized_features[name] = value
                ordered.append(value)
        model = run["model"]
        prediction = model.predict([ordered])[0]
        response: dict[str, Any] = {"run_id": run_id, "prediction": str(prediction)}
        if hasattr(model, "predict_proba"):
            response["probability"] = round(float(max(model.predict_proba([ordered])[0])), 4)
        prediction_id, monitoring = record_prediction(run_id, normalized_features, prediction)
        response.update({
            "prediction_id": prediction_id,
            "monitoring": monitoring,
        })
        PREDICTIONS.labels(task).inc()
        return response
    except HTTPException as error:
        PREDICTION_ERRORS.labels(f"http_{error.status_code}").inc()
        API_ERRORS.labels("predict").inc()
        raise
    except Exception as error:
        PREDICTION_ERRORS.labels("inference").inc()
        API_ERRORS.labels("predict").inc()
        raise HTTPException(status_code=500, detail="Prediction or monitoring failed") from error
    finally:
        PREDICTION_LATENCY.labels(task).observe(perf_counter() - started_at)


@app.get("/api/monitoring/{run_id}")
def monitoring(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        run = restore_workflow_run(run_id)
        if run:
            RUNS[run_id] = run
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        status = monitoring_status(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        **status,
        "status": "armed" if run["model"] else "waiting_for_approval",
    }


@app.post("/api/runs/{run_id}/feedback")
def submit_feedback(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = record_feedback(
            run_id,
            str(payload.get("prediction_id", "")),
            payload.get("actual"),
        )
    except KeyError as error:
        API_ERRORS.labels("feedback").inc()
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        API_ERRORS.labels("feedback").inc()
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    refresh_monitoring_metrics()
    return Response(metrics_payload(), media_type="text/plain; version=0.0.4")