from typing import Any
from time import perf_counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.orchestrator import RunRequest, run_workflow
from app.metrics import API_ERRORS, PREDICTIONS, REQUEST_LATENCY, WORKFLOW_RUNS, metrics_payload

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


@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...), target: str | None = Form(default=None), goal: str = Form(default="Build the best predictive model"), metric: str = Form(default="f1"), minimum_score: str | None = Form(default=None)) -> dict:
    started_at = perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail="A dataset filename is required")
    if not file.filename.lower().endswith((".csv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet")):
        raise HTTPException(status_code=415, detail="Use CSV, JSON, JSONL, Excel, or Parquet format")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty")
    try:
        threshold = float(minimum_score) if minimum_score else None
        request = RunRequest(goal=goal, metric=metric, minimum_score=threshold, target=target or None)
        run_id, result, model, feature_names = run_workflow(file.filename, content, request)
        RUNS[run_id] = {"result": result, "model": model, "feature_names": feature_names, "profile": result["profile"]}
        WORKFLOW_RUNS.labels(result["decision"]["status"]).inc()
        return result
    except (ValueError, UnicodeError, OSError, KeyError) as error:
        API_ERRORS.labels("analyze").inc()
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        REQUEST_LATENCY.labels("analyze").observe(perf_counter() - started_at)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["result"]


@app.post("/api/runs/{run_id}/predict")
def predict(run_id: str, features: dict[str, Any]) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or run["model"] is None:
        raise HTTPException(status_code=409, detail="This run has no approved deployable model")
    numeric_names = {item["name"] for item in run["profile"]["column_profiles"] if item["data_type"] == "numeric"}
    ordered = []
    for name in run["feature_names"]:
        value = features.get(name)
        if value in (None, ""):
            ordered.append(None)
        elif name in numeric_names:
            try:
                ordered.append(float(value))
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"Feature '{name}' must be numeric") from error
        else:
            ordered.append(value)
    model = run["model"]
    prediction = model.predict([ordered])[0]
    response: dict[str, Any] = {"run_id": run_id, "prediction": str(prediction)}
    if hasattr(model, "predict_proba"):
        response["probability"] = round(float(max(model.predict_proba([ordered])[0])), 4)
    PREDICTIONS.labels(run_id).inc()
    return response


@app.get("/api/monitoring/{run_id}")
def monitoring(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "status": "armed" if run["model"] else "waiting", "signals": run["result"]["monitoring"]["signals"]}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(metrics_payload(), media_type="text/plain; version=0.0.4")