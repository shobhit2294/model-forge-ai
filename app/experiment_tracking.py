"""MLflow experiment tracking and model registry integration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import Metric


DEFAULT_EXPERIMENT_NAME = "ModelForge"


def _client() -> MlflowClient:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _safe_params(params: dict[str, Any]) -> dict[str, str]:
    return {key[:250]: str(value)[:500] for key, value in params.items()}


def _log_params(client: MlflowClient, run_id: str, params: dict[str, Any]) -> None:
    for key, value in _safe_params(params).items():
        client.log_param(run_id, key, value)


def _experiment_id(client: MlflowClient) -> str:
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            experiment_id = client.create_experiment(experiment_name)
        except Exception:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise
            experiment_id = experiment.experiment_id
        return experiment_id
    return experiment.experiment_id


def dataset_version(content: bytes) -> str:
    """Return an immutable content identifier without uploading raw dataset contents."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def start_experiment_run(
    run_id: str,
    filename: str,
    content: bytes,
    request: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    client = _client()
    experiment_id = _experiment_id(client)
    dataset_id = dataset_version(content)
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": f"{profile['task']}-{run_id}",
            "modelforge.run_id": run_id,
            "modelforge.dataset_version": dataset_id,
            "modelforge.task": str(profile["task"]),
            "modelforge.target": str(profile.get("target") or ""),
            "modelforge.dataset_filename": filename,
        },
    )
    params = {
        "goal": request.get("goal", ""),
        "target": profile.get("target") or "none",
        "task": profile["task"],
        "metric": request.get("metric", ""),
        "minimum_score": request.get("minimum_score"),
        "validation_strategy": request.get("validation_strategy", "holdout"),
        "max_retries": request.get("max_retries", 2),
        "random_seed": 42,
        "dataset_version": dataset_id,
        "dataset_rows": profile["rows"],
        "dataset_columns": profile["columns"],
        "python_version": ".".join(map(str, os.sys.version_info[:3])),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "xgboost_version": importlib.metadata.version("xgboost"),
        "code_revision": os.environ.get("GIT_COMMIT", "working-tree"),
    }
    _log_params(client, run.info.run_id, params)
    client.log_text(
        run.info.run_id,
        json.dumps({
            "version": dataset_id,
            "filename": filename,
            "rows": profile["rows"],
            "columns": profile["columns"],
            "target": profile.get("target"),
            "task": profile["task"],
            "schema": [
                {"name": item["name"], "data_type": item["data_type"]}
                for item in profile.get("column_profiles", [])
            ],
        }, indent=2, sort_keys=True),
        "dataset/dataset-version.json",
    )
    experiment = client.get_experiment(experiment_id)
    return {
        "status": "tracking",
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_id": experiment_id,
        "experiment_name": experiment.name if experiment else DEFAULT_EXPERIMENT_NAME,
        "run_id": run.info.run_id,
        "dataset_version": dataset_id,
        "candidate_runs": {},
        "model_name": None,
        "model_version": None,
        "model_version_run_id": None,
        "model_uri": None,
        "workflow_run_id": run_id,
        "comparison": [],
    }


def _candidate_run(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
    candidate: dict[str, Any],
) -> str:
    name = str(candidate["model"])
    existing_id = client.get_run(parent_run_id).data.tags.get(f"modelforge.candidate.{name}")
    if existing_id:
        return existing_id
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": f"{name}-comparison",
            "mlflow.parentRunId": parent_run_id,
            "modelforge.candidate_model": name,
        },
    )
    client.set_tag(parent_run_id, f"modelforge.candidate.{name}", run.info.run_id)
    params = candidate.get("params", {})
    if params:
        _log_params(client, run.info.run_id, params)
    return run.info.run_id


def log_candidate_training(tracking: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    candidate_runs = dict(tracking.get("candidate_runs", {}))
    for candidate in training.get("models", []):
        run_id = _candidate_run(client, tracking["experiment_id"], tracking["run_id"], candidate)
        candidate_runs[candidate["model"]] = run_id
    return {**tracking, "candidate_runs": candidate_runs}


def log_evaluation(tracking: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    candidate_runs = dict(tracking.get("candidate_runs", {}))
    comparisons = []
    for candidate in evaluation.get("models", []):
        if candidate.get("params") and candidate["model"] not in candidate_runs:
            candidate_runs[candidate["model"]] = _candidate_run(
                client,
                tracking["experiment_id"],
                tracking["run_id"],
                candidate,
            )
        metrics = {
            key: float(value)
            for key, value in candidate.items()
            if key != "model" and key != "confusion_matrix" and isinstance(value, (int, float))
        }
        run_id = candidate_runs.get(candidate["model"])
        if run_id:
            metric_timestamp = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[Metric(key, value, metric_timestamp, 0) for key, value in metrics.items()],
            )
            client.set_terminated(run_id, status="FINISHED")
        comparisons.append({
            "model": candidate["model"],
            **metrics,
            "candidate_run_id": run_id,
        })
    selected = evaluation.get("selection_metric")
    if selected:
        for item in comparisons:
            if selected in item:
                client.log_metric(tracking["run_id"], f"candidate.{item['model']}.{selected}", item[selected])
    if evaluation.get("status") == "complete":
        client.set_tag(tracking["run_id"], "modelforge.selected_model", evaluation["best_model"])
        client.set_tag(tracking["run_id"], "modelforge.selection_metric", str(selected))
    client.log_text(
        tracking["run_id"],
        json.dumps({
            "selection_metric": selected,
            "validation": evaluation.get("validation"),
            "best_model": evaluation.get("best_model"),
            "models": comparisons,
        }, indent=2, sort_keys=True),
        "evaluation/model-comparison.json",
    )
    return {**tracking, "comparison": comparisons, "best_model": evaluation.get("best_model")}


def log_approval(tracking: dict[str, Any], approved: bool, reviewer: str | None) -> None:
    client = _client()
    client.set_tag(tracking["run_id"], "modelforge.deployment_approval", "approved" if approved else "rejected")
    if reviewer:
        client.set_tag(tracking["run_id"], "modelforge.reviewer", reviewer)


def register_deployed_model(
    tracking: dict[str, Any],
    model: Any,
    profile: dict[str, Any],
    decision: dict[str, Any],
    approval: dict[str, Any],
) -> dict[str, Any]:
    client = _client()
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"ModelForge-{profile['task']}-{profile.get('target') or 'model'}").strip("-.")
    model_info = None
    existing_version = next(
        (
            version
            for version in client.search_model_versions(f"name='{model_name}'")
            if version.run_id == tracking["run_id"]
        ),
        None,
    )
    if existing_version:
        if existing_version.status != "READY":
            raise RuntimeError(
                f"MLflow model version {model_name}/{existing_version.version} is not ready."
            )
        model_version = str(existing_version.version)
        model_uri = existing_version.source
    else:
        with mlflow.start_run(run_id=tracking["run_id"]):
            client.log_metric(tracking["run_id"], f"best.{decision['metric']}", float(decision["best_score"]))
            client.set_tag(tracking["run_id"], "modelforge.deployment_approval", "approved")
            client.set_tag(tracking["run_id"], "modelforge.reviewer", str(approval.get("reviewer") or "unspecified"))
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,
            )
        model_version = str(model_info.registered_model_version or "")
        if not model_version:
            raise RuntimeError("MLflow logged the model but did not return a registry version.")
        model_uri = model_info.model_uri
    client.set_model_version_tag(model_name, model_version, "modelforge.dataset_version", tracking["dataset_version"])
    client.set_model_version_tag(model_name, model_version, "modelforge.workflow_run_id", str(tracking.get("workflow_run_id", "")))
    client.set_tag(tracking["run_id"], "modelforge.model_name", model_name)
    client.set_tag(tracking["run_id"], "modelforge.model_version", model_version)
    client.set_terminated(tracking["run_id"], status="FINISHED")
    return {
        **tracking,
        "status": "complete",
        "model_name": model_name,
        "model_version": model_version,
        "model_version_run_id": str(tracking.get("workflow_run_id", "")),
        "model_uri": model_uri,
        "registered_model_uri": f"models:/{model_name}/{model_version}",
    }


def finish_experiment_run(tracking: dict[str, Any], status: str) -> None:
    if tracking.get("run_id"):
        _client().set_terminated(tracking["run_id"], status=status.upper())
