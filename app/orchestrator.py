"""End-to-end ML engineering workflow state and deployment helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from app.pipeline import (
    DatasetProfile,
    _classification_candidates,
    _is_missing,
    _leakage_risks,
    _parse_rows,
    _preprocessor,
    _regression_candidates,
)
from app.research import research_methods
from app.workflow_graph import build_workflow_graph, get_checkpointed_state, resume_workflow


@dataclass
class RunRequest:
    goal: str
    metric: str
    minimum_score: float | None
    target: str | None
    max_experiments: int = 3
    validation_strategy: str = "holdout"
    max_retries: int = 2
    target_confirmed: bool = False


def _quality_report(profile: DatasetProfile, content: bytes) -> dict[str, Any]:
    _, rows = _parse_rows(profile.filename, content)
    if profile.task == "computer_vision":
        return {"duplicates": 0, "duplicate_rate": 0.0, "outliers": 0, "class_balance": {}, "checks": ["COCO schema detected"]}
    duplicate_count = len(rows) - len({json.dumps(row, sort_keys=True) for row in rows})
    missing_values = sum(item.missing for item in profile.column_profiles)
    leakage_risks = _leakage_risks(profile, rows)
    target_values = [row.get(profile.target) for row in rows] if profile.task == "classification" and profile.target else []
    class_balance: dict[str, int] = {}
    for value in target_values:
        if not _is_missing(value):
            key = str(value)
            class_balance[key] = class_balance.get(key, 0) + 1
    checks = ["schema validated", "missing values profiled", "duplicate records checked"]
    if missing_values:
        checks.append(f"{missing_values} missing values detected; model pipelines will impute within each training split")
    if duplicate_count:
        checks.append("duplicate records require removal before production")
    if leakage_risks:
        checks.append("potential target leakage detected; review flagged features")
    if class_balance and min(class_balance.values()) / max(class_balance.values()) < 0.5:
        checks.append("class imbalance detected; compare weighted metrics and class-aware training")
    return {
        "duplicates": duplicate_count,
        "duplicate_rate": round(duplicate_count / len(rows), 4) if rows else 0.0,
        "missing_values": missing_values,
        "leakage_risks": leakage_risks,
        "outliers": 0,
        "class_balance": class_balance,
        "checks": checks,
    }


def _research_plan(profile: DatasetProfile, recommendation: dict[str, Any]) -> dict[str, Any]:
    catalog = research_methods(profile.task, recommendation["why"])
    return {
        "sources": ["approved internal methods catalog"],
        "methods": recommendation["recommended"],
        "catalog": catalog,
        "experiments": [
            {"name": f"baseline-{model}", "model": model, "change": "candidate model baseline"}
            for model in recommendation["recommended"]
        ],
        "note": "External paper search is intentionally represented as a pluggable research source; no unsupported citations are invented.",
    }


def _fit_for_deployment(profile: DatasetProfile, content: bytes, model_name: str) -> tuple[Pipeline, list[str]] | None:
    if profile.task not in {"classification", "regression"} or not profile.target:
        return None
    _, rows = _parse_rows(profile.filename, content)
    rows = [row for row in rows if not _is_missing(row.get(profile.target))]
    feature_names = [item.name for item in profile.column_profiles if item.name != profile.target]
    numeric_names = {item.name for item in profile.column_profiles if item.data_type == "numeric"}
    features = []
    labels = []
    for row in rows:
        features.append([None if _is_missing(row.get(name)) else float(row[name]) if name in numeric_names else row[name] for name in feature_names])
        labels.append(float(row[profile.target]) if profile.task == "regression" else str(row[profile.target]))
    candidates = _classification_candidates() if profile.task == "classification" else _regression_candidates()
    estimator = candidates.get(model_name)
    if estimator is None:
        return None
    model = Pipeline([("preprocess", _preprocessor(profile, feature_names)), ("model", estimator)])
    model.fit(features, labels)
    return model, feature_names


def run_workflow(filename: str, content: bytes, request: RunRequest) -> tuple[str, dict[str, Any], Pipeline | None, list[str] | None]:
    run_id = uuid.uuid4().hex[:12]
    if not 0 <= request.max_retries <= 5:
        raise ValueError("max_retries must be between 0 and 5.")
    graph = build_workflow_graph()
    state = graph.invoke(
        {
            "run_id": run_id,
            "filename": filename,
            "content": content,
            "request": asdict(request),
            "execution_log": [],
            "node_outputs": {},
            "errors": [],
        },
        {"configurable": {"thread_id": run_id}},
    )
    result = _workflow_result(run_id, state)
    return run_id, result, None, state.get("prepared", {}).get("feature_names")


def _workflow_result(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    evaluation = state.get("evaluation", {"status": "unavailable", "reason": "Workflow failed before evaluation."})
    experiments = [
        {**model, "status": "completed"}
        for model in evaluation.get("models", [])
    ]
    if evaluation.get("status") != "complete" and not experiments:
        experiments = [{"status": "blocked", "reason": evaluation.get("reason", "Evaluation unavailable.")}]
    deployment = state.get("deployment", {"status": "blocked", "reason": "Workflow failed before deployment preparation."})
    decision = state.get("decision", {
        "status": "failed",
        "best_model": None,
        "metric": None,
        "best_score": None,
        "threshold": None,
        "next_step": "Review the workflow error log.",
    })
    outputs = state.get("node_outputs", {})
    stages = [
        {"agent": "Data Profiling Agent", "status": outputs.get("data_profiling", {}).get("status", "blocked"), "output": state.get("quality", {})},
        {"agent": "Preprocessing Agent", "status": outputs.get("preprocessing", {}).get("status", "blocked"), "output": state.get("prepared", {})},
        {"agent": "Model Training Agent", "status": outputs.get("model_training", {}).get("status", "blocked"), "output": state.get("training", {})},
        {"agent": "Evaluation Agent", "status": outputs.get("evaluation", {}).get("status", "blocked"), "output": evaluation},
        {"agent": "Decision Agent", "status": outputs.get("decision", {}).get("status", "blocked"), "output": decision},
        {"agent": "Deployment Preparation Agent", "status": outputs.get("deployment_preparation", {}).get("status", "blocked"), "output": state.get("deployment_preparation", {})},
        {"agent": "Approval Agent", "status": outputs.get("deployment_approval", {}).get("status", "pending"), "output": state.get("deployment_approval", {"status": "pending"})},
        {"agent": "Deployment Agent", "status": outputs.get("deployment", {}).get("status", "pending"), "output": deployment},
    ]
    if state.get("profile"):
        monitoring = {
            "signals": ["latency", "errors", "prediction distribution", "data drift", "model performance"],
            "status": "armed" if deployment.get("status") == "approved" else "waiting",
        }
        return {
            "profile": state["profile"],
            "recommendation": state.get("recommendation", {}),
            "evaluation": evaluation,
            "research": state.get("research", {}),
            "run_id": run_id,
            "request": state.get("request", {}),
            "quality": state.get("quality", {}),
            "tracking": state.get("tracking", {"status": "unavailable"}),
            "experiments": experiments,
            "decision": decision,
            "deployment": deployment,
            "deployment_preparation": state.get("deployment_preparation", {}),
            "deployment_approval": state.get("deployment_approval", {"status": "pending", "approved": False}),
            "monitoring": monitoring,
            "stages": stages,
            "execution_log": state.get("execution_log", []),
            "errors": state.get("errors", []),
            "orchestration": {
                "engine": "LangGraph",
                "checkpointed": True,
                "next_action": "awaiting_approval" if deployment.get("status") == "awaiting_approval" else state.get("workflow_status", "complete"),
                "retry_limit": state.get("request", {}).get("max_retries", 2),
            },
        }
    return {
        "profile": {"filename": "", "format": "unknown", "rows": 0, "columns": 0, "target": None, "task": "unknown", "warnings": [], "column_profiles": []},
        "recommendation": {},
        "evaluation": evaluation,
        "research": {},
        "quality": {},
        "tracking": state.get("tracking", {"status": "unavailable"}),
        "run_id": run_id,
        "request": state.get("request", {}),
        "decision": decision,
        "deployment": deployment,
        "deployment_preparation": state.get("deployment_preparation", {}),
        "deployment_approval": state.get("deployment_approval", {"status": "pending", "approved": False}),
        "stages": stages,
        "execution_log": state.get("execution_log", []),
        "errors": state.get("errors", []),
        "orchestration": {
            "engine": "LangGraph",
            "checkpointed": True,
            "next_action": "failed",
            "retry_limit": state.get("request", {}).get("max_retries", 2),
        },
    }


def approve_workflow(run_id: str, approved: bool, reviewer: str | None = None) -> tuple[dict[str, Any], Pipeline | None, list[str] | None]:
    state = resume_workflow(build_workflow_graph(), run_id, approved, reviewer)
    result = _workflow_result(run_id, state)
    if result["deployment"]["status"] != "approved":
        return result, None, None
    artifact = joblib.load(result["deployment"]["artifact"])
    return result, artifact["model"], artifact["feature_names"]


def restore_workflow_run(run_id: str) -> dict[str, Any] | None:
    state = get_checkpointed_state(run_id)
    if state is None:
        return None
    result = _workflow_result(run_id, state)
    model = None
    feature_names = state.get("feature_names") or state.get("prepared", {}).get("feature_names")
    artifact_path = result.get("deployment", {}).get("artifact")
    if artifact_path and Path(artifact_path).is_file():
        artifact = joblib.load(artifact_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
    return {"result": result, "model": model, "feature_names": feature_names, "profile": result.get("profile", {})}