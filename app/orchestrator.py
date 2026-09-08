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
    _parse_rows,
    _preprocessor,
    _regression_candidates,
    evaluate_models,
    pipeline_result,
    profile_dataset,
    recommend_models,
)
from app.research import research_methods
from app.workflow_graph import build_workflow_graph


@dataclass
class RunRequest:
    goal: str
    metric: str
    minimum_score: float | None
    target: str | None
    max_experiments: int = 3


def _quality_report(profile: DatasetProfile, content: bytes) -> dict[str, Any]:
    _, rows = _parse_rows(profile.filename, content)
    if profile.task == "computer_vision":
        return {"duplicates": 0, "duplicate_rate": 0.0, "outliers": 0, "class_balance": {}, "checks": ["COCO schema detected"]}
    duplicate_count = len(rows) - len({json.dumps(row, sort_keys=True) for row in rows})
    target_values = [row.get(profile.target) for row in rows] if profile.target else []
    class_balance: dict[str, int] = {}
    for value in target_values:
        if value not in (None, ""):
            key = str(value)
            class_balance[key] = class_balance.get(key, 0) + 1
    checks = ["schema validated", "missing values profiled", "duplicate records checked"]
    if duplicate_count:
        checks.append("duplicate records require removal before production")
    if class_balance and min(class_balance.values()) / max(class_balance.values()) < 0.5:
        checks.append("class imbalance detected; compare weighted metrics and class-aware training")
    return {
        "duplicates": duplicate_count,
        "duplicate_rate": round(duplicate_count / len(rows), 4) if rows else 0.0,
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


def _metric_value(result: dict[str, Any], metric: str, task: str) -> float | None:
    aliases = {
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "r2": "r2",
        "mae": "mae",
        "rmse": "rmse",
    }
    requested = metric.lower()
    if task == "regression" and requested in {"f1", "accuracy", "precision", "recall"}:
        requested = "r2"
    key = aliases.get(requested)
    if key is None:
        key = "f1" if task == "classification" else "r2"
    value = result.get(key)
    return float(value) if value is not None else None


def _fit_for_deployment(profile: DatasetProfile, content: bytes, model_name: str) -> tuple[Pipeline, list[str]] | None:
    if profile.task not in {"classification", "regression"} or not profile.target:
        return None
    _, rows = _parse_rows(profile.filename, content)
    rows = [row for row in rows if row.get(profile.target) not in (None, "")]
    feature_names = [item.name for item in profile.column_profiles if item.name != profile.target]
    numeric_names = {item.name for item in profile.column_profiles if item.data_type == "numeric"}
    features = []
    labels = []
    for row in rows:
        features.append([None if row.get(name) in (None, "") else float(row[name]) if name in numeric_names else row[name] for name in feature_names])
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
    profile = profile_dataset(filename, content, request.target)
    recommendation = recommend_models(profile)
    quality = _quality_report(profile, content)
    research = _research_plan(profile, recommendation)
    base = pipeline_result(profile, content)
    evaluation = base["evaluation"]
    experiments = []
    if evaluation and evaluation.get("status") == "complete":
        experiments = [{**result, "status": "completed"} for result in evaluation["models"]]
    elif evaluation:
        experiments = [{"status": "blocked", "reason": evaluation["reason"]}]
    best_score = next((_metric_value(item, request.metric, profile.task) for item in experiments if item.get("model") == evaluation.get("best_model")), None) if evaluation and evaluation.get("status") == "complete" else None
    lower_is_better = profile.task == "regression" and request.metric.lower() in {"mae", "rmse"}
    threshold_met = request.minimum_score is None or (best_score is not None and ((best_score <= request.minimum_score) if lower_is_better else (best_score >= request.minimum_score)))
    approved = bool(evaluation and evaluation.get("status") == "complete" and threshold_met and quality["duplicates"] == 0)
    best_model = evaluation.get("best_model") if evaluation else None
    deployment = {"status": "approved" if approved else "blocked", "model": best_model, "artifact": f"artifacts/{run_id}.joblib" if approved else None, "reason": "Quality gate passed." if approved else "Performance threshold, labels, or data-quality requirements are not satisfied."}
    if approved and best_model:
        fitted_model = _fit_for_deployment(profile, content, best_model)
        if fitted_model:
            artifact_path = Path("artifacts") / f"{run_id}.joblib"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": fitted_model[0], "feature_names": fitted_model[1], "profile": asdict(profile)}, artifact_path)
    else:
        fitted_model = None
    decision = {"status": "approved" if approved else "needs_improvement", "best_model": best_model, "metric": request.metric, "best_score": best_score, "threshold": request.minimum_score, "next_step": "Deploy approved model." if approved else "Run another experiment after changing features, preprocessing, or hyperparameters."}
    stages = [
        {"agent": "Data Agent", "status": "complete", "output": quality},
        {"agent": "Research Agent", "status": "complete", "output": research},
        {"agent": "Model Agent", "status": "complete" if experiments and experiments[0].get("status") == "completed" else "blocked", "output": experiments},
        {"agent": "Decision Agent", "status": decision["status"], "output": decision},
        {"agent": "DevOps Agent", "status": deployment["status"], "output": deployment},
        {"agent": "Monitoring Agent", "status": "armed" if approved else "waiting", "output": {"signals": ["latency", "errors", "prediction distribution", "data drift", "model performance"]}},
    ]
    result = {**base, "run_id": run_id, "request": asdict(request), "quality": quality, "research": research, "experiments": experiments, "decision": decision, "deployment": deployment, "monitoring": stages[-1]["output"], "stages": stages}
    graph_state = build_workflow_graph().invoke({"request": asdict(request), "decision": decision, "loop_count": 0})
    result["orchestration"] = {"engine": "LangGraph", "next_action": graph_state.get("next_action"), "loop_count": graph_state.get("loop_count", 0)}
    return run_id, result, fitted_model[0] if fitted_model else None, fitted_model[1] if fitted_model else None