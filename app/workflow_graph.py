"""Checkpointed, deterministic LangGraph workflow for tabular ML runs."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class WorkflowState(TypedDict, total=False):
    run_id: str
    filename: str
    content: bytes
    request: dict[str, Any]
    profile: dict[str, Any]
    quality: dict[str, Any]
    recommendation: dict[str, Any]
    research: dict[str, Any]
    methods: dict[str, Any]
    prepared: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    decision: dict[str, Any]
    deployment: dict[str, Any]
    deployment_preparation: dict[str, Any]
    deployment_approval: dict[str, Any]
    tracking: dict[str, Any]
    feature_names: list[str]
    execution_log: list[dict[str, Any]]
    node_outputs: dict[str, dict[str, Any]]
    errors: list[dict[str, str]]
    workflow_status: str


_checkpointer: SqliteSaver | None = None
_checkpointer_connection: sqlite3.Connection | None = None


def _get_checkpointer() -> SqliteSaver:
    global _checkpointer, _checkpointer_connection
    if _checkpointer is None:
        checkpoint_path = Path(os.environ.get("LANGGRAPH_CHECKPOINT_PATH", "artifacts/workflow-checkpoints.sqlite"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer_connection = sqlite3.connect(checkpoint_path, check_same_thread=False, timeout=30)
        _checkpointer_connection.execute("PRAGMA journal_mode=WAL")
        _checkpointer_connection.execute("PRAGMA busy_timeout=30000")
        _checkpointer = SqliteSaver(_checkpointer_connection)
        _checkpointer.setup()
    return _checkpointer


def _record_stage(
    state: WorkflowState,
    stage: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    request = state.get("request", {})
    max_attempts = max(1, min(int(request.get("max_retries", 2)) + 1, 6))
    execution_log = list(state.get("execution_log", []))
    node_outputs = dict(state.get("node_outputs", {}))
    errors = list(state.get("errors", []))
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            output = operation()
        except Exception as error:
            execution_log.append({
                "stage": stage,
                "attempt": attempt,
                "status": "retrying" if attempt < max_attempts else "failed",
                "error": f"{type(error).__name__}: {error}",
                "timestamp": timestamp,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            if attempt == max_attempts:
                error_detail = {"stage": stage, "error": f"{type(error).__name__}: {error}"}
                errors.append(error_detail)
                node_outputs[stage] = {"status": "failed", "attempts": attempt, "error": error_detail["error"]}
                return {
                    "execution_log": execution_log,
                    "node_outputs": node_outputs,
                    "errors": errors,
                    "workflow_status": "failed",
                }
        else:
            execution_log.append({
                "stage": stage,
                "attempt": attempt,
                "status": "complete",
                "timestamp": timestamp,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            node_outputs[stage] = {"status": "complete", "attempts": attempt, "output": output}
            return {
                **output,
                "execution_log": execution_log,
                "node_outputs": node_outputs,
                "errors": errors,
            }

    raise RuntimeError(f"Stage '{stage}' exited its bounded retry loop unexpectedly.")


def data_profiling_node(state: WorkflowState) -> dict[str, Any]:
    def profile() -> dict[str, Any]:
        from app.orchestrator import _quality_report, _research_plan
        from app.pipeline import profile_dataset, recommend_models
        from app.research import research_methods

        dataset_profile = profile_dataset(
            state["filename"],
            state["content"],
            state["request"].get("target"),
            infer_target=not state["request"].get("target_confirmed", False),
        )
        quality = _quality_report(dataset_profile, state["content"])
        recommendation = recommend_models(dataset_profile)
        research = _research_plan(dataset_profile, recommendation)
        from app.experiment_tracking import start_experiment_run

        tracking = start_experiment_run(
            state["run_id"],
            state["filename"],
            state["content"],
            state["request"],
            {
                "task": dataset_profile.task,
                "target": dataset_profile.target,
                "rows": dataset_profile.rows,
                "columns": dataset_profile.columns,
                "column_profiles": [item.__dict__ for item in dataset_profile.column_profiles],
            },
        )
        return {
            "profile": {
                "filename": dataset_profile.filename,
                "format": dataset_profile.format,
                "rows": dataset_profile.rows,
                "columns": dataset_profile.columns,
                "target": dataset_profile.target,
                "task": dataset_profile.task,
                "warnings": dataset_profile.warnings,
                "column_profiles": [item.__dict__ for item in dataset_profile.column_profiles],
            },
            "quality": quality,
            "recommendation": recommendation,
            "research": research,
            "methods": research_methods(dataset_profile.task, recommendation["why"]),
            "tracking": tracking,
        }

    return _record_stage(state, "data_profiling", profile)


def _dataset_profile(profile: dict[str, Any]) -> Any:
    from app.pipeline import ColumnProfile, DatasetProfile

    return DatasetProfile(
        filename=profile["filename"],
        format=profile["format"],
        rows=profile["rows"],
        columns=profile["columns"],
        target=profile["target"],
        task=profile["task"],
        warnings=profile["warnings"],
        column_profiles=[ColumnProfile(**column) for column in profile["column_profiles"]],
    )


def preprocessing_node(state: WorkflowState) -> dict[str, Any]:
    def preprocess() -> dict[str, Any]:
        from app.pipeline import prepare_training_data

        prepared = prepare_training_data(
            _dataset_profile(state["profile"]),
            state["content"],
            state["request"].get("validation_strategy", "holdout"),
        )
        return {"prepared": prepared}

    return _record_stage(state, "preprocessing", preprocess)


def model_training_node(state: WorkflowState) -> dict[str, Any]:
    def train() -> dict[str, Any]:
        from app.pipeline import train_candidate_models

        prepared = state["prepared"]
        if prepared["status"] != "ready":
            return {"training": {"status": "unavailable", "reason": prepared["reason"]}}
        training = train_candidate_models(_dataset_profile(state["profile"]), prepared)
        from app.experiment_tracking import log_candidate_training

        return {
            "training": training,
            "tracking": log_candidate_training(state["tracking"], training),
        }

    return _record_stage(state, "model_training", train)


def evaluation_node(state: WorkflowState) -> dict[str, Any]:
    def evaluate() -> dict[str, Any]:
        from app.pipeline import evaluate_predictions

        training = state["training"]
        if training["status"] != "complete":
            return {"evaluation": {"status": "unavailable", "reason": training.get("reason", "Model training did not complete.")}}
        evaluation = evaluate_predictions(
            _dataset_profile(state["profile"]),
            state["prepared"],
            training,
            state["request"].get("metric", "f1"),
        )
        from app.experiment_tracking import log_evaluation

        tracking = log_evaluation(state["tracking"], evaluation)
        return {"evaluation": evaluation, "tracking": tracking}

    return _record_stage(state, "evaluation", evaluate)


def decision_node(state: WorkflowState) -> dict[str, Any]:
    def decide() -> dict[str, Any]:
        evaluation = state["evaluation"]
        request = state["request"]
        quality = state["quality"]
        if evaluation.get("status") != "complete":
            decision = {
                "status": "needs_improvement",
                "best_model": None,
                "metric": request.get("metric"),
                "best_score": None,
                "threshold": request.get("minimum_score"),
                "next_step": evaluation.get("reason", "Evaluation is unavailable."),
            }
            deployment = {"status": "blocked", "model": None, "artifact": None, "reason": decision["next_step"]}
        else:
            metric = evaluation["selection_metric"]
            best_model = evaluation["best_model"]
            best_result = next(model for model in evaluation["models"] if model["model"] == best_model)
            best_score = float(best_result[metric])
            lower_is_better = metric in {"mae", "rmse"}
            threshold = request.get("minimum_score")
            threshold_met = threshold is None or (best_score <= threshold if lower_is_better else best_score >= threshold)
            quality_passed = quality["duplicates"] == 0
            ready = threshold_met and quality_passed
            decision = {
                "status": "awaiting_approval" if ready else "needs_improvement",
                "best_model": best_model,
                "metric": metric,
                "requested_metric": request.get("metric"),
                "best_score": best_score,
                "threshold": threshold,
                "next_step": "Review the evaluation and explicitly approve deployment." if ready else "Quality or performance gates failed; deployment is blocked.",
            }
            deployment = {
                "status": "awaiting_approval" if ready else "blocked",
                "model": best_model,
                "artifact": None,
                "reason": "Human approval is required before saving a deployable model." if ready else "Performance threshold or data-quality requirements are not satisfied.",
                "approval_required": ready,
            }
        return {"decision": decision, "deployment": deployment}

    return _record_stage(state, "decision", decide)


def deployment_preparation_node(state: WorkflowState) -> dict[str, Any]:
    def prepare() -> dict[str, Any]:
        deployment = state.get("deployment", {})
        output = {
            "status": deployment.get("status", "blocked"),
            "model": deployment.get("model"),
            "artifact": deployment.get("artifact"),
            "approval_required": deployment.get("approval_required", False),
            "reason": deployment.get("reason"),
        }
        return {"deployment_preparation": output}

    return _record_stage(state, "deployment_preparation", prepare)


def approval_node(state: WorkflowState) -> dict[str, Any]:
    deployment = state.get("deployment", {})
    if deployment.get("status") != "awaiting_approval":
        return {"deployment_approval": {"status": "not_required", "approved": False}}
    response = interrupt({
        "run_id": state["run_id"],
        "model": deployment.get("model"),
        "decision": state.get("decision"),
        "evaluation": state.get("evaluation"),
        "quality": state.get("quality"),
        "message": "Explicit human approval is required before deployment preparation.",
    })
    approved = bool(response.get("approved")) if isinstance(response, dict) else bool(response)
    reviewer = response.get("reviewer") if isinstance(response, dict) else None
    status = "approved" if approved else "rejected"
    approval = {"status": status, "approved": approved, "reviewer": reviewer}
    logs = list(state.get("execution_log", []))
    outputs = dict(state.get("node_outputs", {}))

    def record_approval() -> dict[str, Any]:
        from app.experiment_tracking import log_approval

        log_approval(state["tracking"], approved, reviewer)
        return {"deployment_approval": approval}

    recorded = _record_stage(state, "deployment_approval", record_approval)
    outputs.update(recorded.get("node_outputs", {}))
    return {
        **recorded,
        "deployment_approval": approval,
        "node_outputs": outputs,
        "workflow_status": recorded.get("workflow_status", state.get("workflow_status", "running")),
    }


def deployment_node(state: WorkflowState) -> dict[str, Any]:
    def deploy() -> dict[str, Any]:
        approval = state.get("deployment_approval", {})
        if not approval.get("approved"):
            from app.experiment_tracking import finish_experiment_run, log_approval

            if approval.get("status") == "rejected":
                log_approval(state["tracking"], False, approval.get("reviewer"))
            tracking = {**state["tracking"], "status": "rejected" if approval.get("status") == "rejected" else "finished"}
            finish_experiment_run(tracking, "FINISHED")
            return {
                "deployment": {
                    **state.get("deployment", {}),
                    "status": "rejected" if approval.get("status") == "rejected" else "blocked",
                    "artifact": None,
                    "reason": "Deployment was not approved by a reviewer.",
                },
                "tracking": tracking,
                "workflow_status": "complete",
            }
        from app.orchestrator import _fit_for_deployment
        import joblib

        profile = _dataset_profile(state["profile"])
        fitted_model = _fit_for_deployment(profile, state["content"], state["decision"]["best_model"])
        if fitted_model is None:
            raise RuntimeError("The winning model could not be fitted for deployment.")
        artifact_path = Path("artifacts") / f"{state['run_id']}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = artifact_path.with_suffix(".joblib.tmp")
        try:
            joblib.dump({
                "model": fitted_model[0],
                "feature_names": fitted_model[1],
                "profile": state["profile"],
            }, temporary_path)
            temporary_path.replace(artifact_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        deployment = {
            **state["deployment"],
            "status": "approved",
            "artifact": str(artifact_path),
            "reason": "Reviewer approval received; deployable model artifact prepared.",
        }
        from app.experiment_tracking import register_deployed_model

        tracking = register_deployed_model(
            state["tracking"],
            fitted_model[0],
            {**state["profile"], "run_id": state["run_id"]},
            state["decision"],
            approval,
        )
        deployment.update({
            "registry_model": tracking["model_name"],
            "model_version": tracking["model_version"],
            "registered_model_uri": tracking["registered_model_uri"],
        })
        return {
            "deployment": deployment,
            "tracking": tracking,
            "feature_names": fitted_model[1],
            "workflow_status": "complete",
        }

    return _record_stage(state, "deployment", deploy)


def _route_on_error(state: WorkflowState) -> str:
    return "failed" if state.get("workflow_status") == "failed" else "continue"


def _route_after_approval(state: WorkflowState) -> str:
    return "failed" if state.get("workflow_status") == "failed" else "deployment"


def build_workflow_graph(checkpointer: Any = None):
    graph = StateGraph(WorkflowState)
    graph.add_node("data_profiling", data_profiling_node)
    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("model_training", model_training_node)
    graph.add_node("evaluate_models", evaluation_node)
    graph.add_node("apply_decision", decision_node)
    graph.add_node("prepare_deployment", deployment_preparation_node)
    graph.add_node("approval", approval_node)
    graph.add_node("prepare_artifact", deployment_node)
    graph.add_node("workflow_failed", workflow_failed_node)

    graph.add_edge(START, "data_profiling")
    graph.add_conditional_edges("data_profiling", _route_on_error, {"continue": "preprocessing", "failed": "workflow_failed"})
    graph.add_conditional_edges("preprocessing", _route_on_error, {"continue": "model_training", "failed": "workflow_failed"})
    graph.add_conditional_edges("model_training", _route_on_error, {"continue": "evaluate_models", "failed": "workflow_failed"})
    graph.add_conditional_edges("evaluate_models", _route_on_error, {"continue": "apply_decision", "failed": "workflow_failed"})
    graph.add_edge("apply_decision", "prepare_deployment")
    graph.add_conditional_edges("prepare_deployment", _route_on_error, {"continue": "approval", "failed": "workflow_failed"})
    graph.add_conditional_edges("approval", _route_after_approval, {"deployment": "prepare_artifact", "failed": "workflow_failed"})
    graph.add_conditional_edges("prepare_artifact", _route_on_error, {"continue": END, "failed": "workflow_failed"})
    graph.add_edge("workflow_failed", END)
    return graph.compile(checkpointer=checkpointer or _get_checkpointer())


def workflow_failed_node(state: WorkflowState) -> dict[str, Any]:
    tracking = state.get("tracking", {})
    if tracking.get("run_id"):
        from app.experiment_tracking import finish_experiment_run

        finish_experiment_run(tracking, "FAILED")
    return {"workflow_status": "failed"}


def resume_workflow(graph: Any, run_id: str, approved: bool, reviewer: str | None = None) -> WorkflowState:
    config = {"configurable": {"thread_id": run_id}}
    return graph.invoke(
        Command(resume={"approved": approved, "reviewer": reviewer}),
        config,
    )


def get_checkpointed_state(run_id: str) -> WorkflowState | None:
    snapshot = build_workflow_graph().get_state({"configurable": {"thread_id": run_id}})
    return snapshot.values or None
