import json
from pathlib import Path

import pytest
from mlflow import MlflowClient
from fastapi.testclient import TestClient

from app.main import RUNS, app
from app.orchestrator import RunRequest, approve_workflow, run_workflow
from app.pipeline import evaluate_models, profile_dataset, recommend_models
from app.workflow_graph import _record_stage, get_checkpointed_state


@pytest.fixture(autouse=True)
def isolated_mlflow_store(tmp_path, monkeypatch):
    tracking_db = (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking_db}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", f"test-{tmp_path.name}")
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "monitoring.sqlite"))


def test_classification_profile_detects_missing_values():
    content = b"age,city,label\n20,London,yes\n,Paris,no\n31,,yes\n"
    profile = profile_dataset("customers.csv", content)
    assert profile.task == "classification"
    assert profile.target == "label"
    assert profile.column_profiles[0].missing == 1
    assert "Missing values detected" in profile.warnings[0]


def test_profile_counts_common_missing_value_tokens():
    profile = profile_dataset("missing.csv", b"feature,label\nNaN,yes\nNA,no\n")
    assert profile.column_profiles[0].missing == 2


def test_xml_profile_flattens_repeating_nested_records():
    content = b"""<dataset>
      <record><age>20</age><address><city>London</city></address><label>yes</label></record>
      <record><age>31</age><address><city>Paris</city></address><label>no</label></record>
    </dataset>"""

    profile = profile_dataset("customers.xml", content)

    assert profile.format == "xml"
    assert profile.rows == 2
    assert profile.target == "label"
    assert {column.name for column in profile.column_profiles} == {"age", "address.city", "label"}


def test_xml_parser_reports_malformed_documents():
    with pytest.raises(ValueError, match="Invalid XML dataset"):
        profile_dataset("broken.xml", b"<dataset><record>")


def test_xml_upload_runs_the_supervised_model_pipeline():
    records = "".join(
        f"<record><value>{index}</value><label>{'yes' if index % 2 else 'no'}</label></record>"
        for index in range(30)
    )
    content = f"<dataset>{records}</dataset>".encode()

    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("labels.xml", content, "application/xml")},
        data={"target": "label", "metric": "accuracy", "minimum_score": "1.1"},
    )

    assert response.status_code == 200
    assert response.json()["profile"]["format"] == "xml"
    assert response.json()["evaluation"]["status"] == "complete"


def test_json_regression_recommends_regressors():
    rows = [{"feature": index, "price": index * 2.5} for index in range(25)]
    profile = profile_dataset("prices.json", json.dumps(rows).encode())
    recommendation = recommend_models(profile)
    assert profile.task == "regression"
    assert "RandomForestRegressor" in recommendation["recommended"]


def test_house_price_target_is_detected_case_insensitively():
    content = b"sqft,bedrooms,SalePrice\n1200,2,220000\n1500,3,280000\n1800,4,350000\n"
    profile = profile_dataset("houses.csv", content, "saleprice")
    assert profile.target == "SalePrice"
    assert profile.task == "regression"


def test_house_price_workflow_selects_a_regressor():
    rows = ["sqft,bedrooms,SalePrice"] + [f"{1000 + index * 50},{2 + index % 3},{180000 + index * 12000}" for index in range(20)]
    content = ("\n".join(rows) + "\n").encode()
    run_id, result, model, _ = run_workflow("houses.csv", content, RunRequest("Predict house price", "r2", 0.0, "SalePrice"))
    assert run_id
    assert result["profile"]["task"] == "regression"
    assert result["evaluation"]["status"] == "complete"
    assert result["decision"]["best_model"] in {"RandomForestRegressor", "Ridge", "XGBRegressor"}
    assert result["deployment"]["status"] == "awaiting_approval"
    assert model is None
    approved_result, model, _ = approve_workflow(run_id, True, "test-reviewer")
    assert approved_result["deployment"]["status"] == "approved"
    assert model is not None
    Path(approved_result["deployment"]["artifact"]).unlink()


def test_classification_evaluation_returns_metrics_and_confusion_matrix():
    rows = [f"value,label\n"] + [f"{index},{'yes' if index % 2 else 'no'}\n" for index in range(20)]
    content = "".join(rows).encode()
    profile = profile_dataset("labels.csv", content)
    evaluation = evaluate_models(profile, content)
    assert evaluation["status"] == "complete"
    assert evaluation["best_model"] in {"RandomForestClassifier", "LogisticRegression", "XGBClassifier"}
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(evaluation["models"][0])
    assert len(evaluation["models"]) == 3


def test_requested_metric_and_cross_validation_control_model_selection():
    rows = ["value,label"] + [f"{index},{'yes' if index % 3 else 'no'}" for index in range(30)]
    content = ("\n".join(rows) + "\n").encode()
    profile = profile_dataset("labels.csv", content)

    evaluation = evaluate_models(profile, content, "accuracy", "cross_validation")

    assert evaluation["selection_metric"] == "accuracy"
    assert evaluation["validation"] == {"strategy": "cross_validation", "folds": 5}
    assert evaluation["best_model"] == max(evaluation["models"], key=lambda item: item["accuracy"])["model"]


def test_profile_detects_class_imbalance_and_possible_target_leakage():
    rows = ["feature,label,target_copy"]
    rows.extend(f"{index},{'yes' if index < 2 else 'no'},{'yes' if index < 2 else 'no'}" for index in range(20))
    profile = profile_dataset("imbalanced.csv", ("\n".join(rows) + "\n").encode())

    assert profile.task == "classification"
    assert any("Class imbalance detected" in warning for warning in profile.warnings)
    assert any("Potential target leakage detected" in warning for warning in profile.warnings)


def test_no_target_is_unsupervised():
    profile = profile_dataset("events.csv", b"feature_a,feature_b\n1,2\n2,3\n")
    assert profile.task == "unsupervised"


def test_no_target_workflow_blocks_deployment():
    content = b"feature_a,feature_b\n1,2\n2,3\n"
    _, result, model, _ = run_workflow("events.csv", content, RunRequest("Find groups", "f1", 0.8, None))
    assert result["decision"]["status"] == "needs_improvement"
    assert result["deployment"]["status"] == "blocked"
    assert result["tracking"]["status"] == "finished"
    client = MlflowClient(tracking_uri=result["tracking"]["tracking_uri"])
    assert client.get_run(result["tracking"]["run_id"]).info.status == "FINISHED"
    assert model is None


def test_duplicate_records_fail_the_quality_gate():
    rows = ["value,label"] + [f"{index},{'yes' if index % 2 else 'no'}" for index in range(10)]
    rows.append(rows[1])
    content = ("\n".join(rows) + "\n").encode()
    _, result, model, _ = run_workflow("duplicates.csv", content, RunRequest("Predict labels", "f1", 0.0, "label"))
    assert result["quality"]["duplicates"] == 1
    assert result["decision"]["status"] == "needs_improvement"
    assert result["deployment"]["status"] == "blocked"
    assert model is None


def test_coco_json_is_not_treated_as_one_tabular_row():
    content = json.dumps({
        "images": [{"id": 1, "file_name": "one.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
        "categories": [{"id": 1, "name": "cat"}],
    }).encode()
    profile = profile_dataset("annotations_coco.json", content)
    assert profile.task == "computer_vision"
    assert profile.rows == 1
    assert "COCO annotations detected" in profile.warnings[0]


def test_orchestrator_runs_quality_gate_and_prepares_prediction_model():
    content = Path("data/customer_churn_demo.csv").read_bytes()
    run_id, result, model, feature_names = run_workflow(
        "customer_churn_demo.csv",
        content,
        RunRequest("Predict churn", "f1", 0.8, "churn"),
    )
    assert run_id
    assert result["decision"]["status"] == "awaiting_approval"
    assert result["deployment"]["status"] == "awaiting_approval"
    assert model is None
    assert "monthly_charges" in feature_names
    assert len(result["experiments"]) == 3
    assert [stage["agent"] for stage in result["stages"][:5]] == [
        "Data Profiling Agent",
        "Preprocessing Agent",
        "Model Training Agent",
        "Evaluation Agent",
        "Decision Agent",
    ]
    assert result["orchestration"]["checkpointed"] is True
    assert result["tracking"]["status"] == "tracking"
    assert result["tracking"]["dataset_version"].startswith("sha256:")
    assert len(result["tracking"]["candidate_runs"]) == 3
    assert len(result["tracking"]["comparison"]) == 3
    mlflow_client = MlflowClient(tracking_uri=result["tracking"]["tracking_uri"])
    parent_run = mlflow_client.get_run(result["tracking"]["run_id"])
    assert parent_run.data.params["random_seed"] == "42"
    assert parent_run.data.params["dataset_version"] == result["tracking"]["dataset_version"]
    candidate_runs = mlflow_client.search_runs(
        [result["tracking"]["experiment_id"]],
        filter_string=f"tags.mlflow.parentRunId = '{result['tracking']['run_id']}'",
    )
    assert len(candidate_runs) == 3
    assert all(run.data.metrics for run in candidate_runs)
    assert [event["stage"] for event in result["execution_log"]] == [
        "data_profiling",
        "preprocessing",
        "model_training",
        "evaluation",
        "decision",
        "deployment_preparation",
    ]
    assert get_checkpointed_state(run_id) is not None

    approved_result, approved_model, approved_features = approve_workflow(run_id, True, "test-reviewer")
    assert approved_result["deployment"]["status"] == "approved"
    assert approved_result["deployment_approval"]["status"] == "approved"
    assert approved_model is not None
    assert approved_features == feature_names
    assert approved_result["tracking"]["status"] == "complete"
    assert int(approved_result["tracking"]["model_version"]) >= 1
    assert approved_result["deployment"]["registry_model"] == approved_result["tracking"]["model_name"]
    assert approved_result["deployment"]["registered_model_uri"].startswith("models:/")
    model_version = mlflow_client.get_model_version(
        approved_result["tracking"]["model_name"],
        approved_result["tracking"]["model_version"],
    )
    assert model_version.tags["modelforge.dataset_version"] == result["tracking"]["dataset_version"]
    assert model_version.tags["modelforge.workflow_run_id"] == run_id
    from app.experiment_tracking import register_deployed_model

    repeated_registration = register_deployed_model(
        approved_result["tracking"],
        approved_model,
        {**result["profile"], "run_id": run_id},
        result["decision"],
        approved_result["deployment_approval"],
    )
    assert repeated_registration["model_version"] == approved_result["tracking"]["model_version"]
    Path(approved_result["deployment"]["artifact"]).unlink()


def test_rejected_deployment_finishes_mlflow_run_without_registering_model():
    rows = ["value,label"] + [f"{index},{'yes' if index % 2 else 'no'}" for index in range(12)]
    content = ("\n".join(rows) + "\n").encode()
    run_id, result, _, _ = run_workflow(
        "rejected.csv",
        content,
        RunRequest("Predict labels", "f1", 0.0, "label"),
    )

    rejected_result, model, _ = approve_workflow(run_id, False, "test-reviewer")

    assert model is None
    assert rejected_result["deployment"]["status"] == "rejected"
    assert rejected_result["tracking"]["status"] == "rejected"
    client = MlflowClient(tracking_uri=result["tracking"]["tracking_uri"])
    parent_run = client.get_run(result["tracking"]["run_id"])
    assert parent_run.info.status == "FINISHED"
    assert parent_run.data.tags["modelforge.deployment_approval"] == "rejected"


def test_workflow_stage_retries_are_bounded_and_logged():
    attempts = 0

    def flaky_stage():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return {"value": "ready"}

    output = _record_stage(
        {"request": {"max_retries": 1}, "execution_log": [], "node_outputs": {}, "errors": []},
        "test_stage",
        flaky_stage,
    )

    assert attempts == 2
    assert output["node_outputs"]["test_stage"]["attempts"] == 2
    assert [event["status"] for event in output["execution_log"]] == ["retrying", "complete"]
    assert output["errors"] == []


def test_failed_workflow_stage_returns_structured_error():
    output = _record_stage(
        {"request": {"max_retries": 1}, "execution_log": [], "node_outputs": {}, "errors": []},
        "test_stage",
        lambda: (_ for _ in ()).throw(ValueError("invalid input")),
    )

    assert output["workflow_status"] == "failed"
    assert output["node_outputs"]["test_stage"]["status"] == "failed"
    assert output["errors"] == [{"stage": "test_stage", "error": "ValueError: invalid input"}]


def test_graph_surfaces_node_failure_after_retry_limit():
    content = b"feature,label\n1,yes\n2,no\n"

    _, result, model, _ = run_workflow(
        "bad-target.csv",
        content,
        RunRequest("Predict", "f1", 0.0, "missing_target", max_retries=1),
    )

    assert result["decision"]["status"] == "failed"
    assert result["errors"][0]["stage"] == "data_profiling"
    assert [entry["status"] for entry in result["execution_log"]] == ["retrying", "failed"]
    assert model is None


def test_metrics_endpoint_is_prometheus_compatible():
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "ai_agent_workflow_runs_total" in response.text


def test_analyze_endpoint_accepts_validation_strategy_and_metric():
    content = ("\n".join(["value,label"] + [f"{index},{'yes' if index % 2 else 'no'}" for index in range(20)]) + "\n").encode()

    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("labels.csv", content, "text/csv")},
        data={"target": "label", "metric": "accuracy", "minimum_score": "1.1", "validation_strategy": "cross_validation"},
    )

    assert response.status_code == 200
    assert response.json()["request"]["validation_strategy"] == "cross_validation"
    assert response.json()["evaluation"]["selection_metric"] == "accuracy"
    assert response.json()["evaluation"]["validation"]["strategy"] == "cross_validation"


def test_profile_endpoint_suggests_target_and_returns_columns():
    content = b"value,label\n1,yes\n2,no\n"

    response = TestClient(app).post(
        "/api/profile",
        files={"file": ("labels.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["candidate_target"] == "label"
    assert {column["name"] for column in response.json()["column_profiles"]} == {"value", "label"}


def test_analyze_requires_confirmation_of_inferred_target():
    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("labels.csv", b"value,label\n1,yes\n2,no\n", "text/csv")},
    )

    assert response.status_code == 409
    assert "Confirm it" in response.json()["detail"]


def test_analyze_can_confirm_profile_only_when_target_is_not_selected():
    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("labels.csv", b"value,label\n1,yes\n2,no\n", "text/csv")},
        data={"target_confirmed": "true"},
    )

    assert response.status_code == 200
    assert response.json()["profile"]["target"] is None
    assert response.json()["profile"]["task"] == "unsupervised"


def test_approval_endpoint_resumes_checkpoint_before_deployment():
    content = Path("data/customer_churn_demo.csv").read_bytes()
    client = TestClient(app)
    analysis = client.post(
        "/api/analyze",
        files={"file": ("customer_churn_demo.csv", content, "text/csv")},
        data={"target": "churn", "metric": "f1", "minimum_score": "0.0"},
    )
    assert analysis.status_code == 200
    run_id = analysis.json()["run_id"]
    assert analysis.json()["deployment"]["status"] == "awaiting_approval"
    RUNS.pop(run_id)

    approval = client.post(f"/api/runs/{run_id}/approve", json={"approved": True, "reviewer": "test-reviewer"})

    assert approval.status_code == 200
    assert approval.json()["deployment"]["status"] == "approved"
    assert approval.json()["deployment_approval"]["reviewer"] == "test-reviewer"
    Path(approval.json()["deployment"]["artifact"]).unlink()


def test_monitoring_detects_feature_drift_and_tracks_delayed_labels():
    from app.monitoring import initialize_monitoring, monitoring_status, record_feedback, record_prediction

    rows = ["feature,label"] + [f"{index},{'yes' if index % 2 else 'no'}" for index in range(100)]
    initialize_monitoring(
        "monitoring-test",
        "training.csv",
        ("\n".join(rows) + "\n").encode(),
        ["feature"],
        "classification",
        "accuracy",
        0.9,
    )
    prediction_ids = [
        record_prediction("monitoring-test", {"feature": 1000}, "no")[0]
        for _ in range(30)
    ]
    feedback = record_feedback("monitoring-test", prediction_ids[0], "yes")
    status = monitoring_status("monitoring-test")

    assert status["drift"]["score"] >= 0.2
    assert status["drift"]["status"] == "drift_detected"
    assert feedback["performance"]["accuracy"] == 0.0
    assert feedback["performance_status"] == "degraded"
    assert status["performance"]["feedback_count"] == 1
    with pytest.raises(ValueError, match="already submitted"):
        record_feedback("monitoring-test", prediction_ids[0], "yes")


def test_prediction_endpoint_exposes_feedback_metrics_and_prometheus_signals():
    from app.monitoring import initialize_monitoring
    from app.metrics import metrics_payload

    class FixedClassifier:
        def predict(self, rows):
            return ["yes"]

    rows = ["feature,label"] + [f"{index},{'yes' if index % 2 else 'no'}" for index in range(40)]
    initialize_monitoring(
        "api-monitoring-test",
        "training.csv",
        ("\n".join(rows) + "\n").encode(),
        ["feature"],
        "classification",
        "accuracy",
        0.9,
    )
    RUNS["api-monitoring-test"] = {
        "result": {"monitoring": {"signals": []}},
        "model": FixedClassifier(),
        "feature_names": ["feature"],
        "profile": {
            "task": "classification",
            "column_profiles": [{"name": "feature", "data_type": "numeric"}],
        },
    }
    client = TestClient(app)
    try:
        prediction = client.post("/api/runs/api-monitoring-test/predict", json={"feature": 1000})
        assert prediction.status_code == 200
        prediction_data = prediction.json()
        assert prediction_data["prediction_id"]

        feedback = client.post(
            "/api/runs/api-monitoring-test/feedback",
            json={"prediction_id": prediction_data["prediction_id"], "actual": "no"},
        )
        assert feedback.status_code == 200
        monitoring = client.get("/api/monitoring/api-monitoring-test")
        assert monitoring.status_code == 200
        assert monitoring.json()["performance"]["metrics"]["accuracy"] == 0.0

        prometheus_metrics = client.get("/metrics").text
        assert 'ai_agent_predictions_total{task="classification"}' in prometheus_metrics
        assert 'ai_agent_prediction_latency_seconds_count{task="classification"}' in prometheus_metrics
        assert 'ai_agent_model_performance_degradation{metric="accuracy",task="classification"}' in prometheus_metrics
    finally:
        RUNS.pop("api-monitoring-test", None)