import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.orchestrator import RunRequest, run_workflow
from app.pipeline import evaluate_models, profile_dataset, recommend_models


def test_classification_profile_detects_missing_values():
    content = b"age,city,label\n20,London,yes\n,Paris,no\n31,,yes\n"
    profile = profile_dataset("customers.csv", content)
    assert profile.task == "classification"
    assert profile.target == "label"
    assert profile.column_profiles[0].missing == 1
    assert "Missing values detected" in profile.warnings[0]


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
    assert result["decision"]["best_model"] in {"RandomForestRegressor", "Ridge"}
    assert model is not None


def test_classification_evaluation_returns_metrics_and_confusion_matrix():
    rows = [f"value,label\n"] + [f"{index},{'yes' if index % 2 else 'no'}\n" for index in range(20)]
    content = "".join(rows).encode()
    profile = profile_dataset("labels.csv", content)
    evaluation = evaluate_models(profile, content)
    assert evaluation["status"] == "complete"
    assert evaluation["best_model"] in {"RandomForestClassifier", "LogisticRegression"}
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(evaluation["models"][0])


def test_no_target_is_unsupervised():
    profile = profile_dataset("events.csv", b"feature_a,feature_b\n1,2\n2,3\n")
    assert profile.task == "unsupervised"


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
    assert result["decision"]["status"] == "approved"
    assert result["deployment"]["status"] == "approved"
    assert model is not None
    assert "monthly_charges" in feature_names
    assert len(result["experiments"]) == 2


def test_metrics_endpoint_is_prometheus_compatible():
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "ai_agent_workflow_runs_total" in response.text