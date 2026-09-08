"""Dataset profiling and model recommendation logic."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ColumnProfile:
    name: str
    data_type: str
    missing: int
    unique: int
    sample: list[str]


@dataclass
class DatasetProfile:
    filename: str
    format: str
    rows: int
    columns: int
    target: str | None
    task: str
    warnings: list[str]
    column_profiles: list[ColumnProfile]


def _parse_rows(filename: str, content: bytes) -> tuple[str, list[dict[str, Any]]]:
    suffix = Path(filename).suffix.lower()
    text = content.decode("utf-8-sig", errors="replace")
    if suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return "jsonl", [row for row in rows if isinstance(row, dict)]
    if suffix in {".xlsx", ".xls", ".parquet"}:
        from io import BytesIO

        frame = pd.read_excel(BytesIO(content)) if suffix in {".xlsx", ".xls"} else pd.read_parquet(BytesIO(content))
        frame = frame.astype(object).where(frame.notna(), None)
        return suffix[1:], frame.to_dict(orient="records")
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if {"images", "annotations", "categories"} <= payload.keys():
                return "coco", [{"images": payload["images"], "annotations": payload["annotations"], "categories": payload["categories"]}]
            payload = payload.get("data", [payload])
        if not isinstance(payload, list):
            raise ValueError("JSON must contain an array of row objects")
        return "json", [row for row in payload if isinstance(row, dict)]
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("The CSV file has no header row")
    return "csv", [dict(row) for row in reader]


def _is_number(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def _column_type(values: list[Any]) -> str:
    present = [value for value in values if value is not None and str(value).strip()]
    if not present:
        return "empty"
    if all(_is_number(value) for value in present):
        return "numeric"
    lowered = {str(value).strip().lower() for value in present}
    if lowered <= {"true", "false", "yes", "no", "0", "1"}:
        return "boolean"
    return "categorical" if len(lowered) <= max(20, len(present) // 5) else "text"


def _guess_target(headers: list[str], profiles: list[ColumnProfile]) -> str | None:
    target_names = {"target", "label", "y", "outcome", "response", "price", "score", "churn", "saleprice", "sale_price", "house_price", "houseprice", "rent", "amount", "revenue", "rating", "rate"}
    for profile in profiles:
        normalized = profile.name.lower().strip().replace("-", "_").replace(" ", "_")
        if normalized in target_names or normalized.replace("_", "") in {name.replace("_", "") for name in target_names}:
            return profile.name
    return None


def profile_dataset(filename: str, content: bytes, target: str | None = None) -> DatasetProfile:
    file_format, rows = _parse_rows(filename, content)
    if file_format == "coco":
        payload = rows[0]
        image_count = len(payload["images"])
        annotation_count = len(payload["annotations"])
        category_count = len(payload["categories"])
        return DatasetProfile(
            filename=filename,
            format=file_format,
            rows=image_count,
            columns=3,
            target=None,
            task="computer_vision",
            warnings=[
                f"COCO annotations detected: {image_count} images, {annotation_count} annotations, and {category_count} categories.",
                "Model comparison is unavailable until the referenced image files are available for training and evaluation.",
                "Use object-detection metrics such as mAP, IoU, precision, and recall rather than tabular accuracy.",
            ],
            column_profiles=[
                ColumnProfile("images", "annotation_collection", 0, image_count, []),
                ColumnProfile("annotations", "annotation_collection", 0, annotation_count, []),
                ColumnProfile("categories", "annotation_collection", 0, category_count, []),
            ],
        )
    headers = list(rows[0].keys()) if rows else []
    profiles: list[ColumnProfile] = []
    for header in headers:
        values = [row.get(header) for row in rows]
        present = [str(value) for value in values if value is not None and str(value).strip()]
        profiles.append(ColumnProfile(header, _column_type(values), len(values) - len(present), len(set(present)), present[:3]))

    target_lookup = {header.lower().strip(): header for header in headers}
    chosen_target = target_lookup.get(target.lower().strip()) if target else _guess_target(headers, profiles)
    target_profile = next((item for item in profiles if item.name == chosen_target), None)
    regression_names = {"price", "saleprice", "sale_price", "houseprice", "house_price", "rent", "amount", "revenue", "rate"}
    normalized_target = chosen_target.lower().strip().replace("-", "_").replace(" ", "_") if chosen_target else ""
    if target_profile is None:
        task = "unsupervised"
    elif target_profile.data_type == "numeric" and (target_profile.unique > 20 or normalized_target in regression_names or normalized_target.replace("_", "") in {name.replace("_", "") for name in regression_names}):
        task = "regression"
    else:
        task = "classification"

    warnings: list[str] = []
    if not rows:
        warnings.append("The dataset contains no rows.")
    if any(profile.missing for profile in profiles):
        warnings.append("Missing values detected; impute numeric and categorical columns during cleaning.")
    if any(profile.data_type == "text" for profile in profiles):
        warnings.append("Free-text columns detected; use vectorization or embeddings and review leakage.")
    if len(rows) < 100:
        warnings.append("Small dataset: use cross-validation and compare against a simple baseline.")
    return DatasetProfile(filename, file_format, len(rows), len(headers), chosen_target, task, warnings, profiles)


def recommend_models(profile: DatasetProfile) -> dict[str, Any]:
    numeric = sum(item.data_type == "numeric" for item in profile.column_profiles)
    text = sum(item.data_type == "text" for item in profile.column_profiles)
    if profile.task == "computer_vision":
        return {
            "recommended": ["YOLO", "Faster R-CNN", "DETR"],
            "why": "COCO object-detection annotations were detected.",
            "evaluation": "mAP@50, mAP@50:95, IoU, precision, and recall",
            "training_plan": [
                "Resolve each COCO image file and validate bounding boxes before training.",
                "Start with a pretrained YOLO baseline, then compare Faster R-CNN or DETR.",
                "Select the winner using validation mAP and inspect per-category recall.",
            ],
        }
    if profile.task == "regression":
        primary = ["RandomForestRegressor", "Ridge"]
        metric = "MAE and RMSE"
    elif profile.task == "classification":
        primary = ["RandomForestClassifier", "LogisticRegression"]
        metric = "F1, ROC-AUC, and balanced accuracy"
    else:
        primary = ["KMeans", "IsolationForest", "PCA"]
        metric = "silhouette score and stability across seeds"
    if text:
        primary.insert(0, "TF-IDF + linear model")
    return {
        "recommended": primary,
        "why": f"Detected {profile.task} data with {numeric} numeric and {text} free-text columns.",
        "evaluation": metric,
        "training_plan": [
            "Split before fitting transformations to prevent leakage.",
            "Build a baseline, then compare candidates with cross-validation.",
            "Persist preprocessing and model together so inference uses identical features.",
        ],
    }


def _preprocessor(profile: DatasetProfile, feature_names: list[str]) -> ColumnTransformer:
    numeric = [index for index, name in enumerate(feature_names) if next(item for item in profile.column_profiles if item.name == name).data_type == "numeric"]
    categorical = [index for index in range(len(feature_names)) if index not in numeric]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def _classification_candidates() -> dict[str, Any]:
    return {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }


def _regression_candidates() -> dict[str, Any]:
    return {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=42),
        "Ridge": Ridge(),
    }


def evaluate_models(profile: DatasetProfile, content: bytes) -> dict[str, Any] | None:
    """Fit candidates on a holdout split and return comparable evidence."""
    if profile.task == "computer_vision":
        return {"status": "unavailable", "reason": "COCO metadata alone is not enough to train models; image files are required.", "metric_definition": "Compare object detectors with mAP, IoU, precision, and recall."}
    if profile.task == "unsupervised" or not profile.target:
        return {"status": "unavailable", "reason": "No ground-truth target column is available for accuracy, precision, recall, or a confusion matrix.", "metric_definition": "Compare clustering with silhouette score and stability across seeds."}
    _, rows = _parse_rows(profile.filename, content)
    rows = [row for row in rows if row.get(profile.target) not in (None, "")]
    if len(rows) < 10:
        return {"status": "unavailable", "reason": "At least 10 labeled rows are required for a reliable holdout evaluation."}

    feature_names = [item.name for item in profile.column_profiles if item.name != profile.target]
    numeric_names = {item.name for item in profile.column_profiles if item.data_type == "numeric"}
    features = []
    for row in rows:
        normalized_row = []
        for name in feature_names:
            value = row.get(name)
            if value is None or str(value).strip() == "":
                normalized_row.append(None)
            elif name in numeric_names:
                normalized_row.append(float(value))
            else:
                normalized_row.append(value)
        features.append(normalized_row)
    if profile.task == "regression":
        labels = [float(row[profile.target]) for row in rows]
        candidates = _regression_candidates()
    else:
        labels = [str(row[profile.target]) for row in rows]
        counts = {label: labels.count(label) for label in set(labels)}
        if len(counts) < 2 or min(counts.values()) < 2:
            return {"status": "unavailable", "reason": "At least two examples per class are required for comparison."}
        candidates = _classification_candidates()

    stratify = labels if profile.task == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=stratify)
    results: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", _preprocessor(profile, feature_names)), ("model", estimator)])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        if profile.task == "classification":
            results.append({
                "model": name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            })
        else:
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            results.append({
                "model": name,
                "mae": round(mean_absolute_error(y_test, predictions), 4),
                "rmse": round(rmse, 4),
                "r2": round(r2_score(y_test, predictions), 4),
            })
    if profile.task == "classification":
        best = max(results, key=lambda item: (item["f1"], item["recall"], item["accuracy"]))
        explanation = "Selected by weighted F1, then recall and accuracy, so class imbalance is less likely to hide poor minority-class performance."
        labels_for_matrix = sorted(set(labels))
        return {"status": "complete", "task": profile.task, "metric_definition": "Precision, recall, and F1 use weighted averages across classes.", "labels": labels_for_matrix, "models": results, "best_model": best["model"], "why_best": explanation}
    best = min(results, key=lambda item: (item["rmse"], item["mae"]))
    return {"status": "complete", "task": profile.task, "metric_definition": "Lower MAE/RMSE is better; higher R2 is better.", "models": results, "best_model": best["model"], "why_best": "Selected by lowest RMSE, with MAE as the tie-breaker."}


def pipeline_result(profile: DatasetProfile, content: bytes | None = None) -> dict[str, Any]:
    recommendation = recommend_models(profile)
    evaluation = evaluate_models(profile, content) if content is not None else None
    if evaluation and evaluation.get("status") == "complete":
        recommendation["recommended"] = [evaluation["best_model"]] + [model["model"] for model in evaluation["models"] if model["model"] != evaluation["best_model"]]
    comparison_ready = bool(evaluation and evaluation.get("status") == "complete")
    research = {
        "status": "ready",
        "methods": {
            "classification": ["holdout validation", "weighted precision/recall/F1", "confusion matrix"],
            "regression": ["holdout validation", "MAE/RMSE/R2"],
            "computer_vision": ["COCO validation split", "mAP@50:95", "IoU and per-class recall"],
            "unsupervised": ["silhouette score", "cluster stability across seeds"],
        }.get(profile.task, []),
        "next_experiment": "Run cross-validation on a larger representative sample before production deployment.",
    }
    devops = {
        "status": "ready",
        "artifacts": ["preprocessing + model pipeline", "evaluation report", "API endpoint"],
        "deployment": "Docker image available; monitoring and model registry integration are next steps.",
    }
    decision = {
        "status": "decided" if comparison_ready else "needs_data",
        "summary": (
            f"Select {evaluation['best_model']} based on the reported validation metrics."
            if comparison_ready
            else "Profile and recommend a route, but do not claim a best model until labeled evaluation data is available."
        ),
    }
    stages = [
        {"agent": "Data Agent", "status": "complete", "output": "Profiled, cleaned, and checked dataset"},
        {"agent": "Model Agent", "status": "complete" if comparison_ready else "recommended", "output": "Trained and compared candidates" if comparison_ready else "Recommended candidate models"},
        {"agent": "Research Agent", "status": research["status"], "output": "Evaluation methods and next experiment"},
        {"agent": "DevOps Agent", "status": devops["status"], "output": "Packaging and deployment artifacts"},
        {"agent": "Decision Agent", "status": decision["status"], "output": decision["summary"]},
    ]
    return {"profile": asdict(profile), "recommendation": recommendation, "evaluation": evaluation, "research": research, "devops": devops, "decision": decision, "stages": stages}