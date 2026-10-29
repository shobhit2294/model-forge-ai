"""Dataset profiling and model recommendation logic."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUPPORTED_DATA_EXTENSIONS = (".csv", ".json", ".jsonl", ".ndjson", ".xml", ".xlsx", ".xls", ".parquet")


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
    if suffix == ".xml":
        try:
            root = ET.fromstring(content)
        except ET.ParseError as error:
            raise ValueError(f"Invalid XML dataset: {error}") from error

        def tag_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        def flatten(element: ET.Element, prefix: str = "") -> dict[str, Any]:
            current = f"{prefix}.{tag_name(element.tag)}" if prefix else tag_name(element.tag)
            values: dict[str, Any] = {
                f"{current}.@{tag_name(name)}": value for name, value in element.attrib.items()
            }
            children = list(element)
            if not children:
                values[current] = (element.text or "").strip()
                return values
            if element.text and element.text.strip():
                values[f"{current}.#text"] = element.text.strip()
            for child in children:
                for name, value in flatten(child, current).items():
                    if name in values:
                        previous = values[name]
                        if not isinstance(previous, list):
                            previous = [previous]
                        previous.append(value)
                        values[name] = previous
                    else:
                        values[name] = value
            return values

        children = list(root)
        if children and len({tag_name(child.tag) for child in children}) == 1 and all(list(child) for child in children):
            rows = []
            for child in children:
                flattened = flatten(child)
                rows.append({
                    key.split(".", 1)[1] if "." in key else key: (
                        json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    )
                    for key, value in flattened.items()
                })
        else:
            flattened = flatten(root)
            rows = [{
                key.split(".", 1)[1] if "." in key else key: (
                    json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                )
                for key, value in flattened.items()
            }]
        if not rows or not rows[0]:
            raise ValueError("XML must contain repeated records with fields or one record with fields.")
        return "xml", rows
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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "null", "none", "na", "n/a"}:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_number(value: Any) -> bool:
    if _is_missing(value):
        return False
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def _column_type(values: list[Any]) -> str:
    present = [value for value in values if not _is_missing(value)]
    if not present:
        return "empty"
    if all(_is_number(value) for value in present):
        return "numeric"
    lowered = {str(value).strip().lower() for value in present}
    if lowered <= {"true", "false", "yes", "no", "0", "1"}:
        return "boolean"
    return "categorical" if len(lowered) <= max(20, len(present) // 5) else "text"


def _guess_target(headers: list[str], profiles: list[ColumnProfile]) -> str | None:
    target_names = {"target", "label", "class", "y", "y_true", "outcome", "response", "price", "score", "churn", "saleprice", "sale_price", "house_price", "houseprice", "rent", "amount", "revenue", "rating", "rate"}
    for profile in profiles:
        normalized = profile.name.lower().strip().replace("-", "_").replace(" ", "_")
        if normalized in target_names or normalized.replace("_", "") in {name.replace("_", "") for name in target_names}:
            return profile.name
    return None


def _is_regression_target(profile: ColumnProfile, row_count: int) -> bool:
    if profile.data_type != "numeric":
        return False
    normalized = profile.name.lower().strip().replace("-", "_").replace(" ", "_")
    regression_names = {"price", "saleprice", "sale_price", "houseprice", "house_price", "rent", "amount", "revenue", "rate", "cost", "income"}
    return (
        profile.unique > max(10, min(20, row_count // 5))
        or normalized in regression_names
        or normalized.replace("_", "") in {name.replace("_", "") for name in regression_names}
    )


def _leakage_risks(profile: DatasetProfile, rows: list[dict[str, Any]]) -> list[str]:
    if not profile.target:
        return []
    risky_name = re.compile(r"(?:^|[_\W])(target|label|outcome|result|prediction|predicted|future|post_event|after_event)(?:$|[_\W])", re.IGNORECASE)
    identifier_name = re.compile(r"(?:^|[_\W])(id|uuid|key|identifier)(?:$|[_\W])", re.IGNORECASE)
    risks = []
    labeled_rows = [row for row in rows if not _is_missing(row.get(profile.target))]
    for column in profile.column_profiles:
        if column.name == profile.target:
            continue
        if risky_name.search(column.name):
            risks.append(f"Feature '{column.name}' has a target/outcome-like name; verify it is available before prediction time.")
            continue
        if identifier_name.search(column.name):
            risks.append(f"Feature '{column.name}' looks like an identifier; check for memorization or leakage before deployment.")
            continue
        if labeled_rows and all(
            not _is_missing(row.get(column.name))
            and str(row.get(column.name)).strip() == str(row.get(profile.target)).strip()
            for row in labeled_rows
        ):
            risks.append(f"Feature '{column.name}' exactly duplicates the target and should be excluded.")
    return risks


def profile_dataset(
    filename: str,
    content: bytes,
    target: str | None = None,
    infer_target: bool = True,
) -> DatasetProfile:
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
        present = [str(value) for value in values if not _is_missing(value)]
        profiles.append(ColumnProfile(header, _column_type(values), len(values) - len(present), len(set(present)), present[:3]))

    target_lookup = {header.lower().strip(): header for header in headers}
    chosen_target = target_lookup.get(target.lower().strip()) if target else (
        _guess_target(headers, profiles) if infer_target else None
    )
    if target and chosen_target is None:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")
    target_profile = next((item for item in profiles if item.name == chosen_target), None)
    if target_profile is None:
        task = "unsupervised"
    else:
        task = "regression" if _is_regression_target(target_profile, len(rows)) else "classification"

    warnings: list[str] = []
    if not rows:
        warnings.append("The dataset contains no rows.")
    if any(profile.missing for profile in profiles):
        warnings.append("Missing values detected; impute numeric and categorical columns during cleaning.")
    if any(profile.data_type == "text" for profile in profiles):
        warnings.append("Free-text columns detected; use vectorization or embeddings and review leakage.")
    if task == "classification" and target_profile and target_profile.unique > 1:
        target_counts: dict[str, int] = {}
        for row in rows:
            value = row.get(chosen_target)
            if not _is_missing(value):
                key = str(value)
                target_counts[key] = target_counts.get(key, 0) + 1
        if target_counts and min(target_counts.values()) / max(target_counts.values()) < 0.5:
            warnings.append("Class imbalance detected; inspect per-class metrics and use stratified validation.")
    dataset_profile = DatasetProfile(filename, file_format, len(rows), len(headers), chosen_target, task, warnings, profiles)
    risks = _leakage_risks(dataset_profile, rows)
    if risks:
        dataset_profile.warnings.append("Potential target leakage detected: " + " ".join(risks))
    if len(rows) < 100:
        dataset_profile.warnings.append("Small dataset: use cross-validation and compare against a simple baseline.")
    return dataset_profile


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
        primary = ["RandomForestRegressor", "XGBRegressor", "Ridge"]
        metric = "MAE and RMSE"
    elif profile.task == "classification":
        primary = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
        metric = "F1, ROC-AUC, and balanced accuracy"
    else:
        primary = ["KMeans", "IsolationForest", "PCA"]
        metric = "silhouette score and stability across seeds"
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
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
        "XGBClassifier": OneVsRestClassifier(XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", n_jobs=1, random_state=42)),
    }


def _regression_candidates() -> dict[str, Any]:
    return {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=42),
        "Ridge": Ridge(),
        "XGBRegressor": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=42),
    }


def _selection_metric(task: str, metric: str) -> str:
    requested = metric.lower().replace("-", "_").strip()
    if task == "classification":
        if requested not in {"f1", "accuracy", "precision", "recall"}:
            raise ValueError("Classification metric must be f1, accuracy, precision, or recall.")
        return requested
    if requested in {"f1", "accuracy", "precision", "recall"}:
        return "r2"
    if requested not in {"r2", "mae", "rmse"}:
        raise ValueError("Regression metric must be r2, mae, or rmse.")
    return requested


def evaluate_models(
    profile: DatasetProfile,
    content: bytes,
    metric: str = "f1",
    validation_strategy: str = "holdout",
) -> dict[str, Any] | None:
    """Run the deterministic preprocessing, training, and evaluation stages."""
    prepared = prepare_training_data(profile, content, validation_strategy)
    if prepared["status"] != "ready":
        return {"status": "unavailable", "reason": prepared["reason"]}
    trained = train_candidate_models(profile, prepared)
    return evaluate_predictions(profile, prepared, trained, metric)


def prepare_training_data(
    profile: DatasetProfile,
    content: bytes,
    validation_strategy: str = "holdout",
) -> dict[str, Any]:
    """Prepare serializable features and validation indices without fitting transforms."""
    validation_strategy = validation_strategy.lower().replace("-", "_").strip()
    if validation_strategy not in {"holdout", "cross_validation"}:
        raise ValueError("Validation strategy must be holdout or cross_validation.")
    if profile.task == "computer_vision":
        return {"status": "unavailable", "reason": "COCO metadata alone is not enough to train models; image files are required."}
    if profile.task == "unsupervised" or not profile.target:
        return {"status": "unavailable", "reason": "No ground-truth target column is available for supervised evaluation."}
    _, rows = _parse_rows(profile.filename, content)
    rows = [row for row in rows if not _is_missing(row.get(profile.target))]
    if len(rows) < 10:
        return {"status": "unavailable", "reason": "At least 10 labeled rows are required for a reliable holdout evaluation."}

    feature_names = [item.name for item in profile.column_profiles if item.name != profile.target]
    numeric_names = {item.name for item in profile.column_profiles if item.data_type == "numeric"}
    features = []
    for row in rows:
        normalized_row = []
        for name in feature_names:
            value = row.get(name)
            if _is_missing(value):
                normalized_row.append(None)
            elif name in numeric_names:
                normalized_row.append(float(value))
            else:
                normalized_row.append(value)
        features.append(normalized_row)
    if profile.task == "regression":
        labels = [float(row[profile.target]) for row in rows]
        class_counts = None
    else:
        labels = [str(row[profile.target]) for row in rows]
        counts = {label: labels.count(label) for label in set(labels)}
        if len(counts) < 2 or min(counts.values()) < 2:
            return {"status": "unavailable", "reason": "At least two examples per class are required for comparison."}
        class_counts = counts

    if validation_strategy == "cross_validation":
        folds = min(5, min(class_counts.values())) if class_counts else min(5, len(labels))
        if folds < 2:
            return {"status": "unavailable", "reason": "At least two folds are required for cross-validation."}
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42) if class_counts else KFold(n_splits=folds, shuffle=True, random_state=42)
        splits = [
            {"train": train_indices.tolist(), "test": test_indices.tolist()}
            for train_indices, test_indices in splitter.split(features, labels)
        ]
        validation_info = {"strategy": validation_strategy, "folds": folds}
    else:
        try:
            train_indices, test_indices = train_test_split(
                list(range(len(labels))),
                test_size=0.2,
                random_state=42,
                stratify=labels if class_counts else None,
            )
        except ValueError as error:
            return {"status": "unavailable", "reason": f"Unable to create a valid holdout split: {error}"}
        splits = [{"train": train_indices, "test": test_indices}]
        validation_info = {"strategy": validation_strategy, "test_size": 0.2, "stratified": bool(class_counts)}

    return {
        "status": "ready",
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
        "splits": splits,
        "validation": validation_info,
        "preprocessing": {
            "numeric_columns": [name for name in feature_names if name in numeric_names],
            "categorical_columns": [name for name in feature_names if name not in numeric_names],
            "numeric": "median imputation and standard scaling",
            "categorical": "most-frequent imputation and one-hot encoding",
            "fit_scope": "training split only",
        },
    }


def train_candidate_models(profile: DatasetProfile, prepared: dict[str, Any]) -> dict[str, Any]:
    """Fit each candidate inside its validation splits and return held-out predictions."""
    candidates = _classification_candidates() if profile.task == "classification" else _regression_candidates()
    predictions_by_model = []
    for name, estimator in candidates.items():
        estimator_params = {
            key: value
            for key, value in estimator.get_params(deep=True).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        held_out_predictions: list[Any] = [None] * len(prepared["labels"])
        for split in prepared["splits"]:
            model = Pipeline([("preprocess", _preprocessor(profile, prepared["feature_names"])), ("model", estimator)])
            train_indices = split["train"]
            test_indices = split["test"]
            x_train = [prepared["features"][index] for index in train_indices]
            x_test = [prepared["features"][index] for index in test_indices]
            y_train = [prepared["labels"][index] for index in train_indices]
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            for index, prediction in zip(test_indices, predictions):
                held_out_predictions[index] = prediction.item() if hasattr(prediction, "item") else prediction
        evaluated_indices = [index for split in prepared["splits"] for index in split["test"]]
        if len(prepared["splits"]) > 1:
            evaluated_indices = sorted(evaluated_indices)
        predictions_by_model.append({
            "model": name,
            "params": estimator_params,
            "truth": [prepared["labels"][index] for index in evaluated_indices],
            "predictions": [held_out_predictions[index] for index in evaluated_indices],
        })
    return {"status": "complete", "models": predictions_by_model}


def evaluate_predictions(
    profile: DatasetProfile,
    prepared: dict[str, Any],
    trained: dict[str, Any],
    metric: str = "f1",
) -> dict[str, Any]:
    """Calculate comparable metrics and select a winner using deterministic Python."""
    metric = _selection_metric(profile.task, metric)
    results: list[dict[str, Any]] = []
    for candidate in trained["models"]:
        y_test = candidate["truth"]
        predictions = candidate["predictions"]
        if profile.task == "classification":
            results.append({
                "model": candidate["model"],
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            })
        else:
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            results.append({
                "model": candidate["model"],
                "mae": round(mean_absolute_error(y_test, predictions), 4),
                "rmse": round(rmse, 4),
                "r2": round(r2_score(y_test, predictions), 4),
            })
    if profile.task == "classification":
        best = max(results, key=lambda item: item[metric])
        explanation = f"Selected by weighted {metric}; per-class outcomes are available in the confusion matrix."
        labels_for_matrix = sorted(set(prepared["labels"]))
        return {"status": "complete", "task": profile.task, "metric_definition": "Precision, recall, and F1 use weighted averages across classes.", "selection_metric": metric, "validation": prepared["validation"], "labels": labels_for_matrix, "models": results, "best_model": best["model"], "why_best": explanation}
    best = min(results, key=lambda item: item[metric]) if metric in {"mae", "rmse"} else max(results, key=lambda item: item[metric])
    direction = "lowest" if metric in {"mae", "rmse"} else "highest"
    return {"status": "complete", "task": profile.task, "metric_definition": "Lower MAE/RMSE is better; higher R2 is better.", "selection_metric": metric, "validation": prepared["validation"], "models": results, "best_model": best["model"], "why_best": f"Selected by {direction} {metric}."}


def pipeline_result(
    profile: DatasetProfile,
    content: bytes | None = None,
    metric: str = "f1",
    validation_strategy: str = "holdout",
) -> dict[str, Any]:
    recommendation = recommend_models(profile)
    evaluation = evaluate_models(profile, content, metric, validation_strategy) if content is not None else None
    if evaluation and evaluation.get("status") == "complete":
        recommendation["recommended"] = [evaluation["best_model"]] + [model["model"] for model in evaluation["models"] if model["model"] != evaluation["best_model"]]
    comparison_ready = bool(evaluation and evaluation.get("status") == "complete")
    research = {
        "status": "ready",
        "methods": {
            "classification": [f"{validation_strategy} validation", "weighted precision/recall/F1", "confusion matrix"],
            "regression": [f"{validation_strategy} validation", "MAE/RMSE/R2"],
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