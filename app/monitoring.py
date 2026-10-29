"""Persistent aggregate monitoring for prediction drift and delayed labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.metrics import (
    INPUT_DRIFT_SCORE,
    MODEL_PERFORMANCE,
    MODEL_PERFORMANCE_DEGRADATION,
)
from app.pipeline import _is_missing, _is_number, _parse_rows


MIN_DRIFT_PREDICTIONS = int(os.environ.get("MONITORING_MIN_PREDICTIONS", "30"))
DRIFT_PSI_THRESHOLD = float(os.environ.get("MONITORING_PSI_THRESHOLD", "0.2"))
PERFORMANCE_DEGRADATION_THRESHOLD = float(
    os.environ.get("MONITORING_PERFORMANCE_DEGRADATION_THRESHOLD", "0.2")
)


def _database_path() -> Path:
    return Path(os.environ.get("MONITORING_DB_PATH", "artifacts/monitoring.sqlite"))


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS monitored_models (
            run_id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            baseline_metric TEXT,
            baseline_score REAL,
            prediction_count INTEGER NOT NULL DEFAULT 0,
            feedback_count INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            sum_absolute_error REAL NOT NULL DEFAULT 0,
            sum_squared_error REAL NOT NULL DEFAULT 0,
            sum_actual REAL NOT NULL DEFAULT 0,
            sum_actual_squared REAL NOT NULL DEFAULT 0,
            drift_score REAL NOT NULL DEFAULT 0,
            performance_degradation REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS monitored_features (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            kind TEXT NOT NULL,
            config TEXT NOT NULL,
            baseline_counts TEXT NOT NULL,
            PRIMARY KEY (run_id, feature),
            FOREIGN KEY (run_id) REFERENCES monitored_models(run_id)
        );
        CREATE TABLE IF NOT EXISTS observed_feature_buckets (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            bucket INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, feature, bucket),
            FOREIGN KEY (run_id, feature) REFERENCES monitored_features(run_id, feature)
        );
        CREATE TABLE IF NOT EXISTS pending_predictions (
            prediction_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            task TEXT NOT NULL,
            predicted_class TEXT,
            predicted_value REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES monitored_models(run_id)
        );
        CREATE TABLE IF NOT EXISTS classification_feedback (
            run_id TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            actual_class TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, predicted_class, actual_class),
            FOREIGN KEY (run_id) REFERENCES monitored_models(run_id)
        );
        """
    )
    model_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(monitored_models)").fetchall()
    }
    if "performance_degradation" not in model_columns:
        connection.execute(
            "ALTER TABLE monitored_models ADD COLUMN performance_degradation REAL NOT NULL DEFAULT 0"
        )
    if "created_at" not in model_columns:
        connection.execute(
            "ALTER TABLE monitored_models ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _category_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _numeric_bucket(value: Any, boundaries: list[float]) -> int:
    if _is_missing(value) or not _is_number(value):
        return len(boundaries) + 1
    number = float(value)
    for index, boundary in enumerate(boundaries):
        if number <= boundary:
            return index
    return len(boundaries)


def _feature_bucket(value: Any, kind: str, config: dict[str, Any]) -> int:
    if kind == "numeric":
        return _numeric_bucket(value, config["boundaries"])
    if _is_missing(value):
        return len(config["categories"]) + 1
    digest = _category_hash(value)
    try:
        return config["categories"].index(digest)
    except ValueError:
        return len(config["categories"])


def initialize_monitoring(
    run_id: str,
    filename: str,
    content: bytes,
    feature_names: list[str],
    task: str,
    baseline_metric: str | None,
    baseline_score: float | None,
) -> None:
    _, rows = _parse_rows(filename, content)
    feature_profiles: list[tuple[str, str, dict[str, Any], list[int]]] = []
    for name in feature_names:
        values = [row.get(name) for row in rows]
        present = [value for value in values if not _is_missing(value)]
        if present and all(_is_number(value) for value in present):
            numeric_values = sorted(float(value) for value in present)
            boundaries = sorted({
                numeric_values[min(len(numeric_values) - 1, int((len(numeric_values) * fraction) // 1))]
                for fraction in (0.2, 0.4, 0.6, 0.8)
            })
            config = {"boundaries": boundaries}
            kind = "numeric"
        else:
            categories = list(dict.fromkeys(_category_hash(value) for value in present))[:50]
            config = {"categories": categories}
            kind = "categorical"
        bucket_counts = [0] * (len(config.get("boundaries", [])) + 2 if kind == "numeric" else len(config["categories"]) + 2)
        buckets = [_feature_bucket(value, kind, config) for value in values]
        for bucket in buckets:
            bucket_counts[bucket] += 1
        feature_profiles.append((name, kind, config, bucket_counts))

    with _database() as connection:
        connection.execute(
            """INSERT OR REPLACE INTO monitored_models
               (run_id, task, baseline_metric, baseline_score)
               VALUES (?, ?, ?, ?)""",
            (run_id, task, baseline_metric, baseline_score),
        )
        connection.execute("DELETE FROM monitored_features WHERE run_id = ?", (run_id,))
        for name, kind, config, counts in feature_profiles:
            connection.execute(
                """INSERT INTO monitored_features
                   (run_id, feature, kind, config, baseline_counts)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, name, kind, json.dumps(config), json.dumps(counts)),
            )


def _psi(expected: list[int], observed: list[int]) -> float:
    expected_total = sum(expected)
    observed_total = sum(observed)
    if not expected_total or not observed_total:
        return 0.0
    bucket_count = max(len(expected), len(observed))
    score = 0.0
    for index in range(bucket_count):
        expected_rate = (expected[index] if index < len(expected) else 0) / expected_total
        observed_rate = (observed[index] if index < len(observed) else 0) / observed_total
        expected_rate = max(expected_rate, 1e-6)
        observed_rate = max(observed_rate, 1e-6)
        score += (observed_rate - expected_rate) * math.log(observed_rate / expected_rate)
    return score


def _classification_performance(connection: sqlite3.Connection, run_id: str) -> dict[str, float]:
    rows = connection.execute(
        "SELECT predicted_class, actual_class, count FROM classification_feedback WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    total = sum(row["count"] for row in rows)
    if not total:
        return {}
    classes = {row["predicted_class"] for row in rows} | {row["actual_class"] for row in rows}
    true_positive = {label: 0 for label in classes}
    predicted_count = {label: 0 for label in classes}
    actual_count = {label: 0 for label in classes}
    correct = 0
    for row in rows:
        predicted, actual, count = row["predicted_class"], row["actual_class"], row["count"]
        predicted_count[predicted] += count
        actual_count[actual] += count
        if predicted == actual:
            true_positive[actual] += count
            correct += count
    weighted_f1 = 0.0
    for label in classes:
        precision_denominator = predicted_count[label]
        recall_denominator = actual_count[label]
        precision = true_positive[label] / precision_denominator if precision_denominator else 0.0
        recall = true_positive[label] / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted_f1 += f1 * recall_denominator / total
    weighted_precision = sum(
        true_positive[label] / predicted_count[label] * actual_count[label] / total
        for label in classes
        if predicted_count[label]
    )
    weighted_recall = sum(
        true_positive[label] / actual_count[label] * actual_count[label] / total
        for label in classes
        if actual_count[label]
    )
    return {
        "accuracy": correct / total,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
    }


def _performance(connection: sqlite3.Connection, model: sqlite3.Row) -> dict[str, float]:
    if model["task"] == "classification":
        return _classification_performance(connection, model["run_id"])
    count = model["feedback_count"]
    if not count:
        return {}
    mse = model["sum_squared_error"] / count
    actual_variance = model["sum_actual_squared"] - model["sum_actual"] ** 2 / count
    return {
        "mae": model["sum_absolute_error"] / count,
        "rmse": math.sqrt(mse),
        "r2": 1 - (model["sum_squared_error"] / actual_variance) if actual_variance > 0 else 0.0,
    }


def record_prediction(run_id: str, features: dict[str, Any], prediction: Any) -> tuple[str, dict[str, Any]]:
    prediction_id = uuid.uuid4().hex
    with _database() as connection:
        model = connection.execute("SELECT * FROM monitored_models WHERE run_id = ?", (run_id,)).fetchone()
        if model is None:
            raise KeyError(f"No monitoring baseline exists for run '{run_id}'.")
        feature_rows = connection.execute("SELECT * FROM monitored_features WHERE run_id = ?", (run_id,)).fetchall()
        connection.execute(
            "UPDATE monitored_models SET prediction_count = prediction_count + 1 WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            """INSERT INTO pending_predictions (prediction_id, run_id, task, predicted_class, predicted_value)
               VALUES (?, ?, ?, ?, ?)""",
            (
                prediction_id,
                run_id,
                model["task"],
                _category_hash(prediction) if model["task"] == "classification" else None,
                None if model["task"] == "classification" else float(prediction),
            ),
        )
        scores = []
        for feature in feature_rows:
            config = json.loads(feature["config"])
            bucket = _feature_bucket(features.get(feature["feature"]), feature["kind"], config)
            connection.execute(
                """INSERT INTO observed_feature_buckets (run_id, feature, bucket, count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(run_id, feature, bucket) DO UPDATE SET count = count + 1""",
                (run_id, feature["feature"], bucket),
            )
            observed_rows = connection.execute(
                "SELECT bucket, count FROM observed_feature_buckets WHERE run_id = ? AND feature = ?",
                (run_id, feature["feature"]),
            ).fetchall()
            observed = [0] * len(json.loads(feature["baseline_counts"]))
            for row in observed_rows:
                if row["bucket"] < len(observed):
                    observed[row["bucket"]] = row["count"]
            observed_total = sum(observed)
            if observed_total >= MIN_DRIFT_PREDICTIONS:
                scores.append(_psi(json.loads(feature["baseline_counts"]), observed))
        drift_score = max(scores, default=0.0)
        connection.execute(
            "UPDATE monitored_models SET drift_score = ? WHERE run_id = ?",
            (drift_score, run_id),
        )
        current_model = connection.execute("SELECT * FROM monitored_models WHERE run_id = ?", (run_id,)).fetchone()
        performance = _performance(connection, current_model)
        maximum_drift = connection.execute(
            "SELECT MAX(drift_score) AS score FROM monitored_models WHERE task = ?",
            (model["task"],),
        ).fetchone()["score"]
        INPUT_DRIFT_SCORE.labels(model["task"]).set(maximum_drift or 0.0)
        return prediction_id, {
            "prediction_count": current_model["prediction_count"],
            "drift_score": drift_score,
            "drift_status": "drift_detected" if drift_score >= DRIFT_PSI_THRESHOLD else (
                "collecting" if current_model["prediction_count"] < MIN_DRIFT_PREDICTIONS else "stable"
            ),
            "feedback_count": current_model["feedback_count"],
            "performance": performance,
        }


def record_feedback(run_id: str, prediction_id: str, actual: Any) -> dict[str, Any]:
    with _database() as connection:
        model = connection.execute("SELECT * FROM monitored_models WHERE run_id = ?", (run_id,)).fetchone()
        if model is None:
            raise KeyError(f"No monitoring baseline exists for run '{run_id}'.")
        prediction = connection.execute(
            "SELECT * FROM pending_predictions WHERE prediction_id = ? AND run_id = ?",
            (prediction_id, run_id),
        ).fetchone()
        if prediction is None:
            raise ValueError("Prediction ID was not found or feedback was already submitted.")
        if model["task"] == "classification":
            if _is_missing(actual):
                raise ValueError("An actual class label is required.")
            predicted_class = prediction["predicted_class"]
            actual_class = _category_hash(actual)
            connection.execute(
                """INSERT INTO classification_feedback (run_id, predicted_class, actual_class, count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(run_id, predicted_class, actual_class) DO UPDATE SET count = count + 1""",
                (run_id, predicted_class, actual_class),
            )
            connection.execute(
                "UPDATE monitored_models SET feedback_count = feedback_count + 1, correct_count = correct_count + ? WHERE run_id = ?",
                (int(predicted_class == actual_class), run_id),
            )
        else:
            if _is_missing(actual) or not _is_number(actual):
                raise ValueError("An actual numeric label is required for a regression model.")
            actual_value = float(actual)
            error = float(prediction["predicted_value"]) - actual_value
            connection.execute(
                """UPDATE monitored_models
                   SET feedback_count = feedback_count + 1,
                       sum_absolute_error = sum_absolute_error + ?,
                       sum_squared_error = sum_squared_error + ?,
                       sum_actual = sum_actual + ?,
                       sum_actual_squared = sum_actual_squared + ?
                   WHERE run_id = ?""",
                (abs(error), error * error, actual_value, actual_value * actual_value, run_id),
            )
        connection.execute("DELETE FROM pending_predictions WHERE prediction_id = ?", (prediction_id,))
        model = connection.execute("SELECT * FROM monitored_models WHERE run_id = ?", (run_id,)).fetchone()
        performance = _performance(connection, model)
        baseline_metric = model["baseline_metric"]
        baseline_score = model["baseline_score"]
        degradation = 0.0
        if baseline_metric in performance and baseline_score is not None:
            observed_score = performance[baseline_metric]
            if baseline_metric in {"mae", "rmse"}:
                degradation = (observed_score - baseline_score) / max(abs(baseline_score), 1e-9)
            else:
                degradation = (baseline_score - observed_score) / max(abs(baseline_score), 1e-9)
            degradation = max(0.0, degradation)
            connection.execute(
                "UPDATE monitored_models SET performance_degradation = ? WHERE run_id = ?",
                (degradation, run_id),
            )
            MODEL_PERFORMANCE.labels(model["task"], baseline_metric).set(observed_score)
            maximum_degradation = connection.execute(
                """SELECT MAX(performance_degradation) AS degradation
                   FROM monitored_models WHERE task = ? AND baseline_metric = ?""",
                (model["task"], baseline_metric),
            ).fetchone()["degradation"]
            MODEL_PERFORMANCE_DEGRADATION.labels(model["task"], baseline_metric).set(
                maximum_degradation or 0.0
            )
        return {
            "run_id": run_id,
            "feedback_count": model["feedback_count"],
            "performance": performance,
            "baseline_metric": baseline_metric,
            "baseline_score": baseline_score,
            "degradation": degradation,
            "performance_status": (
                "degraded" if degradation >= PERFORMANCE_DEGRADATION_THRESHOLD else "within_baseline"
            ) if baseline_metric in performance else "collecting",
        }


def monitoring_status(run_id: str) -> dict[str, Any]:
    with _database() as connection:
        model = connection.execute("SELECT * FROM monitored_models WHERE run_id = ?", (run_id,)).fetchone()
        if model is None:
            raise KeyError(f"No monitoring baseline exists for run '{run_id}'.")
        performance = _performance(connection, model)
        metric = model["baseline_metric"]
        score = performance.get(metric) if metric else None
        degradation = 0.0
        if score is not None and model["baseline_score"] is not None:
            if metric in {"mae", "rmse"}:
                degradation = max(0.0, (score - model["baseline_score"]) / max(abs(model["baseline_score"]), 1e-9))
            else:
                degradation = max(0.0, (model["baseline_score"] - score) / max(abs(model["baseline_score"]), 1e-9))
        return {
            "run_id": run_id,
            "task": model["task"],
            "prediction_count": model["prediction_count"],
            "drift": {
                "method": "population stability index (PSI) over input feature bins",
                "score": model["drift_score"],
                "threshold": DRIFT_PSI_THRESHOLD,
                "status": "drift_detected" if model["drift_score"] >= DRIFT_PSI_THRESHOLD else (
                    "collecting" if model["prediction_count"] < MIN_DRIFT_PREDICTIONS else "stable"
                ),
                "minimum_predictions": MIN_DRIFT_PREDICTIONS,
            },
            "performance": {
                "feedback_count": model["feedback_count"],
                "metrics": performance,
                "baseline_metric": metric,
                "baseline_score": model["baseline_score"],
                "degradation": degradation,
                "degradation_threshold": PERFORMANCE_DEGRADATION_THRESHOLD,
                "status": (
                    "degraded" if degradation >= PERFORMANCE_DEGRADATION_THRESHOLD else "within_baseline"
                ) if metric in performance else "collecting",
            },
        }


def refresh_monitoring_metrics() -> None:
    with _database() as connection:
        models = connection.execute("SELECT * FROM monitored_models").fetchall()
        drift_by_task: dict[str, float] = {}
        degradation_by_metric: dict[tuple[str, str], float] = {}
        latest_performance: dict[tuple[str, str], tuple[str, float]] = {}
        for model in models:
            drift_by_task[model["task"]] = max(
                drift_by_task.get(model["task"], 0.0),
                model["drift_score"],
            )
            metric = model["baseline_metric"]
            if not metric:
                continue
            key = (model["task"], metric)
            degradation_by_metric[key] = max(
                degradation_by_metric.get(key, 0.0),
                model["performance_degradation"],
            )
            performance = _performance(connection, model)
            if metric in performance:
                current = latest_performance.get(key)
                if current is None or model["created_at"] > current[0]:
                    latest_performance[key] = (model["created_at"], performance[metric])
        for task, score in drift_by_task.items():
            INPUT_DRIFT_SCORE.labels(task).set(score)
        for (task, metric), score in degradation_by_metric.items():
            MODEL_PERFORMANCE_DEGRADATION.labels(task, metric).set(score)
        for (task, metric), (_, score) in latest_performance.items():
            MODEL_PERFORMANCE.labels(task, metric).set(score)
