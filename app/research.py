"""Approved research adapter boundary.

This module is deterministic by default. Network and LLM connectors can be
added behind the same interface after credentials and source policies exist.
"""

from __future__ import annotations

from typing import Any


METHOD_CATALOG: dict[str, list[dict[str, str]]] = {
    "classification": [
        {"method": "Logistic Regression", "use": "interpretable baseline"},
        {"method": "Random Forest", "use": "nonlinear tabular baseline"},
        {"method": "Gradient Boosting", "use": "strong tabular candidate"},
    ],
    "regression": [
        {"method": "Ridge Regression", "use": "regularized baseline"},
        {"method": "Random Forest", "use": "nonlinear tabular baseline"},
        {"method": "Gradient Boosting", "use": "strong tabular candidate"},
    ],
    "computer_vision": [
        {"method": "YOLO", "use": "fast object-detection baseline"},
        {"method": "Faster R-CNN", "use": "accuracy-oriented detector"},
        {"method": "DETR", "use": "transformer detector experiment"},
    ],
    "unsupervised": [
        {"method": "KMeans", "use": "cluster baseline"},
        {"method": "Isolation Forest", "use": "anomaly baseline"},
        {"method": "PCA", "use": "dimensionality reduction"},
    ],
}


def research_methods(task: str, query: str | None = None) -> dict[str, Any]:
    return {
        "query": query or task,
        "source_policy": "approved internal catalog; external sources require explicit connector configuration",
        "methods": METHOD_CATALOG.get(task, []),
        "recommended_experiments": [f"Compare {item['method']} as a {item['use']}" for item in METHOD_CATALOG.get(task, [])],
    }