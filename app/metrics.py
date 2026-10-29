from prometheus_client import Counter, Gauge, Histogram, generate_latest

WORKFLOW_RUNS = Counter("ai_agent_workflow_runs_total", "Workflow runs", ["status"])
PREDICTIONS = Counter("ai_agent_predictions_total", "Predictions served", ["task"])
API_ERRORS = Counter("ai_agent_api_errors_total", "API errors", ["endpoint"])
REQUEST_LATENCY = Histogram("ai_agent_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_ERRORS = Counter("ai_agent_prediction_errors_total", "Prediction request errors", ["error_type"])
PREDICTION_LATENCY = Histogram(
    "ai_agent_prediction_latency_seconds",
    "Prediction inference latency",
    ["task"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
INPUT_DRIFT_SCORE = Gauge("ai_agent_input_drift_score", "Maximum population stability index across monitored features", ["task"])
MODEL_PERFORMANCE = Gauge("ai_agent_model_performance", "Observed labeled model performance", ["task", "metric"])
MODEL_PERFORMANCE_DEGRADATION = Gauge(
    "ai_agent_model_performance_degradation",
    "Relative degradation from validation performance (positive is worse)",
    ["task", "metric"],
)


def metrics_payload() -> bytes:
    return generate_latest()