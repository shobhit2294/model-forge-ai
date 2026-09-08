from prometheus_client import Counter, Histogram, generate_latest

WORKFLOW_RUNS = Counter("ai_agent_workflow_runs_total", "Workflow runs", ["status"])
PREDICTIONS = Counter("ai_agent_predictions_total", "Predictions served", ["run_id"])
API_ERRORS = Counter("ai_agent_api_errors_total", "API errors", ["endpoint"])
REQUEST_LATENCY = Histogram("ai_agent_request_latency_seconds", "Request latency", ["endpoint"])


def metrics_payload() -> bytes:
    return generate_latest()