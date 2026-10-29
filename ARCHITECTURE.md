# How It All Works

This doc walks through how the AI Engineering Agent is actually built — what each piece does, how they talk to each other, and why certain choices were made. Written for someone who wants to understand the system, not just run it.

---

## The big picture

You upload a dataset. The system figures out what kind of problem it is, trains a few models, picks the best one, and then waits for you to approve before saving anything. Once approved, you can send new data and get predictions back.

Here's the full flow:

    React UI  →  FastAPI  →  Orchestrator  →  LangGraph workflow
                                                      |
                              Data Agent → Preprocessing → Training → Evaluation → Decision
                                                      |
                                          [passes quality gate]
                                                      |
                                          Human approval pause
                                                      |
                                          Deployment Agent → artifacts/<run_id>.joblib
                                                      |
                                          Prediction endpoint → Monitoring (SQLite)
                                                      ↑
                                          Delayed-label feedback

---

## Step by step — what happens when you hit Analyze

1. You upload a file — the UI sends it to POST /api/profile. The backend reads the columns, guesses types, and suggests a target column if it can find one.

2. You confirm — pick a target column (or skip to profile-only mode). Hit Analyze. The file goes to POST /api/analyze.

3. The workflow kicks off — FastAPI hands it to the orchestrator, which runs a LangGraph pipeline. Six agents run in sequence.

4. Agents do their thing — data profiling → preprocessing plan → model training → evaluation → decision. Each step logs what it did.

5. Quality gate check — if the best model clears the threshold, the run pauses and waits for your approval. If not, it tells you why and stops.

6. You approve — hit the Approve button. The orchestrator resumes from a saved checkpoint (even if you restarted the server), fits the final pipeline, and writes it to artifacts/<run_id>.joblib.

7. Predict away — send new data to POST /api/runs/{run_id}/predict. The saved pipeline preprocesses it and returns a prediction.

---

## What each agent actually does

| Agent | What it does | What it produces |
|---|---|---|
| Data Profiling Agent | Reads the file, detects columns, finds missing values, flags duplicates, checks for leakage, guesses the task type | A structured profile and a list of quality issues |
| Preprocessing Agent | Builds a leakage-safe plan — which columns to impute, scale, or encode, using only training data statistics | A preprocessing spec that becomes a scikit-learn Pipeline |
| Model Training Agent | Trains 2–3 candidate models on the training split (holdout or cross-validation, your choice) | Predictions for the held-out rows from each candidate |
| Evaluation Agent | Scores every candidate with deterministic Python functions, picks the winner | Metrics, confusion matrix, and the winning model |
| Decision Agent | Checks the winning score against your threshold | awaiting_approval, needs_improvement, or failed |
| Deployment Agent | Only runs after you approve — fits the final pipeline on all data, saves it | artifacts/<run_id>.joblib |

---

## How the ML decisions are made

When a dataset comes in, the system works through this logic:

- No target column → profiling only, no model deployed
- Target column is categorical → classification: Logistic Regression + Random Forest + XGBoost → scored on F1, accuracy, precision, recall
- Target column is numeric → regression: Random Forest + XGBoost + Ridge → scored on MAE, RMSE, R²
- Either path ends at a quality gate before any model gets deployed

The system never uses an LLM to pick models or compute metrics. All model selection and scoring is done by scikit-learn with fixed random seeds so results are reproducible.

Every model runs inside a scikit-learn Pipeline with a ColumnTransformer:
- Numeric columns: median imputation, then standard scaling
- Categorical columns: most-frequent imputation, then one-hot encoding

This prevents data leakage — the imputation and scaling statistics are computed only on training rows.

---

## Where things live in the repo

    frontend/src/main.jsx       — The React UI: upload form, results display, approval button
    frontend/vite.config.js     — Dev server config; proxies /api to the backend on port 8010
    app/main.py                 — All the API routes and the in-memory run registry
    app/orchestrator.py         — Coordinates the full workflow from upload to deployment
    app/pipeline.py             — Parsing, profiling, preprocessing, training, and metrics
    app/research.py             — A local catalog of approved methods (no external LLM calls)
    app/workflow_graph.py       — LangGraph nodes and the routing logic between them
    app/metrics.py              — Prometheus counters and latency histograms
    app/monitoring.py           — Drift tracking and delayed-label performance, stored in SQLite
    artifacts/                  — Where approved model pipelines are saved after approval
    infra/alerts.yml            — Prometheus alert rules for latency, drift, and failures
    tests/                      — Backend tests covering the pipeline and workflow behavior

---

## Why these tools were chosen

React + Vite — React handles the upload form and result rendering. Vite makes local development fast and automatically forwards /api requests to the Python backend.

FastAPI — One clean HTTP layer for everything: upload, workflow control, prediction, health check, and Prometheus metrics. FastAPI's async support keeps things snappy under load.

LangGraph + SQLite checkpointer — LangGraph gives the workflow a proper state machine with retry logic and structured logging per node. The SQLite checkpointer means a pending approval survives a server restart — the workflow picks up exactly where it left off.

pandas + format readers — Handles CSV, JSON, JSONL, XML, Excel, and Parquet. XML with nested elements gets flattened to dotted column names automatically.

scikit-learn + XGBoost — Battle-tested, reproducible, and runs on a laptop without a GPU. XGBoost adds a strong gradient boosting candidate without requiring a separate training cluster.

Joblib — Saves the entire fitted Pipeline (preprocessing + model) in one file. Loading it back for predictions is a single line.

Prometheus client + SQLite — The Prometheus client exposes live latency, error rate, and drift signals. SQLite stores persistent aggregate monitoring state between server restarts.

Docker + Docker Compose — A two-stage build: Node builds the React frontend, then Python takes the built assets and runs the API. Compose adds optional Prometheus, Grafana, Postgres-backed MLflow, and a named volume for artifacts.

---

## What happens when something goes wrong

The workflow is designed to stop cleanly rather than silently produce bad results:

- No target column → profiling-only mode, no model deployment
- Too few labeled rows → blocked with a clear message
- Duplicate rows detected → blocked before training
- Score below threshold → needs_improvement, no approval gate shown
- Any node fails → retried up to the configured limit, then the run is marked failed with the error logged

These cases are all covered by backend tests so the demo shows both the happy path and meaningful failure reasons.

---

## Monitoring and feedback

After a model is deployed, every prediction is tracked:

- Input drift — feature distributions are binned and compared to the training baseline using PSI (Population Stability Index). Reported after at least 30 predictions.
- Delayed labels — you can send back the actual outcome for any prediction using its ID. The system updates accuracy/F1 (classification) or MAE/RMSE/R² (regression) against the validation baseline.
- Prometheus — /metrics exposes prediction latency (p95), error rate, drift score, and performance degradation. The infra profile adds Grafana dashboards and alert evaluation via infra/alerts.yml.

Raw inputs are never stored — only aggregate bin counts and hashed categorical values.

---

## Experiment tracking (MLflow)

Every workflow run logs a parent MLflow run. Each candidate model gets a child run with its parameters, runtime versions, validation metrics, and a SHA-256 fingerprint of the dataset (not the raw data). Approved models are registered in the MLflow model registry after the approval step, tagged with the run ID and dataset fingerprint.

Local development uses artifacts/mlflow.db. The optional infra Compose profile adds a proper MLflow UI backed by PostgreSQL.

---

## API endpoints

| Method | Path | What it does |
|---|---|---|
| GET | /health | Quick health check |
| POST | /api/profile | Upload a file and get its schema + suggested target |
| POST | /api/analyze | Start a full workflow run |
| GET | /api/runs/{run_id} | Get the result and status of a run |
| POST | /api/runs/{run_id}/approve | Approve or reject a pending run |
| POST | /api/runs/{run_id}/predict | Run inference with an approved model |
| POST | /api/runs/{run_id}/feedback | Send back the actual label for a past prediction |
| GET | /api/monitoring/{run_id} | Check drift and performance status |
| GET | /metrics | Prometheus metrics endpoint |

---

## What's not production-ready yet

This is an MVP. It works well for demos and local experimentation but a few things would need to be added before running it for real users:

- No authentication — anyone who can reach the API can run workflows and access results
- In-memory run registry — restarts lose all run state (workflow checkpoints survive but the run list doesn't)
- Local artifact storage — model files are on disk, not in durable cloud storage
- No alert delivery — Prometheus rules are defined but there's no Alertmanager configured to actually send alerts
- No fairness or bias testing — the pipeline doesn't check for demographic parity or disparate impact
- Single instance only — the run registry and SQLite stores don't support multiple API replicas
