# AI Engineering Agent

Upload any dataset. Describe what you want to predict. The agents take it from there.

This isn't a template or a notebook you have to configure. You drop in a file, confirm which column you're trying to predict, and a set of agents runs the full ML pipeline — inspecting the data, training several models, picking the best one, and handing you a live prediction endpoint. If the model doesn't meet your score threshold, it tells you why and blocks deployment rather than giving you something that doesn't work.

It runs entirely on your machine. No cloud account, no API keys, no external services required.

---

## Getting started in one command

Make sure Docker Desktop is open and running, then:

    docker compose up --build

Go to http://localhost:8010

That's the whole setup. Docker builds the frontend and starts the backend together.

---

## If you prefer running it without Docker

You'll need Python 3.12 and Node 18+.

Set up Python:

    py -3.12 -m venv .venv312
    .\.venv312\Scripts\Activate.ps1
    pip install -r requirements.txt

Start the backend:

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload

Then in a second terminal:

    cd frontend
    npm install
    npm run dev

Open the URL that Vite prints (usually http://localhost:5173).

---

## What to expect when you use it

When you upload a file, the UI immediately profiles it — before you start training. It shows you the row count, column types, and which column looks like a prediction target. You confirm that target (or pick a different one), set your metric and minimum acceptable score, then click run.

From there the agents work through the pipeline:

1. Data agent — checks for missing values, duplicates, class imbalance, and columns that look like they contain future information (leakage)
2. Research agent — picks which ML methods make sense for your data
3. Model agent — trains multiple candidates on a training split and tests on held-out data it hasn't seen
4. Decision agent — checks if the winner beats your threshold and the data quality checks pass
5. If it passes, you get an approval prompt before anything is deployed
6. DevOps agent — packages the winning model and all its preprocessing into a single file
7. Monitoring agent — starts tracking every prediction you make

When a model is deployed, a prediction form appears in the UI. Fill in the feature values for a new example and it returns a prediction instantly — the same preprocessing pipeline the model was trained with runs automatically.

---

## What kind of data it handles

The system figures out the task type on its own:

- If your target column has categories (yes/no, red/blue/green, churned/retained), it runs classification and trains Logistic Regression, Random Forest, and XGBoost
- If your target column has continuous numbers (house price, temperature, revenue), it runs regression and trains Random Forest, Ridge, and XGBoost
- If there's no obvious target, it profiles your data and recommends clustering or anomaly detection approaches — but it won't deploy a model without a labeled target
- If you upload a COCO object detection JSON, it recognizes it and explains what tools you'd need rather than pretending it can train on annotation metadata alone

File formats that work: CSV, JSON, JSONL, XML, Excel (.xlsx and .xls), Parquet

---

## Running the tests

    .\.venv312\Scripts\python.exe -m pytest -v

---

## Try the included demo

Upload data/customer_churn_demo.csv. The system will suggest churn as the target — confirm it. Use F1 score as the metric and 0.80 as the minimum. Three models will train, the best one will be compared against 0.80, and if it passes you'll see an approval button. Approve it and the prediction panel appears. Enter customer values (tenure, monthly charges, contract type) and get a churn prediction with a confidence percentage.

---

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| POST | /api/profile | Profile a file — get column types and a suggested target without training anything |
| POST | /api/analyze | Run the full pipeline |
| GET | /api/runs/{run_id} | Fetch results for a run |
| POST | /api/runs/{run_id}/approve | Approve or reject a model before it gets deployed |
| POST | /api/runs/{run_id}/predict | Send feature values, get a prediction back |
| POST | /api/runs/{run_id}/feedback | Tell the system what the actual outcome was, for accuracy tracking |
| GET | /api/monitoring/{run_id} | See drift and performance signals for a deployed model |
| GET | /health | Check the service is alive |
| GET | /metrics | Prometheus-format metrics |

---

## Optional monitoring stack

If you want Prometheus, Grafana, MLflow, and PostgreSQL too:

    docker compose --profile infra up --build

Prometheus → http://localhost:9090
Grafana → http://localhost:3001
MLflow → http://localhost:5000

---

## What it can't do yet

A few honest gaps worth knowing before you take this further:

- There's no login system. Anyone who can reach the port can use the API.
- Runs are kept in memory. If the container restarts, the in-memory run list is gone (though model artifacts and workflow checkpoints survive in the volume).
- Monitoring tracks signals but doesn't send alerts. You'd need to configure Alertmanager or a similar tool to actually get notified.
- Models use reasonable default hyperparameters. There's no automatic tuning yet.
- The research agent uses a local catalog of methods. It doesn't search papers or external sources.

See ARCHITECTURE.md for how the internals are structured.
