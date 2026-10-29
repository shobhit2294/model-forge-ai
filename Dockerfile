# ── Stage 1: Build the React frontend ────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json .
RUN npm install
COPY frontend .
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /service

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 180 -r requirements.txt

# Copy application code
COPY app ./app

# Copy demo data (so /api/analyze works with the bundled CSV inside the container)
COPY data ./data

# Copy frontend build output as the static files served by FastAPI
COPY --from=frontend /ui/dist ./app/static

# Runtime artifacts directory (models saved here at request time)
RUN mkdir -p artifacts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]