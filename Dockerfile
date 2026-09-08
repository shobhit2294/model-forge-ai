FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json .
RUN npm install
COPY frontend .
RUN npm run build

FROM python:3.12-slim
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 180 -r requirements.txt
COPY app ./app
COPY --from=frontend /ui/dist ./app/static
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]