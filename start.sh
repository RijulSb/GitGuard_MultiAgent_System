#!/bin/bash
set -e

MODE="${1:-all}"

echo "Starting GitGuard in mode: ${MODE}..."

cleanup() {
    echo "Stopping all processes..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

if [ "$MODE" = "api" ]; then
    echo "Starting GitGuard FastAPI Service on port 8000..."
    exec uvicorn src.service.api:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "ui" ]; then
    echo "Starting GitGuard Streamlit UI on port 8501..."
    exec streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
elif [ "$MODE" = "all" ]; then
    echo "Starting GitGuard FastAPI Service on port 8000..."
    uvicorn src.service.api:app --host 0.0.0.0 --port 8000 &
    API_PID=$!

    # Wait for API to be available
    echo "Waiting for API service to initialize..."
    sleep 3

    echo "Starting GitGuard Streamlit UI on port 8501..."
    streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    UI_PID=$!

    wait -n $API_PID $UI_PID
else
    echo "Executing custom command: $@"
    exec "$@"
fi
