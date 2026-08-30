#!/bin/bash
set -euo pipefail

uvicorn returns_app.api:app --host 127.0.0.1 --port 8000 --workers 2 &
UVICORN_PID=$!

# Wait until app is up before nginx accepts traffic
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

nginx -g "daemon off;" &
NGINX_PID=$!

trap 'kill $UVICORN_PID $NGINX_PID 2>/dev/null || true' SIGTERM SIGINT

wait -n $UVICORN_PID $NGINX_PID
exit $?
