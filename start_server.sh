#!/bin/bash
# vast.ai serverless start script: fetch code, install deps, boot model server, run proxy.
set -e
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)
echo "[diag] pip config:"; pip config list 2>&1 | head -20
echo "[diag] pip env:"; env | grep -i 'pip\|index\|proxy\|mirror' || echo none
echo "[diag] python: $(python3 --version) pip: $(python3 -m pip --version)"
cd /app
if [ ! -d pyworker ]; then
  git clone "$PYWORKER_REPO" pyworker
fi
cd pyworker
git pull --ff-only || true
cp /app/yue2_infer-0.1.5-py3-none-any.whl ./ 2>/dev/null || true
# --no-cache-dir: some hosts serve stale cached wheels whose hashes no longer
# match the index (fatal "HASHES DO NOT MATCH" otherwise).
# --retries/--timeout: host uplinks to PyPI can be flaky (truncated downloads).
python3 -m pip install -q --no-cache-dir --retries 20 --timeout 120 -r requirements.txt
mkdir -p /var/log/model
HF_TOKEN="$HF_TOKEN" BUCKET="$BUCKET" MODEL_PORT=18000 \
  nohup python3 model_server.py > /var/log/model/server.log 2>&1 &
echo "[start] model server booting, launching worker.py"
exec python3 worker.py
