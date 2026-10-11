# YuE2 vast.ai PyWorker (public — workers clone it without auth)

Custom serverless worker: `model_server.py` runs YuE2 once per worker,
`worker.py` proxies the vast router to it. Secrets (`HF_TOKEN`) live in
the vast template env, never here.

Template needs: image with torch/CUDA (e.g. `vastai/pytorch`), the vendored
`yue2_infer` wheel at `/app/`, env `PYWORKER_REPO` = this repo,
`HF_TOKEN`, `BUCKET`, ffmpeg installed, start command `bash start_server.sh`.
