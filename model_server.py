"""YuE2 model server for vast.ai PyWorker setup.

Loads the pipeline ONCE, serves POST /generate on 127.0.0.1:18000.
Started by start_server.sh before worker.py. Logs readiness line for on_load.

Request JSON: {style, lyrics, cot, seed, cfg_scale, title, description, bpm, language}
Response JSON: {id, title, seed, duration_s, bucket_wav, bucket_meta, meta, info}
The finished song is uploaded to the private HF bucket; nothing stays local.
"""

import json
import os
import random
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_REPO = "m-a-p/YuE2-3B"
BUCKET = os.environ.get("BUCKET", "directedbykobyperez/YuE2-mysongs-private")
PORT = int(os.environ.get("MODEL_PORT", "18000"))

PIPE = None


def ensure_wheel():
    try:
        import yue2  # noqa: F401
        return
    except ImportError:
        pass
    import sys
    cands = ["/app/yue2_infer-0.1.5-py3-none-any.whl", "./yue2_infer-0.1.5-py3-none-any.whl"]
    for whl in cands:
        if os.path.exists(whl):
            print(f"[yue2] runtime-installing {whl}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "-q", whl],
                check=True, capture_output=True,
            )
            return
    raise RuntimeError("vendored yue2 wheel not found")


def get_pipe():
    global PIPE
    if PIPE is None:
        ensure_wheel()
        import torch
        from yue2 import YuE2Pipeline

        print("[yue2] loading m-a-p/YuE2-3B ...", flush=True)
        PIPE = YuE2Pipeline.from_pretrained(
            MODEL_REPO, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("[yue2] pipeline ready.", flush=True)
    return PIPE


def render(payload):
    style = (payload.get("style") or "").strip()
    lyrics = (payload.get("lyrics") or "").strip()
    cot = payload.get("cot") or "full"
    if not style or not lyrics:
        return {"error": "style and lyrics are required"}
    try:
        seed = int(payload.get("seed") or 0)
    except Exception:
        seed = 0
    if seed < 1:
        seed = random.randint(1, 10**6)
    try:
        cfg_scale = float(payload.get("cfg_scale") or 1.0)
    except Exception:
        cfg_scale = 1.0

    gen_kwargs = dict(style=style, lyrics=lyrics, cot=cot, seed=seed)
    if cfg_scale != 1.0:
        gen_kwargs["cfg_scale"] = cfg_scale
    song = get_pipe()(**gen_kwargs)

    song_id = secrets.token_hex(6)
    tmp = tempfile.mkdtemp(prefix=f"yue2_{song_id}_")
    flac_path = os.path.join(tmp, "song.flac")
    song.save(flac_path)
    wav_path = os.path.join(tmp, f"{song_id}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-ar", "48000", "-ac", "2", wav_path],
        capture_output=True, check=True,
    )
    try:
        import soundfile as sf
        duration = float(sf.info(wav_path).duration)
    except Exception:
        duration = 0.0

    title = (payload.get("title") or "").strip()
    if not title:
        words = style.replace("\n", " ").split()[:8]
        title = " ".join(words).title()[:80] if words else f"YuE2 Song {song_id[:6]}"
    description = (payload.get("description") or "").strip() or style
    try:
        bpm_val = int(payload.get("bpm")) if payload.get("bpm") else None
    except Exception:
        bpm_val = None

    meta = {
        "id": song_id,
        "title": title,
        "description": description,
        "lyrics": lyrics,
        "caption": description,
        "bpm": bpm_val,
        "language": payload.get("language") or "",
        "duration": duration,
        "seed": seed,
        "generation_mode": cot,
        "lora": None,
        "audio_file": f"{song_id}.wav",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(tmp, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if not os.environ.get("HF_TOKEN", ""):
        return {"error": "HF_TOKEN not configured on worker", "id": song_id, "meta": meta}
    from huggingface_hub import batch_bucket_files
    batch_bucket_files(
        BUCKET,
        add=[
            (wav_path, f"songs/{song_id}/{song_id}.wav"),
            (meta_path, f"songs/{song_id}/meta.json"),
        ],
    )
    return {
        "id": song_id,
        "title": title,
        "seed": seed,
        "duration_s": round(duration, 1),
        "bucket_wav": f"hf://buckets/{BUCKET}/songs/{song_id}/{song_id}.wav",
        "bucket_meta": f"hf://buckets/{BUCKET}/songs/{song_id}/meta.json",
        "meta": meta,
        "info": f"seed={seed} | {duration:.0f}s | uploaded to bucket songs/{song_id}/",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            payload = {}
        try:
            out = render(payload)
            body = json.dumps(out).encode()
            self.send_response(200)
        except Exception as e:  # noqa: BLE001 - report worker errors as JSON
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    get_pipe()  # load before serving so on_load fires once ready
    print(f"[yue2] serving on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
