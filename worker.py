"""vast.ai PyWorker: HTTP proxy  router -> local model_server.py.

Routes: POST /generate (FIFO, one song at a time).
Benchmark runs once at worker start on a ~15s payload (runs=1) so the
readiness probe costs a fraction of a cent, not a full song.
"""

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

MODEL_LOG_FILE = "/var/log/model/server.log"

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,  # one song at a time (FIFO)
            max_queue_time=600.0,
            workload_calculator=lambda payload: 100.0,
            benchmark_config=BenchmarkConfig(
                dataset=[{
                    "style": "Test beep",
                    "lyrics": "[verse]\nbeep",
                    "cot": "off",
                    "seed": 1,
                }],
                runs=1,
                concurrency=1,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["[yue2] pipeline ready."],
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError: Engine",
            "torch.OutOfMemoryError",
        ],
    ),
)

Worker(worker_config).run()
