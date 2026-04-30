import os
import socket
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import deque
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Worker import WorkerNode

app = FastAPI()

# Default to 3 workers per compute node
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 3))

# Delay between worker executions to respect API Rate Limits (in seconds)
RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", 2.0))

# Maximum time a worker can spend on one task before it is considered failed
TASK_TIMEOUT = float(os.environ.get("TASK_TIMEOUT", 60.0))

SIMULATE_WORKER_FAILURE = os.environ.get("SIMULATE_WORKER_FAILURE", "false").lower() == "true"
SIMULATED_FAILURE_DELAY = float(os.environ.get("SIMULATED_FAILURE_DELAY", TASK_TIMEOUT + 5))
simulate_next_worker_failure = SIMULATE_WORKER_FAILURE

class Master:
    def __init__(self, num_workers):
        self.lock = threading.Lock()
        # Initialize available workers
        hostname = socket.gethostname()
        self.available_workers = [WorkerNode(f"worker_{hostname}_{i}") for i in range(num_workers)]
        # Queue for overflow tasks
        self.waiting_tasks = deque()
        # Queue for workers that timed out while processing a task
        self.failed_workers = deque()

    def submit_task(self, task):
        global simulate_next_worker_failure

        event = threading.Event()
        result_container = {}

        with self.lock:
            if self.available_workers:
                worker = self.available_workers.pop()
                worker.simulate_failure = False
                worker.failure_delay = 0

                if simulate_next_worker_failure:
                    worker.simulate_failure = True
                    worker.failure_delay = SIMULATED_FAILURE_DELAY
                    simulate_next_worker_failure = False
                    print(f"[Master] Failure simulation enabled for {worker.id}", flush=True)

                print(f"[Master] Assigning task {task['task_id']} to {worker.id}", flush=True)
                # Start non-blocking thread for execution
                threading.Thread(target=self._execute_task, args=(worker, task, event, result_container)).start()
            else:
                self.waiting_tasks.append((task, event, result_container))

        return event, result_container

    def _execute_task(self, worker, task, event, result_container):
        import time
        timed_out = False
        try:
            # Artificially slow down task execution to avoid instantly hitting the 6000 TPM limit
            time.sleep(RATE_LIMIT_DELAY)

            # The actual execution happens here (calling RAG/LLM inside processTask)
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(worker.processTask, task)
            try:
                result = future.result(timeout=TASK_TIMEOUT)
            except TimeoutError:
                timed_out = True
                executor.shutdown(wait=False, cancel_futures=True)
                result = None
            else:
                executor.shutdown(wait=False)

            if timed_out:
                return

            result_container.update(result)
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            if timed_out:
                with self.lock:
                    self.failed_workers.append(worker)
                    print(f"[Master] {worker.id} timed out. Moving it to failed_workers.", flush=True)
                    if self.available_workers:
                        next_worker = self.available_workers.pop()
                        print(f"[Master] Reassigning task {task['task_id']} to {next_worker.id}", flush=True)
                        threading.Thread(target=self._execute_task, args=(next_worker, task, event, result_container)).start()
                    elif len(self.failed_workers) >= NUM_WORKERS:
                        result_container.update({
                            "ok": False,
                            "task_id": task["task_id"],
                            "worker_id": worker.id,
                            "answer": None,
                            "error": "All workers failed or are unavailable",
                            "status_code": 503
                        })
                        event.set()
                    else:
                        self.waiting_tasks.appendleft((task, event, result_container))
                return

            # Signal that the task is complete so FastAPI can return the response
            event.set()

            # Schedule the next task or return worker to the available pool
            with self.lock:
                if self.waiting_tasks:
                    next_task, next_event, next_result_container = self.waiting_tasks.popleft()
                    threading.Thread(target=self._execute_task, args=(worker, next_task, next_event, next_result_container)).start()
                else:
                    self.available_workers.append(worker)

master = Master(num_workers=NUM_WORKERS)

class QueryRequest(BaseModel):
    id: int = 0
    query: str

async def process_request(req: QueryRequest):
    task = {"task_id": req.id, "user_query": req.query}
    event, result_container = master.submit_task(task)

    # Wait for the thread to finish asynchronously without blocking other FastAPI handlers
    await asyncio.to_thread(event.wait)

    if result_container.get("ok") is False:
        status_code = result_container.get("status_code", 500)
        return JSONResponse(status_code=status_code, content=result_container)

    return result_container

@app.post("/predict")
async def predict_endpoint(req: QueryRequest):
    return await process_request(req)

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    return await process_request(req)
