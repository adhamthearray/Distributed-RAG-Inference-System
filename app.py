import os
import socket
import threading
import asyncio
import time
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Worker import WorkerNode

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 3))
GPU_SERVERS = os.getenv("GPU_SERVERS", "").split(",")
WORKER_FAIL_COOLDOWN = float(os.getenv("WORKER_FAIL_COOLDOWN", 30))
TASK_TIMEOUT_SECONDS = float(os.getenv("TASK_TIMEOUT_SECONDS", 120))

class Master:
    def __init__(self, num_workers):
        self.lock = threading.Lock()
        # Initialize available workers
        hostname = socket.gethostname()
        self.available_workers = deque(WorkerNode(f"worker_{hostname}_{i}" , GPU_SERVERS[i]) for i in range(num_workers))
        # Queue for overflow tasks
        self.waiting_tasks = deque()
        self.failed_workers = []
        threading.Thread(target=self._timeout_monitor, daemon=True).start()

    def submit_task(self, task):
        event = threading.Event()
        result_container = {}
        task = {
            'task' : task,
            'event' : event,
            'result_container': result_container,
            'created_at': time.time()
        }
        with self.lock:
            if self.available_workers:
                worker = self.available_workers.popleft()
                print(f"[Master] Assigning task {task['task']['task_id']} to {worker.id}", flush=True)
                if worker.appendTask(task):
                    threading.Thread(target=self._execute_task, args=(worker,)).start()
                else:
                    self.available_workers.append(worker)
            else:
                self.waiting_tasks.append(task)

        return event, result_container

    def _execute_task(self, worker):
        result = None
        try:
            result = worker.processBatch()
        finally:
            with self.lock:
                if result and result.get("ok") is False:
                    if worker not in self.failed_workers:
                        self.failed_workers.append(worker)

                    batch = result.get("batch")
                    for item in reversed(batch):
                        self.waiting_tasks.appendleft(item)

                    print(f"[Master] Batch failed on {worker.id}. Requeued {len(batch)} tasks.", flush=True)

                    threading.Thread(
                        target=self._recover_worker_after_cooldown,
                        args=(worker,),
                        daemon=True
                    ).start()

                    while self.waiting_tasks and self.available_workers:
                        next_worker = self.available_workers.popleft()
                        self._feed_worker_from_waiting(next_worker)
                    return
                if self._feed_worker_from_waiting(worker):
                    return

                self.available_workers.append(worker)

    def _feed_worker_from_waiting(self, worker):
        while self.waiting_tasks:
            task = self.waiting_tasks.popleft()
            worker_full = worker.appendTask(task)
            if worker_full:
                threading.Thread(target=self._execute_task, args=(worker,)).start()
                return True

        if worker.hasTasks():
            threading.Thread(target=self._execute_task, args=(worker,)).start()
            return True

        return False

    def _recover_worker_after_cooldown(self, worker):
        time.sleep(WORKER_FAIL_COOLDOWN)

        with self.lock:
            if worker in self.failed_workers:
                self.failed_workers.remove(worker)
                self.available_workers.append(worker)
                print(f"[Master] Worker {worker.id} recovered after cooldown", flush=True)

                while self.waiting_tasks and self.available_workers:
                    next_worker = self.available_workers.popleft()
                    self._feed_worker_from_waiting(next_worker)

    def _timeout_monitor(self):
        while True:
            time.sleep(1)

            with self.lock:
                now = time.time()
                still_waiting = deque()

                while self.waiting_tasks:
                    item = self.waiting_tasks.popleft()
                    if now - item["created_at"] > TASK_TIMEOUT_SECONDS:
                        self._fail_task(item, "Task timed out while waiting for an available worker")
                    else:
                        still_waiting.append(item)

                self.waiting_tasks = still_waiting

    def _fail_task(self, item, error):
        task = item["task"]
        item["result_container"].update({
            "ok": False,
            "task_id": task["task_id"],
            "worker_id": None,
            "answer": None,
            "error": error,
            "status_code": 503
        })
        item["event"].set()

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

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    return await process_request(req)
