import os
import socket
import threading
import asyncio
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

class Master:
    def __init__(self, num_workers):
        self.lock = threading.Lock()
        # Initialize available workers
        hostname = socket.gethostname()
        self.available_workers = deque(WorkerNode(f"worker_{hostname}_{i}" , GPU_SERVERS[i]) for i in range(num_workers))
        # Queue for overflow tasks
        self.waiting_tasks = deque()

    def submit_task(self, task):
        event = threading.Event()
        result_container = {}
        task = {
            'task' : task,
            'event' : event,
            'result_container': result_container
        }
        with self.lock:
            if self.available_workers:
                worker = self.available_workers.popleft()
                print(f"[Master] Assigning task {task['task']['task_id']} to {worker.id}", flush=True)
                if worker.appendTask(task):
                    # self.available_workers.pop()
                    threading.Thread(target=self._execute_task, args=(worker,)).start()
                else:
                    self.available_workers.append(worker)
            else:
                self.waiting_tasks.append(task)

        return event, result_container

    def _execute_task(self, worker):
        try:
            worker.processBatch()
        finally:
            with self.lock:
                while self.waiting_tasks:
                    task = self.waiting_tasks.popleft()
                    worker_full = worker.appendTask(task)
                    if worker_full:
                        threading.Thread(target=self._execute_task, args=(worker,)).start()
                        return

                if worker.hasTasks():
                    threading.Thread(target=self._execute_task, args=(worker,)).start()
                    return

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

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    return await process_request(req)
