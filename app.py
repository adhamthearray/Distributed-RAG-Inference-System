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
        self.available_workers = [WorkerNode(f"worker_{hostname}_{i}" , GPU_SERVERS[i]) for i in range(num_workers)]
        # Queue for overflow tasks
        self.waiting_tasks = deque()

    def submit_task(self, task):
        event = threading.Event()
        result_container = {}

        with self.lock:
            if self.available_workers:
                worker = self.available_workers.pop()
                print(f"[Master] Assigning task {task['task_id']} to {worker.id}", flush=True)
                # Start non-blocking thread for execution
                threading.Thread(target=self._execute_task, args=(worker, task, event, result_container)).start()
            else:
                self.waiting_tasks.append((task, event, result_container))

        return event, result_container

    def _execute_task(self, worker, task, event, result_container):
        try:
            # The actual execution happens here (calling RAG/LLM inside processTask)
            result = worker.processTask(task)
            result_container.update(result)
        except Exception as e:
            result_container.update({
                "ok": False,
                "task_id": task["task_id"],
                "worker_id": worker.id,
                "answer": None,
                "error": str(e),
                "status_code": 500
            })
        finally:
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

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    return await process_request(req)
