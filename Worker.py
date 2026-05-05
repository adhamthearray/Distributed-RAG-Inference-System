import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from RAG_system.RAG.ask import ask_question


load_dotenv(Path(__file__).resolve().parent / ".env")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 3))


class WorkerNode:
    def __init__(self, workerID, GPUServer):
        self.id = workerID
        self.batch = []
        self.lock = threading.Lock()
        self.status = "idle"
        self.GPUServer = GPUServer

    def appendTask(self, task):
        with self.lock:
            self.batch.append(task)
            return len(self.batch) >= BATCH_SIZE

    def hasTasks(self):
        with self.lock:
            return len(self.batch) > 0

    def processBatch(self):
        self.status = "busy"

        with self.lock:
            batch = list(self.batch)
            self.batch.clear()

        try:
            batchData = [
                {
                    "task_id": item["task"]["task_id"],
                    "query": item["task"]["user_query"]
                }
                for item in batch
            ]

            response = ask_question(batchData, self.GPUServer)
            return self._response(response, batch)
        except Exception as e:
            return self._error_response(batch, e)
        finally:
            self.status = "idle"

    def _response(self, response, batch):
        pending_by_task_id = {
            item["task"]["task_id"]: item
            for item in batch
        }

        responses = response.get("responses")
        gpu_utilization = response.get("gpu_utilization")

        if response.get("ok") is False:
            error_message = response.get("error", "GPU inference failed")
            for pending in pending_by_task_id.values():
                task = pending["task"]
                pending["result_container"].update({
                    "ok": False,
                    "task_id": task["task_id"],
                    "worker_id": self.id,
                    "answer": None,
                    "error": error_message,
                    "status_code": 500,
                    "gpu_utilization": gpu_utilization
                })
                pending["event"].set()
            return

        for item in responses:
            task_id = item["task_id"]
            pending = pending_by_task_id.get(task_id)

            if pending is None:
                continue

            pending["result_container"].update({
                "ok": True,
                "task_id": task_id,
                "worker_id": self.id,
                "answer": item.get("answer"),
                "gpu_utilization":gpu_utilization
            })
            pending["event"].set()

    def _error_response(self, batch, error):
        error_message = str(error)
        status_code = 429 if "429" in error_message or "rate limit" in error_message.lower() else 500

        for item in batch:
            task = item["task"]
            item["result_container"].update({
                "ok": False,
                "task_id": task["task_id"],
                "worker_id": self.id,
                "answer": None,
                "error": error_message,
                "status_code": status_code
            })
            item["event"].set()

        return {
            "ok": False,
            "worker_id": self.id,
            "error": error_message,
            "status_code": status_code
        }
