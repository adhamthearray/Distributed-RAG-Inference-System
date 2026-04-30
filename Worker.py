from RAG_system.RAG.ask import ask_question

class WorkerNode:
    def __init__(self, workerID):
        self.id = workerID
        self.status = 'idle'
        
    def processTask(self, task):
        self.status = 'busy'

        try:
            answer = ask_question(task['user_query'])
            return self._response(task, answer)
        except Exception as e:
            return self._error_response(task, e)
        finally:
            self.status = 'idle'

    def _response(self, task, answer):
        return {
            "ok": True,
            "task_id": task["task_id"],
            "worker_id": self.id,
            "answer": answer
        }

    def _error_response(self, task, error):
        error_message = str(error)
        status_code = 429 if "429" in error_message or "rate limit" in error_message.lower() else 500

        return {
            "ok": False,
            "task_id": task["task_id"],
            "worker_id": self.id,
            "answer": None,
            "error": error_message,
            "status_code": status_code
        }
