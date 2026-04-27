from RAG_system.RAG.ask import ask_question
class WorkerNode:
    def __init__(self , workerID):
        self.id = workerID,
        self.status = 'idle'
    def processTask(self , task):
        self.status = 'busy'
        response =  ask_question(task['user_query'])
        self.status = 'idle'
        return {
            "task_id": task["task_id"],
            "worker_id": self.id,
            "answer": response
        }
    
        