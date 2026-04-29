import time
import re
from RAG_system.RAG.ask import ask_question

class WorkerNode:
    def __init__(self, workerID):
        # fixed trailing comma bug on workerID
        self.id = workerID
        self.status = 'idle'
        
    def processTask(self, task):
        self.status = 'busy'
        
        max_retries = 5
        answer = "I don't know."
        chunks = []
        
        for attempt in range(max_retries):
            try:
                answer, chunks = ask_question(task['user_query'])
                break # Success!
            except Exception as e:
                error_str = str(e)
                # Parse Groq's specific "Please try again in Xs." message
                match = re.search(r"try again in (\d+\.?\d*)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0 # Add 1s buffer to be safe
                elif "429" in error_str or "rate limit" in error_str.lower():
                    # Fallback exponential backoff: 5s, 10s, 20s, 40s
                    wait_time = (2 ** attempt) * 5 
                else:
                    # Not a rate limit error, break immediately
                    answer = f"Execution Error: {error_str}"
                    break
                    
                if attempt < max_retries - 1:
                    print(f"[{self.id}] 429 Rate Limit. Waiting {wait_time:.2f}s before retry {attempt+1}...", flush=True)
                    time.sleep(wait_time)
                else:
                    answer = f"Error: Rate Limit Exceeded after {max_retries} retries."

        self.status = 'idle'
        return {
            "task_id": task["task_id"],
            "worker_id": self.id,
            "answer": answer,
            "chunks": chunks
        }