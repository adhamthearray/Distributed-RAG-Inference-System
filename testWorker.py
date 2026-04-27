from Worker import WorkerNode
worker1 = WorkerNode(1)
processedTask = worker1.processTask({'user_query':'What is process context' ,'task_id' : '1' })
print(processedTask['answer'])