import asyncio
import time
import aiohttp

async def fetch(session, task_id, query):
    url = "http://localhost:8080/ask"
    payload = {"id": task_id, "query": query}
    
    start_time = time.time()
    try:
        # Send the POST request
        async with session.post(url, json=payload) as response:
            result = await response.json()
            elapsed = time.time() - start_time
            # Print which worker handled it and how long it took
            worker_id = result.get('worker_id', 'Unknown')
            error = result.get('error')
            answer = result.get('answer', 'No answer returned')
            chunks = result.get('chunks', [])
            
            if error:
                print(f"[Task {task_id}] ERROR: {error}\n")
            else:
                print(f"[Task {task_id}] finished in {elapsed:.2f}s | Handled by: {worker_id}")
                print(f"      Q: {query}")
                print(f"      A: {answer}")
                print(f"      [CHUNKS USED]:")
                for c_idx, chunk in enumerate(chunks):
                    # print first 150 chars and remove newlines to keep it readable
                    clean_chunk = chunk.replace('\n', ' ').strip()
                    print(f"        {c_idx+1}. {clean_chunk[:150]}...")
                print("\n")
                
            return result
    except Exception as e:
        print(f"[Task {task_id}] Request failed: {e}")

async def main():
    # 10 test queries to simulate concurrent traffic
    queries = [
        "What is the definition of a distributed system?",
        "What are the main goals of a distributed system?",
        "What is process context?",
        "Explain the difference between loosely coupled and tightly coupled systems.",
        "What are the advantages of distributed systems over centralized systems?",
        "What is a single point of failure?",
        "How is synchronization handled in distributed systems?",
        "What is middleware in the context of distributed systems?",
        "What are the common challenges in designing distributed systems?",
        "Explain the concept of transparency in distributed systems."
    ]
    
    print(f"🚀 Sending {len(queries)} concurrent requests to the cluster (http://localhost:8080/ask)...")
    start = time.time()
    
    # We use aiohttp to send requests non-blocking in parallel
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, query in enumerate(queries):
            tasks.append(fetch(session, i+1, query))
        
        # Fire them all at the exact same time
        await asyncio.gather(*tasks)
        
    print(f"\n✅ All requests finished in {time.time() - start:.2f}s total.")

if __name__ == "__main__":
    # Windows typically requires this policy for asyncio sometimes, but asyncio.run() works fine in 3.10+
    asyncio.run(main())
