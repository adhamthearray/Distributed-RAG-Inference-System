import argparse
import asyncio
import statistics
import time

import aiohttp


QUERIES = [
    "What is the definition of a distributed system?",
    "What are the main goals of a distributed system?",
    "What is process context?",
    "Explain loosely coupled and tightly coupled systems.",
    "What are the advantages of distributed systems?",
    "What is a single point of failure?",
    "How is synchronization handled in distributed systems?",
    "What is middleware in distributed systems?",
    "What are common distributed systems challenges?",
    "Explain transparency in distributed systems.",
]


async def send_request(session, url, task_id, timeout):
    query = QUERIES[task_id % len(QUERIES)]
    payload = {"id": task_id, "query": query}
    start = time.perf_counter()

    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            data = await response.json()
            elapsed = time.perf_counter() - start

            if response.status >= 400 or "error" in data:
                return {
                    "ok": False,
                    "status": response.status,
                    "latency": elapsed,
                    "error": data.get("error", "HTTP error"),
                }

            return {
                "ok": True,
                "status": response.status,
                "latency": elapsed,
                "worker_id": data.get("worker_id", "unknown"),
            }
    except Exception as e:
        return {
            "ok": False,
            "status": "request_failed",
            "latency": time.perf_counter() - start,
            "error": str(e),
        }


async def run_load_test(url, users, concurrency, timeout):
    connector = aiohttp.TCPConnector(limit=concurrency)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def limited_request(task_id):
            async with semaphore:
                return await send_request(session, url, task_id, client_timeout)

        tasks = [limited_request(i + 1) for i in range(users)]
        start = time.perf_counter()

        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)

            if len(results) % 50 == 0 or len(results) == users:
                print(f"Finished {len(results)}/{users} requests", flush=True)

        total_time = time.perf_counter() - start

    return results, total_time


def percentile(values, percent):
    if not values:
        return 0

    sorted_values = sorted(values)
    index = int((percent / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def print_summary(results, total_time):
    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    latencies = [result["latency"] for result in results]

    print("\nLoad test summary")
    print("-----------------")
    print(f"Total requests: {len(results)}")
    print(f"Successful:      {len(successes)}")
    print(f"Failed:          {len(failures)}")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Throughput:      {len(results) / total_time:.2f} requests/sec")

    if latencies:
        print(f"Avg latency:     {statistics.mean(latencies):.2f}s")
        print(f"Min latency:     {min(latencies):.2f}s")
        print(f"P50 latency:     {percentile(latencies, 50):.2f}s")
        print(f"P95 latency:     {percentile(latencies, 95):.2f}s")
        print(f"Max latency:     {max(latencies):.2f}s")

    if failures:
        print("\nFirst few failures:")
        for failure in failures[:5]:
            print(f"- status={failure['status']} latency={failure['latency']:.2f}s error={failure['error']}")


def main():
    parser = argparse.ArgumentParser(description="Load test the distributed RAG API.")
    parser.add_argument("--url", default="http://localhost:8080/ask")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    print(f"URL:         {args.url}")
    print(f"Users:       {args.users}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Timeout:     {args.timeout}s")

    results, total_time = asyncio.run(
        run_load_test(args.url, args.users, args.concurrency, args.timeout)
    )
    print_summary(results, total_time)


if __name__ == "__main__":
    main()
