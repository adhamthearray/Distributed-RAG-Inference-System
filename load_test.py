import argparse
import asyncio
import csv
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
                    "task_id": task_id,
                    "query": query,
                    "status": response.status,
                    "latency": elapsed,
                    "worker_id": data.get("worker_id", "unknown"),
                    "answer": data.get("answer"),
                    "gpu_utilization": data.get("gpu_utilization"),
                    "error": data.get("error", "HTTP error"),
                }

            return {
                "ok": True,
                "task_id": task_id,
                "query": query,
                "status": response.status,
                "latency": elapsed,
                "worker_id": data.get("worker_id", "unknown"),
                "answer": data.get("answer"),
                "gpu_utilization": data.get("gpu_utilization"),
                "error": "",
            }
    except Exception as e:
        return {
            "ok": False,
            "task_id": task_id,
            "query": query,
            "status": "request_failed",
            "latency": time.perf_counter() - start,
            "worker_id": "unknown",
            "answer": "",
            "gpu_utilization": "",
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

    gpu_by_worker = {}
    for result in successes:
        utilization = result.get("gpu_utilization")
        if utilization is None or utilization == "":
            continue

        try:
            utilization = float(utilization)
        except (TypeError, ValueError):
            continue

        worker_id = result.get("worker_id", "unknown")
        gpu_by_worker.setdefault(worker_id, []).append(utilization)

    if gpu_by_worker:
        print("\nGPU utilization summary")
        print("-----------------------")
        all_utilizations = []
        for worker_id, utilizations in sorted(gpu_by_worker.items()):
            all_utilizations.extend(utilizations)
            print(
                f"{worker_id}: "
                f"avg {statistics.mean(utilizations):.1f}% | "
                f"min {min(utilizations):.1f}% | "
                f"max {max(utilizations):.1f}% | "
                f"samples {len(utilizations)}"
            )

        print(f"Overall avg:     {statistics.mean(all_utilizations):.1f}%")


def write_results_csv(results, output_path):
    fields = [
        "task_id",
        "ok",
        "status",
        "latency",
        "worker_id",
        "gpu_utilization",
        "query",
        "answer",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["task_id"]):
            writer.writerow({field: result.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description="Load test the distributed RAG API.")
    parser.add_argument("--url", default="http://localhost:8080/ask")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", default="load_test_results.csv")
    args = parser.parse_args()

    print(f"URL:         {args.url}")
    print(f"Users:       {args.users}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Timeout:     {args.timeout}s")

    results, total_time = asyncio.run(
        run_load_test(args.url, args.users, args.concurrency, args.timeout)
    )
    print_summary(results, total_time)
    write_results_csv(results, args.output)
    print(f"\nSaved request answers to {args.output}")


if __name__ == "__main__":
    main()
