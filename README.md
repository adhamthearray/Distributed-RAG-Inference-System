# Distributed RAG Compute Node Architecture

This document summarizes the architectural overhauls, concurrency implementations, and RAG optimizations made to the Compute Node layer of the distributed RAG system. 

## 1. Core Architecture: FastAPI & Threading Migration
Initially, the compute nodes were running as single-threaded `Flask` applications. To support highly concurrent requests behind the load balancer, we implemented a robust parallel processing pipeline:

* **FastAPI Migration (`app.py`)**: Replaced Flask with an asynchronous FastAPI server. The endpoints (`/ask` and `/predict`) now operate asynchronously, allowing the server to accept thousands of incoming requests without blocking the event loop.
* **Master Node Scheduler (`app.py`)**: A centralized `Master` class was introduced per container. It acts as the task dispatcher.
  * **Worker Pool**: Maintains a pool of `WorkerNode` objects (configurable via `NUM_WORKERS`, defaulting to 3 per container).
  * **Queuing System**: Uses thread-safe logic (`threading.Lock`) to quickly assign incoming requests to available workers, or append them to a `waiting_tasks` overflow queue if the pool is busy.
  * **Non-blocking Execution**: Instead of blocking, the Master spawns a `threading.Thread` to execute the RAG/LLM workload, returning a `threading.Event()` to FastAPI. The API route simply `await`s the event, enabling true concurrent handling.

## 2. Dynamic Rate Limiting & Backpressure Management
The Groq API free tier imposes a strict limit of **6000 Tokens Per Minute (TPM)**. Because each query demands ~1300 tokens, concurrent bursts (e.g., 10 simultaneous requests) instantly exceed this limit, causing `429 Rate Limit Exceeded` errors. We implemented a two-layered defense to overcome this:

* **Staggered Execution (Master Node)**: Added a configurable `RATE_LIMIT_DELAY` (2.0 seconds) in the Master's dispatch loop. This artificially slows down the execution rate across threads to spread the token burn over time rather than instantaneously.
* **Worker Error Handling (`Worker.py`)**: If the LLM call fails, the worker returns a clear execution error instead of crashing the request.

## 3. RAG Pipeline & Vector DB Optimizations
The previous chunking parameters caused the LLM to reply "I don't know" to many questions because critical context was fragmented or missed during retrieval.

* **Chunking Strategy (`fill_db.py`)**: Increased `chunk_size` from `300` to `1000` characters, and increased `chunk_overlap` from `100` to `200`. This ensures paragraphs, definitions, and bulleted lists remain intact.
* **Retrieval Window (`ask.py`)**: Increased `n_results` from `4` to `6`, providing the LLM with a broader context window.

## 4. Deployment Enhancements
* **Baked-in Vector Database (`Dockerfile`)**: We modified the Docker build process to execute `RUN python RAG_system/RAG/fill_db.py` directly during the `--build` phase. 
  * *Why this matters*: Previously, synchronizing the database via volume mounts was error-prone across Windows/Docker hosts. Now, anytime a PDF is modified, running `docker-compose up --build` automatically generates a fresh, embedded SQLite vector database across all nodes seamlessly.
* **Nginx Load Balancer (`nginx.conf`)**: Updated the routing configuration to proxy both `/predict` and `/ask` traffic seamlessly to the backend compute nodes.

## 5. Testing Suite
* **Concurrency Load Tester (`test_concurrent.py`)**: Created an asynchronous Python load-testing script utilizing `aiohttp`. It fires 10 concurrent requests at the exact same millisecond to the Nginx load balancer. 
* **Validation**: This script successfully validates that Nginx distributes the load, the Master Node queues the tasks properly, the Workers execute them in parallel, and the Rate Limiter safely throttles Groq API limits.

---
*Note: If testing with high concurrency, ensure your Nginx `proxy_read_timeout` is high enough (e.g., > 60s), as the exponential backoff might intentionally delay a request longer than the default 60-second Nginx timeout!*
