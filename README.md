# Distributed RAG Inference System

### Fault-Tolerant GPU Task Distribution for High-Concurrency AI Requests

A distributed Retrieval-Augmented Generation (RAG) inference system designed to handle large-scale concurrent AI requests using intelligent load balancing, GPU worker orchestration, batching, and fault recovery mechanisms.

Built as a distributed computing project focused on scalability, resilience, and efficient LLM inference under heavy workloads.

---

# System Overview

This project simulates a real-world distributed AI serving architecture where hundreds or thousands of users send requests simultaneously to a centralized AI system.

The architecture distributes requests across multiple master nodes and GPU workers while maintaining:

* High availability
* Request continuity
* Fault tolerance
* Efficient GPU utilization
* Scalable task scheduling

The system combines:

* Distributed load balancing
* GPU task orchestration
* Retrieval-Augmented Generation (RAG)
* Batched LLM inference
* Worker recovery and task requeueing
* Nginx upstream retry logic

---

# Architecture

```text
Clients / Load Generator
            │
            ▼
     Nginx Load Balancer
            │
            ▼
 ┌─────────────────────┐
 │   Master Nodes      │
 │  node1 / node2      │
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ Local Worker Pools  │
 │ worker_0, worker_1  │
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ Thunder GPU VMs     │
 │ LLM Inference Layer │
 └─────────────────────┘
            │
            ▼
      RAG Responses
```

---

# Core Features

## Distributed Load Balancing

* Nginx distributes requests across multiple master nodes
* Automatic retry on backend failure
* Passive upstream failure handling
* Fault-aware request routing

---

## GPU Worker Orchestration

* Worker-based batch processing
* Dynamic task assignment
* Parallel GPU inference execution
* Worker cooldown recovery

---

## Retrieval-Augmented Generation (RAG)

* ChromaDB vector retrieval
* Context-aware prompt generation
* Course-document-grounded answers
* Embedding-based semantic search

---

## Fault Tolerance

The system can recover from:

* GPU worker failures
* Batch processing failures
* Temporary worker timeouts
* Master node worker exhaustion

Failed tasks are:

1. Requeued automatically
2. Reassigned to healthy workers
3. Processed without exposing internal failure to clients

---

## High-Concurrency Support

Designed to simulate:

* 1000+ concurrent requests
* Batched GPU inference
* Parallel request execution
* Queue-based scheduling

---

# Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

## AI / RAG

* Hugging Face Transformers
* PyTorch
* ChromaDB
* LangChain
* sentence-transformers

## Infrastructure

* Docker Compose
* Nginx
* Thread-based scheduling

## GPU Layer

* CUDA
* NVIDIA GPU VMs
* pynvml

---


```

---

# Request Workflow

## 1. Client Sends Request

```json
{
  "id": 1,
  "query": "What is a distributed system?"
}
```

---

## 2. Nginx Receives Request

Nginx forwards the request to an available master node.

---

## 3. Master Node Schedules Task

The master:

* Creates internal task objects
* Assigns tasks to workers
* Queues overflow requests

---

## 4. Worker Batching

Workers collect requests into batches:

```env
BATCH_SIZE=30
```

This improves GPU throughput by processing multiple prompts together.

---

## 5. RAG Retrieval

The system:

* Embeds the query
* Retrieves relevant context from ChromaDB
* Builds grounded prompts

---

## 6. GPU Inference

The GPU VM:

* Receives batched prompts
* Runs LLM inference
* Returns generated responses

---

## 7. Response Mapping

Responses are mapped back using:

```python
task_id
```

ensuring every client receives the correct answer.

---

# Fault Recovery Workflow

```text
Worker Fails
     │
     ▼
Batch Returned To Master
     │
     ▼
Tasks Requeued
     │
     ▼
Healthy Worker Picks Batch
     │
     ▼
Client Still Receives Response
```

The client remains unaware of internal worker failure.

---

# Nginx Retry Logic

Configured retry conditions:

```nginx
proxy_next_upstream error timeout http_502 http_503 http_504 non_idempotent;
```

This allows:

* Retry on failed master nodes
* POST request retry support
* Automatic failover

---

# Environment Configuration

Example `.env`

```env
NODE1_NUM_WORKERS=2
NODE2_NUM_WORKERS=2

BATCH_SIZE=30

WORKER_FAIL_COOLDOWN=30
TASK_TIMEOUT_SECONDS=120

ENABLE_FAILURE_SIMULATION=true
FAILURE_RATE=0.1
SIMULATED_FAILURE_TIMEOUT=10
```

---

# Running The System

## 1. Clone Repository

```bash
git clone https://github.com/adhamthearray/Distributed-RAG-Inference-System.git
cd Distributed-RAG-Inference-System
```

---

## 2. Build Containers

```bash
docker compose build
```

---

## 3. Start Services

```bash
docker compose up
```

---

## 4. Run Load Test

```bash
python load_test.py --users 1000 --concurrency 1000
```

---

# Example Load Test Goals

Target scenario:

```text
1000 Requests
0 Failed Requests
Simulated GPU Failures Enabled
Automatic Recovery Active
```

---


---

# System Strengths

* Distributed request routing
* Batched GPU inference
* Request continuity under failure
* Automatic worker recovery
* Scalable architecture
* Clean separation of responsibilities
* RAG-grounded AI responses

---

# Current Limitations

* Passive Nginx health handling
* Batch-level failure simulation
* Thread-based local scheduling
* No active GPU health probing
* Queue latency under extreme load

---

# Future Improvements

* Kubernetes deployment
* Dynamic autoscaling
* Redis/RabbitMQ task queues
* Active health monitoring
* Smarter adaptive batching
* Multi-region deployment
* GPU-aware scheduling

---

# Educational Focus

This project demonstrates practical concepts in:

* Distributed Computing
* Fault-Tolerant Systems
* AI Infrastructure
* GPU Scheduling
* Load Balancing
* Retrieval-Augmented Generation
* Concurrent Systems Design

---

# Contributors

* Adham and Team

---

# Final Note

This project was designed to simulate how modern AI infrastructure systems maintain reliability under high concurrency and partial system failure.

The architecture separates:

* external request distribution,
* internal task orchestration,
* and GPU inference execution

to preserve scalability, resilience, and response continuity even during simulated worker failures.
