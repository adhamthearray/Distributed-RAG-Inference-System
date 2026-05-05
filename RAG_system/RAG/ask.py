from pathlib import Path

import chromadb
import requests
from langchain_huggingface import HuggingFaceEmbeddings


BASE_PATH = Path(__file__).parent
CHROMA_PATH = BASE_PATH / "chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="distributedComputing")


def _build_prompt(query, documents):
    return """
You are a helpful assistant for a Distributed Computing course.
Answer only the question using only the course documents below.
Do not use outside knowledge.
Do not write code, examples, labels, headings, or bullet lists.
Do not repeat the question or these instructions.
Do not continue the prompt.
Answer in 1 to 3 concise sentences.
If the documents do not contain the answer, say exactly: I don't know.
--------------------
Course documents:
""" + "\n\n".join(documents) + """

Question:
""" + query + """

Answer:
"""


def ask_question(batchData, GPUserver):
    queries = [item["query"] for item in batchData]
    query_embeddings = [
        embedding_model.embed_query(query)
        for query in queries
    ]

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=6
    )

    requests_batch = []
    for index, item in enumerate(batchData):
        documents = results["documents"][index]
        requests_batch.append({
            "task_id": item["task_id"],
            "prompt": _build_prompt(item["query"], documents)
        })

    payload = {"requests": requests_batch}
    response = requests.post(GPUserver, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "gpu_utilization" not in data:
        data["gpu_utilization"] = data.get("metrics", {}).get("gpu_utilization_percent")

    return data
