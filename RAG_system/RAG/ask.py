import chromadb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
from pathlib import Path




load_dotenv()

# setting the environment

BASE_PATH = Path(__file__).parent
CHROMA_PATH = BASE_PATH / "chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="distributedComputing")


# user_query = input("What do you want to know about distributed computing?\n\n")

def ask_question(user_query):
    results = collection.query(
    query_embeddings=[embedding_model.embed_query(user_query)],
    n_results=6
)
    client = Groq(api_key=os.getenv("GROQAPI"))
    system_prompt = """
You are a helpful assistant for a Distributed Computing course.
Answer the user's question using only the course documents provided below from the data directory.
Do not use outside knowledge and do not make things up.
If the provided course documents do not contain the answer, say: I don't know.
Keep your answer clear, accurate, and focused on distributed computing concepts.
--------------------
Course documents:
"""+str(results['documents'])+"""
"""
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
)
    answer = response.choices[0].message.content
    chunks = results['documents'][0] if results['documents'] else []
    return answer, chunks














#print(system_prompt)


