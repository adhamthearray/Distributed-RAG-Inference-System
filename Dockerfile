FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --upgrade pip

# install everything EXCEPT heavy deps of sentence-transformers
RUN pip install fastapi uvicorn pydantic python-dotenv chromadb \
    langchain langchain-huggingface langchain-community langchain-text-splitters \
    groq pypdf

# install sentence-transformers WITHOUT dependencies
RUN pip install sentence-transformers --no-deps

# manually install ONLY what it actually needs
RUN pip install transformers scikit-learn scipy

# install CPU torch separately (important)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Bake the ChromaDB into the image during build
RUN python RAG_system/RAG/fill_db.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]