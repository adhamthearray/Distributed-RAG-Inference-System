from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from pathlib import Path

# setting the environment

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
CHROMA_PATH = BASE_PATH / "chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="distributedComputing")

# loading the document

loader = PyPDFDirectoryLoader(DATA_PATH, glob="**/*.pdf")

raw_documents = loader.load()

print("Loaded documents:", len(raw_documents))
print("First document preview:", raw_documents[0].page_content[:500] if raw_documents else "No documents loaded")


# splitting the document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

print("Chunks created:", len(chunks))


# preparing to be added in chromadb

documents = []
metadata = []
ids = []

i = 0

for chunk in chunks:
    documents.append(chunk.page_content)
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)

    i += 1

# adding to chromadb


collection.upsert(
    documents=documents,
    embeddings=embedding_model.embed_documents(documents),
    metadatas=metadata,
    ids=ids
)
