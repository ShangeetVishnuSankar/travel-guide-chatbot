"""
VECTOR STORE
============
Embeds document chunks and stores them in Pinecone for similarity search.

CONCEPTS YOU'LL LEARN:
- Embedding models (Google Gemini Embedding 1)
- Cloud vector databases (Pinecone)
- Similarity search (finding relevant chunks)
- Why cloud storage is needed for deployment

WHY PINECONE INSTEAD OF CHROMADB?
- ChromaDB saves vectors to a local folder on your computer.
  That's fine when you're working locally, but when you deploy
  to a cloud server (like Streamlit Community Cloud), the server
  has no memory between restarts — your vector folder would be
  wiped every time the app reboots.
- Pinecone is a cloud-hosted vector database. Your vectors live
  on Pinecone's servers permanently, and your app just connects
  to them — exactly like how your app connects to any other API.
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Gemini free tier limits:
#   - 100 requests per minute (RPM)
#   - 30,000 tokens per minute (TPM)  ← this is the binding constraint
#
# With chunk_size=3000, each chunk is ~750 tokens.
# 30,000 TPM ÷ 750 tokens/chunk = 40 chunks/minute max.
# We use 35 to stay safely under, then wait 65 seconds between batches.
BATCH_SIZE = 35

load_dotenv()

# --- CONFIGURATION ---
# These values are read from your .env file (locally) or
# from Streamlit Cloud secrets (when deployed).
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "travel-guide")

# gemini-embedding-001 produces vectors with 3072 dimensions.
# This MUST match what you set when creating your Pinecone index.
EMBEDDING_DIMENSION = 3072


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize the Gemini embedding model.

    WHAT'S HAPPENING:
    - Google's Gemini Embedding 1 converts text → 3072-dim vectors
    - It's FREE within rate limits (100 RPM, 30K TPM)
    - Each chunk gets embedded once during ingestion
    - Each user query gets embedded once at query time

    TASK TYPES (Gemini-specific optimization):
    - "retrieval_document"  → use when embedding your CHUNKS (stored in DB)
    - "retrieval_query"     → use when embedding the USER'S QUESTION
    - Gemini optimizes the vector differently for each role!
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document",
    )


def get_query_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Embedding model optimized for search queries."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
    )


def create_pinecone_index():
    """
    Create a Pinecone index if it doesn't already exist.

    WHAT IS AN INDEX?
    Think of a Pinecone index like a table in a regular database —
    it's a named storage space for your vectors. You create it once,
    and from then on you just add/search vectors in it.

    KEY SETTINGS:
    - dimension: must match your embedding model (768 for Gemini Embedding 1)
    - metric: "cosine" measures the angle between vectors — the standard
      choice for text similarity. Higher score = more similar.
    - spec: ServerlessSpec means Pinecone manages the infrastructure for you.
      "aws" + "us-east-1" is the free-tier supported region.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"📦 Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Index '{INDEX_NAME}' created.")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists.")


def create_vector_store(chunks: list[Document]) -> PineconeVectorStore:
    """
    Embed all chunks and upload them to Pinecone.

    THIS IS THE INGESTION STEP — you run this once locally before deploying.
    Once your vectors are in Pinecone, your deployed app just reads from there.

    WHAT HAPPENS UNDER THE HOOD:
    1. Each chunk's text is sent to the Gemini Embedding API
    2. Gemini returns a 768-number vector representing the meaning of that text
    3. The vector + the original text + metadata are uploaded to Pinecone
    4. Pinecone indexes it for fast similarity search

    NOTE ON RATE LIMITS:
    Gemini free tier allows 100 RPM for embeddings. With ~100-200 chunks,
    this completes in about 1-2 minutes. LangChain handles batching automatically.
    """
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_minutes = total_batches * 65 // 60

    print(f"\n🔮 Uploading {len(chunks)} chunks to Pinecone in {total_batches} batches...")
    print(f"   Rate limit: 100 requests/min → sending {BATCH_SIZE} chunks per batch")
    print(f"   Estimated time: ~{estimated_minutes} minutes\n")

    create_pinecone_index()

    embeddings = get_embedding_model()
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  📤 Batch {batch_num}/{total_batches} — uploading {len(batch)} chunks...")

        if vector_store is None:
            # First batch: create the vector store and upload
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=INDEX_NAME,
            )
        else:
            # Subsequent batches: add to the existing store
            vector_store.add_documents(batch)

        print(f"  ✅ Batch {batch_num} done.")

        # Wait between batches (skip the wait after the last batch)
        if i + BATCH_SIZE < len(chunks):
            print(f"  ⏳ Waiting 65 seconds to respect rate limit...\n")
            time.sleep(65)

    print(f"\n✅ All {len(chunks)} chunks uploaded to Pinecone index '{INDEX_NAME}'")
    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """
    Connect to an existing Pinecone index.

    WHY THIS IS DIFFERENT FROM CHROMADB:
    With ChromaDB you loaded files from disk. With Pinecone you're making
    a network connection to Pinecone's servers — the same way your app
    connects to any external API. This is what makes it work in the cloud.

    This function is called every time the app starts up (locally or deployed).
    It's fast because we're just establishing a connection, not loading data.
    """
    embeddings = get_query_embedding_model()

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    print(f"✅ Connected to Pinecone index '{INDEX_NAME}'")
    return vector_store


def test_similarity_search(query: str, k: int = 3):
    """
    Test the vector store with a sample query.

    'k' is how many similar chunks to retrieve.

    NOTE: Pinecone scores work opposite to ChromaDB.
    - ChromaDB: lower score = more similar
    - Pinecone:  higher score = more similar (0.0 to 1.0, where 1.0 = identical)
    """
    vector_store = load_vector_store()

    print(f"\n🔍 Searching for: '{query}'")
    print(f"   Retrieving top {k} results...\n")

    results = vector_store.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(results):
        print(f"--- Result {i + 1} (similarity score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:300]}...")
        print()


# --- RUN IT ---
if __name__ == "__main__":
    import sys

    if "--ingest" in sys.argv:
        # Run the full ingestion pipeline — do this once before deploying
        from document_loader import load_all_documents
        from text_splitter import split_documents

        docs = load_all_documents()
        chunks = split_documents(docs)
        create_vector_store(chunks)
    else:
        # Test that search is working (requires ingestion to have run first)
        test_similarity_search("What are the best beaches in Bali?")
        test_similarity_search("How do I get around Tokyo?")
        test_similarity_search("Best street food in Bangkok")
