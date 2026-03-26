"""
DOCUMENT LOADER

Loads travel content from multiple file types into LangChain Document objects.

CONCEPTS YOU'LL LEARN:
- LangChain Document Loaders (TextLoader, PyPDFLoader, WebBaseLoader)
- The Document object (page_content + metadata)
- Why metadata matters for citations in RAG
"""

import os

# Resolve the project root (one level up from this src/ file)
# This makes paths work regardless of which directory you run the script from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def load_text_files(directory: str = "data/web") -> list[Document]:
    """
    Load all .txt files from a directory.

    Each file becomes one or more Document objects.
    Metadata automatically includes the source filename — crucial
    for showing users WHERE an answer came from.
    """
    if not os.path.exists(directory):
        print(f"  ⚠️  Directory not found: {directory}")
        return []

    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  📄 Loaded {len(docs)} text documents from {directory}")
    return docs


def load_tip_files(directory: str = "data/tips") -> list[Document]:
    """Load travel tips text files."""
    return load_text_files(directory)


def load_pdf_files(directory: str = "data/pdfs") -> list[Document]:
    """
    Load all PDF files. Each PAGE becomes a separate Document.

    WHY PER-PAGE?
    - PDFs can be hundreds of pages long.
    - Splitting by page keeps chunks manageable.
    - Page numbers in metadata help with citations.
    """
    if not os.path.exists(directory):
        print(f"  ⚠️  Directory not found: {directory}")
        return []

    pdf_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            pdf_docs.extend(loader.load())

    print(f"  📄 Loaded {len(pdf_docs)} pages from PDFs in {directory}")
    return pdf_docs


def load_all_documents() -> list[Document]:
    """
    Master loader — combines all data sources into one list of Documents.

    This is the single entry point for the ingestion pipeline.
    """
    print("\n📥 Loading all documents...")

    all_docs = []
    all_docs.extend(load_text_files(os.path.join(PROJECT_ROOT, "data/web")))
    all_docs.extend(load_tip_files(os.path.join(PROJECT_ROOT, "data/tips")))
    all_docs.extend(load_pdf_files(os.path.join(PROJECT_ROOT, "data/pdfs")))

    print(f"\n✅ Total documents loaded: {len(all_docs)}")
    return all_docs


# --- TEST IT ---
if __name__ == "__main__":
    docs = load_all_documents()

    # Inspect a sample document
    if docs:
        sample = docs[0]
        print(f"\n--- Sample Document ---")
        print(f"Source: {sample.metadata.get('source', 'unknown')}")
        print(f"Content preview: {sample.page_content[:300]}...")
