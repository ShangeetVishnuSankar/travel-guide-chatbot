"""
TEXT SPLITTER
=============
Splits large documents into smaller, overlapping chunks for embedding.

CONCEPTS YOU'LL LEARN:
- Why chunk size and overlap matter
- RecursiveCharacterTextSplitter (the go-to splitter)
- How splitting affects retrieval quality
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_text_splitter(
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with sensible defaults for travel content.

    PARAMETER GUIDE:
    ┌──────────────┬─────────────────────────────────────────────┐
    │ chunk_size   │ Target size of each chunk (in characters).  │
    │              │ 500-1000 is ideal for retrieval precision.  │
    │              │ We use 3000 here to keep total chunk count  │
    │              │ under 1000 — the Gemini free tier daily     │
    │              │ embedding limit. Larger = fewer API calls.  │
    ├──────────────┼─────────────────────────────────────────────┤
    │ chunk_overlap│ How many characters overlap between chunks. │
    │              │ 10% of chunk_size is a good rule of thumb.  │
    │              │ Prevents losing context at boundaries.      │
    └──────────────┴─────────────────────────────────────────────┘

    WHY RecursiveCharacterTextSplitter?
    - It tries to split on natural boundaries: paragraphs > sentences > words
    - The "recursive" part means it tries "\\n\\n" first, then "\\n", then " ", then ""
    - This keeps paragraphs and sentences intact when possible
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
        add_start_index=True,  # Track where each chunk starts in the original
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of documents into chunks.

    Returns a new list of Document objects (more items, smaller content).
    Metadata from the original document is preserved in each chunk.
    """
    splitter = create_text_splitter()
    chunks = splitter.split_documents(documents)

    print(f"  ✂️  Split {len(documents)} documents into {len(chunks)} chunks")

    if chunks:
        chunk_sizes = [len(c.page_content) for c in chunks]
        print(f"  📊 Chunk sizes — min: {min(chunk_sizes)}, "
              f"max: {max(chunk_sizes)}, "
              f"avg: {sum(chunk_sizes) // len(chunk_sizes)}")

    return chunks


# --- TEST IT ---
if __name__ == "__main__":
    from document_loader import load_all_documents

    docs = load_all_documents()
    chunks = split_documents(docs)

    # Inspect a chunk
    if chunks:
        sample = chunks[0]
        print(f"\n--- Sample Chunk ---")
        print(f"Source: {sample.metadata.get('source', 'unknown')}")
        print(f"Start index: {sample.metadata.get('start_index', 'N/A')}")
        print(f"Size: {len(sample.page_content)} characters")
        print(f"Content:\n{sample.page_content[:500]}")
