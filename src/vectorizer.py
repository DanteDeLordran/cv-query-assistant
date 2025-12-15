import os
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_data(
    md_path: Path, chunk_size: int = 800, chunk_overlap: int = 150
) -> list[Document]:
    """
    Chunk markdown file intelligently

    Args:
        md_path: Path to markdown file
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    print("Reading markdown file...")

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    if not md_text.strip():
        raise ValueError("Markdown file is empty")

    print("Chunking document...")

    # Headers to split on
    headers_to_split_on = [
        ("#", "section"),
        ("##", "subsection"),
    ]

    # CV-optimized separators
    separators = [
        "\n\n",  # Paragraphs
        "\n•",  # Bullet points
        "\n-",  # Dashes
        "\n",  # Lines
        ". ",  # Sentences
        " ",  # Words
        "",  # Characters
    ]

    # First split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    try:
        md_header_splits = header_splitter.split_text(md_text)
    except Exception as e:
        print(f"Header splitting failed: {e}, using fallback")
        md_header_splits = [Document(page_content=md_text, metadata={})]

    # Then split by size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    chunks = text_splitter.split_documents(md_header_splits)

    print(f"Created {len(chunks)} chunks")

    return chunks


def vectorize(
    md_path: Path,
    db_path: str = "./chroma_db",
    collection_name: str = "cv_collection",
    force_reindex: bool = False,
    device: str = "auto",
) -> Chroma:
    """
    Vectorize markdown file and store in ChromaDB

    Args:
        md_path: Path to markdown file
        db_path: ChromaDB directory
        collection_name: Collection name
        force_reindex: Force re-indexing
        device: Device for embeddings (auto, cuda, cpu)

    Returns:
        Chroma vector store instance
    """
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embedding model (device: {device})...")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding model loaded")

    # Create vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    # Check if already indexed
    if os.path.exists(db_path) and not force_reindex:
        try:
            count = vector_store._collection.count()
            if count > 0:
                print(f"Using existing vector store ({count} documents)")
                return vector_store
            else:
                print("Vector store exists but is empty, re-indexing...")
        except Exception as e:
            print(f"Could not check vector store: {e}, re-indexing...")

    # Chunk the document
    chunks = chunk_data(md_path)

    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = str(md_path)
        chunk.metadata["filename"] = md_path.name
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)

        # Add hierarchy
        section = chunk.metadata.get("section", "Unknown")
        subsection = chunk.metadata.get("subsection", "")

        if subsection:
            chunk.metadata["hierarchy"] = f"{section} > {subsection}"
        else:
            chunk.metadata["hierarchy"] = section

    print("Adding documents to vector store...")

    # Batch processing
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"{md_path.stem}_chunk_{j}" for j in range(i, i + len(batch))]

        try:
            vector_store.add_documents(documents=batch, ids=ids)
            print(f"Progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")
        except Exception as e:
            print(f"Batch {i // batch_size} failed: {e}")

    print(f"Vectorized and stored in {db_path}")

    return vector_store
