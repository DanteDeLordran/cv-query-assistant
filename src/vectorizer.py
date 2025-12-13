import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_data(md_path: Path) -> list[Document]:
    print("Reading MD file")

    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    if not md_text.strip():
        raise ValueError("Markdown file is empty")

    print("Chunking MD file")

    headers_to_split_on = [
        ("#", "section"),
        ("##", "subsection"),
    ]

    separators = [
        "\n\n",
        "\n•",
        "\n-",
        "\n",
        ". ",
        " ",
        "",
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_header_splits = header_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=separators
    )

    return text_splitter.split_documents(md_header_splits)


def vectorize(md_path: Path, force_reindex=False) -> Chroma:
    db_path = "./chromma-db"

    print("Loading embbeding model")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embbeding model loaded")

    vector_store = Chroma(
        collection_name="cv_collection",
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    if os.path.exists(db_path) and not force_reindex:
        count = vector_store._collection.count()
        if count > 0:
            print("Using existing vector store")
            return vector_store
        else:
            print("Vector store exists but is empty")

    chunks = chunk_data(md_path)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = md_path
        chunk.metadata["chunk_id"] = i

    print(f"Created {len(chunks)} chunks")

    print("Adding documents to vector store")

    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        vector_store.add_documents(documents=batch, ids=ids)
        print(f"Progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Vectorized and stored in {db_path}")

    return vector_store
