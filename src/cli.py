import os
from pathlib import Path

import torch
import typer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

import parser
import vectorizer

app = typer.Typer()


@app.command()
def upload(path: Path):
    print(f"Uploading {path.suffix} file")
    parser.parse_to_md(path=path)


@app.command()
def list():
    print("Listing loaded CVs...")

    base_dir = Path("./data")
    files = [file for file in base_dir.iterdir() if file.suffix == ".md"]

    if not files:
        print("No CVs at the moment")

    for file in files:
        print(f"- {file.name}")


@app.command()
def init():
    print("CV Query Assistant - Chatbot System")
    is_cuda = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda}")

    file_path = Path(
        input(
            "Load a stored file (run upload or list if you haven't uploaded one yet )"
        )
    )

    if not file_path or not os.path.exists(file_path):
        print("No file path provided, run upload to store a new CV")

    print("Initializing vector store...")

    vector_store = vectorizer.vectorize(file_path)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print("Initializing LLM...")

    model_name = "google/flan-t5-large"

    print(f"Loading {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, dtype=torch.float16 if is_cuda else torch.float32
    )

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline=pipe)

    print("LLM loaded")
