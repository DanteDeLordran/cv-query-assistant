import os
from pathlib import Path

import torch
import typer
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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

    template = """
Analyze every detail of this CV:
{cv}
Give a summary of the candidate
Give the top 5 better questions that you can ask to the candidate in order to know if the candidate is optimal

Also answer any questions I give to you related to the candidate CV in order to know him/her better

Don't answer any non-related question
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm

    print("Chat started!")
    print("Type 'q' to quit")

    while True:
        question = input("Type your question: ").strip()

        if question.lower() == "q":
            break

        if not question or len(question) < 2:
            print("Please enter a valid question")
            continue

        print("Searching in CV")

        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            print("No relevant information found")
            continue

        print("Retrieved sections")

        full_input = template.format(content=retrieved_docs, question=question)

        input_tokens = tokenizer.encode(full_input)

        print("Generating answer")
