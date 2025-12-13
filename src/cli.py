from pathlib import Path

import torch
import typer

import parser

app = typer.Typer()


@app.command()
def parse(path: Path):
    print(f"Parsing {path.suffix} file")
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
    print(f"CUDA available: {torch.cuda.is_available()}")
