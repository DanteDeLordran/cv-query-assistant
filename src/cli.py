from pathlib import Path

import typer

import parser

app = typer.Typer()


@app.command()
def parse(path: Path):
    print(f"Parsing {path.suffix} file")
    parser.parse_to_md(path=path)


@app.command()
def list():
    print("Listing...")


@app.command()
def init():
    print("Initializing...")
