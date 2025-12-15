import warnings
from pathlib import Path

import torch
import typer
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import parser
import vectorizer

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

app = typer.Typer(
    name="CV Query Assistant", help="RAG-based CV analysis tool", add_completion=False
)


@app.command()
def upload(
    path: Path = typer.Argument(..., help="Path to CV file", exists=True),
    output: Path = typer.Option(Path("./data"), help="Output directory"),
):
    """
    Upload and convert a CV to markdown format
    """
    print(f"\nUploading {path.suffix} file: {path.name}")

    try:
        output_path = parser.parse_to_md(path=path, output_dir=output)
        print(f"CV saved to: {output_path}\n")
        print("Run 'init' to start querying this CV")
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def list():
    """
    List all stored CVs
    """
    print("\nStored CVs:")

    base_dir = Path("./data")
    base_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in base_dir.iterdir() if f.suffix == ".md"])

    if not files:
        print("No CVs found\n")
        print("Run 'upload <path>' to add a CV")
        return

    for i, file in enumerate(files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"   {i}. {file.name} ({size_kb:.1f} KB)")

    print()


@app.command()
def init(
    cv: Path = typer.Option(None, help="CV file path (skip prompt)"),
    model: str = typer.Option("google/flan-t5-large", help="LLM model"),
    device: str = typer.Option("auto", help="Device: auto, cuda, cpu"),
):
    """
    Initialize the CV query chatbot
    """
    print("\n" + "=" * 60)
    print("CV Query Assistant - RAG Chatbot")
    print("=" * 60 + "\n")

    # Check CUDA
    is_cuda = torch.cuda.is_available()
    if device == "auto":
        device = "cuda" if is_cuda else "cpu"

    print(f"Device: {device.upper()}")

    # Get CV path
    if not cv:
        list()
        cv_input = input("\nEnter CV filename (from ./data/): ").strip()

        if not cv_input:
            print("No file specified")
            raise typer.Exit(1)

        cv = Path("./data") / cv_input

    # Validate CV exists
    if not cv.exists():
        print(f"CV not found: {cv}")
        print("Run 'list' to see available CVs or 'upload' to add one")
        raise typer.Exit(1)

    print(f"Loading: {cv.name}\n")

    # Initialize vector store
    print("Initializing vector store...")

    try:
        vector_store = vectorizer.vectorize(cv, device=device)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("Vector store ready\n")
    except Exception as e:
        print(f"Failed to initialize vector store: {e}")
        raise typer.Exit(1)

    # Initialize LLM
    print(f"Loading LLM: {model}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)

        model_obj = AutoModelForSeq2SeqLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        pipe = pipeline(
            "text2text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            device=0 if device == "cuda" else -1,
            truncation=True,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print("LLM loaded\n")

    except Exception as e:
        print(f"Failed to load LLM: {e}")
        raise typer.Exit(1)

    # Prompt template
    template = """Answer based on the CV context below.

Context:
{context}

Question: {question}

Provide a clear, professional answer based only on the information in the CV.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    # Chat loop
    print("=" * 60)
    print("Chat Mode - Ask about the CV")
    print("=" * 60)
    print("\nType 'q' to quit, 'help' for examples\n")

    while True:
        try:
            # Get input
            question = input("❯ ").strip()

            if not question:
                continue

            if question.lower() in ["q", "quit", "exit"]:
                print("\nGoodbye!\n")
                break

            if question.lower() == "help":
                print("\nExample questions:")
                print("   • What programming languages does the candidate know?")
                print("   • Describe the candidate's experience")
                print("   • What are the candidate's key skills?")
                print("   • Suggest interview questions for this candidate\n")
                continue

            if len(question) < 3:
                print("Please enter a valid question\n")
                continue

            # Retrieve
            print("Searching CV...")
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                print("No relevant information found\n")
                continue

            # Format context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Truncate if too long
            context_tokens = tokenizer.encode(context)
            if len(context_tokens) > 400:
                print("Context too long, truncating...")
                context = tokenizer.decode(context_tokens[:400])

            # Generate
            print("Generating answer...\n")

            answer = chain.invoke({"context": context, "question": question})

            # Display
            print("─" * 60)
            print(answer)
            print("─" * 60 + "\n")

            # Clear CUDA cache
            if device == "cuda":
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    app()
