from pathlib import Path

from markitdown import MarkItDown


def parse_to_md(path: Path, output_dir: Path = Path("./data")) -> Path:
    """
    Parse any supported file to Markdown using MarkItDown

    Args:
        path: Input file path
        output_dir: Output directory for markdown file

    Returns:
        Path to generated markdown file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If conversion fails
    """
    # Validate input
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to markdown
    md = MarkItDown()

    try:
        result = md.convert(str(path))

        # Generate output path
        output_path = output_dir / f"{path.stem}.md"

        # Write markdown
        output_path.write_text(result.text_content, encoding="utf-8")

        print(f"✓ Converted: {path.name} → {output_path.name}")

        return output_path

    except Exception as e:
        raise ValueError(f"Failed to convert {path}: {e}")
