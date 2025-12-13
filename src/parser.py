from pathlib import Path

from markitdown import MarkItDown


def parse_to_md(path: Path, output: Path = "./data") -> None:
    md = MarkItDown()

    file = md.convert(path)

    with open(f"{output}/{path.name.replace(path.suffix, '')}.md", "w") as f:
        f.write(file.markdown)
