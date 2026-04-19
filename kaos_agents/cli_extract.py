"""Extract structured data from a folder of documents.

The Harvey Review Tables equivalent: load contracts, apply a recipe,
get a spreadsheet with extracted terms and citations.

Usage::

    # Extract lease terms from PDFs
    kaos-extract --recipe lease --files "contracts/*.pdf" --output results.xlsx

    # Extract merger agreement terms
    kaos-extract --recipe merger-agreement --files "deal-room/" --output terms.xlsx

    # List available recipes
    kaos-extract --list-recipes

    # Custom schema from JSON
    kaos-extract --schema schema.json --files "*.docx" --output results.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kaos-extract",
        description="Extract structured data from documents into spreadsheets",
    )
    parser.add_argument(
        "--recipe",
        help="Built-in extraction recipe name (e.g., lease, merger-agreement)",
    )
    parser.add_argument(
        "--schema",
        help="Custom extraction schema JSON file",
    )
    parser.add_argument(
        "--files",
        help="Files or folder to extract from (glob pattern or directory path)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (.xlsx, .csv, or .tsv)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for extraction",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="List available extraction recipes and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args(argv)

    if args.list_recipes:
        _list_recipes()
        return

    if not args.files:
        parser.error("--files is required (provide a glob pattern or directory path)")
    if not args.recipe and not args.schema:
        parser.error("--recipe or --schema is required")

    asyncio.run(_run_extraction(args))


def _list_recipes() -> None:
    """Print available extraction recipes."""
    from kaos_agents.recipes import extraction_recipe_names, load_extraction_recipe

    names = extraction_recipe_names()
    sys.stdout.write(f"Available extraction recipes ({len(names)}):\n\n")
    for name in sorted(names):
        recipe = load_extraction_recipe(name)
        if recipe:
            n_cols = len(recipe.get("schema", {}).get("columns", []))
            desc = recipe.get("description", "")[:80]
            recall = recipe.get("harvey_recall_floor", "")
            recall_str = f" (recall: {recall})" if recall else ""
            sys.stdout.write(f"  {name:<25} {n_cols} columns{recall_str}\n")
            if desc:
                sys.stdout.write(f"    {desc}\n")
    sys.stdout.flush()


async def _run_extraction(args: argparse.Namespace) -> None:
    """Run the extraction pipeline."""
    from kaos_content import serialize_text

    from kaos_agents.cli_chat import _SUPPORTED_EXTENSIONS, _parse_file_to_document
    from kaos_agents.settings import DEFAULT_MODEL

    # 1. Load files
    target = Path(args.files).expanduser()
    if target.is_dir():
        file_paths = sorted(
            f for f in target.iterdir() if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
    elif target.is_file():
        file_paths = [target]
    else:
        file_paths = sorted(Path.cwd().glob(args.files))

    if not file_paths:
        sys.stderr.write(f"No files found matching: {args.files}\n")
        sys.exit(1)

    sys.stdout.write(f"Loading {len(file_paths)} file(s)...\n")
    sys.stdout.flush()

    corpus: dict[str, str] = {}
    for fp in file_paths:
        fp = fp.resolve()
        try:
            doc = _parse_file_to_document(fp)
            text = serialize_text(doc)
            if text.strip():
                corpus[f"file:{fp.name}"] = text
                if args.verbose:
                    sys.stdout.write(f"  {fp.name} ({len(text):,} chars)\n")
        except Exception as exc:
            sys.stderr.write(f"  Failed {fp.name}: {exc}\n")

    if not corpus:
        sys.stderr.write("No documents loaded.\n")
        sys.exit(1)

    sys.stdout.write(f"  {len(corpus)} documents loaded\n\n")
    sys.stdout.flush()

    # 2. Load schema
    if args.recipe:
        from kaos_agents.recipes import load_extraction_recipe

        recipe = load_extraction_recipe(args.recipe)
        if not recipe:
            sys.stderr.write(f"Unknown recipe: {args.recipe}\n")
            sys.stderr.write("Use --list-recipes to see available recipes.\n")
            sys.exit(1)
        schema = recipe["schema"]
        sys.stdout.write(f"Recipe: {args.recipe} ({len(schema.get('columns', []))} columns)\n")
    else:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            sys.stderr.write(f"Schema file not found: {args.schema}\n")
            sys.exit(1)
        schema = json.loads(schema_path.read_text())
        sys.stdout.write(f"Schema: {schema_path.name} ({len(schema.get('columns', []))} columns)\n")

    sys.stdout.flush()

    # 3. Run extraction
    import tempfile

    from kaos_llm_core.programs.extract import extract_corpus

    model = args.model or DEFAULT_MODEL
    sys.stdout.write(f"Model: {model}\n")
    sys.stdout.write(f"Extracting from {len(corpus)} documents...\n\n")
    sys.stdout.flush()

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await extract_corpus(
            schema=schema,
            corpus=corpus,
            output_dir=tmpdir,
            model=model,
            provenance="cited",
        )
    t1 = time.perf_counter()

    tabular_doc = result.table
    total_rows = sum(len(t.rows) for t in tabular_doc.tables)
    total_cols = sum(len(t.columns) for t in tabular_doc.tables)
    sys.stdout.write(
        f"Extracted {total_rows} rows x {total_cols} columns "
        f"({len(tabular_doc.tables)} table(s)) in {t1 - t0:.1f}s\n"
    )
    if hasattr(result, "cost_usd") and result.cost_usd:
        sys.stdout.write(f"Cost: ${result.cost_usd:.4f}\n")
    sys.stdout.flush()

    # 4. Write output
    output_path = (
        Path(args.output) if args.output else Path(f"{args.recipe or 'extraction'}_results.xlsx")
    )
    ext = output_path.suffix.lower()

    if ext == ".xlsx":
        from kaos_office.xlsx.writer import write_xlsx

        write_xlsx(tabular_doc, output_path)
    elif ext == ".csv":
        from kaos_content.serializers.tabular import serialize_csv

        output_path.write_text(serialize_csv(tabular_doc))
    elif ext == ".tsv":
        from kaos_content.serializers.tabular import serialize_tsv

        output_path.write_text(serialize_tsv(tabular_doc))
    else:
        # Default to XLSX
        output_path = output_path.with_suffix(".xlsx")
        from kaos_office.xlsx.writer import write_xlsx

        write_xlsx(tabular_doc, output_path)

    sys.stdout.write(f"\nSaved to {output_path}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
