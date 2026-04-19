"""Load and cache the kl3m EDGAR agreements sample for benchmarking.

Decodes 997 SEC-filed contract exhibits from kl3m tokenized format back to
raw text. Caches the decoded texts to a JSONL file to avoid re-downloading
and re-decoding on subsequent runs.

Usage::

    from tests.benchmarks.kl3m_loader import load_edgar_agreements

    docs = load_edgar_agreements()  # list of (identifier, text) tuples
    print(f"Loaded {len(docs)} documents")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

CACHE_PATH = Path(__file__).parent / "data" / "edgar-agreements-decoded.jsonl"


class _TokenizerWithDecode(Protocol):
    def decode(
        self,
        token_ids: object,
        *,
        skip_special_tokens: bool = False,
    ) -> str | list[str]: ...


def load_edgar_agreements(
    *,
    max_docs: int | None = None,
    max_chars_per_doc: int = 50_000,
) -> list[tuple[str, str]]:
    """Load EDGAR agreements, returning (identifier, text) pairs.

    Args:
        max_docs: Cap on number of documents to return. None = all 997.
        max_chars_per_doc: Truncate documents longer than this (default 50K chars).
            Set to 0 to disable truncation.

    Returns:
        List of (identifier, text) tuples sorted by identifier.
    """
    if CACHE_PATH.exists():
        return _load_from_cache(max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)
    return _download_and_cache(max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)


def _load_from_cache(
    *,
    max_docs: int | None,
    max_chars_per_doc: int,
) -> list[tuple[str, str]]:
    """Load from local JSONL cache."""
    docs: list[tuple[str, str]] = []
    with CACHE_PATH.open() as f:
        for line in f:
            if max_docs is not None and len(docs) >= max_docs:
                break
            record = json.loads(line)
            text = record["text"]
            if max_chars_per_doc > 0 and len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc]
            docs.append((record["identifier"], text))
    return docs


def _download_and_cache(
    *,
    max_docs: int | None,
    max_chars_per_doc: int,
) -> list[tuple[str, str]]:
    """Download from HuggingFace, decode, and cache locally."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("alea-institute/kl3m-data-edgar-agreements-sample", split="train")
    tokenizer = AutoTokenizer.from_pretrained("alea-institute/kl3m-004-128k-cased")
    decode = cast(_TokenizerWithDecode, tokenizer).decode

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_docs: list[tuple[str, str]] = []

    with CACHE_PATH.open("w") as f:
        for i in range(len(ds)):
            identifier = str(ds[i]["identifier"])
            decoded = decode(ds[i]["tokens"], skip_special_tokens=True)
            text = "".join(decoded) if isinstance(decoded, list) else str(decoded)
            record = {"identifier": identifier, "text": text}
            f.write(json.dumps(record))
            f.write("\n")
            all_docs.append((identifier, text))

    if max_docs is not None:
        all_docs = all_docs[:max_docs]

    result: list[tuple[str, str]] = []
    for identifier, text in all_docs:
        if max_chars_per_doc > 0 and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc]
        result.append((identifier, text))

    return result


def load_into_memory(
    max_docs: int | None = None,
    max_chars_per_doc: int = 50_000,
) -> tuple[SessionMemory, list[str]]:
    """Load EDGAR agreements into a SessionMemory DOCUMENTS section.

    Returns:
        (memory, identifiers) — the SessionMemory and list of identifiers loaded.
    """
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryType

    docs = load_edgar_agreements(max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)

    memory = SessionMemory("benchmark-edgar")
    identifiers: list[str] = []

    for identifier, text in docs:
        short_id = identifier.split("/")[-2] if "/" in identifier else identifier
        memory.add(
            MemoryType.DOCUMENTS,
            text,
            metadata={"uri": f"edgar:{short_id}", "identifier": identifier},
        )
        identifiers.append(identifier)

    return memory, identifiers
