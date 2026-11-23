"""Turn dataset rows into a stream of ``(prefix, text)`` pairs.

Rows are plain mappings. A column may hold a string or a list of strings; retrieval datasets
often keep several hard negatives per row in one list column.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SPLITS = ("train", "holdout", "all")


@dataclass(frozen=True)
class TextColumn:
    """A dataset column and the prefix the encoder expects for that kind of text."""

    name: str
    prefix: str = ""

    @classmethod
    def parse(cls, spec: str) -> TextColumn:
        """Parse ``name`` or ``name=prefix``.

        Everything after the first ``=`` is the prefix, verbatim, so a trailing space such as
        the one in ``"query: "`` is kept.

        Raises:
            ValueError: if the column name is empty.
        """
        name, separator, prefix = spec.partition("=")
        if not name:
            raise ValueError(f"Column spec {spec!r} has no column name.")
        return cls(name=name, prefix=prefix if separator else "")


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def is_holdout(text: str, holdout_pct: float) -> bool:
    """Deterministically assign a text to the held-out split by its content hash.

    The same text always lands on the same side, whatever row or column it appears in, so a
    passage that recurs as a negative in many rows cannot leak from training into evaluation.
    ``holdout_pct`` is a percentage with a resolution of 0.01.
    """
    if not 0 <= holdout_pct < 100:
        raise ValueError(f"holdout_pct must be in [0, 100), got {holdout_pct}.")
    return (_hash64(text) % 10_000) < round(holdout_pct * 100)


def iter_texts(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[TextColumn],
    *,
    dedup: bool = True,
    min_chars: int = 1,
    holdout_pct: float = 0.0,
    split: str = "train",
) -> Iterator[tuple[str, str]]:
    """Yield ``(prefix, text)`` for every text in the requested columns.

    Args:
        rows: dataset rows. Missing columns and ``None`` cells are skipped.
        columns: which columns to read and which prefix each one gets.
        dedup: drop exact repeats of ``(prefix, text)``. Retrieval datasets repeat passages
            heavily (shared negatives), which would otherwise over-weight them. This keeps one
            64-bit hash per distinct text in memory.
        min_chars: skip texts shorter than this many characters.
        holdout_pct: percentage of distinct texts reserved for evaluation.
        split: ``"train"`` yields the texts outside the held-out part, ``"holdout"`` the
            held-out ones, ``"all"`` ignores ``holdout_pct``.

    Raises:
        ValueError: on an unknown ``split`` or an out-of-range ``holdout_pct``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    if not 0 <= holdout_pct < 100:
        raise ValueError(f"holdout_pct must be in [0, 100), got {holdout_pct}.")

    want_holdout = split == "holdout"
    seen: set[int] = set()

    for row in rows:
        for column in columns:
            cell = row.get(column.name)
            if cell is None:
                continue

            texts = [cell] if isinstance(cell, str) else list(cell)
            for text in texts:
                if not isinstance(text, str) or len(text) < min_chars:
                    continue
                if split != "all" and is_holdout(text, holdout_pct) != want_holdout:
                    continue

                if dedup:
                    key = _hash64(column.prefix + "\x00" + text)
                    if key in seen:
                        continue
                    seen.add(key)

                yield column.prefix, text


def repeat_texts(
    make_iter: Callable[[], Iterable[tuple[str, str]]],
    max_epochs: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Re-run ``make_iter()`` each time it is exhausted, forever or for ``max_epochs`` passes.

    Deduplication inside ``make_iter`` restarts on every pass, so each pass is one epoch over
    the distinct texts.

    Raises:
        ValueError: if a pass yields nothing, which would otherwise loop forever.
    """
    epoch = 0
    while max_epochs is None or epoch < max_epochs:
        empty = True
        for item in make_iter():
            empty = False
            yield item

        if empty:
            raise ValueError("The text source yielded nothing; check the dataset and column names.")
        epoch += 1
