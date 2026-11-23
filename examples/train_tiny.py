#!/usr/bin/env python3
"""Train and evaluate an SAE on a tiny random encoder, fully offline.

This is an interface smoke test: it builds a 2-layer BERT with random weights and a few
hundred synthetic query/passage rows in a temporary directory, runs ``muscari train-sae`` and
then ``muscari eval-sae`` on them. It downloads nothing and finishes in seconds on a CPU. The
numbers it prints say nothing about SAE quality on a real encoder.

    python examples/train_tiny.py --trainer topk
    python examples/train_tiny.py --trainer saelens   # needs the "saelens" extra
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from muscari.cli import main as muscari

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
WORDS = "search engine index query passage vector sparse dense token feature latent encoder model rank score".split()


def build_encoder(path: Path) -> Path:
    """Save a randomly initialised 2-layer BERT with a word-level tokenizer."""
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    prefix_words = QUERY_PREFIX.split() + PASSAGE_PREFIX.split()
    vocab = {token: index for index, token in enumerate(specials + sorted(set(WORDS + prefix_words)))}

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(path)

    torch.manual_seed(42)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(path)
    return path


def write_rows(path: Path, n_rows: int = 300) -> Path:
    """Write rows with a query, a positive passage and two negative passages each."""
    generator = torch.Generator().manual_seed(42)

    def sentence(length: int) -> str:
        ids = torch.randint(len(WORDS), (length,), generator=generator)
        return " ".join(WORDS[int(i)] for i in ids)

    rows = [
        {"query": sentence(4), "positive": sentence(12), "negatives": [sentence(10), sentence(14)]}
        for _ in range(n_rows)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def run(trainer: str, workdir: Path, device: str) -> dict[str, float]:
    """Train, evaluate again from the exported files, and return the evaluation metrics."""
    encoder = build_encoder(workdir / "tiny-encoder")
    data = write_rows(workdir / "rows.jsonl")
    out = workdir / f"sae-{trainer}"

    muscari(
        [
            "train-sae",
            *("--trainer", trainer),
            *("--out", str(out)),
            *("--model", str(encoder)),
            *("--data-files", str(data)),
            "--columns",
            f"query={QUERY_PREFIX}",
            f"positive={PASSAGE_PREFIX}",
            f"negatives={PASSAGE_PREFIX}",
            *("--max-length", "32"),
            *("--text-batch-size", "8"),
            *("--num-latents", "128"),
            *("--k", "4"),
            *("--total-tokens", "20000"),
            *("--batch-tokens", "128"),
            *("--buffer-tokens", "512"),
            *("--dead-after-tokens", "2000"),
            *("--holdout-pct", "10"),
            *("--eval-tokens", "1000"),
            *("--device", device),
        ]
    )
    muscari(
        [
            "eval-sae",
            *("--sae", str(out)),
            *("--model", str(encoder)),
            *("--data-files", str(data)),
            *("--eval-tokens", "1000"),
            *("--device", device),
            *("--out", str(out / "eval.json")),
        ]
    )
    return dict(json.loads((out / "eval.json").read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer", choices=["topk", "saelens"], default="topk")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workdir", help="keep the encoder, data and SAE here instead of a temporary directory")
    args = parser.parse_args()

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        run(args.trainer, workdir, args.device)
    else:
        with tempfile.TemporaryDirectory(prefix="muscari-tiny-") as temporary:
            run(args.trainer, Path(temporary), args.device)


if __name__ == "__main__":
    main()
