#!/usr/bin/env python3
"""Wrap the weights of a Top-K SAE trained elsewhere in muscari's format, to evaluate it here.

The source is one safetensors file with the tensor names and shapes of muscari's format:

    encoder.weight (num_latents, d_in)   encoder.bias (num_latents,)
    decoder.weight (d_in, num_latents)   decoder.bias (d_in,)

``mahiyama/modernbert-ja-310m-sae`` documents this layout on its model card. muscari stores
``encoder.bias`` with the decoder bias already subtracted (``b_enc - W_enc @ b_dec``); if the
source keeps the raw ``b_enc`` and subtracts ``b_dec`` from the input instead, pass
``--fold-decoder-bias``. Which one a file uses can be checked by evaluating both: the right one
reconstructs far better (``muscari eval-sae``).

    python examples/import_sae.py --weights sae.safetensors --k 16 \\
        --base-model sbintuitions/modernbert-ja-310m --max-length 512 \\
        --column "anc:query=検索クエリ: " --column "pos:document=検索文書: " --out runs/imported

The recorded columns and prefixes only tell ``eval-sae`` and ``probe-sae`` how to read texts;
use the ones the SAE was trained with when they are known. The source is named in the config
by its file name only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from muscari.data import TextColumn
from muscari.export import CONFIG_NAME, save_sae

NAMES = ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias")


def import_sae(
    weights: str | Path,
    out_dir: str | Path,
    *,
    k: int,
    base_model: str,
    max_length: int,
    columns: Sequence[TextColumn] = (),
    layer: str = "last_hidden_state",
    fold_decoder_bias: bool = False,
    source: str | None = None,
) -> Path:
    """Write ``out_dir`` in muscari's format from a safetensors file with the same tensor names.

    Raises:
        ValueError: if a tensor is missing, ``k`` is out of range or the shapes disagree.
    """
    tensors = load_file(str(weights))
    if missing := [name for name in NAMES if name not in tensors]:
        raise ValueError(f"The file has no {missing}; it has {sorted(tensors)}.")
    encoder_weight = tensors["encoder.weight"].float()
    encoder_bias = tensors["encoder.bias"].float()
    decoder_bias = tensors["decoder.bias"].float()
    if not 0 < k <= encoder_weight.shape[0]:
        raise ValueError(f"k must be between 1 and {encoder_weight.shape[0]}.")
    if fold_decoder_bias:
        encoder_bias = encoder_bias - encoder_weight @ decoder_bias

    config: dict[str, Any] = {
        "k": k,
        "base_model": base_model,
        "layer": layer,
        "max_length": max_length,
        "imported_from": source or Path(weights).name,
        "decoder_bias_folded_on_import": fold_decoder_bias,
    }
    if columns:
        config["columns"] = {column.name: column.prefix for column in columns}
        config["labels"] = {column.name: column.label for column in columns if column.label}
    return save_sae(
        out_dir,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=tensors["decoder.weight"].float(),
        decoder_bias=decoder_bias,
        config=config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--weights", required=True, help="safetensors file of the SAE")
    parser.add_argument("--k", type=int, required=True, help="active latents per token")
    parser.add_argument("--base-model", required=True, help="the encoder the SAE was trained on")
    parser.add_argument("--max-length", type=int, required=True, help="token limit to read texts with")
    parser.add_argument("--layer", default="last_hidden_state")
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        type=TextColumn.parse,
        help='text column as "name:label=prefix", as in muscari train-sae --columns; repeatable',
    )
    parser.add_argument("--fold-decoder-bias", action="store_true", help="the source keeps the raw encoder bias")
    parser.add_argument("--source", help="name to record for the source; default: the file name")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = import_sae(
        args.weights,
        args.out,
        k=args.k,
        base_model=args.base_model,
        max_length=args.max_length,
        columns=args.column,
        layer=args.layer,
        fold_decoder_bias=args.fold_decoder_bias,
        source=args.source,
    )
    config = json.loads((out / CONFIG_NAME).read_text(encoding="utf-8"))
    print(json.dumps({key: config[key] for key in ("d_in", "num_latents", "k", "base_model")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
