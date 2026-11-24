"""Command line interface: ``muscari train-sae`` and ``muscari eval-sae``.

``train-sae`` streams texts through a frozen encoder, trains an SAE on the token activations
with one of the two trainers, exports it in the common format and evaluates it on held-out
texts. ``eval-sae`` measures an exported SAE on any texts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .activations import EncoderActivationSource, shuffled_batches
from .data import SPLITS, TextColumn, iter_texts, repeat_texts
from .devices import resolve_device
from .export import CONFIG_NAME, ExportedSAE, evaluate_sae, load_sae, save_topk_sae

_FILE_LOADERS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".csv": "csv", ".txt": "text"}


def _add_encoder_arguments(parser: argparse.ArgumentParser, *, for_training: bool) -> None:
    group = parser.add_argument_group("encoder")
    group.add_argument(
        "--model",
        required=for_training,
        help="Hugging Face Hub id or local directory of the encoder",
    )
    group.add_argument("--revision", help="model revision on the Hub")
    group.add_argument("--layer", type=int, help="hidden_states index; default: last_hidden_state")
    group.add_argument("--max-length", type=int, default=1024 if for_training else None)
    group.add_argument("--text-batch-size", type=int, default=64, help="texts per encoder forward pass")
    group.add_argument("--keep-prefix-tokens", action="store_true", help="also use the prefix tokens")
    group.add_argument("--encoder-amp", action="store_true", help="run the encoder under bf16 autocast (CUDA)")
    group.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")


def _add_data_arguments(parser: argparse.ArgumentParser, *, for_training: bool) -> None:
    group = parser.add_argument_group("data")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Hugging Face dataset id")
    source.add_argument("--data-files", nargs="+", help="local .jsonl, .parquet, .csv or .txt files")
    group.add_argument("--subsets", nargs="*", help="dataset configurations to concatenate")
    group.add_argument("--split", default="train", help="dataset split to read")
    group.add_argument(
        "--columns",
        nargs="+",
        required=for_training,
        help="columns to read, as name or name=prefix; .txt files provide the column 'text'",
    )
    group.add_argument("--min-chars", type=int, default=1, help="skip texts shorter than this")
    group.add_argument("--seed", type=int, default=42, help="seed for row order, shuffling and initialisation")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with both subcommands."""
    parser = argparse.ArgumentParser(
        prog="muscari",
        description="Sparse autoencoders for text encoders.",
    )
    parser.add_argument("--version", action="version", version=f"muscari {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train-sae",
        help="train an SAE on an encoder's token activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument("--trainer", choices=["topk", "saelens"], required=True)
    train.add_argument("--out", required=True, help="output directory")
    _add_encoder_arguments(train, for_training=True)
    _add_data_arguments(train, for_training=True)
    train.add_argument("--holdout-pct", type=float, default=5.0, help="percent of distinct texts held out")
    train.add_argument(
        "--eval-tokens", type=int, help="held-out tokens to evaluate; default: max(200000, 32 * latents)"
    )

    group = train.add_argument_group("SAE")
    group.add_argument("--num-latents", type=int, default=65_536)
    group.add_argument("--k", type=int, default=16)
    group.add_argument("--total-tokens", type=int, default=1_000_000_000)
    group.add_argument("--batch-tokens", type=int, default=8192)
    group.add_argument("--buffer-tokens", type=int, default=524_288, help="shuffle buffer size in tokens")
    group.add_argument("--lr", type=float, default=5e-4)
    group.add_argument("--warmup-frac", type=float, default=0.05)
    group.add_argument("--dead-after-tokens", type=int, default=10_000_000)
    group.add_argument("--amp", action="store_true", help="bf16 autocast for the SAE (CUDA)")

    group = train.add_argument_group("topk trainer")
    group.add_argument("--aux-k", type=int, default=256)
    group.add_argument("--aux-coef", type=float, default=1 / 32)
    group.add_argument("--adam-eps", type=float, default=1e-8)
    group.add_argument("--log-every", type=int, default=100, help="steps between log records")
    group.add_argument("--checkpoint-every", type=int, default=0, help="steps between checkpoints; 0 disables")
    group.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt")

    group = train.add_argument_group("saelens trainer")
    group.add_argument("--architecture", choices=["topk", "batchtopk"], default="topk")
    group.add_argument("--aux-loss-coefficient", type=float, default=1.0)
    group.add_argument("--zero-b-dec", action="store_true", help="start b_dec at zero as stock SAELens does")
    group.add_argument("--wandb", action="store_true", help="log SAELens metrics to Weights & Biases")

    evaluate = commands.add_parser(
        "eval-sae",
        help="evaluate an exported SAE on texts",
        description="Encoder settings that are omitted are taken from the SAE's sae_config.json.",
    )
    evaluate.add_argument("--sae", required=True, help="directory with sae.safetensors and sae_config.json")
    evaluate.add_argument("--out", help="also write the metrics to this JSON file")
    _add_encoder_arguments(evaluate, for_training=False)
    _add_data_arguments(evaluate, for_training=False)
    evaluate.add_argument("--holdout-pct", type=float, help="default: the value recorded with the SAE")
    evaluate.add_argument("--part", choices=SPLITS, help="default: holdout if a holdout percentage is set, else all")
    evaluate.add_argument("--eval-tokens", type=int, help="default: max(200000, 32 * latents)")
    return parser


def load_rows(
    *,
    dataset: str | None,
    subsets: Sequence[str] | None,
    data_files: Sequence[str] | None,
    split: str,
    column_names: Sequence[str],
    seed: int,
) -> Iterable[dict[str, Any]]:
    """Load rows from the Hub or local files, keep the wanted columns and shuffle the row order.

    Several subsets are concatenated before shuffling so that training does not walk through
    them one after another.

    Raises:
        ValueError: on an unsupported file extension, mixed file types, or when none of the
            requested columns exists.
    """
    from datasets import concatenate_datasets, load_dataset

    if data_files:
        kinds = {_FILE_LOADERS.get(Path(name).suffix.lower()) for name in data_files}
        if None in kinds or len(kinds) != 1:
            raise ValueError(f"--data-files must share one extension out of {sorted(_FILE_LOADERS)}.")
        parts = [load_dataset(kinds.pop(), data_files=list(data_files), split="train")]
    else:
        configurations: Sequence[str | None] = subsets or [None]
        parts = [load_dataset(dataset, name, split=split) for name in configurations]

    wanted = [part.select_columns([name for name in column_names if name in part.column_names]) for part in parts]
    if not any(part.column_names for part in wanted):
        raise ValueError(f"None of the columns {list(column_names)} exists in the data.")

    rows: Iterable[dict[str, Any]] = concatenate_datasets(wanted).shuffle(seed=seed)
    return rows


def _public_name(name: str) -> str:
    """Record a local path by its last component so that outputs carry no directory layout."""
    return Path(name).name if Path(name).exists() else name


def _build_source(
    args: argparse.Namespace, *, max_length: int, layer: int | None, exclude_prefix: bool, model: str
) -> EncoderActivationSource:
    return EncoderActivationSource.from_pretrained(
        model,
        revision=args.revision,
        max_length=max_length,
        text_batch_size=args.text_batch_size,
        layer=layer,
        exclude_prefix=exclude_prefix,
        device=args.device,
        amp_dtype=torch.bfloat16 if args.encoder_amp else None,
    )


def _print_json(record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True)


def default_eval_tokens(num_latents: int) -> int:
    """Enough tokens for rare latents to show up: at least 32 per latent, and at least 200,000."""
    return max(200_000, 32 * num_latents)


def evaluate_by_prefix(sae: ExportedSAE, groups: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Evaluate on all activations together and on each prefix (queries, documents) separately.

    The per-prefix entries carry the group's own FVU, measured around the group's own mean, and
    its share of the evaluated tokens.
    """
    total = sum(group.shape[0] for group in groups.values())
    metrics: dict[str, Any] = dict(evaluate_sae(sae, torch.cat(list(groups.values()))))

    metrics["by_prefix"] = {}
    for prefix, group in groups.items():
        group_metrics = evaluate_sae(sae, group)
        metrics["by_prefix"][prefix] = {
            "tokens": group.shape[0],
            "token_share": group.shape[0] / total,
            "fvu": group_metrics["fvu"],
            "l0": group_metrics["l0"],
        }
    return metrics


def _same_data(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    """Whether the data given to ``eval-sae`` is the data the SAE was trained on."""
    if args.data_files:
        return [_public_name(name) for name in args.data_files] == config.get("dataset")
    return args.dataset == config.get("dataset") and (args.subsets or None) == (config.get("subsets") or None)


def train_sae(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ``train-sae`` subcommand and return the final metrics."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and args.trainer != "topk":
        raise SystemExit("--resume is only supported by the topk trainer.")

    device = resolve_device(args.device)
    columns = [TextColumn.parse(spec) for spec in args.columns]
    rows = load_rows(
        dataset=args.dataset,
        subsets=args.subsets,
        data_files=args.data_files,
        split=args.split,
        column_names=[column.name for column in columns],
        seed=args.seed,
    )
    source = _build_source(
        args,
        max_length=args.max_length,
        layer=args.layer,
        exclude_prefix=not args.keep_prefix_tokens,
        model=args.model,
    )
    print(f"muscari {__version__}: encoder on {device}, d_in={source.d_in}", file=sys.stderr, flush=True)

    def texts(split: str) -> Iterator[tuple[str, str]]:
        return iter_texts(
            rows,
            columns,
            holdout_pct=args.holdout_pct,
            split=split,
            min_chars=args.min_chars,
        )

    started = time.perf_counter()
    eval_tokens = default_eval_tokens(args.num_latents) if args.eval_tokens is None else args.eval_tokens
    eval_groups = None
    if args.holdout_pct > 0 and eval_tokens > 0:
        eval_groups = source.collect_by_prefix(texts("holdout"), eval_tokens)
    source.reset_counters()

    epochs_started = 0

    def train_epoch() -> Iterator[tuple[str, str]]:
        nonlocal epochs_started
        epochs_started += 1
        return texts("train")

    provenance = {
        "muscari_version": __version__,
        "base_model": _public_name(args.model),
        "revision": args.revision,
        "layer": "last_hidden_state" if args.layer is None else args.layer,
        "dataset": [_public_name(name) for name in args.data_files] if args.data_files else args.dataset,
        "subsets": None if args.data_files else args.subsets,
        "split": args.split,
        "columns": {column.name: column.prefix for column in columns},
        "prefix_tokens_excluded": not args.keep_prefix_tokens,
        "max_length": args.max_length,
        "holdout_pct": args.holdout_pct,
        "seed": args.seed,
    }

    if args.trainer == "topk":
        from .trainers.topk import CHECKPOINT_NAME, TopKTrainer, TopKTrainerConfig

        topk_config = TopKTrainerConfig(
            num_latents=args.num_latents,
            k=args.k,
            aux_k=args.aux_k,
            aux_coef=args.aux_coef,
            dead_after_tokens=args.dead_after_tokens,
            lr=args.lr,
            warmup_frac=args.warmup_frac,
            adam_eps=args.adam_eps,
            total_tokens=args.total_tokens,
            batch_tokens=args.batch_tokens,
            amp=args.amp,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            device=str(device),
        )
        trainer = TopKTrainer(topk_config, source.d_in)

        train_texts: Iterator[tuple[str, str]] = repeat_texts(train_epoch)
        skipped_texts = 0
        if args.resume:
            if not (out / CHECKPOINT_NAME).is_file():
                raise SystemExit(f"--resume needs {CHECKPOINT_NAME} in the output directory; none was found.")
            data_state = trainer.load_checkpoint(out / CHECKPOINT_NAME)
            skipped_texts = int(data_state.get("texts_seen", 0))
            train_texts = itertools.islice(train_texts, skipped_texts, None)
            print(
                f"resumed at step {trainer.step}; skipping {skipped_texts} texts already encoded",
                file=sys.stderr,
                flush=True,
            )

        batches = shuffled_batches(
            source.stream(train_texts),
            batch_tokens=args.batch_tokens,
            buffer_tokens=args.buffer_tokens,
            seed=args.seed + trainer.step,
        )
        sae = trainer.fit(
            batches,
            out_dir=out,
            log_fn=_print_json,
            data_state=lambda: {"texts_seen": skipped_texts + source.texts_seen},
        )
        save_topk_sae(
            out,
            sae,
            {
                **provenance,
                "trainer": "topk",
                "tokens_trained": trainer.tokens_seen,
                "train_unused_frac": trainer.unused_frac(),
                "train_unused_window_tokens": min(10_000_000, trainer.tokens_seen),
            },
        )
        sae_seconds: float | None = trainer.step_seconds
    else:
        from .trainers.saelens import SAELensTrainerConfig, train_with_saelens

        saelens_config = SAELensTrainerConfig(
            architecture=args.architecture,
            num_latents=args.num_latents,
            k=args.k,
            aux_loss_coefficient=args.aux_loss_coefficient,
            dead_after_tokens=args.dead_after_tokens,
            lr=args.lr,
            lr_end=args.lr / 100,
            warmup_frac=args.warmup_frac,
            total_tokens=args.total_tokens,
            batch_tokens=args.batch_tokens,
            amp=args.amp,
            init_b_dec=not args.zero_b_dec,
            seed=args.seed,
            log_to_wandb=args.wandb,
            device=str(device),
        )
        batches = shuffled_batches(
            source.stream(repeat_texts(train_epoch)),
            batch_tokens=args.batch_tokens,
            buffer_tokens=args.buffer_tokens,
            seed=args.seed,
        )
        train_with_saelens(
            saelens_config,
            source.d_in,
            batches,
            out,
            export_config={
                **provenance,
                "tokens_trained": saelens_config.total_steps * args.batch_tokens,
            },
        )
        sae_seconds = None  # SAELens does not report the time of its own steps

    exported = load_sae(out, device)
    metrics: dict[str, Any] = {
        "texts_encoded": source.texts_seen,
        "tokens_encoded": source.tokens_seen,
        "tokens_encoded_by_prefix": dict(source.tokens_by_prefix),
        "epochs_started": epochs_started,
        "train_unused_frac": exported.config.get("train_unused_frac"),
        "train_unused_window_tokens": exported.config.get("train_unused_window_tokens"),
        "wall_seconds": time.perf_counter() - started,
        "encoder_seconds": source.encode_seconds,
        "sae_seconds": sae_seconds,
    }
    if eval_groups is not None:
        metrics.update(evaluate_by_prefix(exported, eval_groups))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def eval_sae(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ``eval-sae`` subcommand and return the metrics."""
    sae_dir = Path(args.sae)
    config = json.loads((sae_dir / CONFIG_NAME).read_text(encoding="utf-8"))

    model = args.model or config.get("base_model")
    if model is None:
        raise SystemExit("--model is required: the SAE config records no base model.")
    if args.columns:
        columns = [TextColumn.parse(spec) for spec in args.columns]
    elif config.get("columns"):
        columns = [TextColumn(name, prefix) for name, prefix in config["columns"].items()]
    else:
        raise SystemExit("--columns is required: the SAE config records no columns.")

    recorded_layer = config.get("layer", "last_hidden_state")
    layer = (
        args.layer if args.layer is not None else (None if recorded_layer == "last_hidden_state" else recorded_layer)
    )
    holdout_pct = args.holdout_pct if args.holdout_pct is not None else float(config.get("holdout_pct", 0.0))
    # Held-out texts only make sense for the data the SAE was trained on; other data is all new.
    part = args.part or ("holdout" if holdout_pct > 0 and _same_data(args, config) else "all")

    rows = load_rows(
        dataset=args.dataset,
        subsets=args.subsets,
        data_files=args.data_files,
        split=args.split,
        column_names=[column.name for column in columns],
        seed=args.seed,
    )
    source = _build_source(
        args,
        max_length=args.max_length or int(config.get("max_length", 1024)),
        layer=layer,
        exclude_prefix=not args.keep_prefix_tokens and bool(config.get("prefix_tokens_excluded", True)),
        model=model,
    )
    eval_tokens = default_eval_tokens(int(config["num_latents"])) if args.eval_tokens is None else args.eval_tokens
    groups = source.collect_by_prefix(
        iter_texts(
            rows,
            columns,
            holdout_pct=holdout_pct,
            split=part,
            min_chars=args.min_chars,
        ),
        eval_tokens,
    )

    metrics: dict[str, Any] = {"part": part, "holdout_pct": holdout_pct}
    metrics.update(evaluate_by_prefix(load_sae(sae_dir, source.device), groups))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``muscari`` command."""
    args = build_parser().parse_args(argv)
    if args.command == "train-sae":
        train_sae(args)
    else:
        eval_sae(args)


if __name__ == "__main__":
    main()
