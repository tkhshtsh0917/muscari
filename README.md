# muscari

[日本語](README.ja.md)

Sparse autoencoders for text encoders: train SAEs on encoder hidden states and use them as the vocabulary of SAE-based learned sparse retrieval.

muscari takes a frozen text encoder such as [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m), runs your texts through it, and trains a Top-K sparse autoencoder (SAE) on the hidden state of every content token. The result is a dictionary of a few tens of thousands of directions. A retrieval model can then score queries and documents on those directions instead of on the tokenizer's vocabulary, which is the idea behind SAE-SPLADE and SPLARE.

This is an early version (`0.1.0.dev0`). It covers SAE training, export and evaluation. Retrieval training on top of the SAE is planned and not included yet; see [Roadmap](#roadmap).

## Why a separate tool

- **Encoders are not decoder LLMs.** SAELens' training runner loads models through TransformerLens. muscari feeds any Hugging Face encoder through `AutoModel` and hands plain activation tensors to the trainer.
- **Only content tokens should be learned.** Padding, special tokens and the tokens of an instruction prefix (`検索クエリ: `, `query: `) are dropped before training, so the dictionary is spent on the positions a sparse retrieval head pools over.
- **Activations are streamed.** An encoder is cheap to run, so nothing is cached on disk. One billion tokens of 768-dimensional activations would otherwise take about 3 TB.
- **Two trainers, one output format.** A self-contained loop and a SAELens backend consume the same activation stream and write the same files, so results can be compared with the same evaluation code.
- **The backward pass is sparse.** Only `k` latents per token receive gradient, and muscari propagates through those entries only instead of a dense `(tokens, latents)` gradient. On a 2-thread CPU with 65,536 latents and 2,048 tokens of width 768, a step including the auxiliary loss took 2.2 s against 11.0 s with dense autograd. The conditions are in the [design guide](docs/design.md#5-sparse-backward-pass).

## Install

muscari is not on PyPI. Install it from the repository:

```bash
pip install "git+https://github.com/tkhshtsh0917/muscari.git"

# with the optional SAELens backend
pip install "muscari[saelens] @ git+https://github.com/tkhshtsh0917/muscari.git"
```

It needs Python 3.11 or later, PyTorch 2.6 or later, and Transformers 4.48 or later. For development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

Train an SAE on the final layer of ruri-v3 with the data it was fine-tuned on:

```bash
muscari train-sae \
  --trainer shizuku \
  --model cl-nagoya/ruri-v3-310m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets auto-wiki-qa-nemotron jaquad jqara jsquad miracl mkqa mr-tydi quiz-no-mori quiz-works \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --checkpoint-every 1000 \
  --out runs/ruri-v3-310m-sae
```

`--columns` lists the dataset columns to read and the prefix the encoder expects for each. The defaults are 65,536 latents, `k = 16` and one billion tokens, the values reported on the [model card of the SAE used by Shizuku](https://huggingface.co/mahiyama/modernbert-ja-310m-sae). For a first look, shrink the run with `--num-latents`, `--total-tokens` and a smaller model such as `cl-nagoya/ruri-v3-30m`.

The same activations through SAELens:

```bash
muscari train-sae --trainer saelens --architecture topk ...
```

Any corpus works. A plain text file with one text per line provides the column `text`:

```bash
muscari train-sae \
  --trainer shizuku \
  --model cl-nagoya/ruri-v3-310m \
  --data-files corpus.txt \
  --columns "text=検索文書: " \
  --out runs/my-corpus-sae
```

Evaluate an exported SAE on other texts. Encoder settings that you omit are read from the SAE's `sae_config.json`:

```bash
muscari eval-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

Run `muscari train-sae --help` for every option. An offline example that needs no download is in [examples](examples/README.md).

## Python API

```python
from muscari import (
    EncoderActivationSource,
    ShizukuTrainer,
    ShizukuTrainerConfig,
    TextColumn,
    iter_texts,
    load_sae,
    repeat_texts,
    save_topk_sae,
    shuffled_batches,
)

rows = [{"query": "...", "passage": "..."}]  # any iterable of mappings
columns = [
    TextColumn("query", "検索クエリ: "),
    TextColumn("passage", "検索文書: "),
]

source = EncoderActivationSource.from_pretrained("cl-nagoya/ruri-v3-30m")
batches = shuffled_batches(
    source.stream(repeat_texts(lambda: iter_texts(rows, columns))),
    batch_tokens=8192,
    buffer_tokens=524_288,
)

config = ShizukuTrainerConfig(
    num_latents=16_384,
    k=16,
    total_tokens=50_000_000,
)
trainer = ShizukuTrainer(config, d_in=source.d_in)
sae = trainer.fit(batches, out_dir="runs/sae")
save_topk_sae("runs/sae", sae, {"base_model": "cl-nagoya/ruri-v3-30m"})

exported = load_sae("runs/sae", device=source.device)
hidden = source.encode_batch(["検索文書: "], ["名古屋の求人"])  # (tokens, d_in)
features = exported.encode(hidden)  # (tokens, num_latents), k non-zero entries per token
```

## What gets written

| File | Content |
| --- | --- |
| `sae.safetensors` | `encoder.weight (M, d)`, `encoder.bias (M)`, `decoder.weight (d, M)`, `decoder.bias (d)`, and `threshold (M)` for JumpReLU SAEs |
| `sae_config.json` | sizes, activation rule, `k`, base model, layer, dataset, columns and prefixes, holdout percentage, trainer, tokens trained |
| `metrics.json` | held-out evaluation after training |
| `train_log.jsonl`, `trainer_config.json`, `checkpoint.pt` | `shizuku` trainer only |

The decoder bias is folded into the encoder bias, so encoding is one affine map: `pre = x @ encoder.weight.T + encoder.bias`, then keep the `k` largest values per token after ReLU. Local paths are recorded by file name only.

## The two trainers

| | `shizuku` | `saelens` |
| --- | --- | --- |
| Implementation | this repository | [SAELens](https://pypi.org/project/sae-lens/) `SAETrainer` (optional extra) |
| Architectures | Top-K | Top-K, BatchTopK (exported as JumpReLU) |
| Auxiliary loss | AuxK with configurable `aux_k`, normalised by the residual variance | SAELens' own: `d_in // 2` dead latents, not normalised |
| Dead latents | counted in tokens | counted in steps |
| Schedule | 5% linear warmup, cosine decay to zero | linear warmup, cosine decay to `lr / 100` |
| Decoder bias at the start | geometric median of the first batch | the same; SAELens' own default of zero with `--zero-b-dec` |
| Checkpoint and resume | yes (`--checkpoint-every`, `--resume`) | no |
| Logging | JSON lines | SAELens, optionally Weights & Biases |

`shizuku` is an independent implementation of the recipe described on the model card of the SAE behind mahiyama's [Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m) models. Their training code is not public and this project is not affiliated with them. The details and the differences from Gao et al. are in the [design guide](docs/design.md).

## Evaluation

`metrics.json` and `eval-sae` report:

| Metric | Meaning |
| --- | --- |
| `fvu` | fraction of variance unexplained: squared reconstruction error over the variance around the mean activation |
| `l0` | mean number of active latents per token |
| `never_fired_frac` | share of latents that never fire on the evaluated tokens |
| `dense_frac` | share of latents that fire on more than 10% of the tokens |
| `max_frequency` | the largest share of tokens a single latent fires on |

Held-out texts are chosen by a hash of the text itself, so a passage that appears in many rows is never split between training and evaluation. These metrics describe reconstruction and dictionary usage. They do not measure retrieval quality.

## Validation status

| Environment | What ran | Result |
| --- | --- | --- |
| GitHub Actions, Ubuntu 24.04 | the locked `mise run check` on Python 3.12; the tests with the lowest supported versions on Python 3.11 (PyTorch 2.6, Transformers 4.48, Datasets 3.0, safetensors 0.4.4); the tests with the newest versions and SAELens on Python 3.13 | passed |
| macOS 26.6, Apple M5 Pro, Python 3.12, PyTorch 2.15 | `mise run check`, including the four MPS tests and Betterleaks | passed |
| The same Mac, MPS | a trial of 5 million tokens on `cl-nagoya/ruri-v3-30m` with both trainers, and the step benchmark | ran end to end; numbers in the [design guide](docs/design.md#trial-on-a-real-encoder) |
| Linux x86_64 CPU, Python 3.13, PyTorch 2.15 | formatting, lint, mypy, tests, package build, public-content audit, step benchmark | passed |

Not run yet: CUDA (including bfloat16 autocast on CUDA), a full-size run of 65,536 latents and one billion tokens on any encoder, and any measurement of retrieval quality. The automated tests use a randomly initialised two-layer BERT, which checks interfaces and numerics, not SAE quality. The trial on ruri-v3-30m is one short run with one seed.

## Limitations

- One process, one device. No multi-GPU sharding.
- `--resume` restores the SAE, optimizer and dead-latent state exactly. The data stream restarts after the texts already encoded with a new shuffle seed, so the activations that were waiting in the shuffle buffer at checkpoint time are skipped.
- Datasets are loaded in map style and shuffled by row. Streaming datasets are not supported.
- Activations are used as they come out of the encoder, without the per-token normalisation Gao et al. apply.
- The SAELens backend cannot resume, and its auxiliary loss is not configurable from here.

## Roadmap

- A Sentence Transformers module that puts an exported SAE on top of the encoder as a sparse retrieval head, followed by retrieval training.
- A reference run on ruri-v3 with measured reconstruction and retrieval results.
- Streaming datasets.

## Acknowledgements

- Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093): Top-K SAEs, AuxK, the initialisation. Their [reference implementation](https://github.com/openai/sparse_autoencoder) was read to check details and motivated the sparse backward pass.
- mahiyama's [Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m) models and the [SAE model card](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) whose recipe the `shizuku` trainer follows.
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511) (SAE-SPLADE) and [Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277) (SPLARE) for the retrieval setting this tool serves.
- [SAELens](https://pypi.org/project/sae-lens/) for the second training backend.
- [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) and its [fine-tuning dataset](https://huggingface.co/datasets/cl-nagoya/ruri-v3-dataset-ft), used as the running example.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
