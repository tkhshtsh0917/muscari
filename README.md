# muscari

[日本語](README.ja.md)

Sparse autoencoders for text encoders: train SAEs on encoder hidden states and use them as the vocabulary of SAE-based learned sparse retrieval.

muscari takes a frozen text encoder such as [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m), runs your texts through it, and trains a Top-K sparse autoencoder (SAE) on the hidden state of every content token. The result is a dictionary of a few tens of thousands of directions. A retrieval model can then score queries and documents on those directions instead of on the tokenizer's vocabulary, which is the idea behind SAE-SPLADE and SPLARE.

This is an early version (`0.1.0.dev0`). It covers SAE training, export and evaluation, and a Sentence Transformers head that turns an exported SAE into a sparse encoder. That head has been trained only on a small scale so far; see [Roadmap](#roadmap).

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

# with the Sentence Transformers retrieval head
pip install "muscari[retrieval] @ git+https://github.com/tkhshtsh0917/muscari.git"
```

It needs Python 3.11 or later, PyTorch 2.6 or later, and Transformers 4.48 or later. The retrieval head needs Sentence Transformers 6.3 or later, which in turn needs Transformers 5. For development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

Train an SAE on the final layer of ruri-v3 with the data it was fine-tuned on:

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-310m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets auto-wiki-qa-nemotron jaquad jqara jsquad miracl mkqa mr-tydi quiz-no-mori quiz-works \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --checkpoint-every 1000 \
  --out runs/ruri-v3-310m-sae
```

`--columns` lists the dataset columns to read and the prefix the encoder expects for each. Statistics are reported per prefix, which separates queries from documents. For an encoder that takes no prefix, label the columns instead: `--columns "anc:query=" "pos:document=" "neg:document="`. The defaults are 65,536 latents, `k = 16` and one billion tokens, the values reported on the [model card of the SAE used by Shizuku](https://huggingface.co/mahiyama/modernbert-ja-310m-sae). For a first look, shrink the run with `--num-latents`, `--total-tokens` and a smaller model such as `cl-nagoya/ruri-v3-30m`.

The same activations through SAELens:

```bash
muscari train-sae --trainer saelens --architecture topk ...
```

Any corpus works. A plain text file with one text per line provides the column `text`:

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-310m \
  --data-files corpus.txt \
  --columns "text=検索文書: " \
  --out runs/my-corpus-sae
```

Evaluate an exported SAE on other texts. Encoder settings that you omit are read from the SAE's `sae_config.json`:

```bash
muscari eval-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

List what each latent responds to, most frequent first. The table goes to the terminal and the full list to `latents.jsonl` next to the SAE:

```bash
muscari inspect-sae --sae runs/ruri-v3-310m-sae --data-files other.jsonl --columns "text=検索文書: "
```

Rerank the candidates of a reranking dataset with the frozen encoder and the SAE, before any retrieval training. This shows whether the encoder still ranks well with the SAE in its path, and how the SAE's features rank on their own:

```bash
muscari probe-sae --sae runs/ruri-v3-310m-sae --benchmark jqara \
  --query-prefix "検索クエリ: " --passage-prefix "検索文書: "
```

Run `muscari train-sae --help` for every option. An offline example that needs no download is in [examples](examples/README.md).

## Python API

```python
from muscari import (
    EncoderActivationSource,
    TopKTrainer,
    TopKTrainerConfig,
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

config = TopKTrainerConfig(
    num_latents=16_384,
    k=16,
    total_tokens=50_000_000,
)
trainer = TopKTrainer(config, d_in=source.d_in)
sae = trainer.fit(batches, out_dir="runs/sae")
save_topk_sae("runs/sae", sae, {"base_model": "cl-nagoya/ruri-v3-30m"})

exported = load_sae("runs/sae", device=source.device)
hidden = source.encode_batch(["検索文書: "], ["名古屋の求人"])  # (tokens, d_in)
features = exported.encode(hidden)  # (tokens, num_latents), k non-zero entries per token
```

## Sparse retrieval head

`muscari.retrieval` puts an exported SAE on top of its encoder as a [Sentence Transformers](https://sbert.net/) `SparseEncoder`. The head, `SAEPooling`, takes the place of the language-model head and pooling of a SPLADE model: the final hidden state of every content token goes through the SAE's encoder, the active latents are saturated with `log1p`, and the maximum over the tokens gives one dimension per latent.

```python
from muscari.retrieval import build_sparse_encoder

model = build_sparse_encoder(
    "runs/ruri-v3-310m-sae",
    query_prompt="検索クエリ: ",
    document_prompt="検索文書: ",
)
queries = model.encode_query(["名古屋のエンジニア求人"])
documents = model.encode_document(["名古屋市内の開発職を募集しています。", "東京の営業職です。"])
scores = model.similarity(queries, documents)
```

The model trains with the losses, trainer and evaluators of `sentence_transformers.sparse_encoder`. The SAE stays frozen unless the head is built with `trainable=True`:

```python
from sentence_transformers.sparse_encoder import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

loss = SpladeLoss(
    model,
    SparseMultipleNegativesRankingLoss(model),
    document_regularizer_weight=...,  # to be tuned; no value has been validated with this head
    query_regularizer_weight=...,
)
trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
trainer.train()
model.save("runs/ruri-v3-310m-sae-retriever")
```

A complete script that trains this head or, for comparison, a vocabulary-based SPLADE head with the same recipe is in [examples](examples/README.md#retrieval-training).

A saved model loads with `SparseEncoder(path, trust_remote_code=True)` while muscari is installed. Sentence Transformers asks for that flag for every module class outside its own namespace.

Latents that are active in nearly every text cannot tell texts apart. `inspect-sae` lists them by `text_frequency`, and `build_sparse_encoder(..., head_kwargs={"masked_latents": [...]})` removes them from the output.

A first small-scale run on modernbert-ja-30m lifted nDCG@10 on 200 JQaRA queries from 0.37 to about 0.46 in 2,000 steps, in float32; bfloat16 autocast did clearly worse. A vocabulary-based SPLADE head trained with the same recipe collapsed, so the two heads have not been compared yet. Details are in the [design guide](docs/design.md#retrieval-training-on-a-small-scale).

The head is tested for correctness: its vectors equal the ones `probe-sae` scores, on a tiny model and on ruri-v3-30m with its real tokenizer; its gradients equal a dense computation; and a few training steps on a tiny model update the encoder and leave the SAE unchanged.

## What gets written

| File | Content |
| --- | --- |
| `sae.safetensors` | `encoder.weight (M, d)`, `encoder.bias (M)`, `decoder.weight (d, M)`, `decoder.bias (d)`, and `threshold (M)` for JumpReLU SAEs |
| `sae_config.json` | sizes, activation rule, `k`, base model, layer, dataset, columns and prefixes, holdout percentage, trainer, tokens trained |
| `metrics.json` | held-out evaluation, dictionary usage, token counts and timings after training |
| `train_log.jsonl`, `trainer_config.json`, `checkpoint.pt` | `topk` trainer only |
| `latents.jsonl` | written by `inspect-sae`: one line per latent with its firing frequency, position in the text and most frequent tokens |

The decoder bias is folded into the encoder bias, so encoding is one affine map: `pre = x @ encoder.weight.T + encoder.bias`, then keep the `k` largest values per token after ReLU. Local paths are recorded by file name only.

## The two trainers

| | `topk` | `saelens` |
| --- | --- | --- |
| Implementation | this repository | [SAELens](https://pypi.org/project/sae-lens/) `SAETrainer` (optional extra) |
| Architectures | Top-K | Top-K, BatchTopK (exported as JumpReLU) |
| Auxiliary loss | AuxK with configurable `aux_k`, normalised by the residual variance | SAELens' own: `d_in // 2` dead latents, not normalised |
| Dead latents | counted in tokens | counted in steps |
| Schedule | 5% linear warmup, cosine decay to zero | linear warmup, cosine decay to `lr / 100` |
| Decoder bias at the start | geometric median of the first batch | the same; SAELens' own default of zero with `--zero-b-dec` |
| Checkpoint and resume | yes (`--checkpoint-every`, `--resume`) | no |
| Logging | JSON lines | SAELens, optionally Weights & Biases |

`topk` is this repository's own loop. Its mechanics follow Gao et al., and its default hyperparameters are the ones stated on the model card of the SAE behind mahiyama's [Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m) models. The training code of those models is not public and this project is not affiliated with them. What the card states, what it leaves open and how to run with its settings are in the [design guide](docs/design.md#running-with-the-settings-of-the-shizuku-sae-card).

## Evaluation

`metrics.json` and `eval-sae` report, on held-out tokens:

| Metric | Meaning |
| --- | --- |
| `fvu` | fraction of variance unexplained: squared reconstruction error over the variance around the mean activation |
| `l0` | mean number of active latents per token |
| `never_fired_frac` | share of latents that never fire on the evaluated tokens |
| `rare_frac` | share of latents that fire less than a tenth as often as the average latent |
| `dense_frac` | share of latents that fire on more than 10% of the tokens |
| `median_frequency`, `max_frequency` | firing frequency of the median latent and of the most frequent one |
| `by_prefix` | token share, `fvu` and `l0` separately for each prefix, that is for queries and documents. Where columns carry labels, the labels take the place of the prefixes here and below |

A sparse index does not store tokens. It stores one vector per query or document, pooled over the tokens with a maximum, and pays for every non-zero dimension. `text_level` describes the dictionary after that pooling, where a latent counts as active in a text if it fires on at least one token:

| Metric | Meaning |
| --- | --- |
| `texts`, `tokens_per_text` | number of evaluated texts with this prefix and their mean length |
| `l0`, `l0_median`, `l0_p95` | active latents per text: mean, median and 95th percentile |
| `never_fired_frac` | share of latents active in none of the texts |
| `dense_frac` | share of latents active in more than 10% of the texts |
| `max_document_frequency` | largest share of texts in which a single latent is active |
| `flops` | for each pair of prefixes, the sum over latents of the product of their activation probabilities: the expected number of multiplications when one text of the first kind is scored against one of the second through an inverted index |

`flops` for a query prefix and a document prefix is the FLOPS figure that [naver/splade](https://github.com/naver/splade) reports for a query and a document index, after [Paria et al.](https://arxiv.org/abs/2004.05665). It is a property of the SAE alone here: nothing has been trained for retrieval, and retrieval training changes it.

`inspect-sae` writes one record per latent to `latents.jsonl`:

| Field | Meaning |
| --- | --- |
| `frequency`, `frequency_by_prefix` | share of tokens the latent fires on, overall and among the tokens of each prefix |
| `text_frequency` | share of texts in which it fires on at least one token |
| `mean_activation` | its mean value where it fires |
| `first_token_share`, `last_token_share`, `mean_position` | share of its firings on the first and the last content token of a text, and their mean relative position from 0 to 1 |
| `top_tokens` | the tokens it fires on most, each with `share` (part of the latent's firings on this token) and `token_rate` (part of the token's occurrences that fire the latent) |

These fields separate three things a latent can follow other than meaning: one token, one kind of text, or a position in the text.

### Through the encoder's own head

Reconstruction error does not say whether what the SAE loses matters. Two measurements put the SAE back into the encoder and look at what the encoder does with it.

`probe-sae` reads a reranking dataset with one candidate passage per row and ranks the candidates of each query. `--benchmark jqara` reads the test split of [JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA) (`--split dev` the development split) and, as the benchmark's own evaluator does, puts each passage's title in front of its text; `--without-title` leaves the title out, which is how the probe results in the design guide were measured. `--dataset` or `--data-files` with `--query-column`, `--passage-column`, `--label-column`, `--group-column` and `--title-column` read any other. By default 300 queries are sampled, and every figure comes with its standard error over them (`ndcg@10_stderr`).

| Result | Meaning |
| --- | --- |
| `dense.original` | nDCG@10 and MRR@10 when ranking by cosine similarity of the encoder's mean-pooled embeddings |
| `dense.spliced` | the same after every content token state is replaced by the SAE's reconstruction |
| `dense.empty` | the same after every content token state is replaced by the SAE's decoder bias, which is what the SAE returns for an empty code |
| `dense.recovered` | `(spliced - empty) / (original - empty)`: the share of the encoder's ranking quality that survives the SAE |
| `dense.cosine` | cosine similarity between the original and the spliced embedding of the same text: mean, 5th percentile and minimum |
| `sparse.plain` | ranking by the dot product of the pooled SAE features: `log1p`, then the maximum over the content tokens |
| `sparse.idf` | the same with each latent weighted by its inverse document frequency among the candidate passages |
| `sparse.l0` | active latents per query and per passage |
| `hybrid.rrf` | ranking that fuses `dense.original` and `sparse.plain`, which come from the same pass through the encoder, by reciprocal rank fusion (constant 60, nothing to tune) |
| `hybrid.weighted` | the same two rankings fused as a weighted sum of their scores, each standardised over the query's candidates, for dense weights from 0.1 to 0.9; choose the weight on one split and report it on another |

Every `hybrid` entry also holds `over_dense` and `over_sparse`, its mean per-query gain over each of the two rankings with the standard error of that paired difference. The dense and hybrid rows need an encoder trained to produce mean-pooled embeddings, such as ruri-v3. The sparse rows are the state of an SAE-based sparse retriever before it is trained, so they are a starting point and not a result.

For an encoder that still has its language-model head, `eval-sae --mlm-texts 100` adds `mlm`: the masked-language-model loss with the encoder untouched (`loss_clean`), with the final states of the content tokens replaced by the SAE's reconstruction (`loss_spliced`) and by their mean (`loss_mean`), and `recovered = (loss_mean - loss_spliced) / (loss_mean - loss_clean)`. The Shizuku SAE card reports this quantity. It does not state the masking rate; the default here is 15% and the value used is written next to the result.

After training, `metrics.json` also holds:

| Metric | Meaning |
| --- | --- |
| `train_unused_frac` | share of latents that did not fire during the last 10 million training tokens (the whole run if it was shorter) |
| `tokens_encoded_by_prefix` | how many training tokens came from queries and from documents |
| `epochs_started` | how many passes over the deduplicated texts were started |
| `encoder_seconds`, `sae_seconds`, `wall_seconds` | time spent in the encoder, in the SAE updates (`topk` trainer) and in total |

Held-out texts are chosen by a hash of the text itself, so a passage that appears in many rows is never split between training and evaluation. By default 32 tokens per latent are evaluated, and at least 200,000; with fewer, rare latents have no chance to fire and `never_fired_frac` is inflated. `eval-sae` evaluates only the held-out texts when it is given the data the SAE was trained on, and everything otherwise.

The probe measures ranking with nothing trained. No metric here measures the quality of a trained retriever.

## Validation status

| Environment | What ran | Result |
| --- | --- | --- |
| GitHub Actions, Ubuntu 24.04 | the locked `mise run check` on Python 3.12; the tests with the lowest supported versions on Python 3.11 (PyTorch 2.6, Transformers 4.48, Datasets 3.0, safetensors 0.4.4); the tests with the newest versions, SAELens and Sentence Transformers on Python 3.13 | passed |
| macOS 26.6, Apple M5 Pro, Python 3.12, PyTorch 2.15 | `mise run check`, including the hardware tests on MPS and Betterleaks | passed |
| The same Mac, MPS | runs of 5 to 30 million tokens on `cl-nagoya/ruri-v3-30m` with both trainers, their per-text and per-latent statistics, the reranking probe on 200 JQaRA queries, the retrieval head against the probe, small-scale retrieval training with the SAE head, and the step benchmark | ran end to end; numbers in the [design guide](docs/design.md#trial-on-a-real-encoder) |
| Google Colab, NVIDIA L4, Python 3.13, PyTorch 2.14 | the tests, including the hardware tests of the probe, the loss recovered and the retrieval head; loss and gradients on CUDA against the CPU; bfloat16 autocast; the step benchmark; a 5-million-token run in float32 and bfloat16; small-scale retrieval training of both heads | passed; numbers in the design guide |
| Linux x86_64 CPU, Python 3.13, PyTorch 2.15 | formatting, lint, mypy, tests, package build, public-content audit, step benchmark | passed |

Not run yet: a full-size run of 65,536 latents and one billion tokens on any encoder, and retrieval training with the head beyond 2,000 steps on 30m-parameter encoders. The automated tests use a randomly initialised two-layer BERT, which checks interfaces and numerics, not SAE quality. The runs on ruri-v3-30m use a small dictionary and one seed each.

## Limitations

- One process, one device. No multi-GPU sharding.
- `--resume` restores the SAE, optimizer and dead-latent state exactly. The data stream restarts after the texts already encoded with a new shuffle seed, so the activations that were waiting in the shuffle buffer at checkpoint time are skipped.
- Datasets are loaded in map style and shuffled by row. Streaming datasets are not supported.
- Activations are used as they come out of the encoder, without the per-token normalisation Gao et al. apply.
- The SAELens backend cannot resume, and its auxiliary loss is not configurable from here.
- `probe-sae` and `--mlm-texts` need an SAE trained on `last_hidden_state`.

## Roadmap

- A reference run of 65,536 latents and one billion tokens, with measured reconstruction and probe results.
- Retrieval training with the head at a useful scale, and a comparison with a vocabulary-based SPLADE model whose training is stabilised on the same encoder.
- Streaming datasets.

## Acknowledgements

- Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093): Top-K SAEs, AuxK, the initialisation. Their [reference implementation](https://github.com/openai/sparse_autoencoder) was read to check details and motivated the sparse backward pass.
- mahiyama's [Shizuku](https://huggingface.co/mahiyama/sae-splade-ja-310m) models and the [SAE model card](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) whose recipe the `topk` trainer follows.
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511) (SAE-SPLADE) and [Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277) (SPLARE) for the retrieval setting this tool serves.
- [SAELens](https://pypi.org/project/sae-lens/) for the second training backend.
- [ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) and its [fine-tuning dataset](https://huggingface.co/datasets/cl-nagoya/ruri-v3-dataset-ft), used as the running example.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
