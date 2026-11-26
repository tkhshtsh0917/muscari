# Examples

[日本語](README.ja.md)

## Offline smoke run

`train_tiny.py` builds a two-layer BERT with random weights and a few hundred synthetic query/passage rows in a temporary directory, runs `muscari train-sae` on them and then evaluates the exported SAE with `muscari eval-sae`. It downloads nothing and finishes in seconds on a CPU.

```bash
python examples/train_tiny.py --trainer topk
python examples/train_tiny.py --trainer saelens    # needs the "saelens" extra
python examples/train_tiny.py --trainer topk --workdir runs/tiny    # keep the files
```

From a development checkout, `mise run smoke` runs both trainers.

The script prints the training log, the held-out metrics after training and the metrics measured again from the exported files. Use it to see the command line arguments in a working combination and the files a run produces. The encoder is random, so the numbers say nothing about SAE quality.

## A real encoder

The ruri-v3 recipe is in the [README](../README.md#quickstart). For a first run, a small model and a short budget show whether the pipeline works on your machine before committing to a long run:

```bash
muscari train-sae \
  --trainer topk \
  --model cl-nagoya/ruri-v3-30m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets jsquad \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --num-latents 8192 \
  --total-tokens 5000000 \
  --dead-after-tokens 1000000 \
  --out runs/ruri-v3-30m-sae-trial
```

Inside this repository's mise environment the Hugging Face Hub is switched to offline mode so that checks never download anything. There, prefix the command with `mise exec -- env HF_HUB_OFFLINE=0 uv run --frozen --no-sync`. Setting the variable in front of `mise exec` has no effect.

On an Apple M5 Pro with MPS this command took 200 seconds, including the download of the model and the dataset subset and the held-out evaluation. The numbers it produced are in the [design guide](../docs/design.md#trial-on-a-real-encoder).

## Retrieval training

`train_retriever.py` trains a sparse retriever with Sentence Transformers and evaluates it before and after. It needs the `retrieval` extra. Three heads train with the same data, loss and schedule, so that they can be compared:

- `--head sae` uses the latents of an exported SAE as the vocabulary, through `muscari.retrieval`. The SAE encoder stays frozen unless `--trainable-sae` is given.
- `--head vocab` uses the tokenizer's vocabulary through the language-model head (`vocabulary_head.py`). It pools exactly the tokens the SAE head pools, no framing tokens and no prompt tokens, so the two heads differ in their dictionary only. `--shift` subtracts a constant from the logits before ReLU and `--fan-out` keeps that many logits per token, the two changes SPARSEUP ([arXiv 2611.09680](https://arxiv.org/abs/2611.09680)) makes to start training sparse; `--shift auto` estimates the shift as the `--shift-quantile` (default 0.999) of the content-token logits of `--shift-texts` training documents. With no shift, no fan-out and every token pooled the head equals `SpladePooling`; the gradient of the fan-out reaches the kept logits only, which is this project's choice.
- `--head mlm` is Sentence Transformers' `SpladePooling` unchanged, prompt and special tokens included.

The encoder must still have its language-model head for `vocab` and `mlm`.

```bash
python examples/train_retriever.py \
  --head sae --sae runs/ruri-v3-30m-sae-trial \
  --query-prompt "検索クエリ: " --document-prompt "検索文書: " \
  --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \
  --query-column anc --positive-column pos --negative-column neg \
  --eval-benchmark jqara --eval-split dev --max-steps 1000 \
  --out runs/ruri-v3-30m-retriever-trial
```

The loss is a contrastive loss with in-batch negatives and one hard negative per query, plus a FLOPS regulariser on queries and documents whose weight rises quadratically from zero over the first 30% of the steps. The learning rate, the regulariser weight and that schedule are the ones the model card of [sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m) states for its first training stage; the batch size, the length limit and the number of steps are small defaults and not that card's. `--mini-batch-size` encodes each batch in pieces with GradCache (`CachedSpladeLoss`), which gives the same gradients with less memory.

Training saves a checkpoint every tenth of the steps; started again with the same `--out`, the script reuses the evaluation before training and continues from the last checkpoint. `results.json` holds the configuration of the head, the training log and both evaluations; the model is saved in `model/`. Before training, an SAE head gives the `sparse.plain` result of `muscari probe-sae` on the same queries. Train the SAE head in float32: with `--bf16` it ended clearly lower than two float32 runs. With `sbintuitions/modernbert-ja-30m` the unshifted vocabulary head starts nearly dense and collapses within a few hundred steps under these defaults. Both results are in the [design guide](../docs/design.md#retrieval-training-on-a-small-scale).

## Evaluation

`sparse_eval.py` evaluates any Sentence Transformers `SparseEncoder`: a model saved by `train_retriever.py`, an untrained SAE head (`--sae`), or a published sparse retriever. `train_retriever.py` runs the same evaluation with the options prefixed `--eval-`.

```bash
python examples/sparse_eval.py --model runs/ruri-v3-30m-retriever-trial/model \
  --benchmark jqara --split test \
  --retrieval mteb/MIRACLJaRetrievalLite mteb/JaCWIRRetrievalLite \
  --caps full 64/256 64/128 64/64 --out runs/ruri-v3-30m-retriever-trial/eval.json
```

- Reranking ranks each query's own candidates. For JQaRA a passage is `"{title} {text}"`, as in the benchmark's own evaluator (`--without-title` drops the title); all queries of the split are used unless `--queries` draws a sample. The `dev` split serves for choosing settings, `test` for the final numbers.
- Retrieval ranks a whole corpus in the MTEB layout (configurations `corpus`, `queries`, `qrels`, or a directory with those three `.jsonl` files) and reports nDCG@10, MRR@10 and recall@100. The corpus is streamed through the encoder and merged into a running top-100 per query, so neither the corpus vectors nor the full score matrix are kept.
- `--caps QUERY/DOCUMENT` keeps the largest weights of each vector at inference time; `full` keeps them all. The example uses the grid of the SPARSEUP paper: queries capped at 64 terms, documents at 64, 128, 256 or not at all. Every metric comes with the active dimensions per query and per document and the FLOPS, `sum_j p_q(j) p_d(j)`.

Every result also holds the value of each query, keyed by its id. A published model whose modules are not part of Sentence Transformers needs `--trust-remote-code`, which runs code from its repository; read that code first. A local directory saved by these scripts loads without it, unless it contains Python files.

`compare_runs.py` compares two arms on the same queries. An arm may have several result files, for example one per training seed. The script reports both means, the mean of every run and the spread between the runs of an arm, the difference, a 95% bootstrap interval and the p-value of a paired sign-flip randomization test. The interval resamples the queries and, for an arm with more than one run, its runs, so it also covers the variation between seeds; with three seeds that part is coarse. `--resample queries` keeps the runs fixed.

```bash
python examples/compare_runs.py \
  --a runs/sae-seed*/results.json --b runs/vocab-seed*/results.json \
  --task reranking --cap full --metric ndcg@10
```

## Importing an SAE

`import_sae.py` wraps a Top-K SAE trained elsewhere in muscari's format, so that `eval-sae`, `inspect-sae`, `probe-sae` and the retrieval head can use it. The source must be one safetensors file with muscari's tensor names and shapes (`encoder.weight`, `encoder.bias`, `decoder.weight`, `decoder.bias`), which the model card of [mahiyama/modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) documents for that SAE. muscari keeps the decoder bias folded into `encoder.bias`; `--fold-decoder-bias` folds it for a file that keeps the raw encoder bias. The columns and prefixes given with `--column` only tell the evaluation commands how to read texts.

```bash
python examples/import_sae.py --weights sae.safetensors --k 16 \
  --base-model sbintuitions/modernbert-ja-310m --max-length 512 \
  --column "anc:query=検索クエリ: " --column "pos:document=検索文書: " \
  --out runs/imported-sae
```
