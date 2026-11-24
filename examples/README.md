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

`train_retriever.py` trains a sparse retriever with Sentence Transformers and ranks the candidates of a reranking dataset before and after. It needs the `retrieval` extra. Two heads train with the same data, loss and schedule, so that they can be compared:

- `--head sae` uses the latents of an exported SAE as the vocabulary, through `muscari.retrieval`;
- `--head mlm` uses the tokenizer's vocabulary through the language-model head, as SPLADE does. The encoder must still have that head.

```bash
python examples/train_retriever.py \
  --head sae --sae runs/ruri-v3-30m-sae-trial \
  --query-prompt "検索クエリ: " --document-prompt "検索文書: " \
  --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \
  --query-column anc --positive-column pos --negative-column neg \
  --benchmark jqara --max-steps 1000 \
  --out runs/ruri-v3-30m-retriever-trial
```

The loss is a contrastive loss with in-batch negatives and one hard negative per query, plus a FLOPS regulariser on queries and documents whose weight rises quadratically from zero over the first 30% of the steps. The learning rate, the regulariser weight and that schedule are the ones the model card of [sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m) states for its first training stage; the batch size, the length limit and the number of steps are small defaults and not that card's.

The script writes `results.json` with nDCG@10, MRR@10, the active dimensions per query and passage and FLOPS, before and after training, and saves the model. For the evaluation, the query vectors stay on the device and the passages are encoded a chunk at a time and scored against them there; only the scores and per-dimension counts are kept. A vocabulary-sized output, which can be nearly dense before training, is never held for the whole collection, and no sparse tensor is built on the accelerator. Training saves a checkpoint every tenth of the steps; started again with the same `--out`, the script reuses the evaluation before training and continues from the last checkpoint. Before training, an SAE head gives the `sparse.plain` result of `muscari probe-sae` on the same queries. The tests run both heads for three steps on a tiny random encoder. No run on a real encoder has been made with this script yet.
