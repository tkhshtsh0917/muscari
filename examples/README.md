# Examples

[日本語](README.ja.md)

## Offline smoke run

`train_tiny.py` builds a two-layer BERT with random weights and a few hundred synthetic query/passage rows in a temporary directory, runs `muscari train-sae` on them and then evaluates the exported SAE with `muscari eval-sae`. It downloads nothing and finishes in seconds on a CPU.

```bash
python examples/train_tiny.py --trainer shizuku
python examples/train_tiny.py --trainer saelens    # needs the "saelens" extra
python examples/train_tiny.py --trainer shizuku --workdir runs/tiny    # keep the files
```

From a development checkout, `mise run smoke` runs both trainers.

The script prints the training log, the held-out metrics after training and the metrics measured again from the exported files. Use it to see the command line arguments in a working combination and the files a run produces. The encoder is random, so the numbers say nothing about SAE quality.

## A real encoder

The ruri-v3 recipe is in the [README](../README.md#quickstart). For a first run, a small model and a short budget show whether the pipeline works on your machine before committing to a long run:

```bash
muscari train-sae \
  --trainer shizuku \
  --model cl-nagoya/ruri-v3-30m \
  --dataset cl-nagoya/ruri-v3-dataset-ft \
  --subsets jsquad \
  --columns "anc=検索クエリ: " "pos=検索文書: " "neg=検索文書: " \
  --num-latents 8192 \
  --total-tokens 5000000 \
  --dead-after-tokens 1000000 \
  --out runs/ruri-v3-30m-sae-trial
```

Inside this repository's mise environment the Hugging Face Hub is switched to offline mode so that checks never download anything. Set `HF_HUB_OFFLINE=0` for commands that need the Hub.
