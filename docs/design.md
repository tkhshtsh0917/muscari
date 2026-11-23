# Design

[日本語](ja/design.md)

This guide explains what muscari computes, why it is built this way, how it differs from the work it follows, and what has and has not been verified. Usage is in the [README](../README.md).

## Scope

SAE-based learned sparse retrieval replaces the vocabulary of a SPLADE-style model with the latents of a sparse autoencoder. Training such a retriever has two stages: first an SAE is trained to reconstruct the hidden states of the backbone, then the retriever is trained on top of it. muscari implements the first stage for encoder backbones and stops at an exported, evaluated SAE.

## Pipeline

```text
rows -> (prefix, text) -> frozen encoder -> content-token activations
     -> shuffle buffer -> trainer (shizuku | saelens) -> sae.safetensors + sae_config.json
     -> held-out evaluation
```

### 1. Text stream (`muscari.data`)

A row is a mapping. Each requested column is paired with the prefix the encoder expects, and a cell may hold one text or a list of texts, as in datasets that store several hard negatives per row.

- **Deduplication.** Retrieval datasets repeat passages heavily, because the same passage serves as a negative for many queries. Exact repeats of `(prefix, text)` are dropped so that frequent negatives are not over-weighted. This keeps one 64-bit hash per distinct text in memory.
- **Holdout by content.** A text belongs to the held-out part when a hash of the text itself falls below the requested percentage. The decision does not depend on the row or the column, so a passage cannot appear on both sides. Splitting by row would leak shared negatives into the evaluation.
- **Epochs.** When the source is exhausted it restarts, and deduplication restarts with it.

### 2. Activations (`muscari.activations`)

The SAE is trained on `last_hidden_state`, the tensor that Sentence Transformers exposes as `token_embeddings`. For ModernBERT this is the output after the final layer norm. `--layer` selects an entry of `hidden_states` instead.

A position is kept when all of the following hold:

1. it is not padding (`attention_mask`);
2. it is not a special token (`special_tokens_mask`);
3. its character span does not end inside the prefix (`offset_mapping`), unless `--keep-prefix-tokens` is given.

A token that straddles the end of the prefix carries text and is kept. Tokenizers without character offsets fall back to dropping the first `n` content tokens, where `n` is the number of tokens the prefix yields on its own. This is exact when the prefix tokenizes the same way with and without the text after it.

Texts are grouped into pools of 16 batches and sorted by length inside each pool to limit padding. Activations are returned as float32 even when the encoder runs under autocast.

### 3. Shuffle buffer

Tokens of one text are strongly correlated, and length sorting adds correlation between neighbouring texts. Activations therefore pass through a buffer of `--buffer-tokens` rows. Whenever it fills up, a random half is emitted in batches of exactly `--batch-tokens` rows and the other half stays to be mixed with the next refill. Each activation is emitted at most once; only the final partial batch is dropped. The buffer is allocated once on the training device, and peak memory is about 1.5 times its size.

### 4. The SAE (`muscari.sae`)

```text
z     = TopK_k(ReLU(W_enc (x - b_dec) + b_enc))
x_hat = W_dec z + b_dec
```

- **Initialisation.** Decoder rows are random unit vectors and the encoder starts as the transpose of the decoder. The encoder is then scaled by one factor so that random unit-norm inputs are reconstructed with unit norm on average. `b_dec` starts at the geometric median of the first batch (Weiszfeld's algorithm on at most 32,768 rows).
- **Unit-norm decoder.** After every step the decoder rows are renormalised, and before the step the gradient component parallel to each row is removed.
- **Loss.** The reconstruction error is divided by the squared error of predicting the batch mean, which makes it the fraction of variance unexplained (FVU) of the batch.
- **Dead latents.** A latent is dead when it has not fired for `--dead-after-tokens` tokens. Firing means being selected with a positive activation; being selected with an activation of zero does not count.
- **AuxK.** The `aux_k` dead latents with the largest pre-activations reconstruct the residual `x - x_hat` of the live latents, without the decoder bias. The residual is detached, so the term trains only dead latents. It is divided by the variance of the residual, set to zero if it is not finite, and added with weight `aux_coef`.

### 5. Sparse backward pass

For a token, the loss depends on `k` pre-activations (plus `aux_k` for the auxiliary term). The gradient with respect to all other pre-activations is exactly zero. Plain autograd still builds the dense `(batch, num_latents)` gradient and multiplies it with the encoder weights and with the inputs, which costs two more products of the size of the forward pass.

muscari implements the encoder and the decoder as autograd functions that visit the selected entries only. With `g[b, m]` the gradient with respect to the selected pre-activations:

```text
grad b_enc[m] = sum_b g[b, m]
grad W_enc[m] = sum_b g[b, m] (x[b] - b_dec)
grad b_dec    = -sum_m grad b_enc[m] W_enc[m]      (encoder part)
grad W_dec[m] = sum_b z[b, m] r[b]                  (r = d loss / d x_hat)
grad z[b, m]  = <r[b], W_dec[m]>
```

These are accumulated with gathers and `index_add_` in row chunks, so no `(batch, n, d_in)` block is kept for the backward pass. The tests compare losses and gradients against a dense implementation with plain autograd, and against numerical differentiation in double precision.

The auxiliary term selects up to 256 latents per token, sixteen times more than the main term, and a gather moves data far more slowly than a matrix product multiplies it. While the dead latents number at most `64 * aux_k`, the auxiliary path therefore uses dense products restricted to the dead columns; beyond that it uses gathers, whose cost does not depend on the number of dead latents. Both paths are tested against the same oracle.

Measured with `benchmarks/sae_step.py` (65,536 latents, width 768, `k = 16`, `aux_k = 256`, random activations, x86_64 CPU with 2 threads, PyTorch 2.15, median of 3 steps after the first):

| Batch tokens | Dead latents | Auxiliary loss | Sparse backward | Dense autograd |
| ---: | ---: | --- | ---: | ---: |
| 2,048 | 10% | on | 2.2 s, 3.1 GB | 11.0 s, 4.5 GB |
| 2,048 | – | off | 1.8 s, 2.9 GB | 6.9 s, 3.9 GB |
| 8,192 | 10% | on | 7.5 s, 4.7 GB | out of memory (7 GB machine) |
| 8,192 | 50% | on | 16.9 s, 5.7 GB | not run |
| 8,192 | – | off | 5.7 s, 4.6 GB | out of memory (7 GB machine) |

Memory is the peak resident size of the process. These numbers describe one small CPU. They are not a statement about GPUs, where the ratio between gathers and matrix products differs.

### 6. Trainers

**`shizuku`** takes one step per batch:

1. set the learning rate: linear warmup over the first `warmup_frac` of the steps, then cosine decay to zero;
2. compute the loss with the dead mask as it stands before the step;
3. backpropagate, remove the parallel decoder gradient, take an Adam step, renormalise the decoder;
4. update the per-latent count of tokens since the latent last fired.

A checkpoint holds the SAE, the optimizer state, the dead-latent counts, the step and token counters, the configuration and the number of texts already encoded. Resuming restores all of it; the test suite checks that a run interrupted halfway ends with bit-identical parameters when it is given the same batches. From the command line the batches are not the same: the text stream continues after the texts already encoded and the buffer is reshuffled with a new seed, so the activations that were waiting in the buffer are skipped. The number of latents, `k` and the batch size must match the checkpoint; the training length may be extended.

**`saelens`** builds a SAELens `TrainingSAE` (`topk` or `batchtopk`) and runs `SAETrainer.fit()` on the same batches.

| muscari | SAELens |
| --- | --- |
| `num_latents`, `k` | `d_sae`, `k` |
| `total_tokens`, `batch_tokens` | `total_training_samples`, `train_batch_size_samples` |
| `warmup_frac` | `lr_warm_up_steps` |
| `lr`, `lr / 100` | `lr`, `lr_end` with `lr_scheduler_name="cosineannealing"` |
| `dead_after_tokens / batch_tokens` | `dead_feature_window` (steps) |
| `aux_loss_coefficient` | `aux_loss_coefficient` |

Fixed inside SAELens 6.56 and therefore different from `shizuku`: the auxiliary loss uses `d_in // 2` dead latents, is not divided by the residual variance, and is scaled down when fewer dead latents exist; `batchtopk` keeps the `k * batch` largest activations of the whole batch and is saved as a JumpReLU SAE with one threshold per latent, so the number of active latents varies between tokens. Activation normalisation is switched off because it cannot be folded into the exported weights.

### 7. Export format (`muscari.export`)

`sae.safetensors` contains `encoder.weight (M, d)`, `encoder.bias (M)`, `decoder.weight (d, M)`, `decoder.bias (d)` and, for JumpReLU SAEs, `threshold (M)`.

The subtraction of `b_dec` is folded into the encoder bias:

```text
W_enc (x - b_dec) + b_enc = W_enc x + (b_enc - W_enc b_dec)
```

so a retrieval head needs only `encoder.weight` and `encoder.bias`. SAELens stores `W_enc` as `(d, M)` and `W_dec` as `(M, d)`; the conversion transposes both, folds the bias when `apply_b_dec_to_input` is set, and copies `threshold` if present. SAELens' inference export has already folded the decoder norms into the weights. The tests check that the converted SAE and SAELens' own inference model produce the same features and reconstructions.

`sae_config.json` carries `format_version`, the sizes, the activation rule (`topk` or `jumprelu`) and provenance. Local model and data paths are recorded by file name only, so an SAE can be shared without revealing a directory layout.

### 8. Evaluation

Evaluation runs on the exported files with the same code for both trainers. `fvu` is the squared reconstruction error over the variance around the mean activation of the evaluated tokens. `l0` is the mean number of active latents per token. `never_fired_frac`, `dense_frac` (latents firing on more than 10% of tokens) and `max_frequency` describe how evenly the dictionary is used. The frequency figures depend on the number of evaluated tokens: with few tokens, rare latents have had no chance to fire.

A good FVU does not imply good retrieval. Latents that fire on a large share of tokens behave like stop words in an inverted index and make posting lists long, which is why the frequency figures are reported next to FVU.

## Differences from Gao et al.

The paper was read in its arXiv v1 HTML form (Sections 2.1 to 2.4, Appendices A and B), and the [reference implementation](https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942) in `sparse_autoencoder/train.py`.

| Aspect | Gao et al. | `shizuku` trainer |
| --- | --- | --- |
| Input preprocessing | subtract the mean over dimensions, scale each activation to unit norm | none: the SAE must work on raw hidden states at retrieval time |
| Learning rate | constant | 5% warmup, cosine decay (SAE model card of Shizuku) |
| Weight averaging | EMA, 0.999 | none |
| Adam epsilon | 6.25e-10 | PyTorch default 1e-8, configurable |
| `aux_k` | typically 512, a power of two near half the width | 256 (model card; also the default of the reference code) |
| `aux_coef` | 1/32 | 1/32 |
| Dead threshold | 10 million tokens | 10 million tokens |
| Dead bookkeeping (reference code) | in steps, pre-activation above 1e-3 | in tokens, activation above 0 |
| Main loss scale (reference code) | MSE times a fixed inverse variance from a sample | MSE over the variance of each batch |
| Encoder rescaling at initialisation | applied; the paper's ablation reports no effect or a weak negative effect | applied |
| Batch size | 131,072 tokens | 8,192 tokens (model card) |
| Scale-out | sharded over GPUs with custom kernels | one device |

The model card of the SAE used by Shizuku states: 65,536 latents, `k = 16`, AuxK with 256 latents and coefficient 1/32, Adam at 5e-4, 5% warmup with cosine decay, batches of 8,192 tokens, a maximum length of 1,024 and one billion training tokens. Everything the card does not state (initialisation, dead bookkeeping, shuffling, prefix handling, the loss normalisation) is muscari's own choice, mostly following Gao et al. The `shizuku` trainer is therefore a reimplementation of a published recipe, not a reproduction of that model.

## What is verified

- Losses and gradients of the sparse backward pass: against dense autograd and numerical differentiation.
- Checkpoint resumption: bit-identical continuation on identical batches.
- Export: identical outputs to the trained module, and to SAELens' own inference model for `topk` and `batchtopk`.
- Content-token selection: against a manual forward pass, with fast and offset-less tokenizers.
- Holdout split: disjoint, deterministic, independent of prefix and row.

All of this uses a randomly initialised two-layer BERT or synthetic activations. See the [validation status](../README.md#validation-status) for the environments that ran.

## What is not verified

- SAE quality on a real encoder: reconstruction, dead and dense latents at 65,536 latents and one billion tokens.
- That an SAE trained here improves retrieval when used as a sparse vocabulary.
- CUDA, including bfloat16 autocast on CUDA. The autocast code path is exercised on CPU only.
- The lowest supported dependency versions on Python 3.11 outside CI.

## Research background

- Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093), 2024. Top-K SAEs, AuxK, dead latents, initialisation.
- Bussmann et al., [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410), 2024. The `batchtopk` architecture of the SAELens backend.
- Formal et al., [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720), 2021. The retrieval model family whose vocabulary an SAE replaces.
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511) (SAE-SPLADE). Replaces the masked-language-model head of SPLADE with an SAE on an encoder.
- [Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277) (SPLARE). Sparse retrieval with a frozen SAE on an LLM backbone.
- mahiyama, [modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) and [sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m) (Shizuku). The recipe followed by the `shizuku` trainer.
- [SAELens](https://pypi.org/project/sae-lens/). The second training backend.
