# Design

[日本語](ja/design.md)

This guide explains what muscari computes, why it is built this way, how it differs from the work it follows, and what has and has not been verified. Usage is in the [README](../README.md).

## Scope

SAE-based learned sparse retrieval replaces the vocabulary of a SPLADE-style model with the latents of a sparse autoencoder. Training such a retriever has two stages: first an SAE is trained to reconstruct the hidden states of the backbone, then the retriever is trained on top of it. muscari implements the first stage for encoder backbones, up to an exported and evaluated SAE, and supplies the head that the second stage trains on. The second stage itself runs in Sentence Transformers.

## Pipeline

```text
rows -> (prefix, text) -> frozen encoder -> content-token activations
     -> shuffle buffer -> trainer (topk | saelens) -> sae.safetensors + sae_config.json
     -> held-out evaluation
```

### 1. Text stream (`muscari.data`)

A row is a mapping. Each requested column is paired with the prefix the encoder expects, and a cell may hold one text or a list of texts, as in datasets that store several hard negatives per row.

- **Deduplication.** Retrieval datasets repeat passages heavily, because the same passage serves as a negative for many queries. Exact repeats of `(prefix, text)` are dropped so that frequent negatives are not over-weighted. This keeps one 64-bit hash per distinct text in memory.
- **Holdout by content.** A text belongs to the held-out part when a hash of the text itself falls below the requested percentage. The decision does not depend on the row or the column, so a passage cannot appear on both sides. Splitting by row would leak shared negatives into the evaluation.
- **Epochs.** When the source is exhausted it restarts, and deduplication restarts with it.
- **Kinds of text.** Every text carries its kind, and all statistics are reported per kind. By default the kind is the prefix, which separates queries from documents for encoders such as ruri-v3. An encoder without prefixes, such as a masked language model, gives every text the same empty prefix. A column can then carry a label (`name:label=prefix`): the label names the kind, the prefix is still what is prepended, and deduplication works per kind.

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

The auxiliary term selects up to 256 latents per token, sixteen times more than the main term, and a gather moves data far more slowly than a matrix product multiplies it. The auxiliary path therefore prefers dense products restricted to the dead columns. On CUDA it always uses them. On other devices it does so while the dead latents number at most `64 * aux_k`; beyond that it uses gathers, whose time and memory do not depend on the number of dead latents. Both paths are tested against the same oracle.

Measured with `benchmarks/sae_step.py` (65,536 latents, width 768, `k = 16`, `aux_k = 256`, random activations, x86_64 CPU with 2 threads, PyTorch 2.15, median of 3 steps after the first):

| Batch tokens | Dead latents | Auxiliary loss | Sparse backward | Dense autograd |
| ---: | ---: | --- | ---: | ---: |
| 2,048 | 10% | on | 2.2 s, 3.1 GB | 11.0 s, 4.5 GB |
| 2,048 | – | off | 1.8 s, 2.9 GB | 6.9 s, 3.9 GB |
| 8,192 | 10% | on | 7.5 s, 4.7 GB | out of memory (7 GB machine) |
| 8,192 | 50% | on | 16.9 s, 5.7 GB | not run |
| 8,192 | – | off | 5.7 s, 4.6 GB | out of memory (7 GB machine) |

Memory is the peak resident size of the process.

The same benchmark on an Apple M5 Pro (macOS 26.6, PyTorch 2.15, 10% dead latents, median of 4 steps after the first):

| Device | Batch tokens | Auxiliary loss | Sparse backward | Dense autograd |
| --- | ---: | --- | ---: | ---: |
| MPS | 2,048 | on | 0.17 s, 4.9 GB | 0.55 s, 6.0 GB |
| MPS | 8,192 | on | 0.60 s, 13.2 GB | not run |
| MPS | 8,192 | off | 0.52 s, 12.1 GB | not run |
| CPU, 1 thread | 2,048 | on | 0.51 s, 3.2 GB | not run |

On MPS, memory is what the driver has allocated at the end of the run, which includes the allocator's cache and is larger than the tensors in use. At 0.60 s per step of 8,192 tokens, the SAE updates alone would take about 20 hours for one billion tokens on this machine, before the time of the encoder.

On an NVIDIA L4 (Google Colab, PyTorch 2.14, CUDA 13.0, 8,192 tokens per batch, median of 5 steps after the first, peak allocated memory):

| Dead latents | Auxiliary loss | Precision | Sparse backward | Dense autograd |
| ---: | --- | --- | ---: | ---: |
| – | off | float32 | 177 ms, 3.8 GB | 544 ms, 8.1 GB |
| 10% | on | float32 | 220 ms, 4.1 GB | 894 ms, 10.4 GB |
| 50% | on | float32 | 386 ms, 4.9 GB | not run |
| 90% | on | float32 | 551 ms, 5.8 GB | not run |
| – | off | bfloat16 autocast | 101 ms, 2.9 GB | not run |
| 10% | on | bfloat16 autocast | 133 ms, 3.0 GB | not run |

The rule for the auxiliary path comes from these measurements. On the L4, gathers took 580 ms at 50% dead and 620 ms at 90%, slower than the dense products at the same peak memory, so CUDA always uses the dense form. On the 7 GB CPU machine the dense form was no faster at 50% and ran out of memory at 90%, so the threshold stays elsewhere. MPS was measured at 10% only. The first two rows come from `benchmarks/sae_step.py`. The others were timed in a training loop that also renormalises the decoder, where the first two conditions took 191 ms and 237 ms; the 50% and 90% rows force the dense auxiliary path, which is what CUDA now uses.

### 6. Trainers

**`topk`** takes one step per batch:

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

Fixed inside SAELens 6.56 and therefore different from `topk`: the auxiliary loss uses `d_in // 2` dead latents, is not divided by the residual variance, and is scaled down when fewer dead latents exist; `batchtopk` keeps the `k * batch` largest activations of the whole batch and is saved as a JumpReLU SAE with one threshold per latent, so the number of active latents varies between tokens. Activation normalisation is switched off because it cannot be folded into the exported weights.

muscari changes one thing: the decoder bias starts at the geometric median of the first batch, as in the `topk` trainer, where SAELens starts it at zero. Encoder hidden states can share a large common component. With a zero bias the top-k selection then locks onto the few latents aligned with that component; a regression test reproduces this on synthetic activations, where one latent fires on every token and the reconstruction is worse than predicting the mean. `--zero-b-dec` restores SAELens' own initialisation.

A short run is not enough to judge this backend. With SAELens' default decoder scale, most latents go unused early in training and its auxiliary loss brings them back over time. This was seen on synthetic activations. The [trial on a real encoder](#trial-on-a-real-encoder) is consistent with it but does not isolate it.

### 7. Export format (`muscari.export`)

`sae.safetensors` contains `encoder.weight (M, d)`, `encoder.bias (M)`, `decoder.weight (d, M)`, `decoder.bias (d)` and, for JumpReLU SAEs, `threshold (M)`.

The subtraction of `b_dec` is folded into the encoder bias:

```text
W_enc (x - b_dec) + b_enc = W_enc x + (b_enc - W_enc b_dec)
```

so a retrieval head needs only `encoder.weight` and `encoder.bias`. SAELens stores `W_enc` as `(d, M)` and `W_dec` as `(M, d)`; the conversion transposes both, folds the bias when `apply_b_dec_to_input` is set, and copies `threshold` if present. SAELens' inference export has already folded the decoder norms into the weights. The tests check that the converted SAE and SAELens' own inference model produce the same features and reconstructions.

`sae_config.json` carries `format_version`, the sizes, the activation rule (`topk` or `jumprelu`) and provenance. Local model and data paths are recorded by file name only, so an SAE can be shared without revealing a directory layout.

### 8. Evaluation

Evaluation runs on the exported files with the same code for both trainers.

- `fvu` is the squared reconstruction error over the variance around the mean activation of the evaluated tokens. `l0` is the mean number of active latents per token.
- `never_fired_frac`, `rare_frac`, `dense_frac`, `median_frequency` and `max_frequency` describe how evenly the dictionary is used. The average latent fires on `l0 / num_latents` of the tokens; `rare_frac` counts the latents below a tenth of that, and `dense_frac` those above 10% of all tokens.
- `by_prefix` repeats `fvu` and `l0` for each prefix, with the group's share of the tokens. Queries are short, so they are a small part of the tokens and the overall figure mostly describes documents.
- `train_unused_frac` is the share of latents that did not fire during the last 10 million training tokens, or the whole run if it was shorter. It comes from the trainer's own bookkeeping and does not depend on the size of the evaluation sample.

The frequency figures taken from an evaluation sample depend on its size: with few tokens, rare latents have had no chance to fire. The default is therefore 32 tokens per latent and at least 200,000, which is about 2 million tokens for 65,536 latents.

A good FVU does not imply good retrieval. Latents that fire on a large share of tokens behave like stop words in an inverted index and make posting lists long, which is why the frequency figures are reported next to FVU.

#### After pooling (`muscari.analysis`)

An index built on an SAE stores one vector per text: the maximum of each latent over the content tokens, usually after `log1p`. Index size follows the non-zero dimensions per document, and query latency follows the non-zero dimensions per query and the length of the posting lists they hit. `evaluate_texts` therefore keeps text boundaries and measures the pooled sparsity pattern. A latent is active in a text when it fires on at least one token; `log1p` and any other increasing map of the values leave this pattern unchanged.

- `l0` per text, with median and 95th percentile, separately for each prefix. With `k` latents per token a text of `n` tokens has between `k` and `min(k * n, num_latents)` active latents; where it falls in between says how much the tokens of a text share latents.
- `max_document_frequency` and `dense_frac` count latents that are active in a large share of texts. Such a latent has a posting list as long as the collection and cannot discriminate, whatever it fires on.
- `flops` is `sum_j p_a(j) * p_b(j)`, where `p_a(j)` is the share of texts with prefix `a` in which latent `j` is active. For a query prefix and a document prefix this is the expected number of posting-list entries touched per query, divided by the number of documents. It is the quantity the FLOPS regulariser of Paria et al. bounds and the figure naver/splade reports at evaluation time.

All of this is measured on the SAE before any retrieval training. Retrieval training changes the encoder, so these figures are a starting point and not a prediction of the final index.

#### Per latent

`latent_statistics`, and `inspect-sae` on the command line, list every latent with the numbers that tell apart three things a latent can follow other than meaning:

- a token: `top_tokens` with `share` and `token_rate`. A latent whose firings fall almost entirely on one token, and which fires on almost every occurrence of it, is a vocabulary entry;
- a kind of text: `frequency_by_prefix`. A latent that fires on queries and hardly on documents carries the prefix, not the content;
- a position: `first_token_share`, `last_token_share` and `mean_position`, to be compared with `1 / tokens_per_text`. A latent whose firings sit on the first or the last token whatever that token is follows the position.

`text_frequency` is the pooled counterpart of `frequency`. A latent that fires on a few tokens of every text has a low `frequency` and a `text_frequency` near 1, and only the second matters to an index.

#### Through the encoder's own head (`muscari.probe`, `muscari.mlm`)

FVU weighs every direction of the hidden state alike. The encoder does not: its head reads some directions and ignores others. Work on SAEs for language models therefore reports the loss recovered when the reconstruction is spliced into the forward pass. Two counterparts are implemented here, one for each kind of head.

**Embedding models.** For an encoder trained to produce mean-pooled sentence embeddings, the head is the pooling, and the task is ranking. `probe_reranking` computes three embeddings of every text: from the encoder's own token states, with the content token states replaced by the SAE's reconstruction, and with them replaced by the decoder bias, which is the SAE's output for an empty code. Special and prefix positions keep their own states in all three, because the SAE is not trained on them and a retriever built on it does not pass them through it. Mean pooling is linear, so the replaced embeddings follow from sums over the content tokens and need no second forward pass. The candidates of each query are ranked by cosine similarity, and `recovered` is `(spliced - empty) / (original - empty)` for nDCG@10 and MRR@10. The empty-code baseline is not a random ranking: the special and prefix positions are contextual and still carry part of the text.

The same pass gives the pooled sparse vectors, `log1p` of the activations followed by the maximum over the content tokens. This is the pooling the Shizuku SAE card describes as the state of its retriever before retrieval training. Candidates are ranked by the dot product, once as it is and once with each latent weighted by `ln(1 + (N - df + 0.5) / (df + 0.5))`, where `df` counts the candidate passages of all sampled queries in which the latent is active. The gap between the two shows how much the ranking suffers from latents that are active in most passages.

Sparse scores tie often, for example at zero when a query and a passage share no latent. The candidates of each query are shuffled with a fixed seed before ranking, so ties are not resolved by the order of the dataset. Every figure comes with its standard error over the sampled queries, since a sample of a few hundred queries cannot resolve small differences.

**Masked language models.** For an encoder that keeps its language-model head, `mlm_loss_recovered` masks content tokens, replaces the final hidden states of the content positions before the head reads them, and measures the cross-entropy at the masked positions. The baseline is the mean of those states over the evaluated texts, as on the Shizuku SAE card. The states are replaced through a forward hook on the base model, so any architecture that Transformers loads with `AutoModelForMaskedLM` works. A checkpoint without a trained head is refused, since a head initialised at random gives a loss that means nothing.

Both need an SAE trained on `last_hidden_state`.


### 9. Retrieval head (`muscari.retrieval`)

SPLADE scores every vocabulary entry at every token with the language-model head, applies `log(1 + ReLU(.))` and pools with a maximum over the tokens. An SAE-based sparse retriever keeps this shape and swaps the vocabulary for the SAE's dictionary. `SAEPooling` is that swap as a Sentence Transformers module: it reads `token_embeddings` from the `Transformer` module, applies the SAE's encoder to the content tokens, keeps the selected latents, applies ReLU and `log1p`, and writes the maximum over the tokens of each text to `sentence_embedding`. Everything downstream, from `SpladeLoss` and the FLOPS regulariser to the trainer and the evaluators, works on that tensor and needs no change.

- **Same tokens as in SAE training.** Padding and special tokens are left out, and so are the prompt's tokens when the model reports a prompt length. Sentence Transformers measures the prompt by tokenizing it on its own, while SAE training drops the tokens whose characters end inside the prefix. The two agree unless a token straddles the end of the prompt, where they can differ by one token.
- **Frozen by default.** The SAE's encoder is stored as parameters that do not require gradients, so retrieval training changes only the transformer. This is the setting the Shizuku SAE card describes. `trainable=True` trains the SAE's encoder as well.
- **Sparse backward pass.** Only `k` latents per token are selected, so the gradient with respect to the token states is a sum of `k` rows of the encoder matrix. The head reuses the function the trainer uses, which keeps the dense `(tokens, latents)` product inside the forward pass. It runs in float32 with autocast switched off, in chunks of tokens.
- **Both input layouts.** Sentence Transformers either pads a batch or, with flash attention, packs it into one sequence without padding. The head handles both and returns the same vectors.
- **Masked latents.** A latent that is active in nearly every text adds the same amount to every score. `masked_latents` zeroes chosen latents in the output. This does not free their slot among the `k` latents of a token; it only removes them from the index.
- **No decoder.** Retrieval does not reconstruct, so the saved head holds only the encoder half.

The head's vectors are the ones `probe-sae` ranks with, which the tests check on the same texts. What the tests cannot show is how well a model trained with the head retrieves.

## Differences from Gao et al.

The paper was read in its arXiv v1 HTML form (Sections 2.1 to 2.4, Appendices A and B), and the [reference implementation](https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942) in `sparse_autoencoder/train.py`.

| Aspect | Gao et al. | `topk` trainer |
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

### Running with the settings of the Shizuku SAE card

The [model card](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) of the SAE used by Shizuku states the following, and the defaults of the `topk` trainer match all of it:

- the final-layer `last_hidden_state` of a frozen `sbintuitions/modernbert-ja-310m`;
- 65,536 latents, Top-K with `k = 16`, AuxK with 256 latents and coefficient 1/32;
- Adam with betas 0.9 and 0.999 at 5e-4, warmup over the first 5% and cosine decay;
- batches of 8,192 tokens, a maximum length of 1,024 and one billion training tokens;
- deduplicated queries and documents, with 5% held out;
- the tensor names, shapes and the folding of the decoder bias in the saved file.

```bash
muscari train-sae \
  --trainer topk \
  --model sbintuitions/modernbert-ja-310m \
  --data-files corpus.jsonl \
  --columns query document \
  --out runs/modernbert-ja-310m-sae
```

Three things the card describes are outside this command:

- **The data.** The card lists about 846 million tokens (70.1% documents, 14.9% queries, 14.9% Wikipedia articles) from mqa-ja, auto-wiki-qa, Japanese Wikipedia, mmarco-ja and smaller sets, some of them not public, with items that overlap certain evaluation sets removed. muscari ships no recipe for this corpus.
- **The model revision.** The card names revision `77675fc`; pass the full commit hash with `--revision`.
- **The evaluation.** The card reports an FVU of 0.155 on 2 million held-out document tokens, an MLM loss recovery of 0.944 and no feature unused over the last 10 million training tokens. muscari reports the first and the last of these in its own form and has no MLM loss recovery.

Everything the card does not state (initialisation, the dead-latent threshold, shuffling, the loss normalisation, input normalisation, the training library) is muscari's own choice, mostly following Gao et al. The result is therefore a run with a published recipe, not a reproduction of that SAE. The saved tensors have the same layout as the card's file; loading one as the other has not been tried.

## Trial on a real encoder

Three short runs on an Apple M5 Pro with MPS, to check that the pipeline works with a real model, tokenizer and dataset. It is not a statement about SAE quality: the budget is 0.5% of the recipe's one billion tokens, there is one seed, and no reference value exists to compare with.

| Setting | Value |
| --- | --- |
| Encoder | `cl-nagoya/ruri-v3-30m`, `last_hidden_state`, width 256 |
| Data | subset `jsquad` of `cl-nagoya/ruri-v3-dataset-ft`; columns `anc` with `検索クエリ: `, `pos` and `neg` with `検索文書: ` |
| SAE | 8,192 latents, `k = 16` |
| Budget | 5 million tokens (610 steps of 8,192), dead after 1 million tokens, 5% of the texts held out |
| Seed | 0, the default at the time |

| | `topk` | `saelens` `topk`, zero decoder bias | `saelens` `topk`, median decoder bias |
| --- | ---: | ---: | ---: |
| FVU on 200,000 held-out tokens | 0.441 | 0.511 | 0.467 |
| Active latents per token | 16 | 16 | 16 |
| Latents that never fired | 0% | 91.5% | 54.1% |
| Latents firing on more than 10% of tokens | 0% | 0.4% | 0% |
| Largest firing share of one latent | 6.3% | 23.7% | 7.4% |
| Wall time, with evaluation | 200 s | 189 s | 193 s |

The first run includes the download of the model and the dataset subset. Evaluated on 200,000 tokens of the `jaquad` subset, which was not used for training, the `topk` SAE had an FVU of 0.447, no latent that never fired, and one latent above 10% (10.1%). Each run encoded 31,616 distinct texts, 5.52 million content tokens; the `topk` run did so at about 42,000 tokens per second including the encoder.

What the runs show beyond "it works":

- **The prefix rule does what it should with this tokenizer.** It is a fast tokenizer. `検索クエリ: ` becomes the four tokens `検索`, `クエリ`, `:` and `▁`; all of them end inside the prefix and are dropped, and every token of the text is kept. This was checked by hand on three texts, with and without a prefix.
- **These hidden states have a large common component.** Over 100,000 activations, the mean activation has a norm of 35.0, as large as the average distance of an activation from it (35.3); it carries half of the squared norm.
- **The decoder bias matters for the SAELens backend.** Starting it at the geometric median instead of zero lowered the latents that never fired from 91.5% to 54.1% and the FVU from 0.511 to 0.467.

What they do not show:

- The log of the `topk` run shows no dead latent and a zero auxiliary loss at every logged step, so AuxK was not exercised on real activations.
- Why half of the SAELens latents still never fired. The short budget is a candidate, as described for the SAELens backend above, but a longer run was not made.
- A like-for-like comparison of the trainers. The columns differ in auxiliary loss, dead-latent bookkeeping, decoder scaling and initialisation.

### Training length

The same setting on the nine retrieval subsets of the dataset (`auto-wiki-qa-nemotron`, `jaquad`, `jqara`, `jsquad`, `miracl`, `mkqa`, `mr-tydi`, `quiz-no-mori`, `quiz-works`), seed 42, one independent run per length and trainer, evaluated on 200,000 held-out tokens. The SAELens runs start the decoder bias at the geometric median.

| Training tokens | `topk` FVU | `saelens` FVU | `saelens` latents that never fired | Largest firing share, `topk` / `saelens` |
| ---: | ---: | ---: | ---: | ---: |
| 5 million | 0.444 | 0.474 | 54.1% | 5.3% / 6.4% |
| 10 million | 0.396 | 0.420 | 17.1% | 6.6% / 5.2% |
| 20 million | 0.360 | 0.365 | 0.9% | 6.6% / 13.3% |
| 30 million | 0.341 | 0.342 | 0.8% | 14.2% / 14.1% |

- The unused latents of the SAELens backend were a matter of training length. With enough tokens the two trainers reach the same FVU; in short runs the `topk` trainer is ahead, and every one of its latents fired at every length.
- FVU was still falling at 30 million tokens, by less each time the length doubled.
- No latent of the `topk` runs was ever flagged dead, so AuxK was not exercised at this size (width 256, 8,192 latents).
- In the longer runs of both trainers one latent fires on about 14% of the tokens. What it responds to has not been examined.
- Throughput halved from the fourth run on, from about 43,000 to about 23,000 tokens per second, across separate processes. The cause was not isolated; these runs did not record encoder and SAE time separately.

### CUDA

The 5-million-token `jsquad` run above, repeated on an NVIDIA L4 with seed 0, gave an FVU of 0.4406 in float32 and 0.4405 with bfloat16 autocast for the encoder and the SAE, against 0.441 on MPS. Throughput was about 55,000 and 76,000 tokens per second.

A short run at the full recipe width (`cl-nagoya/ruri-v3-310m`, 65,536 latents, 3 million tokens, bfloat16) ran at about 15,000 tokens per second on the L4 including the encoder. The SAE updates alone take 133 ms per 8,192 tokens there, about 62,000 tokens per second, so the encoder is the larger part. At that rate one billion tokens would take about 19 hours on an L4. This is an extrapolation from a 3-million-token run, not a measurement.

## What is verified

- Losses and gradients of the sparse backward pass: against dense autograd and numerical differentiation.
- Checkpoint resumption: bit-identical continuation on identical batches.
- Export: identical outputs to the trained module, and to SAELens' own inference model for `topk` and `batchtopk`.
- Content-token selection: against a manual forward pass, with fast and offset-less tokenizers.
- Holdout split: disjoint, deterministic, independent of prefix and row.
- Apple MPS and CUDA: losses, gradients and activations agree with the CPU, and the runs above completed. On CUDA, bfloat16 autocast gave the same result as float32 to three decimals.
- Prefix exclusion with the ruri-v3 tokenizer: by hand, on three texts.
- Text-level statistics, latent statistics and ranking metrics: against hand-built cases.
- The retrieval head: its vectors equal the probe's on the same texts; gradients equal a dense computation, for the token states and, when trainable, for the SAE's encoder; padded and unpadded batches agree; a saved model loads with the same head; three training steps on a tiny model update the transformer and leave the SAE unchanged.
- The reranking probe and the loss recovered: an SAE that reconstructs exactly recovers 1; pooled vectors and replaced embeddings agree with a manual computation; the clean masked-language-model loss equals the model's own loss on the same masked batch.

Except for the runs above, all of this uses a randomly initialised two-layer BERT or synthetic activations. See the [validation status](../README.md#validation-status) for the environments that ran.

## What is not verified

- SAE quality on a real encoder: reconstruction, dead and dense latents at 65,536 latents and one billion tokens.
- That an SAE trained here improves retrieval when used as a sparse vocabulary. No retrieval model has been trained with the head.
- The unpadded input layout of Sentence Transformers end to end. It needs flash attention; the head is tested on hand-built features in that layout.
- AuxK on real activations.
- A run of the full length on any device. The time estimates above are extrapolations.
- The pooled, per-text figures, the per-latent listing, the reranking probe and the loss recovered on a real SAE. The code is tested against hand-built cases; no real dictionary has been measured with it yet.

## Research background

- Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093), 2024. Top-K SAEs, AuxK, dead latents, initialisation.
- Bussmann et al., [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410), 2024. The `batchtopk` architecture of the SAELens backend.
- Formal et al., [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720), 2021. The retrieval model family whose vocabulary an SAE replaces.
- [From Tokens to Concepts: Leveraging SAE for SPLADE](https://arxiv.org/abs/2604.21511) (SAE-SPLADE). Replaces the masked-language-model head of SPLADE with an SAE on an encoder.
- [Learning Retrieval Models with Sparse Autoencoders](https://arxiv.org/abs/2603.13277) (SPLARE). Sparse retrieval with a frozen SAE on an LLM backbone.
- mahiyama, [modernbert-ja-310m-sae](https://huggingface.co/mahiyama/modernbert-ja-310m-sae) and [sae-splade-ja-310m](https://huggingface.co/mahiyama/sae-splade-ja-310m) (Shizuku). The recipe followed by the `topk` trainer.
- [SAELens](https://pypi.org/project/sae-lens/). The second training backend.
