"""Activation extraction: which positions are kept, and how batches are mixed."""

from __future__ import annotations

import pytest
import torch
from tiny import COLUMNS, DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows

from muscari import EncoderActivationSource, iter_texts, shuffled_batches


class SlowTokenizer:
    """Wraps the fast tokenizer but, like a slow tokenizer, cannot report character offsets."""

    is_fast = False

    def __init__(self, fast):
        self.fast = fast

    def __call__(self, text, **kwargs):
        kwargs.pop("return_offsets_mapping", None)
        return self.fast(text, **kwargs)


def reference_hidden(source: EncoderActivationSource, text: str) -> torch.Tensor:
    """Hidden states of every position of one unpadded text, special tokens included."""
    encoding = source.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        return source.model(**encoding).last_hidden_state[0]


def test_only_content_tokens_are_kept_in_order(source):
    texts = ["東京 求人 検索", "名古屋 営業"]
    activations = source.encode_batch([DOC_PREFIX, QUERY_PREFIX], texts)

    # Positions: [CLS], prefix, words..., [SEP]. Only the words remain.
    first = reference_hidden(source, DOC_PREFIX + texts[0])[2:5]
    second = reference_hidden(source, QUERY_PREFIX + texts[1])[2:4]

    assert activations.shape == (5, HIDDEN_SIZE)
    assert activations.dtype == torch.float32
    assert torch.allclose(activations, torch.cat([first, second]), atol=1e-5)
    assert (source.texts_seen, source.tokens_seen) == (2, 5)


def test_prefix_tokens_can_be_kept(source):
    source.exclude_prefix = False
    activations = source.encode_batch([DOC_PREFIX, QUERY_PREFIX], ["東京 求人 検索", "名古屋 営業"])

    assert activations.shape[0] == 5 + 2  # each prefix is one token


def test_texts_without_prefix_keep_every_content_token(source):
    assert source.encode_batch(["", ""], ["東京 求人 検索", "名古屋"]).shape[0] == 4


def test_slow_tokenizer_fallback_matches_offset_masking(source):
    prefixes = [DOC_PREFIX, QUERY_PREFIX, ""]
    texts = ["東京 求人 検索", "名古屋 営業", "開発 経理"]
    expected = source.encode_batch(prefixes, texts)

    source.tokenizer = SlowTokenizer(source.tokenizer)
    fallback = source.encode_batch(prefixes, texts)

    assert torch.equal(fallback, expected)


def test_truncation_limits_the_content_tokens(source):
    long_text = " ".join(["東京"] * 100)
    activations = source.encode_batch([DOC_PREFIX], [long_text])

    assert activations.shape[0] == source.max_length - 3  # [CLS], prefix token and [SEP]


def test_empty_texts_yield_no_rows_and_are_skipped_by_the_stream(source):
    assert source.encode_batch([DOC_PREFIX], [""]).shape == (0, HIDDEN_SIZE)
    assert list(source.stream([(DOC_PREFIX, "")])) == []


def test_layer_index_selects_a_hidden_state(tiny_model_dir):
    last = EncoderActivationSource.from_pretrained(str(tiny_model_dir), device="cpu")
    final_layer = EncoderActivationSource.from_pretrained(str(tiny_model_dir), layer=-1, device="cpu")
    embeddings = EncoderActivationSource.from_pretrained(str(tiny_model_dir), layer=0, device="cpu")
    batch = ([DOC_PREFIX], ["東京 求人 検索"])

    assert torch.allclose(final_layer.encode_batch(*batch), last.encode_batch(*batch), atol=1e-6)
    assert not torch.allclose(embeddings.encode_batch(*batch), last.encode_batch(*batch), atol=1e-3)


def test_mismatched_prefixes_and_texts_are_rejected(source):
    with pytest.raises(ValueError):
        source.encode_batch([DOC_PREFIX], ["a", "b"])


def test_stream_covers_every_text_across_pools(source):
    texts = list(iter_texts(make_rows(60), COLUMNS))
    expected = sum(len(text.split()) for _, text in texts)

    chunks = list(source.stream(texts))

    assert len(texts) > source.text_batch_size * 16  # more than one length-sorted pool
    assert sum(chunk.shape[0] for chunk in chunks) == expected
    assert source.texts_seen == len(texts)


def test_collect_caps_the_token_count_and_rejects_an_empty_source(source):
    texts = list(iter_texts(make_rows(20), COLUMNS))

    assert source.collect(texts, 100).shape == (100, HIDDEN_SIZE)
    assert source.collect(texts, 100).device.type == "cpu"
    with pytest.raises(ValueError, match="No activations"):
        source.collect([], 10)


def test_encode_texts_reports_tokens_per_text_and_counts_by_prefix(source):
    prefixes = [DOC_PREFIX, QUERY_PREFIX, DOC_PREFIX, DOC_PREFIX]
    texts = ["東京 求人 検索", "名古屋 営業", "", "開発"]

    activations, tokens_per_text = source.encode_texts(prefixes, texts)

    assert tokens_per_text.tolist() == [3, 2, 0, 1]
    assert activations.shape[0] == 6
    assert dict(source.tokens_by_prefix) == {DOC_PREFIX: 4, QUERY_PREFIX: 2}
    assert source.encode_seconds > 0

    source.reset_counters()
    assert (source.texts_seen, source.tokens_seen, dict(source.tokens_by_prefix), source.encode_seconds) == (
        0,
        0,
        {},
        0.0,
    )


def test_collect_by_prefix_groups_activations_and_respects_the_cap(source):
    texts = list(iter_texts(make_rows(20), COLUMNS))
    queries = sum(len(text.split()) for prefix, text in texts if prefix == QUERY_PREFIX)
    documents = sum(len(text.split()) for prefix, text in texts if prefix == DOC_PREFIX)

    everything = source.collect_by_prefix(texts, 10**9)
    capped = source.collect_by_prefix(texts, 150)

    assert {prefix: group.shape[0] for prefix, group in everything.items()} == {
        QUERY_PREFIX: queries,
        DOC_PREFIX: documents,
    }
    assert sum(group.shape[0] for group in capped.values()) == 150
    assert all(group.shape[1] == HIDDEN_SIZE and group.device.type == "cpu" for group in capped.values())


def test_grouped_activations_belong_to_their_prefix(source):
    texts = [(QUERY_PREFIX, "東京 求人"), (DOC_PREFIX, "名古屋 営業 開発"), (QUERY_PREFIX, "経理")]

    groups = source.collect_by_prefix(texts, 100)
    expected_queries = torch.cat([source.encode_batch([QUERY_PREFIX], [text]) for text in ("経理", "東京 求人")])

    assert torch.allclose(groups[QUERY_PREFIX], expected_queries, atol=1e-5)  # the pool is sorted by length
    assert torch.allclose(groups[DOC_PREFIX], source.encode_batch([DOC_PREFIX], ["名古屋 営業 開発"]), atol=1e-5)


def numbered_stream(n_chunks: int, rows: int) -> list[torch.Tensor]:
    """Chunks whose single feature is a unique row id, so emitted rows can be traced."""
    return [torch.arange(i * rows, (i + 1) * rows).float().unsqueeze(1) for i in range(n_chunks)]


def test_shuffled_batches_emit_every_row_once_in_full_batches():
    batches = list(shuffled_batches(numbered_stream(40, 10), batch_tokens=16, buffer_tokens=64, seed=42))
    values = torch.cat(batches).flatten()

    assert all(batch.shape == (16, 1) for batch in batches)
    assert values.numel() == 400
    assert values.unique().numel() == 400


def test_shuffled_batches_drop_only_the_final_partial_batch():
    batches = list(shuffled_batches(numbered_stream(7, 10), batch_tokens=16, buffer_tokens=32, seed=42))
    values = torch.cat(batches).flatten()

    assert values.numel() == 64  # 70 rows, four full batches
    assert values.unique().numel() == 64


def test_shuffled_batches_mix_rows_from_different_chunks():
    first = next(iter(shuffled_batches(numbered_stream(40, 10), batch_tokens=16, buffer_tokens=64, seed=42)))
    source_chunks = (first.flatten() // 10).unique()

    assert source_chunks.numel() > 2


def test_shuffled_batches_are_reproducible_per_seed():
    def run(seed: int) -> torch.Tensor:
        return torch.cat(list(shuffled_batches(numbered_stream(20, 10), batch_tokens=8, buffer_tokens=32, seed=seed)))

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


def test_emitted_batches_do_not_alias_the_buffer():
    iterator = shuffled_batches(numbered_stream(40, 10), batch_tokens=16, buffer_tokens=64, seed=42)
    first = next(iterator)
    snapshot = first.clone()
    list(iterator)  # refills and compacts the buffer many times

    assert torch.equal(first, snapshot)


def test_a_chunk_larger_than_the_buffer_is_consumed_completely():
    stream = [torch.arange(1000).float().unsqueeze(1)]
    values = torch.cat(list(shuffled_batches(stream, batch_tokens=10, buffer_tokens=40, seed=42))).flatten()

    assert values.numel() == 1000
    assert values.unique().numel() == 1000


@pytest.mark.parametrize(("batch_tokens", "buffer_tokens"), [(16, 31), (0, 64)])
def test_invalid_buffer_settings_are_rejected(batch_tokens, buffer_tokens):
    with pytest.raises(ValueError):
        list(shuffled_batches([], batch_tokens=batch_tokens, buffer_tokens=buffer_tokens))
