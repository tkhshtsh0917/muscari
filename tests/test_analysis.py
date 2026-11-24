"""Pooled per-text statistics and per-latent token statistics, against hand-built cases."""

from __future__ import annotations

import pytest
import torch

from muscari import EncodedText, evaluate_texts, latent_statistics, load_sae
from muscari.export import save_sae

E0, E1, E2, E3 = torch.eye(4)


@pytest.fixture
def sae(tmp_path):
    """Latents 0-3 carry ReLU(x_i) and latents 4-7 carry ReLU(-x_i), so each token below fires one."""
    identity = torch.eye(4)
    out = save_sae(
        tmp_path,
        encoder_weight=torch.cat([identity, -identity]),
        encoder_bias=torch.zeros(8),
        decoder_weight=torch.cat([identity, -identity], dim=1),
        decoder_bias=torch.zeros(4),
        config={"k": 4},
    )
    return load_sae(out)


def text(prefix: str, *tokens: tuple[torch.Tensor, int]) -> EncodedText:
    return EncodedText(
        prefix,
        torch.stack([vector for vector, _ in tokens]),
        torch.tensor([token_id for _, token_id in tokens]),
    )


TEXTS = [
    text("q", (E0, 10), (E1, 11)),  # latents 0 and 1
    text("q", (E0, 10), (E0, 10)),  # latent 0 only
    text("d", (E0, 10), (E1, 11), (E2, 12), (-E3, 13)),  # latents 0, 1, 2 and 7
    text("q", (E1, 10)),  # latent 1, on the token that usually fires latent 0
]


@pytest.mark.parametrize("batch_tokens", [1, 3, 1000])
def test_pooled_statistics_per_prefix(sae, batch_tokens):
    result = evaluate_texts(sae, TEXTS, batch_tokens=batch_tokens)
    queries, documents = result["by_prefix"]["q"], result["by_prefix"]["d"]

    assert (queries["texts"], documents["texts"]) == (3, 1)
    assert queries["tokens_per_text"] == pytest.approx(5 / 3)
    assert queries["l0"] == pytest.approx(4 / 3)  # 2, 1 and 1 active latents
    assert queries["l0_median"] == 1.0
    assert documents["l0"] == 4.0
    # Latent 0 is in two of the three queries, latent 1 in two as well; six latents never appear.
    assert queries["max_document_frequency"] == pytest.approx(2 / 3)
    assert queries["never_fired_frac"] == pytest.approx(6 / 8)
    assert queries["dense_frac"] == pytest.approx(2 / 8)
    assert documents["max_document_frequency"] == 1.0


def test_flops_is_the_expected_overlap_of_two_kinds_of_text(sae):
    flops = {tuple(entry["prefixes"]): entry["flops"] for entry in evaluate_texts(sae, TEXTS)["flops"]}

    # Query probabilities: latent 0 and latent 1 at 2/3 each. Document: latents 0, 1, 2, 7 at 1.
    assert flops == {
        ("q", "q"): pytest.approx(8 / 9),
        ("q", "d"): pytest.approx(4 / 3),
        ("d", "d"): pytest.approx(4.0),
    }


def test_a_latent_active_on_several_tokens_counts_once_per_text(sae):
    repeated = [text("d", (E0, 10), (2 * E0, 10), (3 * E0, 10))]

    assert evaluate_texts(sae, repeated)["by_prefix"]["d"]["l0"] == 1.0


def test_latent_statistics_rank_tokens_by_how_often_they_fire_a_latent(sae):
    records = latent_statistics(sae, TEXTS, top_n=5)
    by_latent = {record["latent"]: record for record in records}

    assert [record["latent"] for record in records[:4]] == [0, 1, 2, 7]  # most frequent first
    assert len(records) == 8
    assert by_latent[0]["frequency"] == pytest.approx(4 / 9)
    assert by_latent[0]["mean_activation"] == pytest.approx(1.0)
    # Latent 0 fires only on token 10, which occurs five times and fires it four times.
    assert by_latent[0]["top_tokens"] == [{"token_id": 10, "share": 1.0, "token_rate": pytest.approx(4 / 5)}]
    assert by_latent[1]["top_tokens"] == [
        {"token_id": 11, "share": pytest.approx(2 / 3), "token_rate": 1.0},
        {"token_id": 10, "share": pytest.approx(1 / 3), "token_rate": pytest.approx(1 / 5)},
    ]
    assert by_latent[7]["top_tokens"] == [{"token_id": 13, "share": 1.0, "token_rate": 1.0}]


@pytest.mark.parametrize("batch_tokens", [1, 3, 1000])
def test_latent_statistics_separate_kind_of_text_and_position(sae, batch_tokens):
    by_latent = {record["latent"]: record for record in latent_statistics(sae, TEXTS, batch_tokens=batch_tokens)}

    # Latent 0: first token of texts 1, 2 and 3, and the second, last token of text 2.
    assert by_latent[0]["frequency_by_prefix"] == {"q": pytest.approx(3 / 5), "d": pytest.approx(1 / 4)}
    assert by_latent[0]["text_frequency"] == pytest.approx(3 / 4)
    assert by_latent[0]["first_token_share"] == pytest.approx(3 / 4)
    assert by_latent[0]["last_token_share"] == pytest.approx(1 / 4)
    assert by_latent[0]["mean_position"] == pytest.approx(1 / 4)
    # Latent 1: last of text 1, second of four in text 3, and the only token of text 4.
    assert by_latent[1]["first_token_share"] == pytest.approx(1 / 3)
    assert by_latent[1]["last_token_share"] == pytest.approx(2 / 3)
    assert by_latent[1]["mean_position"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert by_latent[7]["frequency_by_prefix"] == {"q": 0.0, "d": pytest.approx(1 / 4)}
    assert (by_latent[7]["last_token_share"], by_latent[7]["mean_position"]) == (1.0, 1.0)
    assert by_latent[4] == {
        "latent": 4,
        "frequency": 0.0,
        "frequency_by_prefix": {"q": 0.0, "d": 0.0},
        "text_frequency": 0.0,
        "mean_activation": 0.0,
        "first_token_share": 0.0,
        "last_token_share": 0.0,
        "mean_position": 0.5,
        "top_tokens": [],
    }


def test_top_n_limits_the_tokens_per_latent(sae):
    by_latent = {record["latent"]: record for record in latent_statistics(sae, TEXTS, top_n=1, batch_tokens=2)}

    assert [entry["token_id"] for entry in by_latent[1]["top_tokens"]] == [11]


def test_empty_input_and_invalid_top_n_are_rejected(sae):
    with pytest.raises(ValueError):
        evaluate_texts(sae, [])
    with pytest.raises(ValueError):
        latent_statistics(sae, [])
    with pytest.raises(ValueError):
        latent_statistics(sae, TEXTS, top_n=0)
