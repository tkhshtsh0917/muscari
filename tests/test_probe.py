"""Reranking probe: metrics, grouping, pooling and the dense and sparse rankings."""

from __future__ import annotations

import math

import pytest
import torch
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, perfect_sae

from muscari import load_sae
from muscari.export import save_sae
from muscari.probe import (
    PooledText,
    RerankingExample,
    group_reranking_rows,
    pool_texts,
    probe_reranking,
    ranking_metrics,
    sparse_scores,
)

TEXTS = ["東京 求人 検索", "名古屋 営業", "在宅 勤務 経験 資格 正社員", "給与"]


def test_ranking_metrics_against_a_hand_computed_case():
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0])
    labels = torch.tensor([0, 1, 0, 1])

    metrics = ranking_metrics(scores, labels, k=3)

    # The top three are irrelevant, relevant, irrelevant; the ideal top three hold both relevant ones.
    assert metrics["ndcg@3"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert metrics["mrr@3"] == 0.5
    assert ranking_metrics(scores, labels, k=1) == {"ndcg@1": 0.0, "mrr@1": 0.0}
    assert ranking_metrics(-scores, labels, k=10)["ndcg@10"] == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def test_ties_keep_the_input_order_and_a_query_needs_a_relevant_candidate():
    assert ranking_metrics(torch.zeros(3), torch.tensor([0, 0, 1]), k=2)["mrr@2"] == 0.0
    assert ranking_metrics(torch.zeros(3), torch.tensor([1, 0, 0]), k=2)["mrr@2"] == 1.0

    with pytest.raises(ValueError):
        ranking_metrics(torch.zeros(3), torch.tensor([0, 0, 0]))


def test_rows_are_grouped_per_query():
    rows = [
        {"id": "a", "q": "東京", "p": "p1", "y": 0},
        {"id": "b", "q": "名古屋", "p": "p2", "y": 2},
        {"id": "a", "q": "東京", "p": "p3", "y": 1},
        {"id": "c", "q": "大阪", "p": "p4", "y": 0},  # no relevant candidate: dropped
        {"id": "b", "q": "名古屋", "p": "p5", "y": 0},
    ]
    columns = {"query_column": "q", "passage_column": "p", "label_column": "y"}

    examples = group_reranking_rows(rows, **columns, group_column="id")
    assert examples == [
        RerankingExample("東京", ("p1", "p3"), (0, 1), "a"),
        RerankingExample("名古屋", ("p2", "p5"), (1, 0), "b"),
    ]
    by_text = group_reranking_rows(rows, **columns)  # the query text identifies the query
    assert [(e.query, e.passages, e.labels, e.query_id) for e in by_text] == [
        ("東京", ("p1", "p3"), (0, 1), "東京"),
        ("名古屋", ("p2", "p5"), (1, 0), "名古屋"),
    ]
    assert group_reranking_rows(rows, **columns, group_column="id", max_queries=1) == examples[:1]

    titled = [dict(row, t=f"T{row['id']}") for row in rows]
    with_titles = group_reranking_rows(titled, **columns, group_column="id", title_column="t")
    assert with_titles[0].passages == ("Ta p1", "Ta p3")  # the JQaRA evaluator's "{title} {text}"

    with pytest.raises(ValueError, match="relevant candidate"):
        group_reranking_rows(rows[3:4], **columns)
    with pytest.raises(ValueError, match="no column"):
        group_reranking_rows(rows, **{**columns, "label_column": "relevance"})


def test_pooled_vectors_match_a_manual_computation(source, tmp_path):
    sae = perfect_sae(tmp_path)
    pooled = pool_texts(source, sae, DOC_PREFIX, TEXTS, token_chunk=3)

    for text, vector in zip(TEXTS, pooled, strict=True):
        batch = source.encode_texts([DOC_PREFIX], [text])
        states = batch.activations
        expected = torch.log1p(torch.cat([states.relu(), (-states).relu()], dim=1)).amax(dim=0)
        dense = torch.zeros(2 * HIDDEN_SIZE)
        dense[vector.indices] = vector.values

        assert torch.allclose(dense, expected, atol=1e-6)
        assert (vector.values > 0).all() and (vector.indices.diff() > 0).all()

        original = batch.hidden_sums[0] / batch.attended[0]
        without_content = (batch.hidden_sums[0] - states.sum(dim=0)) / batch.attended[0]
        assert torch.allclose(vector.embeddings[0], original, atol=1e-6)
        assert torch.allclose(vector.embeddings[1], original, atol=1e-5)  # the SAE reconstructs exactly
        assert torch.allclose(vector.embeddings[2], without_content, atol=1e-5)  # its decoder bias is zero


def test_empty_code_embedding_uses_the_decoder_bias(source, tmp_path):
    bias = torch.arange(HIDDEN_SIZE, dtype=torch.float32)
    silent = load_sae(
        save_sae(
            tmp_path,
            encoder_weight=torch.zeros(8, HIDDEN_SIZE),
            encoder_bias=torch.zeros(8),
            decoder_weight=torch.zeros(HIDDEN_SIZE, 8),
            decoder_bias=bias,
            config={"k": 2},
        )
    )
    batch = source.encode_texts([QUERY_PREFIX], [TEXTS[0]])
    vector = pool_texts(source, silent, QUERY_PREFIX, [TEXTS[0]])[0]

    replaced = (batch.hidden_sums[0] - batch.activations.sum(dim=0) + batch.tokens_per_text[0] * bias) / batch.attended[
        0
    ]
    assert vector.indices.numel() == 0
    assert torch.allclose(vector.embeddings[1], replaced, atol=1e-5)  # nothing fires: only the bias comes back
    assert torch.allclose(vector.embeddings[2], replaced, atol=1e-5)


def test_sparse_scores_are_weighted_dot_products():
    def vector(indices: list[int], values: list[float]) -> PooledText:
        return PooledText(torch.tensor(indices, dtype=torch.long), torch.tensor(values), torch.zeros(3, 2))

    query = vector([0, 2], [1.0, 2.0])
    passages = [vector([0, 1], [3.0, 5.0]), vector([2, 3], [4.0, 7.0]), vector([], []), vector([0, 2], [1.0, 1.0])]

    assert sparse_scores(query, passages, torch.ones(4, dtype=torch.float64)).tolist() == [3.0, 8.0, 0.0, 3.0]
    assert sparse_scores(query, passages, torch.tensor([2.0, 1.0, 0.5, 1.0], dtype=torch.float64)).tolist() == [
        6.0,
        4.0,
        0.0,
        3.0,
    ]


def test_a_perfect_sae_recovers_the_dense_ranking(source, tmp_path):
    examples = [
        RerankingExample("東京 求人", ("東京 求人 検索", "名古屋 営業", "経理 給与 残業", "在宅 勤務"), (1, 0, 0, 0)),
        RerankingExample("営業 経験", ("派遣 休日", "営業 経験 資格", "東京 開発"), (0, 1, 0)),
    ]

    result = probe_reranking(source, perfect_sae(tmp_path), examples, query_kind=QUERY_PREFIX, passage_kind=DOC_PREFIX)

    assert (result["queries"], result["passages"]) == (2, 7)
    assert result["candidates_per_query"] == 3.5 and result["relevant_per_query"] == 1.0
    assert result["dense"]["spliced"] == pytest.approx(result["dense"]["original"])
    assert result["dense"]["recovered"]["ndcg@10"] == pytest.approx(1.0)
    assert result["dense"]["cosine"]["passage"]["min"] == pytest.approx(1.0, abs=1e-5)
    assert set(result["sparse"]["plain"]) == {"ndcg@10", "mrr@10", "ndcg@10_stderr", "mrr@10_stderr"}
    assert set(result["dense"]["recovered"]) == {"ndcg@10", "mrr@10"}
    assert 0 < result["sparse"]["l0"]["query"] <= result["sparse"]["l0"]["passage"] <= 2 * HIDDEN_SIZE
    assert all(0 <= result["sparse"][name]["ndcg@10"] <= 1 for name in ("plain", "idf"))


def test_ties_are_not_broken_by_the_order_of_the_dataset(source, tmp_path):
    silent = load_sae(
        save_sae(
            tmp_path,
            encoder_weight=torch.zeros(8, HIDDEN_SIZE),
            encoder_bias=torch.zeros(8),
            decoder_weight=torch.zeros(HIDDEN_SIZE, 8),
            decoder_bias=torch.zeros(HIDDEN_SIZE),
            config={"k": 2},
        )
    )
    passages = tuple(
        f"東京 求人 {word}" for word in "名古屋 検索 営業 開発 経理 給与 勤務 経験 資格 正社員 派遣 在宅".split()
    )
    relevant_first = [RerankingExample(f"休日 {n}", passages, (1,) + (0,) * 11) for n in range(40)]

    result = probe_reranking(source, silent, relevant_first)

    # Every sparse score is zero. Ranking in dataset order would put the relevant passage first.
    assert 0.1 < result["sparse"]["plain"]["mrr@10"] < 0.5
    assert 0.01 < result["sparse"]["plain"]["mrr@10_stderr"] < 0.1
    assert result["sparse"]["l0"] == {"query": 0.0, "passage": 0.0}


def test_an_empty_list_of_examples_is_rejected(source, tmp_path):
    with pytest.raises(ValueError):
        probe_reranking(source, perfect_sae(tmp_path), [])
