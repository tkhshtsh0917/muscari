"""The evaluation example must score exactly what a direct computation scores, capped or not."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows, write_jsonl

from muscari import TopKSAE, save_topk_sae
from muscari.probe import ranking_metrics

pytest.importorskip("sentence_transformers")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def sparse_eval():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("sparse_eval")


@pytest.fixture
def reranking_file(tmp_path):
    rows = [
        {"q_id": f"q{index}", "query": row["anc"], "title": f"題{index}", "text": text, "label": label}
        for index, row in enumerate(make_rows(8, seed=7))
        for text, label in ((row["neg"][0], 0), (row["pos"], 1), (row["neg"][1], 0), (make_rows(1)[0]["pos"], 0))
    ]
    return write_jsonl(tmp_path / "rerank.jsonl", rows)


@pytest.fixture
def retrieval_dir(tmp_path):
    rows = make_rows(12, seed=3)
    directory = tmp_path / "corpus-set"
    directory.mkdir()
    corpus = [{"id": f"d{index}", "title": "", "text": row["pos"]} for index, row in enumerate(rows)]
    corpus += [{"id": f"n{index}", "title": "題", "text": row["neg"][0]} for index, row in enumerate(rows)]
    write_jsonl(directory / "corpus.jsonl", corpus)
    write_jsonl(
        directory / "queries.jsonl", [{"id": f"q{index}", "text": row["anc"]} for index, row in enumerate(rows)]
    )
    qrels = [{"query-id": f"q{index}", "corpus-id": f"d{index}", "score": 1 + index % 2} for index in range(10)]
    qrels.append({"query-id": "q10", "corpus-id": "d10", "score": 0})  # never relevant: not evaluated
    write_jsonl(directory / "qrels.jsonl", qrels)
    return directory


@pytest.fixture
def mlm_model(tiny_mlm_dir):
    from sentence_transformers.sparse_encoder import SparseEncoder
    from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer

    return SparseEncoder(
        modules=[Transformer(str(tiny_mlm_dir), transformer_task="fill-mask", max_seq_length=32), SpladePooling("max")],
        prompts={"query": QUERY_PREFIX, "document": DOC_PREFIX},
        similarity_fn_name="dot",
        device="cpu",
    )


def _dense(model, method: str, texts):
    return getattr(model, method)(list(texts), convert_to_tensor=True, convert_to_sparse_tensor=False).float().cpu()


def test_caps_parse_and_keep_the_largest_weights(sparse_eval):
    assert sparse_eval.parse_cap("full") == (None, None)
    assert sparse_eval.parse_cap("64/400") == (64, 400)
    assert sparse_eval.parse_cap("full/16") == (None, 16)
    for bad in ("64", "0/4", "a/b", "1/2/3"):
        with pytest.raises(argparse.ArgumentTypeError):
            sparse_eval.parse_cap(bad)
    assert [sparse_eval.cap_label(cap) for cap in ((None, None), (64, None), (8, 4))] == ["full", "64/full", "8/4"]

    vectors = torch.tensor([[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 0.0, 0.0]])
    assert torch.equal(sparse_eval.keep_largest(vectors, 2), torch.tensor([[0.0, 3.0, 0.0, 2.0], [5.0, 0, 0, 0]]))
    assert sparse_eval.keep_largest(vectors, None) is vectors
    assert sparse_eval.keep_largest(vectors, 9) is vectors


def test_rerank_matches_a_direct_computation_for_every_cap(sparse_eval, mlm_model, reranking_file):
    examples = sparse_eval.reranking_examples(data_files=[str(reranking_file)], group_column="q_id")
    caps = [(None, None), (3, 5), (None, 2)]

    result = sparse_eval.rerank(mlm_model, examples, caps=caps, chunk=3, seed=42)

    queries = list(dict.fromkeys(item.query for item in examples))
    passages = list(dict.fromkeys(passage for item in examples for passage in item.passages))
    query_vectors = _dense(mlm_model, "encode_query", queries)
    passage_vectors = _dense(mlm_model, "encode_document", passages)
    generator = torch.Generator().manual_seed(42)
    shuffles = [torch.randperm(len(item.passages), generator=generator) for item in examples]
    assert (result["queries"], result["passages"], result["dimensions"]) == (8, len(passages), 23)
    for cap in caps:
        label = sparse_eval.cap_label(cap)
        capped_queries = sparse_eval.keep_largest(query_vectors, cap[0])
        capped_passages = sparse_eval.keep_largest(passage_vectors, cap[1])
        for item, shuffle in zip(examples, shuffles, strict=True):
            rows = [passages.index(passage) for passage in item.passages]
            scores = (capped_passages[rows] @ capped_queries[queries.index(item.query)]).double()
            expected = ranking_metrics(scores[shuffle], torch.tensor(item.labels)[shuffle])
            assert result["per_query"][label][item.query_id] == pytest.approx(expected)
        query_share = (capped_queries > 0).double().mean(dim=0)
        passage_share = (capped_passages > 0).double().mean(dim=0)
        assert result["caps"][label]["l0_query"] == pytest.approx((capped_queries > 0).sum(1).double().mean().item())
        assert result["caps"][label]["l0_document"] == pytest.approx(
            (capped_passages > 0).sum(1).double().mean().item()
        )
        assert result["caps"][label]["flops"] == pytest.approx((query_share * passage_share).sum().item())
    assert result["caps"]["3/5"]["l0_query"] <= 3 and result["caps"]["full/2"]["l0_document"] <= 2
    assert result == sparse_eval.rerank(mlm_model, examples, caps=caps, chunk=1000, seed=42)  # chunks change nothing


def test_reranking_examples_take_titles_and_samples(sparse_eval, reranking_file):
    plain = sparse_eval.reranking_examples(data_files=[str(reranking_file)], group_column="q_id")
    titled = sparse_eval.reranking_examples(data_files=[str(reranking_file)], group_column="q_id", title_column="title")
    sample = sparse_eval.reranking_examples(data_files=[str(reranking_file)], group_column="q_id", max_queries=3)

    by_id = {item.query_id: item for item in plain}
    for item in titled:
        index = item.query_id[1:]
        assert item.passages == tuple(f"題{index} {passage}" for passage in by_id[item.query_id].passages)
    assert len(plain) == 8 and len(sample) == 3 and {item.query_id for item in sample} <= set(by_id)


def test_retrieval_metrics_use_graded_gains(sparse_eval):
    retrieval_metrics = sparse_eval.retrieval_metrics
    relevance = {"a": 2, "b": 1, "z": 0}
    metrics = retrieval_metrics(["x", "b", "a"], relevance, k=2, depth=3)

    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert metrics["ndcg@2"] == pytest.approx((1 / math.log2(3)) / ideal)
    assert metrics["mrr@2"] == pytest.approx(0.5)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert retrieval_metrics(["x"], relevance, k=10, depth=1)["recall@1"] == 0.0
    with pytest.raises(ValueError, match="no relevant"):
        retrieval_metrics(["a"], {"a": 0})


def test_retrieve_matches_an_exhaustive_ranking(sparse_eval, mlm_model, retrieval_dir):
    queries, corpus, qrels = sparse_eval.retrieval_data(str(retrieval_dir))
    caps = [(None, None), (4, 6)]

    result = sparse_eval.retrieve(mlm_model, queries, corpus, qrels, caps=caps, top_k=5, chunk=7)

    evaluated = [query_id for query_id in queries if query_id in qrels and max(qrels[query_id].values()) > 0]
    assert (result["queries"], result["documents"]) == (10, 24) and set(result["per_query"]["full"]) == set(evaluated)
    assert corpus["n0"].startswith("題 ") and not corpus["d0"].startswith(" ")
    documents = list(corpus)
    document_vectors = _dense(mlm_model, "encode_document", corpus.values())
    query_vectors = _dense(mlm_model, "encode_query", [queries[query_id] for query_id in evaluated])
    for cap in caps:
        label = sparse_eval.cap_label(cap)
        scores = sparse_eval.keep_largest(query_vectors, cap[0]) @ sparse_eval.keep_largest(document_vectors, cap[1]).T
        for row, query_id in enumerate(evaluated):
            ranked = [documents[int(i)] for i in scores[row].argsort(descending=True, stable=True)[:5]]
            expected = sparse_eval.retrieval_metrics(ranked, qrels[query_id], k=10, depth=5)
            # Equal scores may be ordered differently by top-k; compare the scores reached instead.
            assert result["per_query"][label][query_id]["recall@5"] == pytest.approx(expected["recall@5"])
            assert result["per_query"][label][query_id]["ndcg@10"] == pytest.approx(expected["ndcg@10"], abs=1e-6)
    sample, _, _ = sparse_eval.retrieval_data(str(retrieval_dir), max_queries=4)
    assert len(sample) == 4 and all(max(qrels[query_id].values()) > 0 for query_id in sample)


def test_saved_project_models_load_without_remote_code(sparse_eval, tiny_model_dir, tmp_path):
    from muscari.retrieval import build_sparse_encoder

    torch.manual_seed(42)
    sae_dir = save_topk_sae(tmp_path / "sae", TopKSAE(HIDDEN_SIZE, 48, 4), {"max_length": 32})
    model = build_sparse_encoder(sae_dir, str(tiny_model_dir), query_prompt=QUERY_PREFIX, document_prompt=DOC_PREFIX)
    model.save(str(tmp_path / "model"))
    assert not sparse_eval.needs_remote_code(str(tmp_path / "model"))
    assert sparse_eval.needs_remote_code("someone/published-model")

    modules = json.loads((tmp_path / "model" / "modules.json").read_text(encoding="utf-8"))
    modules[1]["type"] = "custom_code.Pooling"
    (tmp_path / "model" / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    assert sparse_eval.needs_remote_code(str(tmp_path / "model"))


def test_the_script_evaluates_a_saved_model(sparse_eval, tiny_model_dir, reranking_file, retrieval_dir, tmp_path):
    from muscari.retrieval import build_sparse_encoder

    torch.manual_seed(42)
    sae_dir = save_topk_sae(tmp_path / "sae", TopKSAE(HIDDEN_SIZE, 48, 4), {"max_length": 32})
    build_sparse_encoder(
        sae_dir, str(tiny_model_dir), query_prompt=QUERY_PREFIX, document_prompt=DOC_PREFIX, max_seq_length=32
    ).save(str(tmp_path / "model"))
    out = tmp_path / "eval" / "result.json"
    common = [*("--device", "cpu"), *("--files", str(reranking_file)), *("--group-column", "q_id")]

    saved = sparse_eval.run(
        sparse_eval.build_parser().parse_args(
            [
                *("--model", str(tmp_path / "model")),
                *common,
                *("--retrieval", str(retrieval_dir)),
                *("--caps", "full", "2/3"),
                *("--out", str(out)),
            ]
        )
    )
    untrained = sparse_eval.run(
        sparse_eval.build_parser().parse_args(
            [
                *("--sae", str(sae_dir)),
                *("--model", str(tiny_model_dir)),
                *("--query-prompt", QUERY_PREFIX),
                *("--document-prompt", DOC_PREFIX),
                *("--max-seq-length", "32"),
                *common,
                *("--out", str(tmp_path / "untrained.json")),
            ]
        )
    )

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["model"] == "model" and written["retrieval"]["corpus-set"]["caps"].keys() == {"full", "2/3"}
    assert written["reranking"]["titles"] is False and written["reranking"]["split"] == "test"
    assert saved["reranking"]["per_query"]["full"] == untrained["reranking"]["per_query"]["full"]
    with pytest.raises(SystemExit, match="Nothing to evaluate"):
        sparse_eval.run(sparse_eval.build_parser().parse_args(["--model", str(tmp_path / "model"), "--out", str(out)]))
