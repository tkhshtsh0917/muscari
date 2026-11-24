"""The retrieval example must train both heads and agree with the probe before training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows, write_jsonl

from muscari import TopKSAE, load_sae, save_topk_sae
from muscari.probe import probe_reranking

pytest.importorskip("sentence_transformers")
pytest.importorskip("accelerate")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "train_retriever.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("muscari_train_retriever_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", make_rows(64))
    candidates = [
        {"q_id": f"q{index}", "query": row["anc"], "text": text, "label": label}
        for index, row in enumerate(make_rows(10, seed=7))
        for text, label in ((row["neg"][0], 0), (row["pos"], 1), (row["neg"][1], 0))
    ]
    return train, write_jsonl(tmp_path / "rerank.jsonl", candidates)


def arguments(example, head: str, model: Path, data, out: Path, *extra: str):
    train, rerank = data
    return example.build_parser().parse_args(
        [
            *("--head", head),
            *("--model", str(model)),
            *("--out", str(out)),
            *("--query-prompt", QUERY_PREFIX),
            *("--document-prompt", DOC_PREFIX),
            *("--max-seq-length", "32"),
            *("--data-files", str(train)),
            *("--query-column", "anc"),
            *("--positive-column", "pos"),
            *("--negative-column", "neg"),
            *("--max-steps", "3"),
            *("--batch-size", "8"),
            *("--learning-rate", "1e-3"),
            *("--eval-files", str(rerank)),
            *("--eval-group-column", "q_id"),
            *("--eval-queries", "0"),
            *("--device", "cpu"),
            *extra,
        ]
    )


def test_sae_head_starts_at_the_probe_result_and_trains(example, tiny_model_dir, source, data, tmp_path):
    torch.manual_seed(42)
    sae_dir = save_topk_sae(tmp_path / "sae", TopKSAE(HIDDEN_SIZE, 96, 4), {"max_length": 32})
    args = arguments(example, "sae", tiny_model_dir, data, tmp_path / "out", "--sae", str(sae_dir))

    result = example.run(args)

    probe = probe_reranking(
        source, load_sae(sae_dir), example.evaluation_examples(args), query_kind=QUERY_PREFIX, passage_kind=DOC_PREFIX
    )
    assert result["before"]["ndcg@10"] == pytest.approx(probe["sparse"]["plain"]["ndcg@10"])
    assert result["before"]["l0_query"] == pytest.approx(probe["sparse"]["l0"]["query"])
    assert result["before"]["l0_passage"] == pytest.approx(probe["sparse"]["l0"]["passage"])
    assert (result["before"]["dimensions"], result["before"]["queries"], result["train_rows"]) == (96, 10, 24)
    assert result["after"] != result["before"] and result["after"]["flops"] > 0
    assert json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))["head"] == "sae"
    assert (tmp_path / "out" / "model" / "modules.json").exists()


def test_a_stopped_run_continues_from_its_checkpoint(example, tiny_model_dir, data, tmp_path, monkeypatch):
    torch.manual_seed(42)
    sae_dir = save_topk_sae(tmp_path / "sae", TopKSAE(HIDDEN_SIZE, 96, 4), {"max_length": 32})
    out = tmp_path / "out"
    args = arguments(example, "sae", tiny_model_dir, data, out, "--sae", str(sae_dir), "--max-steps", "10")
    first = example.run(args)
    assert not first["resumed"] and sorted(path.name for path in (out / "checkpoints").iterdir()) == ["checkpoint-10"]

    calls = []
    original = example.rerank
    monkeypatch.setattr(example, "rerank", lambda *a, **k: calls.append(1) or original(*a, **k))
    (out / "results.json").unlink()
    again = example.run(args)

    assert again["resumed"] and again["before"] == first["before"]
    assert len(calls) == 1  # only the evaluation after training ran again


def test_rerank_matches_a_sparse_computation_for_a_vocabulary_head(example, tiny_mlm_dir, data, tmp_path):
    args = arguments(example, "mlm", tiny_mlm_dir, data, tmp_path / "out")
    model = example.build_model(args)
    examples = example.evaluation_examples(args)

    result = example.rerank(model, examples, chunk=4)

    queries = list(dict.fromkeys(item.query for item in examples))
    passages = list(dict.fromkeys(passage for item in examples for passage in item.passages))
    query_vectors = model.encode_query(queries, convert_to_tensor=True, convert_to_sparse_tensor=False).cpu()
    passage_vectors = model.encode_document(passages, convert_to_tensor=True, convert_to_sparse_tensor=False).cpu()
    query_presence = (query_vectors > 0).double().mean(dim=0)
    passage_presence = (passage_vectors > 0).double().mean(dim=0)
    assert result["l0_query"] == pytest.approx((query_vectors > 0).sum(dim=1).double().mean().item())
    assert result["l0_passage"] == pytest.approx((passage_vectors > 0).sum(dim=1).double().mean().item())
    assert result["flops"] == pytest.approx((query_presence * passage_presence).sum().item())
    assert result == example.rerank(model, examples, chunk=1000)  # the chunk size changes nothing


def test_mlm_head_trains_with_the_same_recipe(example, tiny_mlm_dir, data, tmp_path):
    result = example.run(arguments(example, "mlm", tiny_mlm_dir, data, tmp_path / "out", "--negatives", "2"))

    assert result["before"]["dimensions"] == 23  # the tiny vocabulary
    assert set(result["before"]) == {"ndcg@10", "mrr@10", "queries", "dimensions", "l0_query", "l0_passage", "flops"}
    assert 0 <= result["after"]["ndcg@10"] <= 1 and result["train_seconds"] > 0


def test_each_head_needs_its_own_argument(example, tiny_model_dir, data, tmp_path):
    with pytest.raises(SystemExit, match="--sae"):
        example.run(arguments(example, "sae", tiny_model_dir, data, tmp_path / "out"))
