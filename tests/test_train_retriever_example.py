"""The retrieval example must train both heads and agree with the probe before training."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows, write_jsonl

from muscari import TopKSAE, load_sae, save_topk_sae
from muscari.probe import probe_reranking

pytest.importorskip("sentence_transformers")
pytest.importorskip("accelerate")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE = EXAMPLES / "train_retriever.py"


@pytest.fixture(scope="module")
def example():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))  # as when the script runs: it imports sparse_eval beside it
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

    examples = importlib.import_module("sparse_eval").reranking_examples(data_files=[str(data[1])], group_column="q_id")
    probe = probe_reranking(source, load_sae(sae_dir), examples, query_kind=QUERY_PREFIX, passage_kind=DOC_PREFIX)
    before, after = result["before"]["reranking"], result["after"]["reranking"]
    assert before["caps"]["full"]["ndcg@10"] == pytest.approx(probe["sparse"]["plain"]["ndcg@10"])
    assert before["caps"]["full"]["l0_query"] == pytest.approx(probe["sparse"]["l0"]["query"])
    assert before["caps"]["full"]["l0_document"] == pytest.approx(probe["sparse"]["l0"]["passage"])
    assert (before["dimensions"], before["queries"], result["train_rows"]) == (96, 10, 24)
    assert after["per_query"] != before["per_query"] and after["caps"]["full"]["flops"] > 0
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
    original = example.evaluate
    monkeypatch.setattr(example, "evaluate", lambda *a, **k: calls.append(1) or original(*a, **k))
    (out / "results.json").unlink()
    again = example.run(args)

    assert again["resumed"] and again["before"] == first["before"]
    assert len(calls) == 1  # only the evaluation after training ran again


def test_mlm_head_trains_with_the_same_recipe(example, tiny_mlm_dir, data, tmp_path):
    result = example.run(arguments(example, "mlm", tiny_mlm_dir, data, tmp_path / "out", "--negatives", "2"))

    assert result["before"]["reranking"]["dimensions"] == 23  # the tiny vocabulary
    assert set(result["before"]["reranking"]["caps"]["full"]) == {
        "ndcg@10",
        "mrr@10",
        "l0_query",
        "l0_document",
        "flops",
    }
    assert 0 <= result["after"]["reranking"]["caps"]["full"]["ndcg@10"] <= 1 and result["train_seconds"] > 0


def test_each_head_needs_its_own_argument(example, tiny_model_dir, data, tmp_path):
    with pytest.raises(SystemExit, match="--sae"):
        example.run(arguments(example, "sae", tiny_model_dir, data, tmp_path / "out"))


def test_vocabulary_head_estimates_its_shift_and_trains_with_gradcache(example, tiny_mlm_dir, data, tmp_path):
    extra = ("--shift", "auto", "--shift-quantile", "0.9", "--fan-out", "4", "--mini-batch-size", "3")
    result = example.run(arguments(example, "vocab", tiny_mlm_dir, data, tmp_path / "out", *extra))
    pooled = example.build_model(
        arguments(example, "vocab", tiny_mlm_dir, data, tmp_path / "specials", "--pool-special-tokens")
    )
    assert pooled[-1].special_token_ids == []  # nothing is left out but the prompt

    config = result["head_config"]
    assert config["fan_out"] == 4 and config["shift"] != 0 and config["special_token_ids"]
    assert result["mini_batch_size"] == 3 and len(result["train_log"]) > 0
    before = result["before"]["reranking"]["caps"]["full"]
    assert before["l0_query"] > 0 and 0 <= result["after"]["reranking"]["caps"]["full"]["ndcg@10"] <= 1


def test_a_trainable_sae_head_changes_its_encoder(example, tiny_model_dir, data, tmp_path):
    from sentence_transformers.sparse_encoder import SparseEncoder

    torch.manual_seed(42)
    sae = TopKSAE(HIDDEN_SIZE, 96, 4)
    sae_dir = save_topk_sae(tmp_path / "sae", sae, {"max_length": 32})
    args = arguments(example, "sae", tiny_model_dir, data, tmp_path / "out", "--sae", str(sae_dir), "--trainable-sae")

    result = example.run(args)

    trained = SparseEncoder(str(tmp_path / "out" / "model"), device="cpu", trust_remote_code=True)
    assert result["head_config"]["trainable"] is True
    assert not torch.equal(trained[1].encoder_weight.detach(), load_sae(sae_dir).encoder_weight)


def test_listed_queries_are_kept_out_of_the_training_rows(example, tiny_mlm_dir, data, tmp_path):
    rows = [json.loads(line) for line in data[0].read_text(encoding="utf-8").splitlines()]
    listed = tmp_path / "evaluation-queries.txt"
    listed.write_text(f"  {rows[0]['anc']}\n\n{rows[3]['anc']}\nnot a training query\n", encoding="utf-8")
    args = arguments(example, "mlm", tiny_mlm_dir, data, tmp_path / "out", "--exclude-queries", str(listed))

    kept = example.training_rows(rows, args)
    everything = example.training_rows(rows, arguments(example, "mlm", tiny_mlm_dir, data, tmp_path / "out"))

    dropped = {rows[0]["anc"], rows[3]["anc"]}
    assert not dropped & set(kept["anchor"]) and dropped <= set(everything["anchor"])
    assert kept["anchor"] == [row["anc"] for row in rows if row["anc"] not in dropped][: len(kept)]
    assert len(kept) == len(everything)  # later rows fill the places of the dropped ones
    assert example.normalise_query(" ＡＢＣ　") == "ABC"
