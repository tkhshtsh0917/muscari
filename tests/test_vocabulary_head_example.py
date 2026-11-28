"""The example vocabulary head must equal SPLADE when neutral and pool the SAE head's tokens otherwise."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from tiny import DOC_PREFIX, QUERY_PREFIX, make_rows

pytest.importorskip("sentence_transformers")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TEXTS = [row["pos"] for row in make_rows(6)]


@pytest.fixture(scope="module")
def head_module():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("vocabulary_head")


def _capture(model):
    """Record the features that reach the head and what it returns, batch by batch.

    ``encode`` sorts texts by length, so the head's rows are compared with its own inputs.
    """
    seen = []
    handle = model[-1].register_forward_hook(lambda module, args, output: seen.append(dict(output)))
    return seen, handle


def test_a_neutral_head_equals_splade_pooling(head_module, tiny_mlm_dir):
    from sentence_transformers.sparse_encoder import SparseEncoder
    from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer

    neutral = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir),
        keep_special_tokens=True,
        exclude_prompt=False,
        query_prompt=QUERY_PREFIX,
        max_seq_length=32,
        device="cpu",
    )
    splade = SparseEncoder(
        modules=[Transformer(str(tiny_mlm_dir), transformer_task="fill-mask", max_seq_length=32), SpladePooling("max")],
        prompts={"query": QUERY_PREFIX},
        device="cpu",
    )

    ours = neutral.encode_query(TEXTS, convert_to_tensor=True, convert_to_sparse_tensor=False)
    theirs = splade.encode_query(TEXTS, convert_to_tensor=True, convert_to_sparse_tensor=False)

    assert neutral[-1].special_token_ids == [] and (ours > 0).any()
    assert torch.allclose(ours, theirs, atol=1e-6)


@pytest.mark.parametrize("fan_out", [None, 3])
def test_shift_and_fan_out_apply_to_content_tokens_only(head_module, tiny_mlm_dir, fan_out):
    model = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir), shift=-0.5, fan_out=fan_out, document_prompt=DOC_PREFIX, max_seq_length=32, device="cpu"
    )
    seen, handle = _capture(model)
    model.encode_document(TEXTS, convert_to_tensor=True, convert_to_sparse_tensor=False)
    handle.remove()

    features = seen[0]
    pooled = features["sentence_embedding"]
    special = torch.tensor(model[-1].special_token_ids)
    prompt_length = int(features["prompt_length"])
    assert prompt_length > 0 and special.numel() > 0
    for row in range(len(TEXTS)):
        positions = torch.arange(features["input_ids"].shape[1])
        attended = features["attention_mask"][row] > 0
        content = attended & ~torch.isin(features["input_ids"][row], special)
        content &= (attended.long().cumsum(0) - 1) >= prompt_length
        shifted = features["token_embeddings"][row, positions[content]].float() + 0.5
        if fan_out is not None:
            kept = shifted.topk(fan_out, dim=1)
            shifted = torch.full_like(shifted, -1.0).scatter(1, kept.indices, kept.values)
        expected = torch.log1p(torch.relu(shifted)).amax(dim=0)
        assert torch.allclose(pooled[row], expected, atol=1e-6)
        if fan_out is not None:
            assert int((pooled[row] > 0).sum()) <= fan_out * int(content.sum())


def test_gradients_reach_the_encoder_through_the_fan_out(head_module, tiny_mlm_dir):
    model = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir), shift=-1.0, fan_out=2, max_seq_length=32, device="cpu"
    )
    model.train()
    features = model.preprocess(TEXTS[:2])
    pooled = model(features)["sentence_embedding"]
    pooled.sum().backward()

    gradients = [parameter.grad for parameter in model[0].parameters() if parameter.grad is not None]
    assert gradients and any(gradient.abs().sum() > 0 for gradient in gradients)


def test_the_estimated_shift_is_the_quantile_of_the_content_logits(head_module, tiny_mlm_dir):
    model = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir), document_prompt=DOC_PREFIX, max_seq_length=32, device="cpu"
    )
    seen, handle = _capture(model)
    shift = head_module.estimate_shift(model, TEXTS, quantile=0.9, samples_per_batch=10**9, batch_size=64)
    handle.remove()

    logits, _, _ = model[-1]._content_tokens(seen[0])
    assert shift == pytest.approx(float(torch.quantile(logits.float().reshape(-1), 0.9)))
    built = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir),
        shift="auto",
        shift_texts=TEXTS,
        shift_quantile=0.9,
        document_prompt=DOC_PREFIX,
        device="cpu",
    )
    sampled = head_module.estimate_shift(built, TEXTS, quantile=0.9, samples_per_batch=50)
    assert built[-1].shift == pytest.approx(shift, rel=1e-5)
    assert abs(sampled - shift) < 1.0  # a small sample lands near the exact quantile
    with pytest.raises(ValueError, match="shift_texts"):
        head_module.build_vocabulary_encoder(str(tiny_mlm_dir), shift="auto")


def test_a_saved_vocabulary_model_reloads_through_the_evaluation_script(head_module, tiny_mlm_dir, tmp_path):
    sparse_eval = importlib.import_module("sparse_eval")
    model = head_module.build_vocabulary_encoder(
        str(tiny_mlm_dir), shift=-0.25, fan_out=4, query_prompt=QUERY_PREFIX, max_seq_length=32, device="cpu"
    )
    model.save(str(tmp_path / "model"))
    assert not sparse_eval.needs_remote_code(str(tmp_path / "model"))

    args = sparse_eval.build_parser().parse_args(
        ["--model", str(tmp_path / "model"), "--device", "cpu", "--out", str(tmp_path / "eval.json")]
    )
    loaded = sparse_eval.load_model(args)

    assert isinstance(loaded[-1], head_module.VocabularyPooling)
    assert loaded[-1].get_config_dict() == model[-1].get_config_dict()
    assert torch.equal(
        loaded.encode_query(TEXTS, convert_to_tensor=True, convert_to_sparse_tensor=False),
        model.encode_query(TEXTS, convert_to_tensor=True, convert_to_sparse_tensor=False),
    )
    (tmp_path / "model" / "vocabulary_head.py").write_text("raise SystemExit\n", encoding="utf-8")
    assert sparse_eval.needs_remote_code(str(tmp_path / "model"))


def test_invalid_fan_out_is_rejected(head_module):
    with pytest.raises(ValueError, match="fan_out"):
        head_module.VocabularyPooling(10, fan_out=11)
    with pytest.raises(ValueError, match="fan_out"):
        head_module.VocabularyPooling(10, fan_out=0)
