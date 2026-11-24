"""Masked-language-model loss recovered, on a tiny random BERT that keeps its head."""

from __future__ import annotations

import pytest
import torch
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows, perfect_sae
from transformers import AutoModelForMaskedLM

from muscari import EncoderActivationSource, load_sae
from muscari.export import save_sae
from muscari.mlm import load_masked_lm, mlm_loss_recovered


@pytest.fixture
def masked_lm(tiny_mlm_dir):
    return load_masked_lm(str(tiny_mlm_dir))


@pytest.fixture
def mlm_source(tiny_mlm_dir):
    return EncoderActivationSource.from_pretrained(str(tiny_mlm_dir), text_batch_size=8, max_length=32, device="cpu")


@pytest.fixture
def texts():
    return [(DOC_PREFIX, str(row["pos"])) for row in make_rows(12)] + [
        (QUERY_PREFIX, str(row["anc"])) for row in make_rows(6)
    ]


def constant_sae(directory, value: float):
    return load_sae(
        save_sae(
            directory,
            encoder_weight=torch.zeros(8, HIDDEN_SIZE),
            encoder_bias=torch.zeros(8),
            decoder_weight=torch.zeros(HIDDEN_SIZE, 8),
            decoder_bias=torch.full((HIDDEN_SIZE,), value),
            config={"k": 2},
        )
    )


def test_a_perfect_sae_recovers_the_whole_loss(masked_lm, mlm_source, texts, tmp_path):
    result = mlm_loss_recovered(perfect_sae(tmp_path), masked_lm, mlm_source, texts, mask_probability=0.3)

    assert result["loss_spliced"] == pytest.approx(result["loss_clean"], rel=1e-5)
    assert abs(result["loss_mean"] - result["loss_clean"]) > 1e-3 * result["loss_clean"]
    assert result["recovered"] == pytest.approx(1.0, abs=1e-3)
    assert result["texts"] == 18 and result["masked_tokens"] > 0
    assert not masked_lm.base_model._forward_hooks  # the hook is gone


def test_the_clean_loss_is_the_models_own_masked_lm_loss(masked_lm, mlm_source, texts, tmp_path):
    result = mlm_loss_recovered(
        perfect_sae(tmp_path), masked_lm, mlm_source, texts, mask_probability=0.3, batch_size=len(texts), seed=7
    )

    # Rebuild the same masked batch and let the model compute its loss from labels.
    ordered = sorted(texts, key=lambda item: len(item[1]))
    encoding, content = mlm_source.tokenize([kind for kind, _ in ordered], [text for _, text in ordered])
    masked = content & (torch.rand(content.shape, generator=torch.Generator().manual_seed(7)) < 0.3)
    labels = torch.where(masked, encoding["input_ids"], torch.tensor(-100))
    input_ids = torch.where(masked, torch.tensor(mlm_source.tokenizer.mask_token_id), encoding["input_ids"])
    with torch.no_grad():
        expected = masked_lm(input_ids=input_ids, attention_mask=encoding["attention_mask"], labels=labels).loss

    assert result["masked_tokens"] == int(masked.sum())
    assert result["loss_clean"] == pytest.approx(expected.item(), rel=1e-5)


def test_only_content_tokens_are_masked(masked_lm, mlm_source, texts, tmp_path):
    result = mlm_loss_recovered(perfect_sae(tmp_path), masked_lm, mlm_source, texts, mask_probability=0.999999)
    content_tokens = sum(
        int(mlm_source.tokenize([kind], [text])[1].sum()) for kind, text in texts
    )  # prefix and special tokens excluded

    assert result["masked_tokens"] == content_tokens


def test_the_reference_losses_do_not_depend_on_the_sae(masked_lm, mlm_source, texts, tmp_path):
    first = mlm_loss_recovered(constant_sae(tmp_path / "a", 0.0), masked_lm, mlm_source, texts)
    second = mlm_loss_recovered(constant_sae(tmp_path / "b", 3.0), masked_lm, mlm_source, texts)

    assert (first["loss_clean"], first["loss_mean"]) == (second["loss_clean"], second["loss_mean"])
    assert first["loss_spliced"] != pytest.approx(second["loss_spliced"], rel=1e-3)
    assert first == mlm_loss_recovered(constant_sae(tmp_path / "c", 0.0), masked_lm, mlm_source, texts)


def test_a_checkpoint_without_a_head_is_rejected(tiny_model_dir, tiny_mlm_dir):
    with pytest.raises(ValueError, match="no trained language-model head"):
        load_masked_lm(str(tiny_model_dir))

    assert isinstance(load_masked_lm(str(tiny_mlm_dir)), type(AutoModelForMaskedLM.from_pretrained(str(tiny_mlm_dir))))


def test_invalid_inputs_are_rejected(masked_lm, mlm_source, source, texts, tmp_path):
    sae = perfect_sae(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        mlm_loss_recovered(sae, masked_lm, mlm_source, [])
    with pytest.raises(ValueError, match="mask_probability"):
        mlm_loss_recovered(sae, masked_lm, mlm_source, texts, mask_probability=1.0)
    with pytest.raises(ValueError, match="mask token"):
        mlm_loss_recovered(sae, masked_lm, source, texts)  # the plain tiny tokenizer has none
