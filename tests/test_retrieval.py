"""Sentence Transformers head: pooling, gradients, persistence and a short training run."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tiny import DOC_PREFIX, HIDDEN_SIZE, QUERY_PREFIX, make_rows, perfect_sae

from muscari import TopKSAE, load_sae, save_topk_sae
from muscari.export import save_sae
from muscari.probe import pool_texts

pytest.importorskip("sentence_transformers")

from sentence_transformers.sparse_encoder import SparseEncoder  # noqa: E402

from muscari.retrieval import SAEPooling, build_sparse_encoder  # noqa: E402

TEXTS = ["東京 求人 検索", "名古屋 営業", "在宅 勤務 経験 資格 正社員", "給与"]
SPECIAL = [0, 2, 3]  # padding, classifier and separator; the unknown token (1) is content
UNKNOWN_WORDS = "未知語 東京 知らない言葉"


@pytest.fixture
def random_sae(tmp_path):
    torch.manual_seed(42)
    return load_sae(save_topk_sae(tmp_path / "sae", TopKSAE(HIDDEN_SIZE, 96, 4), {"max_length": 32}))


def dense_reference(states, keep, weight, bias, k):
    """Pooled vectors computed the plain way, with every intermediate tensor dense."""
    pre = states @ weight.T + bias
    values, indices = pre.topk(k, dim=-1)
    features = torch.zeros_like(pre).scatter(-1, indices, torch.log1p(values.relu()))
    return features.masked_fill(~keep.unsqueeze(-1), 0.0).amax(dim=1)


def features_for(states, input_ids, attention_mask, **extra):
    return {"token_embeddings": states, "input_ids": input_ids, "attention_mask": attention_mask, **extra}


def test_encoding_matches_the_probe_pooling(tiny_model_dir, source, random_sae):
    model = build_sparse_encoder(
        random_sae, str(tiny_model_dir), query_prompt=QUERY_PREFIX, document_prompt=DOC_PREFIX, device="cpu"
    )

    texts = [*TEXTS, UNKNOWN_WORDS]  # unknown tokens are pooled, as they were when the SAE was trained
    for encode, prefix in ((model.encode_query, QUERY_PREFIX), (model.encode_document, DOC_PREFIX)):
        embeddings = encode(texts, convert_to_sparse_tensor=False)
        assert embeddings.shape == (len(texts), 96)
        for row, vector in zip(embeddings, pool_texts(source, random_sae, prefix, texts), strict=True):
            expected = torch.zeros(96)
            expected[vector.indices] = vector.values
            assert torch.allclose(row, expected, atol=1e-5)

    assert isinstance(model, SparseEncoder) and model.get_embedding_dimension() == 96
    assert model.max_seq_length == 32 and model.similarity_fn_name == "dot"
    assert model[1].special_token_ids == SPECIAL  # taken from the tokenizer


def test_special_prompt_and_padded_positions_are_not_pooled():
    torch.manual_seed(42)
    head = SAEPooling(8, 32, k=3, special_token_ids=[0, 9])
    torch.nn.init.normal_(head.encoder_weight)
    states = torch.randn(2, 6, 8)
    input_ids = torch.tensor([[0, 5, 6, 7, 9, 1], [0, 5, 7, 9, 1, 1]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])

    pooled = head(features_for(states, input_ids, attention, prompt_length=2))["sentence_embedding"]

    # Position 0 is special, position 1 is the prompt; the first text keeps 2 and 3, the second only 2.
    keep = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    assert torch.allclose(pooled, dense_reference(states, keep, head.encoder_weight, head.encoder_bias, 3), atol=1e-6)
    assert (pooled[1] > 0).sum() <= 3

    with_prompt = SAEPooling(8, 32, k=3, special_token_ids=[0, 9], exclude_prompt=False)
    with_prompt.load_state_dict(head.state_dict())
    keep[:, 1] = True
    pooled = with_prompt(features_for(states, input_ids, attention, prompt_length=2))["sentence_embedding"]
    assert torch.allclose(pooled, dense_reference(states, keep, head.encoder_weight, head.encoder_bias, 3), atol=1e-6)


def test_left_padding_and_a_tensor_prompt_length_are_handled():
    torch.manual_seed(42)
    head = SAEPooling(8, 32, k=3, special_token_ids=[0])
    torch.nn.init.normal_(head.encoder_weight)
    states = torch.randn(1, 5, 8)
    input_ids = torch.tensor([[1, 1, 0, 5, 6]])
    attention = torch.tensor([[0, 0, 1, 1, 1]])

    pooled = head(features_for(states, input_ids, attention, prompt_length=torch.tensor([2])))["sentence_embedding"]

    keep = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool)  # the special token and one prompt token go
    assert torch.allclose(pooled, dense_reference(states, keep, head.encoder_weight, head.encoder_bias, 3), atol=1e-6)


def test_an_unpadded_batch_gives_the_same_vectors_as_a_padded_one():
    torch.manual_seed(42)
    head = SAEPooling(8, 32, k=3, special_token_ids=[0, 9], token_chunk=2)
    torch.nn.init.normal_(head.encoder_weight)
    states = torch.randn(2, 5, 8)
    input_ids = torch.tensor([[0, 5, 6, 7, 9], [0, 5, 9, 1, 1]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    padded = head(features_for(states, input_ids, attention, prompt_length=1))["sentence_embedding"]

    flat = {
        "token_embeddings": torch.cat([states[0], states[1, :3]]).unsqueeze(0),
        "input_ids": torch.cat([input_ids[0], input_ids[1, :3]]).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, 5, 8]),
        "prompt_length": 1,
    }
    assert torch.allclose(head(flat)["sentence_embedding"], padded, atol=1e-6)


@pytest.mark.parametrize("trainable", [False, True])
def test_gradients_match_the_dense_computation(trainable):
    torch.manual_seed(42)
    head = SAEPooling(8, 64, k=4, special_token_ids=[0], trainable=trainable, token_chunk=3)
    torch.nn.init.normal_(head.encoder_weight)
    torch.nn.init.normal_(head.encoder_bias)
    states = torch.randn(3, 5, 8, requires_grad=True)
    input_ids = torch.tensor([[0, 5, 6, 7, 8], [0, 5, 6, 1, 1], [0, 5, 6, 7, 1]])
    attention = (input_ids != 1).long()
    keep = attention.bool() & (input_ids != 0)
    target = torch.randn(3, 64)

    loss = (head(features_for(states, input_ids, attention))["sentence_embedding"] * target).sum()
    grads = torch.autograd.grad(loss, [states, *(p for p in head.parameters() if p.requires_grad)])

    weight = head.encoder_weight.detach().clone().requires_grad_(True)
    bias = head.encoder_bias.detach().clone().requires_grad_(True)
    reference_states = states.detach().clone().requires_grad_(True)
    reference = (dense_reference(reference_states, keep, weight, bias, 4) * target).sum()
    expected = torch.autograd.grad(reference, [reference_states, weight, bias])

    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)
    assert len(grads) == (3 if trainable else 1)
    for got, want in zip(grads, expected, strict=False):
        assert torch.allclose(got, want, atol=1e-5)
    assert [p.requires_grad for p in head.parameters()] == [trainable, trainable]


def test_a_jumprelu_sae_keeps_what_exceeds_its_thresholds(tmp_path):
    torch.manual_seed(42)
    sae = load_sae(
        save_sae(
            tmp_path,
            encoder_weight=torch.randn(16, 8),
            encoder_bias=torch.randn(16),
            decoder_weight=torch.randn(8, 16),
            decoder_bias=torch.zeros(8),
            threshold=torch.rand(16),
            config={"activation": "jumprelu"},
        )
    )
    head = SAEPooling.from_sae(sae, special_token_ids=[0])
    states = torch.randn(2, 4, 8)
    input_ids = torch.tensor([[0, 5, 6, 7], [0, 5, 1, 1]])
    attention = (input_ids != 1).long()

    pooled = head(features_for(states, input_ids, attention))["sentence_embedding"]

    assert (head.activation, head.k) == ("jumprelu", None)
    for row, text_states, kept in ((0, states[0], [1, 2, 3]), (1, states[1], [1])):
        expected = torch.log1p(sae.encode(text_states[kept])).amax(dim=0)
        assert torch.allclose(pooled[row], expected, atol=1e-6)


def test_masked_latents_are_always_zero(random_sae):
    states = torch.randn(1, 6, HIDDEN_SIZE)
    features = features_for(states, torch.arange(4, 10).unsqueeze(0), torch.ones(1, 6, dtype=torch.long))
    plain = SAEPooling.from_sae(random_sae, special_token_ids=SPECIAL)(dict(features))["sentence_embedding"]
    active = plain[0].nonzero().flatten().tolist()

    masked = SAEPooling.from_sae(random_sae, special_token_ids=SPECIAL, masked_latents=active[:2])
    pooled = masked(dict(features))["sentence_embedding"]

    assert (pooled[0, active[:2]] == 0).all()
    assert torch.equal(pooled[0, active[2:]], plain[0, active[2:]])


def test_a_saved_model_loads_with_the_same_head(tiny_model_dir, random_sae, tmp_path):
    model = build_sparse_encoder(
        random_sae,
        str(tiny_model_dir),
        query_prompt=QUERY_PREFIX,
        document_prompt=DOC_PREFIX,
        head_kwargs={"masked_latents": [5, 3]},
        device="cpu",
    )
    model.save(str(tmp_path / "model"))

    # The head lives outside the sentence_transformers namespace, so loading has to be allowed.
    with pytest.raises(ValueError, match="trust_remote_code"):
        SparseEncoder(str(tmp_path / "model"), device="cpu")
    loaded = SparseEncoder(str(tmp_path / "model"), device="cpu", trust_remote_code=True)

    assert isinstance(loaded[1], SAEPooling)
    assert loaded[1].get_config_dict() == model[1].get_config_dict()
    assert loaded[1].masked_latents == [3, 5] and not loaded[1].encoder_weight.requires_grad
    assert torch.equal(loaded[1].encoder_weight, model[1].encoder_weight)
    assert torch.equal(
        loaded.encode_query(TEXTS, convert_to_sparse_tensor=False),
        model.encode_query(TEXTS, convert_to_sparse_tensor=False),
    )


def test_invalid_heads_are_rejected(random_sae, tmp_path):
    with pytest.raises(ValueError, match="activation"):
        SAEPooling(8, 16, k=2, activation="relu")
    with pytest.raises(ValueError, match="needs k"):
        SAEPooling(8, 16)
    with pytest.raises(ValueError, match="needs k"):
        SAEPooling(8, 16, k=17)
    with pytest.raises(RuntimeError, match="special tokens"):
        SAEPooling(8, 16, k=2)(features_for(torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.ones(1, 2)))

    middle = load_sae(save_topk_sae(tmp_path / "middle", TopKSAE(HIDDEN_SIZE, 16, 2), {"layer": 1}))
    with pytest.raises(ValueError, match="last_hidden_state"):
        build_sparse_encoder(middle, "unused")
    with pytest.raises(ValueError, match="model_name_or_path"):
        build_sparse_encoder(random_sae)


def test_a_perfect_sae_makes_related_texts_overlap(tiny_model_dir, tmp_path):
    model = build_sparse_encoder(perfect_sae(tmp_path), str(tiny_model_dir), device="cpu")
    embeddings = model.encode(["東京 求人 検索", "東京 求人 検索", "残業 休日"], convert_to_sparse_tensor=False)

    scores = model.similarity(embeddings, embeddings)
    assert scores[0, 1] == pytest.approx(scores[0, 0].item()) and scores[0, 1] > scores[0, 2]


def test_training_updates_the_encoder_and_leaves_the_sae_frozen(tiny_model_dir, random_sae, tmp_path):
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from sentence_transformers.sparse_encoder import SparseEncoderTrainer, SparseEncoderTrainingArguments
    from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

    model = build_sparse_encoder(
        random_sae, str(tiny_model_dir), query_prompt=QUERY_PREFIX, document_prompt=DOC_PREFIX, device="cpu"
    )
    rows = make_rows(32)
    train = datasets.Dataset.from_dict(
        {"anchor": [str(row["anc"]) for row in rows], "positive": [str(row["pos"]) for row in rows]}
    )
    loss = SpladeLoss(
        model,
        SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=1e-3,
        query_regularizer_weight=1e-3,
    )
    sae_before = model[1].encoder_weight.clone()
    encoder_before = F.normalize(model[0].model.embeddings.word_embeddings.weight.detach().clone(), dim=-1)

    trainer = SparseEncoderTrainer(
        model=model,
        args=SparseEncoderTrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=3,
            per_device_train_batch_size=8,
            learning_rate=1e-3,
            prompts={"anchor": QUERY_PREFIX, "positive": DOC_PREFIX},
            report_to="none",
            save_strategy="no",
            use_cpu=True,
        ),
        train_dataset=train,
        loss=loss,
    )
    trainer.train()

    encoder_after = F.normalize(model[0].model.embeddings.word_embeddings.weight.detach(), dim=-1)
    assert trainer.state.global_step == 3
    assert torch.equal(model[1].encoder_weight, sae_before)
    assert not torch.allclose(encoder_after, encoder_before)
