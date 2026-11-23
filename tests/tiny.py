"""A tiny offline encoder and synthetic retrieval rows shared by the tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from muscari import TextColumn

QUERY_PREFIX = "検索クエリ: "
DOC_PREFIX = "検索文書: "
WORDS = "東京 名古屋 求人 検索 営業 開発 経理 給与 勤務 経験 資格 正社員 派遣 在宅 残業 休日".split()
COLUMNS = [
    TextColumn("anc", QUERY_PREFIX),
    TextColumn("pos", DOC_PREFIX),
    TextColumn("neg", DOC_PREFIX),
]
HIDDEN_SIZE = 32
MAX_POSITIONS = 64


def make_rows(n: int, seed: int = 42) -> list[dict[str, object]]:
    """Rows shaped like a retrieval dataset: a query, a positive and a list of negatives."""
    generator = torch.Generator().manual_seed(seed)

    def sentence(length: int) -> str:
        ids = torch.randint(len(WORDS), (length,), generator=generator)
        return " ".join(WORDS[int(i)] for i in ids)

    return [{"anc": sentence(4), "pos": sentence(12), "neg": [sentence(10), sentence(14)]} for _ in range(n)]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def build_tiny_encoder(path: Path) -> Path:
    """Save a 2-layer random BERT and a word-level tokenizer; each prefix is one token."""
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    prefix_words = QUERY_PREFIX.split() + DOC_PREFIX.split()
    vocab = {token: index for index, token in enumerate(specials + sorted(set(WORDS + prefix_words)))}

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )

    torch.manual_seed(42)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=MAX_POSITIONS,
    )
    BertModel(config).save_pretrained(path)
    fast.save_pretrained(path)
    return path
