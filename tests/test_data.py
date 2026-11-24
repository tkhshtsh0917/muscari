"""Text stream: column parsing, deduplication and the content-hash holdout split."""

from __future__ import annotations

import pytest
from tiny import COLUMNS, DOC_PREFIX, QUERY_PREFIX, make_rows

from muscari import TextColumn, is_holdout, iter_texts, repeat_texts


def test_column_spec_keeps_the_prefix_verbatim():
    assert TextColumn.parse("anc=検索クエリ: ") == TextColumn("anc", "検索クエリ: ")
    assert TextColumn.parse("text") == TextColumn("text", "")
    assert TextColumn.parse("q=a=b") == TextColumn("q", "a=b")

    with pytest.raises(ValueError):
        TextColumn.parse("=prefix")


def test_column_spec_can_label_the_kind_of_text():
    assert TextColumn.parse("anc:query=") == TextColumn("anc", "", "query")
    assert TextColumn.parse("anc:query") == TextColumn("anc", "", "query")
    assert TextColumn.parse("pos:document=passage: a:b") == TextColumn("pos", "passage: a:b", "document")
    assert TextColumn.parse("anc=検索クエリ: ").label is None  # a colon in the prefix is not a label

    assert TextColumn("anc", QUERY_PREFIX).kind == QUERY_PREFIX
    assert TextColumn("anc", QUERY_PREFIX, "query").kind == "query"

    with pytest.raises(ValueError):
        TextColumn.parse("anc:=prefix")
    with pytest.raises(ValueError):
        TextColumn.parse(":query=prefix")


def test_labelled_columns_yield_their_label_and_are_deduplicated_per_kind():
    rows = [{"anc": "東京 求人", "pos": "東京 求人", "neg": ["東京 求人", "名古屋 営業"]}]
    columns = [TextColumn("anc", "", "query"), TextColumn("pos", "", "document"), TextColumn("neg", "", "document")]

    assert list(iter_texts(rows, columns)) == [
        ("query", "東京 求人"),
        ("document", "東京 求人"),
        ("document", "名古屋 営業"),
    ]


def test_split_is_disjoint_deduplicated_and_decided_by_content():
    rows = make_rows(50) * 2  # every row appears twice

    train = list(iter_texts(rows, COLUMNS, holdout_pct=20, split="train"))
    held = list(iter_texts(rows, COLUMNS, holdout_pct=20, split="holdout"))
    everything = list(iter_texts(rows, COLUMNS, holdout_pct=20, split="all"))

    assert train and held
    assert len(train) == len(set(train)) and len(held) == len(set(held))
    assert set(train).isdisjoint(held)
    assert set(everything) == set(train) | set(held)
    assert all(is_holdout(text, 20) for _, text in held)
    assert not any(is_holdout(text, 20) for _, text in train)
    assert {prefix for prefix, _ in train} == {QUERY_PREFIX, DOC_PREFIX}


def test_a_text_falls_on_the_same_side_under_any_prefix():
    rows = [{"anc": "東京 求人", "pos": "東京 求人"}]
    columns = [TextColumn("anc", QUERY_PREFIX), TextColumn("pos", DOC_PREFIX)]

    kept = list(iter_texts(rows, columns, holdout_pct=50, split="all"))
    held = list(iter_texts(rows, columns, holdout_pct=50, split="holdout"))

    # The prefix is part of the deduplication key, not of the split decision.
    assert kept == [(QUERY_PREFIX, "東京 求人"), (DOC_PREFIX, "東京 求人")]
    assert len(held) in (0, 2)


def test_list_cells_are_expanded_and_unusable_cells_are_skipped():
    rows = [
        {"pos": "a", "neg": ["bb", "ccc", None, 7]},
        {"pos": None},
        {"other": "ignored"},
    ]
    columns = [TextColumn("pos"), TextColumn("neg")]

    assert list(iter_texts(rows, columns)) == [("", "a"), ("", "bb"), ("", "ccc")]
    assert list(iter_texts(rows, columns, min_chars=2)) == [("", "bb"), ("", "ccc")]


def test_dedup_can_be_disabled():
    rows = [{"pos": "a"}, {"pos": "a"}]

    assert len(list(iter_texts(rows, [TextColumn("pos")]))) == 1
    assert len(list(iter_texts(rows, [TextColumn("pos")], dedup=False))) == 2


@pytest.mark.parametrize("holdout_pct", [-1, 100])
def test_out_of_range_holdout_is_rejected(holdout_pct):
    with pytest.raises(ValueError):
        list(iter_texts([{"pos": "a"}], [TextColumn("pos")], holdout_pct=holdout_pct))
    with pytest.raises(ValueError):
        is_holdout("a", holdout_pct)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        list(iter_texts([], [TextColumn("pos")], split="test"))


def test_holdout_share_is_close_to_the_requested_percentage():
    texts = [f"text {i}" for i in range(20_000)]
    share = sum(is_holdout(text, 5) for text in texts) / len(texts)

    assert 0.04 < share < 0.06
    assert not any(is_holdout(text, 0) for text in texts[:100])


def test_repeat_restarts_the_source_for_each_epoch():
    calls = []

    def make_iter():
        calls.append(None)
        return iter([("", "a"), ("", "b")])

    assert list(repeat_texts(make_iter, max_epochs=3)) == [("", "a"), ("", "b")] * 3
    assert len(calls) == 3


def test_repeat_refuses_an_empty_source():
    with pytest.raises(ValueError, match="yielded nothing"):
        next(repeat_texts(lambda: iter([])))
