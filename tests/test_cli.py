"""Command line: training, provenance, resumption and evaluation, end to end and offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tiny import DOC_PREFIX, QUERY_PREFIX, make_rows, write_jsonl

from muscari import load_sae
from muscari.cli import load_rows, main

COLUMN_SPECS = [f"anc={QUERY_PREFIX}", f"pos={DOC_PREFIX}", f"neg={DOC_PREFIX}"]


def train_arguments(model: Path, data: Path, out: Path, *, trainer: str = "topk") -> list[str]:
    return [
        "train-sae",
        *("--trainer", trainer),
        *("--out", str(out)),
        *("--model", str(model)),
        *("--data-files", str(data)),
        "--columns",
        *COLUMN_SPECS,
        *("--max-length", "32"),
        *("--text-batch-size", "8"),
        *("--num-latents", "128"),
        *("--k", "4"),
        *("--total-tokens", "20000"),
        *("--batch-tokens", "128"),
        *("--buffer-tokens", "512"),
        *("--eval-tokens", "1000"),
        *("--dead-after-tokens", "2000"),
        *("--holdout-pct", "10"),
        *("--device", "cpu"),
    ]


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    return write_jsonl(tmp_path / "rows.jsonl", make_rows(300))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_writes_an_evaluated_sae_with_provenance(tiny_model_dir, data_file, tmp_path, capsys):
    out = tmp_path / "run"
    main(train_arguments(tiny_model_dir, data_file, out))

    metrics = read_json(out / "metrics.json")
    assert 0 < metrics["fvu"] < 1.0
    assert metrics["eval_tokens"] == 1000 and metrics["l0"] <= 4
    assert metrics["tokens_encoded"] >= 20_000

    config = read_json(out / "sae_config.json")
    assert config["trainer"] == "topk" and config["tokens_trained"] == 156 * 128
    assert (config["num_latents"], config["k"], config["d_in"]) == (128, 4, 32)
    assert config["columns"] == {"anc": QUERY_PREFIX, "pos": DOC_PREFIX, "neg": DOC_PREFIX}
    assert config["layer"] == "last_hidden_state" and config["prefix_tokens_excluded"] is True

    log_lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"step"')]
    assert log_lines[-1]["step"] == 156
    assert load_sae(out).num_latents == 128


def test_outputs_record_file_names_without_directories(tiny_model_dir, data_file, tmp_path):
    out = tmp_path / "run"
    main(train_arguments(tiny_model_dir, data_file, out))

    config_text = (out / "sae_config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)

    assert config["base_model"] == tiny_model_dir.name
    assert config["dataset"] == ["rows.jsonl"]
    assert str(tmp_path) not in config_text


def test_resume_continues_from_the_checkpoint(tiny_model_dir, data_file, tmp_path, capsys):
    out = tmp_path / "run"
    arguments = train_arguments(tiny_model_dir, data_file, out)
    main([*arguments, "--checkpoint-every", "50"])
    capsys.readouterr()

    checkpoint = torch.load(out / "checkpoint.pt", weights_only=True)
    assert checkpoint["step"] == 156 and checkpoint["data"]["texts_seen"] > 0

    extended = [argument if argument != "20000" else "40000" for argument in arguments]
    main([*extended, "--checkpoint-every", "50", "--resume"])
    captured = capsys.readouterr()

    steps = [json.loads(line)["step"] for line in captured.out.splitlines() if line.startswith('{"step"')]
    assert steps[0] == 200 and steps[-1] == 312
    assert "resumed at step 156" in captured.err
    assert read_json(out / "sae_config.json")["tokens_trained"] == 312 * 128


def test_resume_without_a_checkpoint_is_refused(tiny_model_dir, data_file, tmp_path):
    arguments = train_arguments(tiny_model_dir, data_file, tmp_path / "run")

    with pytest.raises(SystemExit, match="none was found"):
        main([*arguments, "--resume"])


def test_resume_is_refused_for_the_saelens_trainer(tiny_model_dir, data_file, tmp_path):
    arguments = train_arguments(tiny_model_dir, data_file, tmp_path / "run", trainer="saelens")

    with pytest.raises(SystemExit, match="only supported by the topk trainer"):
        main([*arguments, "--resume"])


@pytest.mark.parametrize("architecture", ["topk", "batchtopk"])
def test_saelens_trainer_runs_end_to_end(tiny_model_dir, data_file, tmp_path, architecture):
    pytest.importorskip("sae_lens")
    out = tmp_path / "run"
    main([*train_arguments(tiny_model_dir, data_file, out, trainer="saelens"), "--architecture", architecture])

    assert read_json(out / "sae_config.json")["trainer"] == f"saelens-{architecture}"
    assert 0 < read_json(out / "metrics.json")["fvu"] < 1.5


def test_plain_text_files_provide_a_text_column(tiny_model_dir, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(str(row["pos"]) for row in make_rows(400)) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"
    arguments = train_arguments(tiny_model_dir, corpus, out)
    start = arguments.index("--columns")
    arguments[start + 1 : start + 4] = [f"text={DOC_PREFIX}"]

    main(arguments)

    assert read_json(out / "sae_config.json")["columns"] == {"text": DOC_PREFIX}


def test_evaluation_takes_encoder_settings_from_the_sae(tiny_model_dir, data_file, tmp_path, capsys):
    out = tmp_path / "run"
    main(train_arguments(tiny_model_dir, data_file, out))
    capsys.readouterr()

    report = tmp_path / "eval.json"
    main(
        [
            "eval-sae",
            *("--sae", str(out)),
            *("--model", str(tiny_model_dir)),
            *("--data-files", str(data_file)),
            *("--eval-tokens", "1000"),
            *("--text-batch-size", "8"),
            *("--device", "cpu"),
            *("--out", str(report)),
        ]
    )

    evaluated = read_json(report)
    trained = read_json(out / "metrics.json")
    assert evaluated["part"] == "holdout" and evaluated["holdout_pct"] == 10
    # Same held-out texts, same prefixes, same truncation as the evaluation after training.
    assert evaluated["fvu"] == pytest.approx(trained["fvu"], rel=1e-5)
    assert evaluated["l0"] == pytest.approx(trained["l0"])
    assert json.loads(capsys.readouterr().out) == evaluated


def test_evaluation_part_can_be_overridden(tiny_model_dir, data_file, tmp_path, capsys):
    out = tmp_path / "run"
    main(train_arguments(tiny_model_dir, data_file, out))
    capsys.readouterr()

    main(
        [
            "eval-sae",
            *("--sae", str(out)),
            *("--model", str(tiny_model_dir)),
            *("--data-files", str(data_file)),
            *("--part", "all"),
            *("--eval-tokens", "500"),
            *("--device", "cpu"),
        ]
    )

    assert json.loads(capsys.readouterr().out)["part"] == "all"


def test_rows_are_shuffled_reproducibly_and_keep_only_wanted_columns(data_file):
    def load(seed: int) -> list[dict]:
        rows = load_rows(
            dataset=None,
            subsets=None,
            data_files=[str(data_file)],
            split="train",
            column_names=["anc", "pos"],
            seed=seed,
        )
        return list(rows)

    original = [row["anc"] for row in make_rows(300)]
    first = load(0)

    assert set(first[0]) == {"anc", "pos"}
    assert [row["anc"] for row in first] != original
    assert sorted(row["anc"] for row in first) == sorted(original)
    assert first == load(0)
    assert first != load(1)


def test_unusable_data_files_are_rejected(data_file, tmp_path):
    arguments = {"dataset": None, "subsets": None, "split": "train", "seed": 0}

    with pytest.raises(ValueError, match="one extension"):
        load_rows(data_files=["rows.tsv"], column_names=["anc"], **arguments)
    with pytest.raises(ValueError, match="one extension"):
        load_rows(data_files=["rows.jsonl", "rows.csv"], column_names=["anc"], **arguments)
    with pytest.raises(ValueError, match="None of the columns"):
        load_rows(data_files=[str(data_file)], column_names=["missing"], **arguments)


def test_missing_required_arguments_exit_with_usage(capsys):
    with pytest.raises(SystemExit):
        main(["train-sae", "--trainer", "topk", "--out", "x"])

    assert "--model" in capsys.readouterr().err
