"""Package metadata and the public import surface."""

from __future__ import annotations

import tomllib
from pathlib import Path

import muscari

ROOT = Path(__file__).resolve().parents[1]


def test_exported_version_matches_the_project_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert muscari.__version__ == project["version"]


def test_every_public_name_is_importable():
    assert all(hasattr(muscari, name) for name in muscari.__all__)
    assert sorted(muscari.__all__) == muscari.__all__


def test_typing_marker_is_shipped_with_the_package():
    assert (Path(muscari.__file__).parent / "py.typed").is_file()
