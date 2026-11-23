"""Run pinned Betterleaks against public sources, unpacked artifacts and Git history.

Reports are redacted and stored in the ignored build directory. This performs local
detection only; it neither validates credentials with providers nor uploads reports.
Run after the distribution build; this also performs the public-content audit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

from check_public import ROOT, inspect_artifacts, inspect_public, public_files


def write_member(root: Path, name: str, data: bytes) -> None:
    """Write regular archive content, rejecting paths outside the staging directory."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def main() -> None:
    """Fail on missing inputs, a version mismatch, findings or scanner errors."""
    expected = tomllib.loads((ROOT / "mise.toml").read_text())["tools"]["betterleaks"]
    version = subprocess.check_output(["betterleaks", "version"], text=True).strip().removeprefix("v")
    if version != expected:
        raise SystemExit(f"Betterleaks {expected} required; got {version}. Run mise install.")

    files = public_files()
    errors = inspect_public(files) + inspect_artifacts()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Public content and license checks passed ({len(files)} files; artifacts=True).", flush=True)
    reports = ROOT / "build" / "security"
    reports.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("BETTERLEAKS_", "GITLEAKS_"))}

    with tempfile.TemporaryDirectory(prefix="muscari-secrets-") as temporary:
        stage = Path(temporary)
        empty_ignores = stage / "ignore-files"
        empty_ignores.mkdir()
        for path in files:
            destination = stage / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

        for artifact in (ROOT / "dist").glob("*.whl"):
            with zipfile.ZipFile(artifact) as archive:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        write_member(stage / "wheel", name, archive.read(name))
        for artifact in (ROOT / "dist").glob("*.tar.gz"):
            with tarfile.open(artifact) as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        stream = archive.extractfile(member)
                        assert stream is not None
                        write_member(stage / "sdist", member.name, stream.read())

        common = [
            "--redact=100",
            "--no-banner",
            "--no-color",
            "--validation=false",
            "--ignore-gitleaks-allow",
            "--gitleaks-ignore-path",
            str(empty_ignores),
            "--exit-code=1",
            "--report-format=json",
        ]
        subprocess.run(
            ["betterleaks", "dir", str(stage), *common, "--report-path", str(reports / "files.json")],
            cwd=stage,
            env=environment,
            check=True,
        )

        if not (ROOT / ".git").exists():
            print("Git history: unavailable in this source archive; files and artifacts scanned.")
            return
        commit_count = int(subprocess.check_output(["git", "rev-list", "--all", "--count"], cwd=ROOT, text=True))
        if commit_count == 0:
            print("Git history: no commits yet; files and artifacts scanned.")
            return
        shallow = subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=ROOT,
            text=True,
        ).strip()
        if shallow == "true":
            raise SystemExit("Full Git history is required for the history scan; fetch the missing history first.")
        subprocess.run(
            [
                "betterleaks",
                "git",
                str(ROOT),
                "--log-opts=--all --full-history",
                *common,
                "--report-path",
                str(reports / "history.json"),
            ],
            cwd=stage,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
