# Contributing to muscari

[日本語](CONTRIBUTING.ja.md)

## Project settings

| Setting | Source of truth |
| --- | --- |
| Purpose and deliverable | A Python library and the `muscari` command that train, export and evaluate SAEs on text-encoder hidden states. See [design](docs/design.md) |
| Target environments | One process on one device: CPU, CUDA or Apple MPS. See [validation status](README.md#validation-status) for what has run where |
| Python | Development version in `.python-version`; supported minimum in `pyproject.toml` |
| Tools | mise pins uv and Betterleaks in `mise.toml` |
| Dependencies | Runtime ranges and the `saelens` and `retrieval` extras in `pyproject.toml`. Developer tools use the `dev` group |
| Reproducibility | `uv.lock` pins the main development and CI environment |
| Quality | Ruff formatting and lint, mypy, pytest, package build, public-content and license audit, Betterleaks |
| Documentation | English and Japanese README, design guide, examples guide and this file |
| Distribution | Source repository only. No package publishing job is configured |
| License | Apache-2.0; LICENSE and NOTICE are included in the wheel and the sdist |

The package version is defined by `project.version`; the exported `__version__` must agree. Changes to the export format, the token selection rule, the loss or a default require matching documentation and behavioural tests.

## Setup and validation

Install [mise](https://mise.jdx.dev/getting-started.html), inspect the checkout configuration, then run:

```bash
git clone https://github.com/tkhshtsh0917/muscari.git
cd muscari

mise trust
mise install
mise run setup
mise run check
```

`setup` installs the locked dependencies, the `saelens` and `retrieval` extras and the `dev` group. `check` runs formatting, lint, mypy, the tests, the package build and the audits with `uv run --frozen --no-sync`, so it never resolves or updates dependencies. For an intentional dependency change, edit `pyproject.toml`, run `mise exec -- uv lock`, review the diff, then run setup and check. `mise run format` applies formatting.

The main CI job runs the same locked setup and check tasks. Separate compatibility jobs resolve the lowest supported versions on Python 3.11 without the SAELens extra, and the newest versions on Python 3.13, record what was installed and run the tests. Actions are pinned to commit SHAs; update the SHA and the version comment together.

The mise environment sets `HF_HUB_OFFLINE=1`. No check downloads a model or a dataset. For a run that needs the Hugging Face Hub, set the variable inside the mise environment:

```bash
mise exec -- env HF_HUB_OFFLINE=0 uv run --frozen --no-sync muscari train-sae ...
```

Writing `HF_HUB_OFFLINE=0` in front of `mise exec` has no effect, because mise applies `mise.toml` afterwards.

## Tests

Tests at the root of `tests/` cover the library with a randomly initialised two-layer BERT and synthetic activations. They check numerics and interfaces: the sparse backward pass against dense autograd and numerical differentiation, exact checkpoint resumption, the export against SAELens' own inference model, and the command line end to end. `tests/tooling/` checks the development audits. The SAELens tests and the retrieval-head tests are skipped when their extra is not installed.

On a machine where PyTorch can use CUDA or Apple MPS, run the hardware tests as well:

```bash
mise exec -- uv run --frozen --no-sync pytest -q -m accelerator -rs
```

They compare losses, gradients, activations and the evaluation statistics with the CPU, train a small SAE on the device and, on CUDA, check bfloat16 autocast. They are skipped where neither device exists, which includes hosted CI. Report which groups ran and which were skipped.

## Offline example and benchmark

`mise run smoke` trains and evaluates a tiny SAE with both trainers. It needs no download and is optional: the standard check already runs the example through the tests.

`benchmarks/sae_step.py` times one optimisation step with the sparse backward pass or with dense autograd:

```bash
mise exec -- uv run --frozen --no-sync python benchmarks/sae_step.py --mode sparse
mise exec -- uv run --frozen --no-sync python benchmarks/sae_step.py --mode dense --batch-tokens 2048
```

It uses random activations, so only time and memory are meaningful. Keep the device, thread count, arguments and library versions with any number you publish, and state what was not run.

## Reviews and readable code

Keep responsibilities separate across `src/muscari`, `examples`, `benchmarks`, `tests` and `scripts`. Use meaningful blank lines between input validation, computation and state changes. In multiline calls, put one argument per line with trailing commas; keep simple short calls compact. Explain responsibilities, shapes, failure conditions and numerical constraints in docstrings.

Fixes need tests that would fail before the fix, preferably against an independent oracle such as a dense reference or a hand-built SAE with a known answer. Cover relevant boundaries without repeating the same property across every parameter combination. Tests on a random encoder do not establish SAE quality on a real one. Do not suppress a check globally to hide a failure.

Update the English and Japanese documents together. Public files must be sufficient without local notes and must not contain personal paths, private links, tokens or keys. Agent-specific rules are in [AGENTS.md](AGENTS.md).

## Builds, licensing and secret scanning

`mise run build` produces one wheel and one sdist in `dist`. The `secrets` task first performs the public-content and license audit, then runs Betterleaks; `check` invokes this combined task once. `mise run audit` runs the audit alone.

The audit checks the license metadata and the byte-for-byte LICENSE and NOTICE contents, the typing marker, the documents in both languages and their relative links, and that every sdist member belongs to the public source inventory or to named build-generated metadata. Setuptools 77.0.3 or later is required for PEP 639 metadata. Remove stale artifacts from `dist` when the package version changes.

Betterleaks uses the pinned built-in rules and scans the public files, the extracted artifacts and all locally available Git refs. Provider validation is disabled; findings and scanner errors fail the task. Redacted reports stay in the ignored `build/security/`. Shallow clones fail the history check: fetch the missing history before retrying. Pattern detection does not guarantee that no secret exists.

Dependencies retain their own licenses. Preserve required third-party notices if you incorporate their material. Before any package release, confirm hosted CI, inspect the artifacts and their licensing, and follow the maintainer's release process.
