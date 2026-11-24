# muscari development rules

- Read `CONTRIBUTING.md` for project settings, version ownership and verification commands.
- Keep package code, examples, benchmarks, tests and development scripts separate. The SAELens backend belongs to the `saelens` extra and the Sentence Transformers head to the `retrieval` extra; both must stay optional at import time; development tools belong to the `dev` dependency group.
- Keep the two trainers on one activation stream and one export format. A change to the export format needs a `format_version` decision, matching documentation and a round-trip test.
- The sparse backward pass must stay numerically equal to plain autograd. Change it only together with the dense-oracle and gradient-check tests.
- Explain responsibilities, shapes, failure conditions and numerical constraints in docstrings. Keep attribution and development history in documentation, preserving required copyright notices.
- State what a recipe is taken from. Separate what a paper or model card says from this project's own choices, and do not describe a reimplementation as a reproduction.
- Update English and Japanese public documentation together. Public documentation must be sufficient without personal notes, private files or machine-specific paths. Outputs must not record local directory layouts.
- Use `mise run setup` after an intentional dependency change; use `mise run check` for validation. Checks must not resolve or update dependencies, and must not download models or datasets.
- Add behaviour-based regression tests for fixes and examine related input boundaries. Do not suppress checks globally to hide failures.
- Keep source-only edits, commit creation, history rewriting, push, tags and publication as distinct authorizations. Preserve other contributors' work.
- Report actual commands and results. CI configuration is not evidence that hosted CI ran; tests on a random encoder are not evidence of SAE quality on a real one; a CPU benchmark is not evidence about GPUs.
- Use one argument per line in multiline calls, including documentation examples; keep simple short calls compact. Run the pinned Betterleaks scan as part of the shared checks.
