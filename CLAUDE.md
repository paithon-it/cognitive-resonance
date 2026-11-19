# cognitive-resonance

The experiment behind the paithon.it article *Risonanza cognitiva*: a small Python
library (`src/cognitive_resonance/`), one analysis script per experiment
(`analysis/`), the GPU training scripts (`gpu/`), the protocol (`PROTOCOL.it.md`,
English in `PROTOCOL.md`) and the results (`results/`). Tests: `uv run --extra dev
pytest -q`. What is registered in the protocol is not edited after a run: what
changes goes in its "Deviations" section.

## Security review

- **Every push**: `.githooks/pre-push` runs the repo skill `security-review-range`
  on exactly what is being pushed, and blocks on HIGH or MEDIUM findings with
  confidence of at least 8/10, or when the review cannot run. Once per clone:
  `git config core.hooksPath .githooks`. The bypass (`SKIP_SECURITY_REVIEW=1 git
  push`) is used only with the explicit ok of the repository owner. Reports are in
  `.git/security-reviews/`. The areas and the exclusions are in
  `.githooks/security-review-areas`; the hook has its own test, `bash
  tool/test/pre_push_hook_test.sh`.
- **Releases**: this repo has no production deploy. Before a version is archived
  with a DOI, run `/security-review-range <tag or sha of the previous
  version>..HEAD`, always with the explicit range, and tag the archived commit.
- **Findings**: a defect in our code is fixed here, with a test that fails first.
  A defect in code owned elsewhere (the `turing-meter` clone that
  `analysis/people.py` imports, third-party packages) is reported there, never
  patched locally.
