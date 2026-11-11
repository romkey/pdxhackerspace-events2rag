# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- Deliberate differences from the rules: Python 3.14 rather than 3.15
  (onnxruntime has no 3.15 wheels yet); no `/healthz` or compose health
  check, because the ingester listens on no port; the repo is
  `pdxhackerspace-events2rag` while the package, command and
  distribution are `events2rag`.
