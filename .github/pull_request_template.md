<!-- Thank you for contributing! Please see CONTRIBUTING.md before opening a PR. -->

## Motivation

<!-- What problem does this PR solve, and why is the change needed? -->

## Modifications

<!-- Summarize the changes by component: contracts/, rust/crates/, controller,
     rollout/, weights/, reward/, docs/, CI, ... -->

## Related Issues

<!-- e.g. "Fixes #123", "Part of #456" -->

## Test Plan

<!-- Which tests did you run? Include commands and results, e.g. `pytest tests -q`,
     `cargo test --manifest-path rust/Cargo.toml --locked`. -->

## Accuracy / Performance Evidence

<!-- For algorithm or backend changes: accuracy/logprob parity checks. For
     optimizations: before/after numbers, hardware, model and request set, and
     the exact benchmark command. Delete if not applicable. -->

## Checklist

- [ ] The title and description are accurate (they become the squash-commit message).
- [ ] `ruff check src tests` and `python -m pytest tests -q` pass locally.
- [ ] `cargo check`/`cargo test --manifest-path rust/Cargo.toml --locked` pass for Rust changes.
- [ ] Wire-contract changes update both the Rust struct and the Python mirror.
- [ ] Tests and documentation were added or updated as needed.
- [ ] A user-visible change is recorded under `Unreleased` in `CHANGELOG.md`.

## Reviewer Notes

<!-- Optional: areas you would like reviewers to focus on, known limitations. -->
