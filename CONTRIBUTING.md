# Contributing to megaRL

Thanks for taking the time to improve megaRL. This document describes the
development setup, repository layout, checks, and pull-request process.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
For security vulnerabilities, please follow [SECURITY.md](SECURITY.md) instead
of opening a public issue.

## 🌱 Your first PR

New here? Pick an issue labelled
[`good first issue`](https://github.com/ohtensorplay/megarl/labels/good%20first%20issue),
comment that you would like to work on it, then:

1. Follow [Development setup](#development-setup) and create a branch.
2. Make one small, focused change — one issue per PR.
3. Run the [Checks](#checks) locally.
4. Open a PR against `main` with the default template and link the issue
   (`Fixes #123`).
5. Respond to review feedback — a few rounds of iteration are normal.

If you have never opened a PR before, GitHub's
[First Pull Request course](https://skills.github.com/#first-day-on-github) is a
good starting point. Ask task-specific questions in the issue; general
questions belong in the [community forum](https://forum.tensorplay.org).

## Development setup

Requirements:

- Python 3.12 or newer
- A stable Rust toolchain (the Python extension is a PyO3/maturin module)
- Linux with CUDA-capable hardware for GPU rollout and training; CPU-only
  development and the full unit-test suite work without a GPU

TensorPlay wheels are published on the project index, not only on PyPI, so set
`PIP_EXTRA_INDEX_URL` when installing:

```bash
git clone https://github.com/ohtensorplay/megarl.git
cd megarl
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]" \
  --extra-index-url https://download.tensorplay.cn/simple
```

The editable install builds the Rust extension through maturin. To work on the
Rust side directly, use `cargo` in the `rust/` workspace:

```bash
cargo check --manifest-path rust/Cargo.toml --locked
cargo test  --manifest-path rust/Cargo.toml --locked
```

The Rust gateway binary used by the HTTP tests is built with:

```bash
cargo build --manifest-path rust/Cargo.toml -p megarl-rs --bin megarl-gateway
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/megarl/contracts.py` | Authoritative Python mirror of the versioned MGRL wire contract |
| `src/megarl/control.py` | Rust-backed staleness budget, version ledger and lease table |
| `src/megarl/controller.py`, `async_controller.py` | Sync and fully asynchronous training loops |
| `src/megarl/rollout.py`, `gateway.py`, `worker.py` | megaLM rollout adapter and Rust HTTP gateway |
| `src/megarl/weights.py`, `rdma.py`, `moe.py` | Delta weight fabric, RDMA transport, MoE shard conversion |
| `src/megarl/reward.py`, `losses.py`, `batching.py` | Reward/verifier pool, GRPO loss, token-budget packing |
| `src/megarl/observability.py`, `grafana/` | RL telemetry, metrics and dashboards |
| `rust/crates/protocol` | Wire frame definition and validation (source of truth) |
| `rust/crates/control` | Staleness budget, version ledger, leases |
| `rust/crates/transport` | Weight delta planning, RDMA provider |
| `rust/crates/gateway` | axum rollout gateway |
| `tests/` | Unit tests (`test_*.py`) |
| `devdocs/` | Accepted architecture baseline |
| `refs/` | Pinned, read-only upstream reference checkouts (see `UPSTREAMS.md`) |

Keep new code in the matching layer. In particular, public wire shapes belong
in `contracts.py` **and** `rust/crates/protocol` — the cross-language drift test
must be kept green. Runtime code must never import from `refs/`.

## Checks

Run these before opening a pull request — the same gates run in CI:

```bash
ruff check src tests                                   # syntax / undefined-name gate
python -m pytest tests -q                              # full suite, CPU-only
cargo check --manifest-path rust/Cargo.toml --locked   # Rust compile gate
cargo test  --manifest-path rust/Cargo.toml --locked   # Rust unit tests
```

To install the repository hygiene hooks:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

A change to the wire contract requires updating both the Rust struct and the
Python dataclass plus its `decode()` handling. A change to upstream benchmark
comparisons requires updating the commit pins in `refs/UPSTREAMS.md` in the
same commit.

## Pull requests

- Use the default PR template and keep the title/description accurate — they
  become the squash-commit message used when bisecting.
- Keep PRs focused. Separate features, fixes, and refactors into separate PRs.
- Add or update tests for behavioral changes; update `devdocs/architecture.md`
  when an accepted design decision changes.
- For performance work, include before/after numbers, the model and request
  set, hardware/software versions, and the exact command, following the
  evidence requirements in `refs/UPSTREAMS.md`.
- Add an entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for
  user-visible changes.
- AI-assisted contributions are welcome, but authors are expected to
  understand and be able to explain their changes during review.

CI runs on pull requests into `main` and on pushes to `main`. Draft PRs are
supported; mark a PR ready for review when you want the full check suite and
reviewer attention.

## Releases

Versions follow `major.minor.patch` in `pyproject.toml` and
`rust/Cargo.toml`. Maintainers publish by pushing a `v*` tag, which triggers
the maturin build-and-publish workflow for wheels and the source distribution.
Update `CHANGELOG.md` first.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
