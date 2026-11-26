# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rollout offload pause/resume lifecycle: versioned `LifecycleCommand` /
  `LifecycleAck` wire messages (Kinds 8/9), a Rust gateway orchestrator with
  epoch guarding, idempotent repeats, compensating rollback and a routing fence
  (paused workers are never routed), and worker-side `EngineLifecycle` callbacks
  that drain in-flight generation before offloading. Python stays a thin HTTP
  and callback adapter; state is exposed at `/rollout/lifecycle` and as
  `megarl_lifecycle_*` metrics.
- Weight-sync phase timing (Q6): workers echo a versioned `WeightUpdateAck`
  (protocol kind 10) with pause/barrier/transfer/apply durations; the gateway
  combines it with route/wire measurements and exposes `weight_reload_*_ms`
  phase counters in `/metrics` and the `/rollout/weights` response.
- Paced asynchronous admission (Q4): `AsyncController` now keeps deferred
  prompts in an `asyncio.Condition`-guarded FIFO and releases them through the
  Rust `AsyncPacer` in bounded batches, with priority over new arrivals,
  backoff from `retry_after_ms()` and a `wait_for_capacity()` driver API.
- Checkpointed rollout resume state (Q9): trainer manifests persist
  `prompt_cursor` and `model_version`; both controllers advance/restore them
  and checkpoint after the weight reload so a resumed run does not replay
  consumed prompts against a stale engine.
- Syscall-level reward-sandbox hardening: a fail-closed seccomp-BPF syscall
  blacklist (networking, ptrace, mount/namespace, keyring, module loading, host
  admin) installed before `exec`, plus an optional Firejail wrapper with a
  network-less private namespace. Both are configurable per request or by
  environment, reported in grade/verify results and `/sandbox/profile`, and
  covered by end-to-end tests that prove `socket()` returns `EPERM` under the
  filter.

- RL observability layer on top of the shared `tensorplay-observability`
  package (`tensorplay-observability>=0.1.0` dependency): `Telemetry.emit()`
  still writes the same JSONL events but now also mirrors controller events into
  `megarl_*` counters, gauges and real histograms, with bounded events, a
  `Timer`, `render_prometheus()`, `configure()` and re-exported `Tracking`.
  Documented in `devdocs/observability.md` with a Grafana dashboard, alert
  rules and datasource provisioning under `grafana/`. Optional `otel` and
  `tracking` extras enable OTLP export and training-trackers fan-out.
- Recycled pinned host-memory pool (`src/megarl/hostmem.py`), ported from the
  pinned ROLL `HostMemPool`: exact-size `(numel, dtype)` free lists,
  `cudaHostRegister` page locking via `torch.cuda.cudart()`, strong references
  keyed by storage identity with view-aware, idempotent recycling, ROLL-style
  pinned-byte stats and a process-wide singleton. `PinnedPayloadStore` in
  `src/megarl/dataplane.py` stages payload bytes in those recycled pages and
  runs under `LeasedPayloadQueue`. `tests/test_hostmem.py` verifies real pinning
  and pointer-stable reuse on CUDA hardware; pageable mode keeps CPU-only tests
  runnable.

- Agentic RL layer: multi-turn tool-use episodes with a Rust-validated
  `AgentTrajectory` wire message (Kind 7), an `AgentLoop` that preserves exact
  sampled tokens/logprobs/versions per turn, CLEAN/REALIGN/FORK prefix
  reconciliation, bounded turns/tokens/wall-clock, turn-discounted rewards,
  and `AgenticController` / `AsyncAgenticController` adapters reusing the
  existing staleness, reward, packing, loss and trainer pipeline.
- Rust `megarl-agent` crate: all-or-nothing JSON/OpenAI/XML tool-call parsing
  and a dependency-free JSON-Schema subset validator exposed through PyO3;
  Python `ToolRegistry` with schema-checked parallel dispatch, structured
  timeouts/failures, result truncation, plus built-in calculator and
  resource-limited sandboxed Python tools.
- Assistant-only loss masks for agentic batches: `RLSample.loss_mask` flows from
  `AgentTrajectory` through `collate` into `TensorPlayTrainer`, so tool
  observations stay in context without receiving policy gradients (verified
  with a real TensorPlay gradient test).
- Design and evidence document `devdocs/agentic-rl.md` mapping each decision to
  the pinned AReaL / ROLL / rllm / SkyRL / slime / trl / verl references.

### Added (network tool layer)

- Rust `megarl-nettools` crate (port of MegaLM's
  `remote_media.rs`) provides an SSRF-hardened HTTP core — validating DNS
  resolver, IP-literal guard, per-hop redirect re-validation, layered timeouts,
  classified retries and streaming size caps — exposed through the
  `http_request_json` PyO3 binding. Python `nettools.py` builds `fetch` and
  pluggable `search` tools plus an authenticated `ToolHTTPProxy` with dynamic
  HTTP-backed handlers on top of it.

### Changed (production hardening)

- **Trainer migrated onto the `megatrain` core.** `TensorPlayTrainer` now
  subclasses `megatrain.BaseTrainer` (new `megatrain>=0.1,<0.2` dependency), so
  optimizer construction, FSDP, compile, autocast/GradScaler, `no_sync`
  gradient accumulation, gradient clipping, scheduler stepping and atomic
  checkpoint/resume come from the shared core. `TrainerConfig` subclasses
  `megatrain.TrainerConfig`, inheriting `amp`, `max_grad_norm`, `seed`,
  `ce_chunk_tokens` and future core fields; `rl_config` is the same object.
  `update()`/`save_checkpoint()`/`load_checkpoint()` keep their signatures, so
  both controllers are unchanged. The RL normalization families are mapped to
  the core `_keep_tokens` hook (rows for GRPO/SAPO/LUSPO/Dr.GRPO, fresh tokens
  for all token-mean objectives); PPO's critic is evaluated per microbatch
  (fixes a shared-graph double-backward under accumulation), and the frozen
  teacher is forwarded only for reverse-KL/OPD. `dpo_update` gained the same
  accumulation/scaler/clipping path. `kl.py` and `stateless_adam.py` are now
  re-exports of the core modules.
- `Tool` caches its canonical JSON-Schema once and validates calls from the
  Rust parser's canonical arguments, avoiding per-call re-serialization.
- `ToolRegistry.execute` starts each call's deadline at submission (not at
  collection) and bounds retries in the worker; `Tool.retries` is now honoured.
- `AgentRollout(loop, max_workers=N)` runs a GRPO group's episodes in a reusable
  thread pool and yields results in order; exposed as `max_workers` on both
  agentic controllers. `Telemetry.emit` is lock-guarded for parallel episodes.
- `AgentLoop`, `AgentRollout`, `ToolRegistry` and both agentic controllers
  support `close()` and context-manager cleanup. `AgentConfig.strict=False`
  turns tool failures into error observations instead of aborting the episode.

- Distillation and preference objectives from the pinned references: sampled
  token reverse-KL loss (`loss_type="reverse_kl"`, with `opd` alias) driven by a
  frozen teacher model, selectable `k1/k2/k3/low_var_kl` estimators in
  `src/megarl/kl.py`, and the orthogonal OPD advantage penalty
  (`opd_kl_coef`, slime/SkyRL/ROLL recipe) available on every policy loss.
- DPO in `src/megarl/dpo.py` (sigmoid/cDPO and IPO, optional frozen reference
  model, label smoothing, length normalization) with `TensorPlayTrainer.dpo_update`
  and best/worst pair construction in both controllers (`ControllerConfig.dpo`).

- Q8 closure: quantized weight deltas (blockwise FP8 E4M3/E5M2 via TensorPlay
  kernels, NVFP4 E2M1 nibble codec) with checksummed `MGDQ` frames and
  byte-exact stacked-expert split/merge; stateless Adam with moment-free
  checkpoints; versioned LoRA adapter publish/activate/GC with restart-safe
  manifests and load-before-version-bump ordering.
- Optimizer moments now checkpoint in MEGA (stringified parameter indices)
  instead of an opaque pickle blob, with checksum verification.
- Full policy-gradient algorithm suite on TensorPlay: GRPO, BNPO, Dr.GRPO,
  DAPO, PPO (GAE + clipped value loss), CISPO, SAPO, LUSPO and VESPO behind a
  single `policy_loss` entry point, with clip-higher, two-sided `delta` clip,
  token/sequence importance levels, temperature/gamma weighting, off-policy
  sequence masking and per-algorithm diagnostics.
- Advantage estimators and reward-scaling modes: group (GRPO), batch
  (PPO-lite), none (Dr.GRPO), RLOO leave-one-out, REINFORCE mean baseline and
  token-level GAE with an explicit bootstrap slot; controllers select the
  estimator through typed config.
- Production trainer with real optimizer steps, gradient accumulation, adaptive
  KL, token-level staleness masking and atomic checkpoint/resume.
- Binary weight-delta protocol with SHA-256 checksums, sparse block diffs,
  shard-aware descriptors and a monotonic version handshake.
- Hardened Rust gateway with worker pools, sticky routing, bearer-token auth,
  timeouts, frame-size limits and retry metrics.
- POSIX shared-memory data plane with leased handles, checksummed slots and
  cross-process payload reads.
- Rust reward crate: math equivalence verification, sandboxed code execution
  with resource limits, domain routing and a bounded retrying pool, exposed to
  Python through PyO3.
- Four-mode train/inference logprob importance correction with NaN
  neutralization, carried through the wire contract, batch and GRPO loss.
- Multi-stage production Dockerfile and Rust throughput benchmark binary.
- AReaL-style staleness re-basing on checkpoint restore, exposed through the
  Rust control plane and both controllers.
- Per-reference gap-closure checklist with evidence, deferrals and acceptance
  criteria (`devdocs/gap-closure.md`).

- GitHub community infrastructure: issue and pull-request templates, code
  ownership, Dependabot, code of conduct, security policy and contributing
  guide.
- GitHub Actions workflows for linting, Python/Rust unit tests and wheel/sdist
  publishing via maturin.
- Bilingual README (English / 中文) and Apache-2.0 licensing.

### Removed

- The pre-release `version_1/` sample checkpoint and the `optimizer.pkl` legacy
  read path. The project has not shipped, so checkpoints use only
  `model.mega` / `optimizer.mega` / `scheduler.mega` / artifact files.
- All bespoke dashboards: the offline HTML report (`src/megarl/dashboard.py`
  and its test), the gateway's `/dashboard` page, and the legacy JSON metrics
  endpoint (`GET /metrics.json`). The gateway keeps `/health` and Prometheus
  `/metrics`; training curves go to the multi-backend `Tracking` layer and
  Grafana dashboards under `grafana/`.

## [0.1.0]

### Added

- Versioned MGRL wire contract with per-token policy versions and out-of-band
  payload handles, mirrored in Python and validated against the Rust definition.
- Rust control plane (staleness budget, version ledger, lease table) with PyO3
  bindings.
- Synchronous and fully asynchronous controllers with zero-variance group
  dropping, token-budget packing and GRPO loss on TensorPlay.
- Embedded megaLM rollout adapter with per-token processed logprobs and
  alignment guards; axum rollout gateway with a Python frame worker.
- Delta weight fabric with SHA-256 verification, filesystem and shared-memory
  transports, plus an RDMA one-sided write transport with tensor-parallel shard
  mapping.
- Reward verifiers, batched model-based reward pool, expert-parallel MoE delta
  conversion and a multimodal trajectory wire message with a modality-aware
  collator.
- Live Rust dashboard endpoint, offline HTML telemetry report and TensorPlay
  backend probing.

[Unreleased]: https://github.com/ohtensorplay/megarl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ohtensorplay/megarl/releases/tag/v0.1.0
