<div align="center">

# megaRL

**RL post-training infrastructure for the [TensorPlay](https://github.com/lexing-2026/TensorPlay) ecosystem.**

<p>
  <a href="#about">About</a> ·
  <a href="#-built-on-tensorplay">Built on TensorPlay</a> ·
  <a href="#-key-features">Key Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-repository-layout">Repository Layout</a> ·
  <a href="#-documentation">Documentation</a> ·
  <a href="./CONTRIBUTING.md">Contributing</a>
</p>

<p>
  <a href="./README.md"><strong>EN</strong></a> ·
  <a href="./README.zh.md"><strong>中文</strong></a>
</p>

<p>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.12-23347A?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/rust-stable-23347A?style=flat-square&logo=rust&logoColor=white" alt="Rust stable">
  <img src="https://img.shields.io/badge/platform-Linux-23347A?style=flat-square" alt="Linux">
  <a href="https://github.com/ohtensorplay/megarl/stargazers"><img src="https://img.shields.io/github/stars/ohtensorplay/megarl?style=flat-square&label=stars" alt="stars"></a>
  <a href="https://github.com/ohtensorplay/megarl/actions/workflows/lint.yml"><img src="https://github.com/ohtensorplay/megarl/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/ohtensorplay/megarl/actions/workflows/test.yml"><img src="https://github.com/ohtensorplay/megarl/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-23347A?style=flat-square" alt="License: Apache-2.0"></a>
</p>

</div>

---

## About

megaRL is the **RL post-training layer** of the TensorPlay ecosystem. It turns
a policy model into a training loop that is asynchronous by default, keeps
tensors out of the control plane, and treats the *version of every generated
token* as a first-class value.

The design follows one boundary: **Rust is the spine, Python is the brain.**

| Layer | Language | Responsibility |
| --- | --- | --- |
| Control & data plane, weight-sync coordinator, rollout gateway, wire protocol, metrics | Rust | scheduling, versions, leases, zero-copy transport |
| Model definition, loss / advantage, optimizer, reward-model forward, algorithm iteration | Python | research productivity |
| Kernels, autograd DAG, compiler, device plugins | TensorPlay C++ | unchanged |

The dominant cost in RL post-training is orchestration and data movement —
serialization, rank-0 full gathers, straggler GPU idle, and full weight
conversion — not training compute. megaRL targets that bottleneck with a narrow
control surface and a versioned wire format, while leaving all algorithmic
semantics in Python.

> Requirements: Linux · Python ≥ 3.12 · a stable Rust toolchain. A GPU is
> needed for real rollout/training; the full unit-test suite runs on CPU.

---

## 🧱 Built on TensorPlay

| Layer | Project | Responsibility |
| --- | --- | --- |
| **RL post-training** | **megaRL** | Async control plane, staleness budget, rollout gateway, delta weight fabric |
| **Rollout engine** | [MegaLM](https://github.com/ohtensorplay/megalm) | Model execution, continuous batching, KV cache, logprobs, serving gateway |
| **Tensor & kernel foundation** | [TensorPlay](https://github.com/lexing-2026/TensorPlay) | Tensor runtime, operator dispatch, kernels, serialization, device abstraction |
| **Compiler backends** (optional) | [tensorplay-tensorrt](https://github.com/lexing-2026/tensorplay-tensorrt) and others | Compilation and backend acceleration |

TensorPlay's `PrivateUse1` device plugins let megaRL target MetaX / MLU / NPU
alongside CUDA without rewriting the control plane.

---

## ✨ Key Features

- **Token-level versioning** — every trajectory token carries
  `(policy_version, processed_logprob, sampling_params)`. Partial rollouts and
  in-flight weight updates stay mathematically usable instead of being labelled
  as one coarse trajectory version.
- **One staleness budget** — a single typed knob (`max_staleness`) with
  capacity-based admission, instead of config sprawl.
- **Out-of-band data plane** — the controller sees metadata only; tensor
  payloads travel through leased handles over shared memory / RDMA. No Ray
  tensor serialization and no GIL tax on the control path.
- **Delta-first, shard-local weight sync** — convert once on the trainer side
  into final rollout coordinates and transfer only changed bytes; no rank-0
  full staging and no global synchronization barrier.
- **Rollout as a service, plus an embedded fast path** — a Rust axum gateway
  with sticky routing for prefix locality by default; in-process megaLM
  `Engine.step()` consumption for single-node experiments. Per-token logprobs
  are preserved across both paths, with alignment guards that refuse malformed
  importance ratios.
- **Narrow backend traits** — `RolloutService`, `WeightTransport` and
  `RewardFn` are replaceable through small typed configs with stable semantics.
- **Fully asynchronous controller** — generation, reward and training overlap;
  long-tail requests support `abort | wait | keep` pause modes; zero-variance
  groups are dropped and refilled rather than wasting a forward/backward pass.
- **Agentic RL (multi-turn tool use)** — bounded `AgentLoop` episodes keep the
  exact sampled tokens, logprobs and per-token versions across turns; a
  Rust-validated `AgentTrajectory` frame and assistant-only loss masks ensure
  tool observations stay in context without receiving gradients; pluggable
  schema-validated tool registry, CLEAN/REALIGN/FORK prefix reconciliation,
  session affinity and turn-discounted rewards plug into the same controller.
  See [devdocs/agentic-rl.md](devdocs/agentic-rl.md).
- **Network tool layer** — Rust-backed `fetch` (validating DNS resolver, blocked
  private/IP-literal hosts, redirect re-validation, streaming size caps) and a
  pluggable `search` tool, plus an authenticated `ToolHTTPProxy` with dynamic
  HTTP-backed handlers and schema discovery for external agents. Ported from
  MegaLM's SSRF-safe media fetcher; patterns from the pinned AReaL / SkyRL /
  vLLM tool services.
- **Diffusion LM RL (dLLM)** — GRPO-style RL for masked and block diffusion
  language models via d1's one-step log-prob estimator and wd1's ratio-free
  objective; deterministic prompt-mask canvas, `mean`/`mean_loo`
  parameterisations, MegaLM rollout adapter, per-step denoising-trajectory
  recording/replay with an out-of-band payload store, sync + async controllers,
  continuous Flow-GRPO / DiffNFT objectives for flow-matching generators,
  JustGRPO (ICML 2026 Flexibility Trap) with exact AR likelihood and top-entropy
  fast mode, AGRPO (ICML 2026 amortized step-level GRPO) with unbiased MC
  timestep estimation, GDSD (likelihood-free guided denoiser
  self-distillation), dOPSD (trajectory-privileged on-policy
  self-distillation), T3D (few-step trajectory self-distillation via DDO),
  TIPS (arXiv 2610.22099, tiny inference-time visibility policy), ReMaRL
  (arXiv 2611.22576, learned remasking with Plackett–Luce position policy) and
  PURL (arXiv 2611.11551v1, joint unmask/edit/token likelihood), all under the
  namespaced `megarl.dllm.algorithms.<algorithm>` packages, a learned process
  value head with stepwise GAE, and a trainer path that reuses the versioned
  wire contract unchanged. Real-checkpoint GPU smoke scripts are included.
  See [devdocs/dllm-rl.md](devdocs/dllm-rl.md).
- **Reward & algorithm toolkit** — one production implementation per algorithm:
  GRPO, BNPO, Dr.GRPO, DAPO, PPO (GAE + value clip), CISPO, SAPO, LUSPO,
  VESPO and reverse-KL distillation (`reverse_kl` / `opd`) behind
  `policy_loss`; selectable k1/k2/k3/low-var KL estimators, orthogonal OPD
  advantage penalty (`opd_kl_coef`, slime/SkyRL recipe), and DPO (sigmoid/cDPO
  / IPO with optional reference model) through `dpo_update`; group/batch/none
  reward scaling, RLOO and REINFORCE baselines, token-budget packing with
  `cu_seqlens`, a batched model-based reward pool, external verifiers and a
  multimodal collator.
- **Four-signal observability** — Prometheus `/metrics` (counters, gauges, real
  histograms), optional OTLP metrics/traces, single-line JSON logs and training
  trackers. RL event semantics live in megaRL; primitives come from the shared
  `tensorplay-observability` package. See
  [devdocs/observability.md](devdocs/observability.md).
- **Production reward plane (Rust)** — semantic math verification (`1/2` ≡
  `0.5` ≡ `\frac{1}{2}`), code execution under hard CPU/memory/time limits with
  process-group kill, domain routing and a bounded retrying pool.
- **Train/inference logprob correction** — TRL-style token/sequence truncate or
  mask modes neutralize the vLLM ↔ trainer logprob gap before the importance
  ratio is formed; `sampling_logprobs` travel in the versioned wire contract.
- **Quantized weight fabric** — FP8/NVFP4 deltas in the rollout's final
  stacked-expert layout, checksummed `MGDQ` frames, and moment-free stateless
  Adam checkpoints.
- **Versioned LoRA adapters** — immutable per-version publishes, load-before-
  version-bump activation and staleness-aware GC for in-flight requests.
- **Real checkpoint/resume** — atomic `LATEST` pointer over immutable version
  directories, a SHA-256 manifest, optimizer state and RNG restoration.
- **Benchmarkable** — `megarl-bench` prints JSON throughput for the Rust reward
  plane; a multi-stage Dockerfile ships the compiled gateway and extension.

---

## 🏗 Architecture

```
L5  Python DSL / algorithms
    GRPO / BNPO / Dr.GRPO / DAPO / PPO / CISPO / SAPO / LUSPO / VESPO
    reverse_kl / OPD (k1/k2/k3 KL estimators) in losses.py + kl.py
    DPO (sigmoid / IPO) in dpo.py; GAE + RLOO in advantages.py
L4  Control plane (Rust, megarl-control)
    versioned async loop · staleness admission · pause/resume · leases
L3  Data plane (Rust, megarl-transport)
    typed TrajectoryRecord frames · shm / RDMA · zero-copy
L2  Execution backends
    Rollout (MegaLM engine / gateway) | Trainer (megaTrain core + TensorPlay FSDP/stax) | Reward
L1  Weight fabric
    delta generation · CUDA IPC / NCCL / shm / fs / RDMA · version handshake
L0  Substrate
    TensorPlay p10 / tpx / stax + MegaLM + device plugins
    (CUDA / MetaX / MLU / NPU / Vulkan / MPS)
```

The authoritative design baseline, including rationale and known gaps in the
upstream references, is [devdocs/architecture.md](devdocs/architecture.md).

---

## 🚀 Quick Start

### 1. Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

TensorPlay wheels are published on the project index. Set
`PIP_EXTRA_INDEX_URL` when installing locally:

```bash
python -m pip install -e ".[test]" \
  --extra-index-url https://download.tensorplay.cn/simple
```

> **Shared development host:** do **not** run the command above (or plain
> `maturin develop`) here — this machine imports TensorPlay from the source
> checkout via a `.pth` file, and pip re-resolving the wheel breaks the
> environment (`libnccl.so.2`). See
> [`devdocs/local-environment.md`](devdocs/local-environment.md) for the rule,
> verification command and recovery steps.

The editable install builds the PyO3/maturin extension. Build the Rust gateway
binary separately if you want to run the HTTP end-to-end tests:

```bash
cargo build --manifest-path rust/Cargo.toml -p megarl-rs --bin megarl-gateway
```

To use the embedded megaLM rollout backend, install the `rollout` extra (or a
matching `megalm-server` for the service path):

```bash
python -m pip install -e ".[rollout]"
```

### 2. Verify the install

```bash
ruff check src tests
python -m pytest tests -q
cargo check --manifest-path rust/Cargo.toml --locked
cargo test --manifest-path rust/Cargo.toml --locked
```

Measure the Rust reward plane (JSON output, suitable for CI regression tracking):

```bash
cargo run --release --manifest-path rust/Cargo.toml --bin megarl-bench -- 4096 8
```

Build the production image (gateway + compiled extension, non-root runtime):

```bash
docker build -t megarl:latest .
docker run --rm -p 8080:8080 megarl:latest
```

### 3. Contracts and control primitives

```python
from megarl import (
    RolloutRequest,
    SamplingParams,
    decode,
    encode,
    validate,
)

request = RolloutRequest(
    request_id=1,
    policy_version=0,
    prompt_token_ids=[128000, 872, 29871],
    max_new_tokens=128,
    n=8,
    sampling=SamplingParams(1.0, 1.0, 0, None, True, 0),
    session_id=None,
)
validate(request)          # checks the Python mirror against the Rust struct
frame = encode(request)    # versioned MGRL frame, metadata + handles only
kind, message = decode(frame)
assert kind == "RolloutRequest" and message == request
```

### 4. Drive a training loop

```python
import asyncio
from megarl.async_controller import AsyncController, AsyncControllerConfig
from megarl.controller import Prompt

config = AsyncControllerConfig(max_staleness=4, group_size=8)
controller = AsyncController(
    config,
    rollout=my_rollout,      # implements RolloutService
    reward_fn=my_reward_fn,  # TrajectoryRecord -> float
    trainer=my_trainer,      # implements Trainer.update
)
results = asyncio.run(controller.run([Prompt(1, [128000, 872])], steps=100))
```

Minimal, fully working implementations of all three roles live in
`tests/test_controller.py` and `tests/test_async_controller.py`.

### 5. Drive an agentic (multi-turn tool-use) loop

```python
from megarl import AgenticController, AgentConfig
from megarl.controller import ControllerConfig, Prompt
from megarl.tools import ToolRegistry, CALCULATOR_TOOL

controller = AgenticController(
    ControllerConfig(group_size=8, max_tokens_per_batch=8192),
    rollout=my_rollout,                 # RolloutService (one turn per call)
    tokenizer=my_tokenizer,             # encode(text) / decode(tokens)
    registry=ToolRegistry([CALCULATOR_TOOL]),
    reward_fn=my_agentic_reward_fn,  # AgentTrajectory -> float
    trainer=my_trainer,
    agent_config=AgentConfig(max_turns=8, max_total_tokens=4096),
    max_workers=4,  # optional: run a GRPO group's episodes in parallel
)
result = controller.step([Prompt(1, [128000, 872])])
controller.close()  # or use `with AgenticController(...) as controller:`
```

The loop keeps the exact sampled tokens/logprobs/versions per turn, appends tool
observations with `loss_mask=0`, and feeds the episode through the same
staleness-aware controller and trainer. Production knobs: canonical schemas are
cached once, tool deadlines start at submission, `AgentConfig.strict=False`
converts tool failures into error observations, and `max_workers`/`close()`
provide bounded parallelism with deterministic cleanup. See
`devdocs/agentic-rl.md`; working fakes live in `tests/test_agent.py`.

---

## 📁 Repository Layout

| Path | Contents |
| --- | --- |
| `src/megarl/contracts.py` | Versioned MGRL wire contract, mirrored from Rust |
| `src/megarl/control.py` | Rust-backed staleness budget, version ledger, lease table |
| `src/megarl/controller.py`, `async_controller.py` | Sync and async training loops |
| `src/megarl/agent.py`, `tools.py` | Agentic episode loop, tool registry and controller adapters |
| `src/megarl/nettools.py` | Network tools: hardened fetch/search and authenticated tool proxy (thin adapter over Rust) |
| `src/megarl/rollout.py`, `gateway.py`, `worker.py` | megaLM adapter and Rust HTTP gateway |
| `src/megarl/weights.py`, `rdma.py`, `moe.py` | Delta weight fabric, RDMA transport, MoE shards |
| `src/megarl/reward.py`, `losses.py`, `batching.py` | Reward pool, GRPO loss, token-budget packing |
| `src/megarl/observability.py` | RL telemetry → shared metrics/tracking |
| `grafana/` | Prometheus datasource, dashboard and alert rules |
| `rust/crates/` | `protocol`, `control`, `transport`, `gateway`, `reward`, `agent`, `nettools` |
| `devdocs/` | Accepted architecture baseline |
| `refs/` | Pinned, read-only upstream references (`UPSTREAMS.md`) |

---

## ✅ Implementation status

Implemented: the versioned wire contract with cross-language drift checks;
Rust control plane and PyO3 bindings; sync and async controllers; embedded
megaLM rollout with per-token logprobs; reward verifiers and zero-variance
filtering; GRPO loss on TensorPlay; delta weight fabric with SHA-256
verification plus shared-memory and RDMA transports; batched reward pool and
MoE shard conversion; multimodal trajectories; the Rust gateway with Prometheus
metrics and Grafana dashboards;
and backend probing.

Hardware-dependent work remains: exercising the RDMA provider against a real
NIXL/ibverbs fabric, and validating out-of-tree device plugins on MetaX / MLU /
NPU hardware. Interfaces and fallbacks are in place; only the physical devices
and fabric credentials are needed.

---

## 📚 Documentation

- [Architecture baseline](devdocs/architecture.md)
- [Agentic RL design and evidence](devdocs/agentic-rl.md)
- [Observability: metrics, OTLP, logs and trackers](devdocs/observability.md)
- [Gap-by-gap closure checklist](devdocs/gap-closure.md)
- [Pinned upstream references](refs/UPSTREAMS.md) — required evidence for any benchmark claim
- [Contributing guide](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Security policy](SECURITY.md) · [Changelog](CHANGELOG.md)
- [TensorPlay source](https://github.com/lexing-2026/TensorPlay) · [MegaLM source](https://github.com/ohtensorplay/megalm) · [TensorPlay website](https://tensorplay.cn) · [Community forum](https://forum.tensorplay.org) · [ohtensorplay organization](https://github.com/ohtensorplay)

---

## Contributing & License

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup
and review expectations. megaRL is released under the
[Apache License 2.0](LICENSE).

<div align="center">
  <p>megaRL is part of the TensorPlay ecosystem. If it's useful to you, feel free to Star and share feedback.</p>
</div>
