<div align="center">

# megaRL

**[TensorPlay](https://github.com/lexing-2026/TensorPlay) 生态的 RL 后训练基础设施。**

<p>
  <a href="#关于">关于</a> ·
  <a href="#-基于-tensorplay-构建">技术栈</a> ·
  <a href="#-核心特性">核心特性</a> ·
  <a href="#-架构">架构</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-仓库结构">仓库结构</a> ·
  <a href="#-文档">文档</a> ·
  <a href="./CONTRIBUTING.md">参与贡献</a>
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

## 关于

megaRL 是 TensorPlay 生态的 **RL 后训练层**。它把策略模型变成一个默认异步的训练循环：张量不进入控制面，**每个生成 token 的版本**都是一等公民。

设计遵循一条边界：**Rust 是骨架，Python 是大脑。**

| 层 | 语言 | 职责 |
| --- | --- | --- |
| 控制面 / 数据面、权重同步协调、rollout 网关、线协议、指标 | Rust | 调度、版本、租约、零拷贝传输 |
| 模型定义、损失 / 优势、优化器、奖励模型前向、算法迭代 | Python | 研究生产力 |
| 算子、autograd DAG、编译器、设备插件 | TensorPlay C++ | 保持不变 |

RL 后训练的主要开销在编排和数据搬运——序列化、rank-0 全量收集、掉队 GPU 空转、全量权重转换——而不是训练计算。megaRL 用窄控制面和版本化线协议直接针对这个瓶颈，同时把所有算法语义留在 Python。

> 环境要求：Linux · Python ≥ 3.12 · 稳定的 Rust 工具链。真实 rollout/训练需要 GPU，但完整单元测试可在 CPU 上运行。

---

## 🧱 基于 TensorPlay 构建

| 层 | 项目 | 职责 |
| --- | --- | --- |
| **RL 后训练** | **megaRL** | 异步控制面、陈旧度预算、rollout 网关、增量权重传输 |
| **Rollout 引擎** | [MegaLM](https://github.com/ohtensorplay/megalm) | 模型执行、连续批处理、KV 缓存、logprobs、服务网关 |
| **张量与算子基础** | [TensorPlay](https://github.com/lexing-2026/TensorPlay) | 张量运行时、算子分发、内核、序列化、设备抽象 |
| **编译后端**（可选） | [tensorplay-tensorrt](https://github.com/lexing-2026/tensorplay-tensorrt) 等 | TensorPlay 的编译与后端加速 |

TensorPlay 的 `PrivateUse1` 设备插件让 megaRL 无需重写控制面即可支持 MetaX / MLU / NPU（与 CUDA 并列）。

---

## ✨ 核心特性

- **Token 级版本化**——每条轨迹的每个 token 都携带 `(policy_version, processed_logprob, sampling_params)`。部分 rollout 和进行中的权重更新因此保持数学可用，而不是被粗糙地标成同一个轨迹版本。
- **单一陈旧度预算**——用一个类型化旋钮（`max_staleness`）加基于容量的准入控制，避免配置膨胀。
- **带外数据面**——控制面只看元数据；张量载荷通过带租约的句柄走共享内存 / RDMA。没有 Ray 张量序列化，控制路径也没有 GIL 税。
- **增量优先、分片本地的权重同步**——在训练侧一次性转换到 rollout 的最终坐标，只传输变化的字节；没有 rank-0 全量暂存，也没有全局同步屏障。
- **Rollout 即服务 + 嵌入式快速路径**——默认使用 Rust axum 网关和粘性路由来保留前缀局部性；单机实验可用进程内 megaLM `Engine.step()` 消费。两条路径都保留逐 token logprobs，并有对齐检查拒绝畸形的重要性采样比。
- **窄后端 trait**——`RolloutService`、`WeightTransport`、`RewardFn` 可通过小而稳定的类型化配置替换。
- **全异步控制器**——生成、奖励与训练重叠；长尾请求支持 `abort | wait | keep` 暂停模式；零方差组被丢弃并重填，而不是浪费一次前反向。
- **Agentic RL（多轮工具调用）**——`AgentLoop` 在每一轮保留模型真实采样到的 token、logprob 与逐 token 版本；Rust 校验的 `AgentTrajectory` 帧与「仅 assistant token」损失掩码保证工具观测只进入上下文、不接收梯度；可插拔的 schema 校验工具注册表、CLEAN/REALIGN/FORK 前缀漂移处理、会话亲和与轮次折扣奖励直接复用同一控制器。生产选项：schema 只序列化一次、工具超时从提交时计时、`strict=False` 把工具失败变成错误观测、`max_workers` 并行执行同组多个 episode，`close()`／上下文管理器确定性回收线程。网络工具层的 HTTP 核心在 Rust（`megarl-nettools`，参考 MegaLM `remote_media.rs`）：校验式 DNS 解析、IP 字面量拦截、逐跳重定向校验、流式大小上限；`ToolHTTPProxy` 提供鉴权与动态 HTTP 工具注册。详见 [devdocs/agentic-rl.md](devdocs/agentic-rl.md)。
- **扩散语言模型 RL（dLLM）**——用 d1 的单步 log-prob 估计与 wd1 的无比率目标，在掩码式与分块式扩散语言模型上做 GRPO 式 RL；确定性提示词掩码画布、`mean`/`mean_loo` 参数化、MegaLM rollout 适配器、逐去噪步轨迹录制／回放（out-of-band payload store）、同步与异步控制器、连续流匹配的 Flow-GRPO / DiffNFT 目标、JustGRPO（ICML 2026 Flexibility Trap，AR 精确似然 + top-entropy 快速模式）、AGRPO（ICML 2026 摊销式逐步 GRPO，无偏 MC 时间步估计）、GDSD（无似然引导式去噪器自蒸馏）、dOPSD（基于自身去噪轨迹特权信息的在线自蒸馏）、T3D（DDO 少步轨迹自蒸馏）、TIPS（arXiv 2610.22099，微型推理期可见性策略）、ReMaRL（arXiv 2611.22576，基于 Plackett–Luce 位置策略的学习式重掩码）与 PURL（arXiv 2611.11551v1，解掩码／编辑／token 联合似然），三者均位于命名空间化的 `megarl.dllm.algorithms.<algorithm>` 子包中；另有过程价值头与逐步 GAE，以及复用既有版本化线契约的训练通路；附真实 checkpoint GPU smoke 脚本。详见 [devdocs/dllm-rl.md](devdocs/dllm-rl.md)。
- **奖励与算法工具箱**——每种算法一份生产实现：GRPO、BNPO、Dr.GRPO、DAPO、PPO（GAE + value clip）、CISPO、SAPO、LUSPO、VESPO 与反向 KL 蒸馏（`reverse_kl` / `opd`）统一走 `policy_loss`；支持 k1/k2/k3/low-var KL 估计器、正交的 OPD 优势惩罚项（`opd_kl_coef`，slime/SkyRL 方案），以及 `dpo_update` 中的 DPO（sigmoid/cDPO/IPO，可选参考模型）；另有 group/batch/none 奖励缩放、RLOO 与 REINFORCE 基线、`cu_seqlens` token 预算微批打包、批量模型奖励池、外部验证器和多模态 collator。
- **四信号可观测性**——Prometheus `/metrics`（counter、gauge、真 histogram）、可选 OTLP 指标/链路、单行 JSON 日志与训练 trackers；RL 事件语义在 megaRL，底层原语来自共享包 `tensorplay-observability`。详见 [devdocs/observability.md](devdocs/observability.md)。

---

## 🏗 架构

```
L5  Python DSL / 算法
    GRPO / BNPO / Dr.GRPO / DAPO / PPO / CISPO / SAPO / LUSPO / VESPO
    reverse_kl / OPD（k1/k2/k3 KL 估计器）在 losses.py + kl.py
    DPO（sigmoid / IPO）在 dpo.py；advantages.py 提供 GAE 与 RLOO
L4  控制面 (Rust, megarl-control)
    版本化异步循环 · 陈旧度准入 · 暂停/恢复 · 租约
L3  数据面 (Rust, megarl-transport)
    类型化 TrajectoryRecord 帧 · shm / RDMA · 零拷贝
L2  执行后端
    Rollout (MegaLM engine / gateway) | Trainer (megaTrain 内核 + TensorPlay FSDP/stax) | Reward
L1  权重织物
    增量生成 · CUDA IPC / NCCL / shm / fs / RDMA · 版本握手
L0  底座
    TensorPlay p10 / tpx / stax + MegaLM + 设备插件
    (CUDA / MetaX / MLU / NPU / Vulkan / MPS)
```

权威设计基线（含设计理由与上游参考的已知缺口）见 [devdocs/architecture.md](devdocs/architecture.md)。

---

## 🚀 快速开始

### 1. 环境

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

TensorPlay wheel 发布在项目源上，本地安装时设置 `PIP_EXTRA_INDEX_URL`：

```bash
python -m pip install -e ".[test]" \
  --extra-index-url https://download.tensorplay.cn/simple
```

> **共享开发机注意：** 本机**不要**执行上面的命令，也不要直接运行
> `maturin develop`。本机通过 `.pth` 从源码检出导入 TensorPlay；pip 重新解析
> wheel 会破坏环境（报 `libnccl.so.2`）。规则、验证命令和恢复步骤见
> [`devdocs/local-environment.md`](devdocs/local-environment.md)。

可编辑安装会通过 maturin 构建 PyO3 扩展。若要运行 HTTP 端到端测试，需单独构建 Rust 网关二进制：

```bash
cargo build --manifest-path rust/Cargo.toml -p megarl-rs --bin megarl-gateway
```

使用嵌入式 megaLM rollout 后端时安装 `rollout` 额外依赖（服务路径则部署配套的 `megalm-server`）：

```bash
python -m pip install -e ".[rollout]"
```

### 2. 验证安装

```bash
ruff check src tests
python -m pytest tests -q
cargo check --manifest-path rust/Cargo.toml --locked
cargo test --manifest-path rust/Cargo.toml --locked
```

### 3. 契约与控制原语

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
validate(request)          # 用 Rust 结构体校验 Python 镜像
frame = encode(request)    # 版本化 MGRL 帧，仅含元数据与句柄
kind, message = decode(frame)
assert kind == "RolloutRequest" and message == request
```

### 4. 驱动训练循环

```python
import asyncio
from megarl.async_controller import AsyncController, AsyncControllerConfig
from megarl.controller import Prompt

config = AsyncControllerConfig(max_staleness=4, group_size=8)
controller = AsyncController(
    config,
    rollout=my_rollout,      # 实现 RolloutService
    reward_fn=my_reward_fn,  # TrajectoryRecord -> float
    trainer=my_trainer,      # 实现 Trainer.update
)
results = asyncio.run(controller.run([Prompt(1, [128000, 872])], steps=100))
```

`tests/test_controller.py` 与 `tests/test_async_controller.py` 中提供了三个角色最小但完整可运行的示例。

---

## 📁 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/megarl/contracts.py` | 从 Rust 镜像的版本化 MGRL 线契约 |
| `src/megarl/control.py` | Rust 支撑的陈旧度预算、版本账本、租约表 |
| `src/megarl/controller.py`、`async_controller.py` | 同步与异步训练循环 |
| `src/megarl/agent.py`、`tools.py` | Agentic 回合循环、工具注册表与控制器适配器 |
| `src/megarl/nettools.py` | 网络工具：加固的 fetch / search 与带鉴权的工具代理（Rust 核心的薄适配层） |
| `src/megarl/rollout.py`、`gateway.py`、`worker.py` | megaLM 适配器与 Rust HTTP 网关 |
| `src/megarl/weights.py`、`rdma.py`、`moe.py` | 增量权重织物、RDMA 传输、MoE 分片 |
| `src/megarl/reward.py`、`losses.py`、`batching.py` | 奖励池、GRPO 损失、token 预算打包 |
| `src/megarl/observability.py`、`grafana/` | RL 遥测、指标与面板 |
| `rust/crates/` | `protocol`、`control`、`transport`、`gateway`、`reward`、`agent`、`nettools` |
| `devdocs/` | 已接受的设计基线 |
| `refs/` | 固定版本的只读上游参考（`UPSTREAMS.md`） |

---

## ✅ 实现状态

已实现：带跨语言漂移检查的版本化线契约；Rust 控制面与 PyO3 绑定；同步与异步控制器；带逐 token logprobs 的嵌入式 megaLM rollout；奖励验证器与零方差过滤；TensorPlay 上的 GRPO 损失；带 SHA-256 校验的增量权重织物及共享内存 / RDMA 传输；批量奖励池与 MoE 分片转换；多模态轨迹；带 Prometheus 指标与 Grafana 面板的 Rust 网关；以及后端探测。

仍依赖硬件的工作：在真实 NIXL/ibverbs 网络上验证 RDMA provider，以及在 MetaX / MLU / NPU 硬件上验证外部设备插件。接口和回退路径已就位，只缺物理设备和网络凭据。

---

## 📚 文档

- [架构基线](devdocs/architecture.md)
- [Agentic RL 设计与证据](devdocs/agentic-rl.md)
- [可观测性：指标、OTLP、日志与 trackers](devdocs/observability.md)
- [逐项缺口闭环清单](devdocs/gap-closure.md)
- [固定版本的上游参考](refs/UPSTREAMS.md)——任何基准声明的必需证据
- [贡献指南](CONTRIBUTING.md) · [行为准则](CODE_OF_CONDUCT.md) · [安全策略](SECURITY.md) · [更新日志](CHANGELOG.md)
- [TensorPlay 源码](https://github.com/lexing-2026/TensorPlay) · [MegaLM 源码](https://github.com/ohtensorplay/megalm) · [TensorPlay 官网](https://tensorplay.cn) · [社区论坛](https://forum.tensorplay.org) · [ohtensorplay 组织](https://github.com/ohtensorplay)

---

## 贡献与许可

欢迎贡献——开发环境和评审要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。megaRL 基于 [Apache License 2.0](LICENSE) 发布。

<div align="center">
  <p>megaRL 属于 TensorPlay 生态。如果对你有帮助，欢迎 Star 并反馈。</p>
</div>
