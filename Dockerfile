# syntax=docker/dockerfile:1

# Builder: compile the Rust control plane (PyO3 extension + gateway binary).
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.cargo/bin:$PATH \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain 1.94.0 \
    && python -m pip install --no-cache-dir "maturin>=1.7,<2"

WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock ./rust/
COPY rust/crates ./rust/crates
COPY rust/src ./rust/src
# The gateway embeds these assets at compile time (include_str!).
COPY brand/assets ./brand/assets
COPY brand/design-tokens.css ./brand/design-tokens.css
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

ARG PIP_EXTRA_INDEX_URL=https://download.tensorplay.cn/simple
ARG TENSORPLAY_SPEC="tensorplay>=1.3.0.dev0,<2"
ARG MEGATRAIN_SPEC="megatrain>=0.1,<0.2"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --pre "$TENSORPLAY_SPEC" "$MEGATRAIN_SPEC" \
        --extra-index-url "$PIP_EXTRA_INDEX_URL"
RUN maturin build --manifest-path rust/Cargo.toml --release --out /dist \
    && cargo build --manifest-path rust/Cargo.toml --release --locked \
       -p megarl-rs --bin megarl-gateway --bin megarl-bench \
    && cargo build --manifest-path rust/Cargo.toml --release --locked \
       -p megarl-reward-service --bin megarl-reward-router

# Runtime: Python algorithms + the compiled Rust spine, no Rust toolchain.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    MEGARL_GATEWAY_BIN=/usr/local/bin/megarl-gateway

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin megarl \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist/*.whl /tmp/megarl/
COPY --from=builder /src/rust/target/release/megarl-gateway /usr/local/bin/megarl-gateway
COPY --from=builder /src/rust/target/release/megarl-bench /usr/local/bin/megarl-bench
COPY --from=builder /src/rust/target/release/megarl-reward-router /usr/local/bin/megarl-reward-router

ARG PIP_EXTRA_INDEX_URL=https://download.tensorplay.cn/simple
ARG TENSORPLAY_SPEC="tensorplay>=1.3.0.dev0,<2"
ARG MEGATRAIN_SPEC="megatrain>=0.1,<0.2"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --pre "$TENSORPLAY_SPEC" "$MEGATRAIN_SPEC" \
        --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    && python -m pip install --no-cache-dir --no-deps /tmp/megarl/*.whl \
    && rm -rf /tmp/megarl \
    && chmod 0755 /usr/local/bin/megarl-gateway /usr/local/bin/megarl-bench \
       /usr/local/bin/megarl-reward-router

USER megarl
WORKDIR /home/megarl

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "from megarl import megarl_rs; assert megarl_rs.protocol_version() == 1"

ENTRYPOINT ["/usr/local/bin/megarl-gateway"]
CMD ["0.0.0.0:8080"]
