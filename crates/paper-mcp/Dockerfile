# syntax=docker/dockerfile:1
# paper-mcp: the Rust MCP server for WG21 papers. paper-mcp is its own Cargo
# workspace, so this builds from crates/paper-mcp alone.

# The tag matches the toolchain rust-toolchain.toml pins at the repository root.
FROM rust:1.98.0-slim-trixie AS build
# g++: the statically linked ONNX Runtime needs libstdc++ at link time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
    && install -D target/release/paper-mcp /out/paper-mcp

FROM debian:trixie-slim
# ca-certificates: the embedding model is downloaded from Hugging Face over
# HTTPS on first start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 paper-mcp \
    && mkdir /models \
    && chown paper-mcp /models
COPY --from=build /out/paper-mcp /usr/local/bin/paper-mcp
USER paper-mcp
ENV PAPER_MCP_MODEL_CACHE_DIR=/models \
    PAPER_MCP_BIND=0.0.0.0:8003 \
    PAPER_MCP_KEYS_FILE=/run/secrets/paper-mcp-keys
EXPOSE 8003
ENTRYPOINT ["paper-mcp"]
CMD ["serve"]
