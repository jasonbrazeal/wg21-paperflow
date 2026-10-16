# paper-mcp Server Setup

This guide runs `paper-mcp` on the host that already serves `cpp-mcp` as described in [`packages/cpp-mcp/docs/server-setup.md`](../../../packages/cpp-mcp/docs/server-setup.md). It assumes that guide is complete: the `cppmcp` user, the `~/wg21-paperflow` checkout, the `/etc/cpp-mcp/keys` file, Caddy, and the `cpp-mcp` systemd unit on `127.0.0.1:8001`.

The two servers coexist by sharing only the keys file. Each has its own systemd unit, port, and data, and a single Caddy site routes to both.

| | `cpp-mcp` | `paper-mcp` |
|---|---|---|
| Unit | `cpp-mcp.service` | `paper-mcp.service` |
| Bind | `127.0.0.1:8001` | `127.0.0.1:8002` |
| Data | `$CPP_MCP_DATA_DIR` | `$WG21_DATA_DIR` (paperstore plus `paperindex/`) |
| Keys | `/etc/cpp-mcp/keys` | `/etc/cpp-mcp/keys` (same file) |
| Public URL | `https://mcpserver1.cpp.al/mcp` | `https://mcpserver1.cpp.al/papers/mcp` |

## 1. Build the binary

Install the build prerequisites. LanceDB needs `protoc`, and the Rust toolchain version is pinned by `rust-toolchain.toml`:

```bash
sudo apt install -y build-essential protobuf-compiler
sudo su - cppmcp
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
source ~/.cargo/env
```

As the `cppmcp` user, build from the existing checkout. The Cargo workspace includes `paperweight`, whose `vendor/promptforge` submodule must be checked out even though `paper-mcp` does not use it:

```bash
cd ~/wg21-paperflow
git submodule update --init --recursive
cargo build --release -p paper-mcp --locked
~/wg21-paperflow/target/release/paper-mcp --help
```

## 2. Build the index

`paper-mcp` reads the paperstore under `$WG21_DATA_DIR` and keeps its index in `$WG21_DATA_DIR/paperindex/`. Populate the paperstore on this host with paperflow (`paperflow full`), or copy `paperstore.db` and `paperstore/` from another workspace. Then build the index from this host's data:

```bash
export WG21_DATA_DIR=/home/cppmcp/wg21-data
export FASTEMBED_CACHE_DIR=/home/cppmcp/.cache/fastembed
~/wg21-paperflow/target/release/paper-mcp index
```

The first run downloads the embedding model into `$FASTEMBED_CACHE_DIR`; without it the model lands in `.fastembed_cache` under the current directory. Indexing is incremental: later runs re-embed only papers whose Markdown changed. Run it again after each paperflow ingest, then restart the service (section 5) so it serves the new index.

## 3. Share the keys file

`paper-mcp` reads the same format as `cpp-mcp`: one key per line, blank lines and `#` comments ignored. Point it at `/etc/cpp-mcp/keys` and every key issued for `cpp-mcp` also works for `paper-mcp`. Run both services as `cppmcp`, since the file is mode 600 and owned by that user.

`paper-mcp` refuses to start when the file has no keys, unless `--no-auth` is passed. Never use `--no-auth` on a public-facing server.

## 4. Systemd service

Create `/etc/systemd/system/paper-mcp.service`:

```ini
[Unit]
Description=WG21 Paper MCP Server
After=network.target

[Service]
Type=simple
User=cppmcp
Group=cppmcp
WorkingDirectory=/home/cppmcp
Environment=WG21_DATA_DIR=/home/cppmcp/wg21-data
Environment=FASTEMBED_CACHE_DIR=/home/cppmcp/.cache/fastembed
ExecStart=/home/cppmcp/wg21-paperflow/target/release/paper-mcp serve --host 127.0.0.1 --port 8002 --keys-file /etc/cpp-mcp/keys
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The port must differ from `cpp-mcp`'s 8001. Keep `--host 127.0.0.1` and let Caddy terminate TLS; port 8002 is never opened in ufw or the cloud firewall.

```bash
sudo systemctl daemon-reload
sudo systemctl enable paper-mcp
sudo systemctl start paper-mcp
sudo systemctl status paper-mcp
curl -H "Authorization: Bearer YOUR_API_KEY" http://127.0.0.1:8002/mcp
```

Requests without a valid key get HTTP 401.

### Reloading keys

Both servers re-read the keys file on SIGHUP without dropping connections. After adding or revoking a key in `/etc/cpp-mcp/keys`, reload both:

```bash
sudo systemctl reload cpp-mcp paper-mcp
```

## 5. Rebuilding after a manifest mismatch

The index records the embedding model, vector dimension, and chunker version it was built with. When an upgraded binary differs in any of them, `paper-mcp serve` exits at startup with an error naming the differing setting and the rebuild command, visible in `journalctl -u paper-mcp`. It also exits when the index was never built. Rebuild, then restart:

```bash
sudo -u cppmcp WG21_DATA_DIR=/home/cppmcp/wg21-data \
    FASTEMBED_CACHE_DIR=/home/cppmcp/.cache/fastembed \
    /home/cppmcp/wg21-paperflow/target/release/paper-mcp index
sudo systemctl restart paper-mcp
```

The index is derived data. If it is damaged, delete `$WG21_DATA_DIR/paperindex/` and run the same command to rebuild it from scratch.

## 6. Caddy route

Serve both from one site by routing a path prefix to `paper-mcp`. `handle_path` strips `/papers`, so `paper-mcp` still sees requests at `/mcp`. Edit `/etc/caddy/Caddyfile`:

```
mcpserver1.cpp.al {
    handle_path /papers/* {
        reverse_proxy 127.0.0.1:8002
    }
    handle {
        reverse_proxy 127.0.0.1:8001
    }
}
```

```bash
sudo systemctl reload caddy
curl -H "Authorization: Bearer YOUR_API_KEY" https://mcpserver1.cpp.al/papers/mcp
```

With keys configured, `paper-mcp` accepts any `Host` header, so the public name Caddy forwards is not rejected.

## 7. Connect from Cursor

Add a second entry next to `cpp-standard` in `.cursor/mcp.json`, using the same key:

```json
{
  "mcpServers": {
    "wg21-papers": {
      "url": "https://mcpserver1.cpp.al/papers/mcp",
      "headers": {
        "Authorization": "Bearer <api-key>"
      }
    }
  }
}
```

## Upgrades

```bash
cd /home/cppmcp/wg21-paperflow
git pull --ff-only
git submodule update --init --recursive
cargo build --release -p paper-mcp --locked
sudo systemctl restart paper-mcp
```

If the service fails to start, follow section 5.

## Monitoring

```bash
sudo journalctl -u paper-mcp -f
```

`RUST_LOG` overrides the default log filter, for example `Environment=RUST_LOG=info` in the unit.
