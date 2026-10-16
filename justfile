# Command hub. Each package owns its recipes in packages/<name>/justfile and is
# mounted here as a module, so commands read as `just <package> <verb>`, e.g.
# `just tomd score P4228R0`. Run `just --list` to see everything.

mod tomd 'packages/tomd/justfile'

# List available recipes.
default:
    @just --list

# Bring the paper-mcp index under $WG21_DATA_DIR up to date.
paper-mcp-index:
    cargo run --release -p paper-mcp --locked -- index

# Serve the paper index over MCP; arguments pass through to `paper-mcp serve`.
paper-mcp-serve *args:
    cargo run --release -p paper-mcp --locked -- serve {{args}}
