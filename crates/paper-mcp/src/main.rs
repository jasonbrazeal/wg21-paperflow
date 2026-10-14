// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The `paper-mcp` command: index WG21 papers and serve search over MCP.

use std::process::ExitCode;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use paperstore_sqlite::SqliteBackend;
use tracing_subscriber::EnvFilter;

/// Index WG21 papers held in paperstore and serve search over MCP.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Index papers: one or more paper ids, a four-digit year, or `all`.
    Index {
        /// Paper ids, years, or `all`.
        #[arg(required = true, value_name = "SELECTOR")]
        selectors: Vec<String>,
    },
    /// Serve search over MCP Streamable HTTP.
    Serve,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Index { selectors } => {
            let parsed = paper_mcp::parse_selectors(&selectors)?;
            let store = SqliteBackend::from_env()?;
            let papers = paper_mcp::resolve(&store, &parsed)?;
            tracing::info!(count = papers.len(), "selected papers");
            bail!("indexing is not implemented yet")
        }
        Command::Serve => bail!("serve is not implemented yet"),
    }
}
