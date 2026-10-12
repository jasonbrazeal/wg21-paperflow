//! The `runvil` command: run any runvil prompt through the gateway.
//!
//! `runvil <PROMPT> [--input <PATH>] [--args <TEXT>] [--output <PATH>]
//! [--model <ID>] [--mcp <PATH>] [--verbose]` runs the prompt file through
//! the Harness. `--input` is written to the prompt's declared `input:` file,
//! `--args` is the run's argument text, and the prompt's declared `output:`
//! file is written to stdout, or to `--output` when given. A prompt that
//! declares no `output:` file writes the run's final text instead. The run
//! takes its name from the prompt's file stem.
//!
//! The gateway credentials `PROMPTFORGE_GATEWAY_URL` (the gateway origin; a
//! trailing `/v1` is dropped) and `PROMPTFORGE_GATEWAY_API_KEY` must both be
//! set: every model round and web search goes through the gateway.
//! `--model` selects the gateway model; without it, the run binds the first
//! chat model the gateway lists.
//!
//! Every run has the `web` Plugin installed, which gives it `web/fetch` and
//! `web/search`, and one MCP Plugin per server of the MCP file (see
//! [`mcp`]), named after the server. Progress goes to stderr; `--verbose`
//! adds every tool call, its full result, Lua log messages, and model
//! replies. Ctrl-C cancels the run and exits with status 130.

mod mcp;
mod progress;

use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use harness::plugin::{HostContext, HostServices, PackageRegistry, PluginId};
use harness::record::RunOutcome;
use harness::vfs::VfsRef;
use harness::{BoxFuture, Harness, HostSnapshot, OutputError, RunControl, RunRequest, Timer};
use harness_gateway_client::{GatewayBroker, GatewayEndpoint, GatewaySearch, SecretString};
use plugin_web::SEARCH_PROVIDER;
use promptforge_plugin::{ServiceKey, TOKIO_RUNTIME};

use crate::progress::ProgressRecorder;

/// The key the Host's tokio runtime is provided under, for the Plugins
/// that spawn their work on it.
const RUNTIME: ServiceKey<tokio::runtime::Handle> = ServiceKey::new(TOKIO_RUNTIME);

/// The command-line interface.
#[derive(Debug, Parser)]
#[command(version, about = "Run a runvil prompt through the gateway")]
struct Cli {
    /// The prompt file to run.
    prompt: PathBuf,
    /// Write the file at PATH to the prompt's declared `input:` file.
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,
    /// The prompt's arguments, passed verbatim: a JSON object for a prompt
    /// that declares `args:`, or plain text.
    #[arg(long, value_name = "TEXT", default_value = "")]
    args: String,
    /// Write the prompt's output to PATH instead of stdout.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// The gateway model id the prompt's model roles bind to; defaults to the
    /// first chat model the gateway lists.
    #[arg(long, value_name = "ID")]
    model: Option<String>,
    /// The `mcp.json` file whose `mcpServers` the run connects to; defaults
    /// to the first `mcp.json` beside the executable, in the working
    /// directory, or in ~/.promptforge.
    #[arg(long, value_name = "PATH")]
    mcp: Option<PathBuf>,
    /// Also print every tool call with its arguments, every tool result in
    /// full, Lua log messages, and model replies to stderr.
    #[arg(short, long)]
    verbose: bool,
}

/// The run was cancelled by Ctrl-C before it completed.
#[derive(Debug)]
struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("run cancelled")
    }
}

impl Error for Cancelled {}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:?}");
            let cancelled = error.chain().any(<dyn Error>::is::<Cancelled>);
            ExitCode::from(if cancelled { 130 } else { 1 })
        }
    }
}

/// Runs the prompt `cli` names and writes its declared output.
///
/// # Errors
/// Returns an error if either gateway variable is unset or blank, the
/// prompt, input, or MCP file cannot be read, the gateway model list cannot
/// be fetched or omits the selected model, the run fails, the run is
/// cancelled ([`Cancelled`]), the prompt did not produce its declared
/// output, or the output cannot be written.
async fn run(cli: Cli) -> Result<()> {
    let name = prompt_name(&cli.prompt)?;
    let (base_url, key) = gateway_from_env()?;
    let source = tokio::fs::read_to_string(&cli.prompt)
        .await
        .with_context(|| format!("read prompt file {}", cli.prompt.display()))?;
    let input_text = match &cli.input {
        Some(path) => Some(
            tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("read input file {}", path.display()))?,
        ),
        None => None,
    };
    let servers = match mcp::locate(cli.mcp.as_deref())? {
        Some(path) => {
            eprintln!("MCP servers from {}", path.display());
            mcp::read_servers(&path)?
        }
        None => Vec::new(),
    };

    let endpoint = GatewayEndpoint::new(&format!("{base_url}/v1"))
        .context("read the gateway URL from PROMPTFORGE_GATEWAY_URL")?;
    let key =
        SecretString::new(key).context("read the gateway key from PROMPTFORGE_GATEWAY_API_KEY")?;
    let host = host_context(
        host_services(GatewaySearch::new(endpoint.clone(), key.clone())),
        servers,
    );
    let harness = Harness::new(
        Arc::new(ProgressRecorder {
            verbose: cli.verbose,
        }),
        Arc::new(GatewayBroker::new(endpoint, key)),
        Arc::new(TokioTimer),
        Arc::new(host),
        HostServices::new(),
    );
    let ctrl_c = tokio::spawn(cancel_on_ctrl_c(harness.control()));
    let report = harness
        .run(RunRequest {
            name: name.clone(),
            source,
            args: cli.args,
            input_text,
            vfs: VfsRef::default(),
            host: HostSnapshot {
                selected_model: selected_model(cli.model.as_deref()),
                workspace_roots: Vec::new(),
            },
        })
        .await;
    ctrl_c.abort();
    let report = report.with_context(|| format!("run the {name} prompt"))?;

    let final_text = match report.outcome {
        RunOutcome::Completed { final_text } => final_text,
        RunOutcome::Failed { kind, message } => bail!("run failed ({kind}): {message}"),
        RunOutcome::Cancelled => return Err(Cancelled.into()),
    };
    let text = match report.output {
        Ok(text) => text,
        Err(OutputError::Undeclared) => final_text,
        Err(error @ OutputError::Missing { .. }) => {
            return Err(error).context("the prompt did not produce its declared output");
        }
        Err(error) => return Err(error).context("read the run's declared output"),
    };
    match &cli.output {
        Some(path) => {
            tokio::fs::write(path, &text)
                .await
                .with_context(|| format!("write the output to {}", path.display()))?;
            println!("{}", path.display());
        }
        None => write_stdout(text.as_bytes())?,
    }
    Ok(())
}

/// Returns the Host-wide services the Plugins read at install: the search
/// provider `web/search` runs through, and the runtime `web` and the MCP
/// servers spawn their work on.
fn host_services(search: GatewaySearch) -> HostServices {
    let mut services = HostServices::new();
    // Two valid, distinct keys into an empty map, so neither call can be
    // refused.
    let _ = services.provide(&SEARCH_PROVIDER, Arc::new(search));
    let _ = services.provide(&RUNTIME, Arc::new(tokio::runtime::Handle::current()));
    services
}

/// Installs the native Plugins, `web`, and one MCP Plugin per server in
/// `servers`, each under its name lowercased.
///
/// A Plugin that fails to install is reported to stderr and left out; a
/// run that declares it under `plugins:` then fails at startup naming it.
fn host_context(services: HostServices, servers: Vec<(String, serde_json::Value)>) -> HostContext {
    let mut packages = PackageRegistry::new();
    for package in [plugin_web::PACKAGE, plugin_mcp::PACKAGE] {
        if let Err(error) = packages.add(package) {
            eprintln!(
                "warning: the {} package was not added: {error}",
                package.name
            );
        }
    }
    let mut host = HostContext::new(packages, services);
    for package in ["promptforge/subagents", plugin_web::PACKAGE.name] {
        if let Err(error) = host.install(package, None, serde_json::Value::Null) {
            eprintln!("warning: the {package} Plugin was not installed: {error}");
        }
    }
    for (server, entry) in servers {
        let id = match PluginId::parse(&server.to_ascii_lowercase()) {
            Ok(id) => id,
            Err(error) => {
                eprintln!("warning: the MCP server {server:?} was skipped: {error}");
                continue;
            }
        };
        if let Err(error) = host.install(plugin_mcp::PACKAGE.name, Some(id), entry) {
            eprintln!("warning: the MCP server {server:?} was not installed: {error}");
        }
    }
    host
}

/// Cancels the run on the first Ctrl-C.
///
/// A failure to register the signal listener is reported to stderr rather
/// than discarded: the run simply stays non-cancellable in that case.
async fn cancel_on_ctrl_c(control: RunControl) {
    match tokio::signal::ctrl_c().await {
        Ok(()) => control.cancel(),
        Err(error) => {
            eprintln!("warning: cannot listen for Ctrl-C, run is not cancellable: {error}");
        }
    }
}

/// Sleeps on tokio's timer.
#[derive(Debug)]
struct TokioTimer;

impl Timer for TokioTimer {
    fn sleep(&self, seconds: f64) -> BoxFuture<()> {
        // The Engine refuses a timeout that `Duration` cannot hold before it
        // issues the effect, so the zero fallback never fires.
        let duration = Duration::try_from_secs_f64(seconds).unwrap_or(Duration::ZERO);
        Box::pin(tokio::time::sleep(duration))
    }
}

/// Returns the name the run takes from its prompt file: the file stem.
fn prompt_name(prompt: &Path) -> Result<String> {
    prompt
        .file_stem()
        .and_then(OsStr::to_str)
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
        .with_context(|| {
            format!(
                "the prompt path {} has no usable file name",
                prompt.display()
            )
        })
}

/// Returns the explicit model selection, trimmed.
///
/// A missing or blank selection is `None`, which the Harness resolves to the
/// first model the gateway lists.
fn selected_model(model: Option<&str>) -> Option<String> {
    model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_owned)
}

/// Reads the gateway environment, requiring both variables.
fn gateway_from_env() -> Result<(String, String)> {
    let endpoint = env_optional("PROMPTFORGE_GATEWAY_URL")?;
    let token = env_optional("PROMPTFORGE_GATEWAY_API_KEY")?;
    let endpoint = endpoint.map(|value| gateway_origin(&value));
    let token = token.map(|value| value.trim().to_owned());
    match (endpoint, token) {
        (Some(endpoint), Some(token)) if !endpoint.is_empty() && !token.is_empty() => {
            Ok((endpoint, token))
        }
        _ => bail!(
            "PROMPTFORGE_GATEWAY_URL and PROMPTFORGE_GATEWAY_API_KEY must both be set: \
             every model round goes through the gateway"
        ),
    }
}

/// Returns the gateway origin for `url`: trimmed, without trailing slashes
/// or a `/v1` suffix.
fn gateway_origin(url: &str) -> String {
    let url = url.trim().trim_end_matches('/');
    url.strip_suffix("/v1")
        .unwrap_or(url)
        .trim_end_matches('/')
        .to_owned()
}

/// Reads an optional environment variable, distinguishing an absent
/// variable (`Ok(None)`) from a present-but-unreadable one.
fn env_optional(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error @ std::env::VarError::NotUnicode(_)) => {
            Err(error).with_context(|| format!("read environment variable {name}"))
        }
    }
}

/// Writes bytes to stdout verbatim, flushing before return.
fn write_stdout(bytes: &[u8]) -> Result<()> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
        .context("write the output to stdout")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use super::{Cli, gateway_origin, prompt_name, selected_model};

    #[test]
    fn parser_takes_the_prompt_and_every_option() {
        let cli = Cli::parse_from(["runvil", "prompts/agora-author-advocacy.md"]);
        assert_eq!(cli.prompt, Path::new("prompts/agora-author-advocacy.md"));
        assert_eq!(cli.input, None);
        assert_eq!(cli.args, "");
        assert_eq!(cli.output, None);
        assert_eq!(cli.model, None);
        assert_eq!(cli.mcp, None);
        assert!(!cli.verbose);

        let cli = Cli::parse_from([
            "runvil",
            "p.md",
            "--input",
            "paper.md",
            "--args",
            r#"{"id": "P2300R10"}"#,
            "--output",
            "out.json",
            "--model",
            "analyst",
            "--mcp",
            "mcp.json",
            "-v",
        ]);
        assert_eq!(cli.input.as_deref(), Some(Path::new("paper.md")));
        assert_eq!(cli.args, r#"{"id": "P2300R10"}"#);
        assert_eq!(cli.output.as_deref(), Some(Path::new("out.json")));
        assert_eq!(cli.model.as_deref(), Some("analyst"));
        assert_eq!(cli.mcp.as_deref(), Some(Path::new("mcp.json")));
        assert!(cli.verbose);
    }

    #[test]
    fn parser_requires_a_prompt() {
        assert!(Cli::try_parse_from(["runvil"]).is_err());
        assert!(Cli::try_parse_from(["runvil", "p.md", "extra"]).is_err());
    }

    #[test]
    fn prompt_name_uses_file_stem() {
        let name = |path: &str| prompt_name(Path::new(path)).ok();
        assert_eq!(
            name("/code/prompts/agora-author-advocacy.md").as_deref(),
            Some("agora-author-advocacy")
        );
        assert_eq!(name("hello").as_deref(), Some("hello"));
        assert_eq!(name("/"), None);
        assert_eq!(name(""), None);
    }

    #[test]
    fn selected_model_trims_or_defers() {
        assert_eq!(selected_model(None), None);
        assert_eq!(selected_model(Some("  ")), None);
        assert_eq!(
            selected_model(Some(" analyst ")).as_deref(),
            Some("analyst")
        );
    }

    #[test]
    fn gateway_origin_strips_v1() {
        for url in [
            "http://gw:8080",
            "http://gw:8080/",
            "http://gw:8080/v1",
            " http://gw:8080/v1/ ",
        ] {
            assert_eq!(gateway_origin(url), "http://gw:8080", "for {url:?}");
        }
    }
}
