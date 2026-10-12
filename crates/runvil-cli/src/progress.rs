//! Progress on stderr: one line per Engine event as the Harness records it,
//! and with `--verbose` each event's detail lines.

use std::io::{self, Write as _};

use harness::record::{
    Record, RecordKind, RecorderFuture, RunId, RunMeta, RunOutcome, RunRecorder,
};

/// A recorder that prints each event of the run to stderr as the Harness
/// records it, and keeps nothing.
#[derive(Debug)]
pub(crate) struct ProgressRecorder {
    /// Print each event's [`detail_lines`] after its progress line.
    pub(crate) verbose: bool,
}

impl RunRecorder for ProgressRecorder {
    fn begin_run(&self, _meta: RunMeta) -> RecorderFuture<'_, RunId> {
        Box::pin(async { Ok(RunId::from_raw(1)) })
    }

    fn append(&self, _run: RunId, record: Record) -> RecorderFuture<'_, ()> {
        if record.kind == RecordKind::Event {
            print_progress(&record.payload, self.verbose);
        }
        Box::pin(async { Ok(()) })
    }

    fn end_run(&self, _run: RunId, _outcome: RunOutcome) -> RecorderFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

/// Writes one progress line for an Engine event to stderr, followed by its
/// [`detail_lines`] when `verbose`.
fn print_progress(event: &serde_json::Value, verbose: bool) {
    let mut stderr = io::stderr().lock();
    // Progress is a side channel, so a failed write to stderr is
    // deliberately dropped rather than surfaced.
    let _ignored = writeln!(
        stderr,
        "[{}] {}: {}",
        event_field(event, "execution"),
        event_field(event, "section"),
        event_field(event, "kind"),
    );
    if verbose {
        for line in detail_lines(event) {
            let _ignored = writeln!(stderr, "{line}");
        }
    }
}

/// Returns the string field `name` of the event, or `-`.
fn event_field<'a>(event: &'a serde_json::Value, name: &str) -> &'a str {
    event
        .get(name)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("-")
}

/// Returns the indented detail lines `--verbose` prints for an Engine event.
///
/// A tool call shows its id, tool name, and JSON arguments; a tool result
/// shows the id of the call it answers and its full content, which carries
/// the error text when the call failed. Lua log messages and model replies
/// are shown too. Every other kind has no detail lines.
fn detail_lines(event: &serde_json::Value) -> Vec<String> {
    let text = |name: &str| {
        event
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
    };
    match text("kind") {
        "assistant_tool_calls" => event
            .get("calls")
            .and_then(serde_json::Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        let field = |name: &str| {
                            call.get(name)
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("-")
                        };
                        let arguments = call
                            .get("arguments")
                            .map_or_else(|| "null".to_owned(), serde_json::Value::to_string);
                        format!("    -> call {} {} {arguments}", field("id"), field("name"))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        "tool_result" => {
            let mut lines = vec![format!(
                "    <- result {} {}:",
                text("tool_call_id"),
                text("alias")
            )];
            lines.extend(indented(text("content")));
            lines
        }
        "lua" => vec![format!("    lua: {}", text("message"))],
        "assistant_reply" => {
            let mut lines = vec!["    reply:".to_owned()];
            lines.extend(indented(text("text")));
            lines
        }
        _ => Vec::new(),
    }
}

/// Returns each line of `text` indented under a detail header.
fn indented(text: &str) -> impl Iterator<Item = String> + '_ {
    text.lines().map(|line| format!("        {line}"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::detail_lines;

    #[test]
    fn detail_lines_show_tool_calls_results_logs_and_replies() {
        let calls = json!({
            "kind": "assistant_tool_calls",
            "calls": [{ "id": "c1", "name": "web_search", "arguments": { "query": "Yihe Li" } }],
        });
        assert_eq!(
            detail_lines(&calls),
            [r#"    -> call c1 web_search {"query":"Yihe Li"}"#]
        );

        let result = json!({
            "kind": "tool_result",
            "tool_call_id": "c1",
            "alias": "web_search",
            "content": "status 401\nUnauthorized",
        });
        assert_eq!(
            detail_lines(&result),
            [
                "    <- result c1 web_search:",
                "        status 401",
                "        Unauthorized"
            ]
        );

        assert_eq!(
            detail_lines(&json!({ "kind": "lua", "message": "hi" })),
            ["    lua: hi"]
        );
        assert_eq!(
            detail_lines(&json!({ "kind": "assistant_reply", "text": "a\nb" })),
            ["    reply:", "        a", "        b"]
        );
        assert_eq!(
            detail_lines(&json!({ "kind": "tool_call_failed" })),
            Vec::<String>::new()
        );
    }
}
