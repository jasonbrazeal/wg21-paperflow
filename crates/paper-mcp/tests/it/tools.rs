//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Tool handler tests: each MCP tool called directly against the indexed
//! fixture corpus.

use std::path::Path;
use std::sync::Arc;

use paper_mcp::embed::HashEmbedder;
use paper_mcp::index::{INDEX_DIR_NAME, Indexer, PaperIndex};
use paper_mcp::tools::{
    FilterParams, GetPaperLinesParams, ListPapersParams, PaperIdParams, PaperTools,
    SearchPapersParams, ToolError,
};
use paperstore::StorageBackend;
use rmcp::ServerHandler;
use rmcp::handler::server::wrapper::Parameters;

use super::{FIXTURE_PAPER_IDS, METADATA_ONLY_ID, num, open_from_env, with_fixture};

const MODEL_ID: &str = "test/hash";
const DIM: usize = 16;
const QUERY: &str = "Paper Title hazard pointer cells";
const TOOL_NAMES: [&str; 5] = [
    "get_paper_lines",
    "get_paper_metadata",
    "list_papers",
    "list_revisions",
    "search_papers",
];

/// Latest revision of each fixture paper, text-bearing or not.
const LATEST_IDS: [&str; 5] = ["N5001", "P1000R2", "P2000R0", "P3000R0", "SD-4"];

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("build runtime: {e}"))
        .block_on(future)
}

/// Index the fixture workspace and build the tool router over it.
async fn tools(workspace: &Path) -> PaperTools {
    let embedder = Arc::new(HashEmbedder::new(MODEL_ID, DIM));
    let index = PaperIndex::open(&workspace.join(INDEX_DIR_NAME))
        .await
        .unwrap_or_else(|e| panic!("open index: {e}"));
    let backend = open_from_env();
    let report = Indexer::new(&backend, embedder.as_ref(), &index)
        .run()
        .await
        .unwrap_or_else(|e| panic!("index run: {e}"));
    assert!(report.is_success(), "{report:?}");
    PaperTools::new(open_from_env(), index, embedder)
}

fn with_tools(f: impl AsyncFnOnce(&PaperTools)) {
    with_fixture(|workspace| {
        block_on(async {
            let tools = tools(workspace).await;
            f(&tools).await;
        });
    });
}

fn search_params(search_type: &str, filters: FilterParams) -> SearchPapersParams {
    SearchPapersParams {
        query: QUERY.to_owned(),
        search_type: search_type.to_owned(),
        filters,
        limit: None,
    }
}

fn paper_filter(ids: &[&str]) -> FilterParams {
    FilterParams {
        paper_ids: Some(ids.iter().map(|id| (*id).to_owned()).collect()),
        ..FilterParams::default()
    }
}

fn paper_id(id: &str) -> Parameters<PaperIdParams> {
    Parameters(PaperIdParams {
        paper_id: id.to_owned(),
    })
}

fn lines(id: &str, start: Option<i64>, end: Option<i64>) -> Parameters<GetPaperLinesParams> {
    Parameters(GetPaperLinesParams {
        paper_id: id.to_owned(),
        start,
        end,
    })
}

fn expect_err<T>(result: Result<T, ToolError>, context: &str) -> String {
    match result {
        Ok(_) => panic!("{context}: expected a tool error"),
        Err(e) => e.to_string(),
    }
}

fn assert_names(message: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            message.contains(needle),
            "{message:?} should name {needle:?}"
        );
    }
}

#[test]
fn router_exposes_exactly_the_five_tools_with_output_schemas() {
    with_tools(async |tools| {
        let listed = tools.tools();
        let names: Vec<&str> = listed.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(names, TOOL_NAMES);
        for tool in &listed {
            let schema = tool
                .output_schema
                .as_ref()
                .unwrap_or_else(|| panic!("{} has no output schema", tool.name));
            assert_eq!(
                schema.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "{}",
                tool.name
            );
        }
        let capabilities = tools.get_info().capabilities;
        assert!(capabilities.tools.is_some());
        assert!(capabilities.resources.is_none());
        assert!(capabilities.prompts.is_none());
    });
}

#[test]
fn search_hits_round_trip_through_get_paper_lines() {
    with_tools(async |tools| {
        for search_type in ["hybrid", "vector", "keyword"] {
            let out = tools
                .search_papers(Parameters(search_params(
                    search_type,
                    paper_filter(&["p1000r1"]),
                )))
                .await
                .unwrap_or_else(|e| panic!("{search_type}: {e}"))
                .0;
            assert!(!out.hits.is_empty(), "{search_type}");
            for hit in &out.hits {
                assert_eq!(hit.paper_id, "P1000R1");
                assert_eq!(hit.revision, Some(1));
                assert_eq!(hit.url, "https://wg21.link/P1000R1");
                let slice = tools
                    .get_paper_lines(lines(
                        &hit.paper_id,
                        Some(i64::from(hit.start_line)),
                        Some(i64::from(hit.end_line)),
                    ))
                    .await
                    .unwrap_or_else(|e| panic!("lines: {e}"))
                    .0;
                assert_eq!(slice.text, hit.text, "{search_type} {}", hit.start_line);
            }
        }
    });
}

#[test]
fn search_respects_limit_and_default_scope() {
    with_tools(async |tools| {
        let mut params = search_params("keyword", FilterParams::default());
        params.limit = Some(3);
        let out = tools
            .search_papers(Parameters(params))
            .await
            .unwrap_or_else(|e| panic!("search: {e}"))
            .0;
        assert_eq!(out.hits.len(), 3);
        for hit in &out.hits {
            assert!(
                LATEST_IDS.contains(&hit.paper_id.as_str()),
                "{}",
                hit.paper_id
            );
        }
    });
}

#[test]
fn search_input_errors_are_tool_errors_naming_the_input() {
    with_tools(async |tools| {
        let cases: [(SearchPapersParams, &[&str]); 7] = [
            (
                search_params("bm25", FilterParams::default()),
                &["search_type", "bm25"],
            ),
            (
                search_params("keyword", paper_filter(&["P9999R0"])),
                &["paper_ids", "P9999R0"],
            ),
            (
                search_params("keyword", paper_filter(&["P1000"])),
                &["paper_ids", "P1000"],
            ),
            (
                search_params("keyword", paper_filter(&["not-a-paper"])),
                &["paper_ids", "not-a-paper"],
            ),
            (
                search_params(
                    "keyword",
                    FilterParams {
                        mailing: Some("2025-13".to_owned()),
                        ..FilterParams::default()
                    },
                ),
                &["mailing", "2025-13"],
            ),
            (
                SearchPapersParams {
                    query: "  ".to_owned(),
                    ..search_params("keyword", FilterParams::default())
                },
                &["query"],
            ),
            (
                SearchPapersParams {
                    limit: Some(0),
                    ..search_params("keyword", FilterParams::default())
                },
                &["limit"],
            ),
        ];
        for (params, needles) in cases {
            let context = format!("{params:?}");
            let message = expect_err(tools.search_papers(Parameters(params)).await, &context);
            assert_names(&message, needles);
        }
    });
}

#[test]
fn get_paper_metadata_returns_the_paper_and_its_revisions() {
    with_tools(async |tools| {
        let out = tools
            .get_paper_metadata(paper_id("p1000r1"))
            .await
            .unwrap_or_else(|e| panic!("metadata: {e}"))
            .0;
        let row = open_from_env()
            .meta(&num("P1000R1"))
            .unwrap_or_else(|e| panic!("meta: {e}"));
        assert_eq!(out.paper.paper_id, "P1000R1");
        assert_eq!(out.paper.revision, Some(1));
        assert_eq!(out.paper.title, row.title);
        assert_eq!(out.paper.authors, row.authors);
        assert_eq!(out.paper.mailing_date, row.mailing_date);
        assert_eq!(out.paper.previous_version, "p1000r0");
        assert_eq!(out.paper.url, "https://wg21.link/P1000R1");
        assert!(out.paper.has_text);
        let ids: Vec<&str> = out.revisions.iter().map(|r| r.paper_id.as_str()).collect();
        assert_eq!(ids, ["P1000R0", "P1000R1", "P1000R2"]);

        let failed = tools
            .get_paper_metadata(paper_id(METADATA_ONLY_ID))
            .await
            .unwrap_or_else(|e| panic!("metadata-only: {e}"))
            .0;
        assert!(!failed.paper.has_text);
        assert!(!failed.paper.title.is_empty());
    });
}

#[test]
fn exact_id_tools_reject_unknown_and_revisionless_ids() {
    with_tools(async |tools| {
        for bad in ["P9999R0", "P1000", "xyz", "P1000R9"] {
            let message = expect_err(tools.get_paper_metadata(paper_id(bad)).await, bad);
            assert_names(&message, &["paper_id", bad]);
            let message = expect_err(tools.get_paper_lines(lines(bad, None, None)).await, bad);
            assert_names(&message, &["paper_id", bad]);
        }
    });
}

#[test]
fn list_revisions_accepts_ids_with_and_without_revision() {
    with_tools(async |tools| {
        for query in ["P1000", "p1000r1", "P1000R2"] {
            let out = tools
                .list_revisions(paper_id(query))
                .await
                .unwrap_or_else(|e| panic!("{query}: {e}"))
                .0;
            assert_eq!(out.paper, "P1000", "{query}");
            let got: Vec<(&str, Option<u32>, &str)> = out
                .revisions
                .iter()
                .map(|r| (r.paper_id.as_str(), r.revision, r.previous_version.as_str()))
                .collect();
            assert_eq!(
                got,
                [
                    ("P1000R0", Some(0), ""),
                    ("P1000R1", Some(1), "p1000r0"),
                    ("P1000R2", Some(2), "p1000r1"),
                ],
                "{query}"
            );
            assert!(out.revisions.iter().all(|r| !r.mailing_date.is_empty()));
        }
        let single = tools
            .list_revisions(paper_id("SD-4"))
            .await
            .unwrap_or_else(|e| panic!("SD-4: {e}"))
            .0;
        let ids: Vec<&str> = single
            .revisions
            .iter()
            .map(|r| r.paper_id.as_str())
            .collect();
        assert_eq!(ids, ["SD-4"]);
        for bad in ["P9999", "bogus"] {
            let message = expect_err(tools.list_revisions(paper_id(bad)).await, bad);
            assert_names(&message, &["paper_id", bad]);
        }
    });
}

#[test]
fn list_papers_filters_and_paginates() {
    with_tools(async |tools| {
        let list = async |filters: FilterParams, offset, limit| {
            tools
                .list_papers(Parameters(ListPapersParams {
                    filters,
                    offset,
                    limit,
                }))
                .await
                .unwrap_or_else(|e| panic!("list_papers: {e}"))
                .0
        };
        let latest = list(FilterParams::default(), None, None).await;
        let ids: Vec<&str> = latest.papers.iter().map(|p| p.paper_id.as_str()).collect();
        assert_eq!(ids, LATEST_IDS);
        assert_eq!(
            (latest.total, latest.offset, latest.next_offset),
            (5, 0, None)
        );

        let all = FilterParams {
            revision_scope: Some("all".to_owned()),
            ..FilterParams::default()
        };
        let mut pages = Vec::new();
        let mut offset = Some(0);
        while let Some(start) = offset {
            let page = list(all.clone(), Some(start), Some(3)).await;
            assert_eq!(page.total, FIXTURE_PAPER_IDS.len());
            assert_eq!(page.offset, start);
            assert!(page.papers.len() <= 3);
            pages.extend(page.papers.into_iter().map(|p| p.paper_id));
            offset = page.next_offset;
        }
        assert_eq!(pages, FIXTURE_PAPER_IDS);

        let past_end = list(all.clone(), Some(100), None).await;
        assert!(past_end.papers.is_empty());
        assert_eq!(past_end.next_offset, None);

        let picked = list(paper_filter(&["P1000R0", "P3000R0"]), None, None).await;
        let ids: Vec<&str> = picked.papers.iter().map(|p| p.paper_id.as_str()).collect();
        assert_eq!(ids, ["P1000R0", "P3000R0"]);

        let wording = FilterParams {
            chunk_kind: Some("wording".to_owned()),
            ..all
        };
        let with_wording = list(wording, None, None).await;
        let ids: Vec<&str> = with_wording
            .papers
            .iter()
            .map(|p| p.paper_id.as_str())
            .collect();
        assert!(!ids.contains(&METADATA_ONLY_ID), "{ids:?}");
        assert!(!ids.is_empty());
    });
}

#[test]
fn list_papers_input_errors_are_tool_errors() {
    with_tools(async |tools| {
        let cases: [(ListPapersParams, &[&str]); 3] = [
            (
                ListPapersParams {
                    filters: paper_filter(&["P1000"]),
                    offset: None,
                    limit: None,
                },
                &["paper_ids", "P1000"],
            ),
            (
                ListPapersParams {
                    filters: FilterParams {
                        revision_scope: Some("newest".to_owned()),
                        ..FilterParams::default()
                    },
                    offset: None,
                    limit: None,
                },
                &["revision_scope", "newest"],
            ),
            (
                ListPapersParams {
                    filters: FilterParams::default(),
                    offset: None,
                    limit: Some(0),
                },
                &["limit"],
            ),
        ];
        for (params, needles) in cases {
            let context = format!("{params:?}");
            let message = expect_err(tools.list_papers(Parameters(params)).await, &context);
            assert_names(&message, needles);
        }
    });
}

#[test]
fn get_paper_lines_slices_the_whole_markdown() {
    with_tools(async |tools| {
        let md = open_from_env()
            .paper_md(&num("N5001"))
            .unwrap_or_else(|e| panic!("paper_md: {e}"));
        let all: Vec<&str> = md.lines().collect();
        let whole = tools
            .get_paper_lines(lines("n5001", None, None))
            .await
            .unwrap_or_else(|e| panic!("lines: {e}"))
            .0;
        assert_eq!(whole.paper_id, "N5001");
        assert_eq!((whole.start, whole.end), (0, all.len()));
        assert_eq!(whole.total_lines, all.len());
        assert_eq!(whole.text, all.join("\n"));
        assert!(whole.text.starts_with("---\ntitle: "));

        let tail = tools
            .get_paper_lines(lines("N5001", Some(-3), None))
            .await
            .unwrap_or_else(|e| panic!("lines: {e}"))
            .0;
        assert_eq!((tail.start, tail.end), (all.len() - 3, all.len()));
        assert_eq!(tail.text, all[all.len() - 3..].join("\n"));

        let empty = tools
            .get_paper_lines(lines("N5001", Some(5), Some(2)))
            .await
            .unwrap_or_else(|e| panic!("lines: {e}"))
            .0;
        assert_eq!((empty.start, empty.end, empty.text.as_str()), (5, 5, ""));

        let message = expect_err(
            tools
                .get_paper_lines(lines(METADATA_ONLY_ID, None, None))
                .await,
            METADATA_ONLY_ID,
        );
        assert_names(&message, &["paper_id", METADATA_ONLY_ID, "no text"]);
    });
}
