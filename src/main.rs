use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use regex::Regex;
use reqwest::blocking::Client;
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use url::Url;

const APP_NAME: &str = "ato-mcp";
const DEFAULT_RELEASES_URL: &str = "https://github.com/gunba/ato-mcp/releases/latest/download";
const DEFAULT_K: usize = 8;
const MAX_K: usize = 50;
const SNIPPET_CHARS: usize = 280;
const EMBEDDING_DIM: usize = 256;
const MAX_TOKENS: usize = 1024;
const QUERY_PREFIX: &str = "task: search result | query: ";
const EMBEDDINGGEMMA_HF_FINGERPRINT: &str =
    "5d4d31914cdb65cd84d3248390946461efdd4ec4f99afd13d23218cd4060d706";
const OLD_CONTENT_CUTOFF: &str = "2000-01-01";
const DEFAULT_EXCLUDED_TYPES: &[&str] = &["Edited_private_advice"];
const LEGISLATION_TYPE: &str = "Legislation_and_supporting_material";

#[derive(Parser)]
#[command(name = "ato-mcp", version, about = "Standalone ATO MCP server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the MCP stdio server.
    Serve {
        /// Serve the installed corpus without checking for updates first.
        #[arg(long)]
        no_update: bool,
    },
    /// First-run install of the corpus into the local data directory.
    Init {
        #[arg(long)]
        manifest_url: Option<String>,
    },
    /// Apply a manifest delta to the local corpus.
    Update {
        #[arg(long)]
        manifest_url: Option<String>,
    },
    /// Verify the local corpus, optionally restoring the previous DB snapshot.
    Doctor {
        #[arg(long)]
        rollback: bool,
    },
    /// Print index version and counts.
    Stats {
        #[arg(long, default_value = "markdown")]
        format: OutputFormat,
    },
    /// Run a search from the CLI.
    Search {
        query: String,
        #[arg(short, long, default_value_t = DEFAULT_K)]
        k: usize,
        #[arg(long, value_delimiter = ',')]
        types: Vec<String>,
        #[arg(long)]
        date_from: Option<String>,
        #[arg(long)]
        date_to: Option<String>,
        #[arg(long)]
        doc_scope: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: SearchMode,
        #[arg(long, default_value = "relevance")]
        sort_by: SortBy,
        #[arg(long)]
        include_old: bool,
        #[arg(long, default_value = "markdown")]
        format: OutputFormat,
    },
    /// Search document titles and citations only.
    SearchTitles {
        query: String,
        #[arg(short, long, default_value_t = 20)]
        k: usize,
        #[arg(long, value_delimiter = ',')]
        types: Vec<String>,
        #[arg(long)]
        include_old: bool,
        #[arg(long, default_value = "markdown")]
        format: OutputFormat,
    },
    /// Fetch a document or a slice of it.
    GetDocument {
        doc_id: String,
        #[arg(long, default_value = "outline")]
        format: DocumentFormat,
        #[arg(long)]
        anchor: Option<String>,
        #[arg(long)]
        heading_path: Option<String>,
        #[arg(long)]
        from_ord: Option<i64>,
        #[arg(long)]
        include_children: bool,
        #[arg(long)]
        count: Option<usize>,
        #[arg(long)]
        max_chars: Option<usize>,
    },
    /// Documents most recently published by the corpus date field.
    WhatsNew {
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        before: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, value_delimiter = ',')]
        types: Vec<String>,
        #[arg(long, default_value = "markdown")]
        format: OutputFormat,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Markdown,
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum SortBy {
    Relevance,
    Recency,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DocumentFormat {
    Outline,
    Card,
    Markdown,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { no_update } => {
            if no_update || env_truthy("ATO_MCP_OFFLINE") {
                ensure_installed_db()?;
            } else {
                update_before_serve()?;
            }
            serve()
        }
        Command::Init { manifest_url } => {
            let url = manifest_url.unwrap_or_else(default_manifest_url);
            let stats = apply_update(&url)?;
            println!(
                "init complete: +{} ~{} -{} ({:.1} MB downloaded)",
                stats.added,
                stats.changed,
                stats.removed,
                stats.bytes_downloaded as f64 / 1_000_000.0
            );
            Ok(())
        }
        Command::Update { manifest_url } => {
            let url = manifest_url.unwrap_or_else(default_manifest_url);
            let stats = apply_update(&url)?;
            println!(
                "update complete: +{} ~{} -{} ({:.2} MB downloaded)",
                stats.added,
                stats.changed,
                stats.removed,
                stats.bytes_downloaded as f64 / 1_000_000.0
            );
            Ok(())
        }
        Command::Doctor { rollback } => doctor(rollback),
        Command::Stats { format } => {
            println!("{}", stats(format)?);
            Ok(())
        }
        Command::Search {
            query,
            k,
            types,
            date_from,
            date_to,
            doc_scope,
            mode,
            sort_by,
            include_old,
            format,
        } => {
            let types = empty_vec_as_none(types);
            println!(
                "{}",
                search(
                    &query,
                    SearchOptions {
                        k,
                        types: types.as_deref(),
                        date_from: date_from.as_deref(),
                        date_to: date_to.as_deref(),
                        doc_scope: doc_scope.as_deref(),
                        mode,
                        sort_by,
                        include_old,
                        format,
                    },
                    None,
                )?
            );
            Ok(())
        }
        Command::SearchTitles {
            query,
            k,
            types,
            include_old,
            format,
        } => {
            let types = empty_vec_as_none(types);
            println!(
                "{}",
                search_titles(&query, k, types.as_deref(), include_old, format)?
            );
            Ok(())
        }
        Command::GetDocument {
            doc_id,
            format,
            anchor,
            heading_path,
            from_ord,
            include_children,
            count,
            max_chars,
        } => {
            println!(
                "{}",
                get_document(
                    &doc_id,
                    GetDocumentOptions {
                        format,
                        anchor: anchor.as_deref(),
                        heading_path: heading_path.as_deref(),
                        from_ord,
                        include_children,
                        count,
                        max_chars,
                    },
                )?
            );
            Ok(())
        }
        Command::WhatsNew {
            since,
            before,
            limit,
            types,
            format,
        } => {
            let types = empty_vec_as_none(types);
            println!(
                "{}",
                whats_new(
                    since.as_deref(),
                    before.as_deref(),
                    limit,
                    types.as_deref(),
                    format
                )?
            );
            Ok(())
        }
    }
}

fn empty_vec_as_none(values: Vec<String>) -> Option<Vec<String>> {
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn default_manifest_url() -> String {
    format!("{}/manifest.json", releases_url().trim_end_matches('/'))
}

fn releases_url() -> String {
    std::env::var("ATO_MCP_RELEASES_URL").unwrap_or_else(|_| DEFAULT_RELEASES_URL.to_string())
}

fn data_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ATO_MCP_DATA_DIR") {
        let path = PathBuf::from(path);
        fs::create_dir_all(&path)?;
        return Ok(path);
    }
    let mut path =
        dirs::data_dir().ok_or_else(|| anyhow!("could not resolve user data directory"))?;
    path.push(APP_NAME);
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn live_dir() -> Result<PathBuf> {
    let path = data_dir()?.join("live");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn staging_dir() -> Result<PathBuf> {
    let path = data_dir()?.join("staging");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn backups_dir() -> Result<PathBuf> {
    let path = data_dir()?.join("backups");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn db_path() -> Result<PathBuf> {
    Ok(live_dir()?.join("ato.db"))
}

fn installed_manifest_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("installed_manifest.json"))
}

fn lock_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("LOCK"))
}

fn model_path() -> Result<PathBuf> {
    Ok(live_dir()?.join("model.onnx"))
}

fn tokenizer_path() -> Result<PathBuf> {
    Ok(live_dir()?.join("tokenizer.json"))
}

fn lock_file() -> Result<File> {
    let path = lock_path()?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    file.lock_exclusive()?;
    Ok(file)
}

fn open_read() -> Result<Connection> {
    let path = db_path()?;
    if !path.exists() {
        bail!(
            "no live DB found at {}; run `ato-mcp init` first",
            path.display()
        );
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .context("opening local corpus database")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    Ok(conn)
}

fn open_write() -> Result<Connection> {
    let path = db_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path).context("opening local corpus database for writing")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    Ok(conn)
}

fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS documents (
            doc_id         TEXT PRIMARY KEY,
            type           TEXT NOT NULL,
            title          TEXT NOT NULL,
            date           TEXT,
            downloaded_at  TEXT NOT NULL,
            content_hash   TEXT NOT NULL,
            pack_sha8      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(type);
        CREATE INDEX IF NOT EXISTS idx_doc_date ON documents(date);

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id      INTEGER PRIMARY KEY,
            doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            ord           INTEGER NOT NULL,
            heading_path  TEXT,
            anchor        TEXT,
            text          BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_ord ON chunks(doc_id, ord);

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id   INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding  BLOB NOT NULL CHECK(length(embedding) = 256)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS title_fts USING fts5(
            doc_id UNINDEXED,
            title,
            headings,
            tokenize = "porter unicode61 remove_diacritics 2"
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            heading_path,
            tokenize = "porter unicode61 remove_diacritics 2"
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS empty_shells (
            doc_id          TEXT PRIMARY KEY,
            first_seen_at   TEXT NOT NULL,
            last_checked_at TEXT NOT NULL,
            check_count     INTEGER NOT NULL DEFAULT 1,
            source          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_shells_last_checked ON empty_shells(last_checked_at);
        "#,
    )?;
    set_meta(conn, "schema_version", "5")?;
    Ok(())
}

fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?")?;
    let mut rows = stmt.query([key])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get(0)?))
    } else {
        Ok(None)
    }
}

fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

fn canonical_url(doc_id: &str) -> String {
    format!("https://www.ato.gov.au/law/view/document?docid={}", doc_id)
}

fn decompress_text(blob: Vec<u8>) -> Result<String> {
    let bytes = zstd::stream::decode_all(Cursor::new(blob))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn compress_text(text: &str) -> Result<Vec<u8>> {
    Ok(zstd::stream::encode_all(Cursor::new(text.as_bytes()), 3)?)
}

fn fts_query(query: &str) -> String {
    let re = Regex::new(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*").expect("valid regex");
    let tokens: Vec<String> = re
        .find_iter(query)
        .map(|m| m.as_str())
        .filter(|t| t.len() >= 2)
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    if tokens.is_empty() {
        "\"\"".to_string()
    } else {
        tokens.join(" ")
    }
}

fn glob_to_like(pattern: &str) -> String {
    let mut out = String::new();
    for ch in pattern.chars() {
        match ch {
            '*' => out.push('%'),
            '%' | '_' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

#[derive(Default)]
struct SqlFilter {
    sql: String,
    params: Vec<Value>,
}

fn build_doc_filter(
    alias: &str,
    types: Option<&[String]>,
    date_from: Option<&str>,
    date_to: Option<&str>,
    doc_scope: Option<&str>,
    include_old: bool,
) -> SqlFilter {
    let mut clauses = Vec::new();
    let mut params_out = Vec::new();

    if let Some(types) = types {
        if !types.is_empty() {
            let mut ors = Vec::new();
            for t in types {
                if t.contains('*') {
                    ors.push(format!("{alias}.type LIKE ? ESCAPE '\\'"));
                    params_out.push(Value::Text(glob_to_like(t)));
                } else {
                    ors.push(format!("{alias}.type = ?"));
                    params_out.push(Value::Text(t.clone()));
                }
            }
            clauses.push(format!("({})", ors.join(" OR ")));
        }
    } else if !DEFAULT_EXCLUDED_TYPES.is_empty() {
        let placeholders = vec!["?"; DEFAULT_EXCLUDED_TYPES.len()].join(",");
        clauses.push(format!("{alias}.type NOT IN ({placeholders})"));
        for t in DEFAULT_EXCLUDED_TYPES {
            params_out.push(Value::Text((*t).to_string()));
        }
    }

    if let Some(date_from) = date_from {
        clauses.push(format!("{alias}.date >= ?"));
        params_out.push(Value::Text(date_from.to_string()));
    }
    if let Some(date_to) = date_to {
        clauses.push(format!("{alias}.date <= ?"));
        params_out.push(Value::Text(date_to.to_string()));
    }
    if let Some(doc_scope) = doc_scope {
        clauses.push(format!("{alias}.doc_id LIKE ? ESCAPE '\\'"));
        params_out.push(Value::Text(glob_to_like(doc_scope)));
    }
    if !include_old && date_from.is_none() {
        clauses.push(format!(
            "({alias}.date IS NULL OR {alias}.date >= ? OR {alias}.type = ?)"
        ));
        params_out.push(Value::Text(OLD_CONTENT_CUTOFF.to_string()));
        params_out.push(Value::Text(LEGISLATION_TYPE.to_string()));
    }

    SqlFilter {
        sql: clauses.join(" AND "),
        params: params_out,
    }
}

#[derive(Debug, Serialize)]
struct Hit {
    doc_id: String,
    title: String,
    #[serde(rename = "type")]
    doc_type: String,
    date: Option<String>,
    heading_path: String,
    anchor: Option<String>,
    snippet: String,
    canonical_url: String,
    score: Option<f64>,
    chunk_id: Option<i64>,
    ord: Option<i64>,
    next_call: Option<String>,
    ranking: Option<RankingDetails>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct RankingDetails {
    overall_score: Option<f64>,
    vector_rank: Option<usize>,
    vector_score: Option<f64>,
    lexical_rank: Option<usize>,
    lexical_score: Option<f64>,
}

#[derive(Debug, Clone)]
struct VectorHit {
    chunk_id: i64,
    score: f64,
}

struct SearchOptions<'a> {
    k: usize,
    types: Option<&'a [String]>,
    date_from: Option<&'a str>,
    date_to: Option<&'a str>,
    doc_scope: Option<&'a str>,
    mode: SearchMode,
    sort_by: SortBy,
    include_old: bool,
    format: OutputFormat,
}

fn search(
    query: &str,
    opts: SearchOptions<'_>,
    server_state: Option<&mut ServerState>,
) -> Result<String> {
    let conn = open_read()?;
    let k = opts.k.clamp(1, MAX_K);
    let filter = build_doc_filter(
        "d",
        opts.types,
        opts.date_from,
        opts.date_to,
        opts.doc_scope,
        opts.include_old,
    );
    let internal_limit = std::cmp::max(k * 5, 50);
    let lexical_hits = if matches!(opts.mode, SearchMode::Hybrid | SearchMode::Keyword) {
        lexical_search(&conn, query, &filter, internal_limit)?
    } else {
        Vec::new()
    };
    let mut vector_hits = Vec::new();
    let ranked_hits = match opts.mode {
        SearchMode::Hybrid | SearchMode::Vector => {
            ensure_vector_search_ready(&conn)?;
            let query_embedding = match server_state {
                Some(state) => state.encode_query_embedding(query)?,
                None => encode_query_embedding(query)?,
            };
            vector_hits = vector_search(&conn, &query_embedding, &filter, internal_limit)?;
            if matches!(opts.mode, SearchMode::Hybrid) {
                rrf_fuse(&vector_hits, &lexical_hits)
            } else {
                vector_hits.clone()
            }
        }
        SearchMode::Keyword => lexical_hits.clone(),
    };
    let candidate_count = ranked_hits.len();
    let mut provenance = rank_provenance(&vector_hits, &lexical_hits);
    let mut records = Vec::new();
    let frontier = match opts.sort_by {
        SortBy::Relevance => k,
        SortBy::Recency => std::cmp::max(k * 5, 50),
    };
    for ranked_hit in ranked_hits.into_iter().take(frontier) {
        if let Some(mut hit) = load_hit(&conn, ranked_hit.chunk_id, query)? {
            hit.score = Some(ranked_hit.score);
            let mut ranking = provenance.remove(&ranked_hit.chunk_id).unwrap_or_default();
            ranking.overall_score = Some(ranked_hit.score);
            hit.ranking = Some(ranking);
            records.push(hit);
        }
    }
    if matches!(opts.sort_by, SortBy::Recency) {
        records.sort_by(|a, b| b.date.cmp(&a.date));
        records.truncate(k);
    }
    let next_call = if candidate_count > records.len() && k < MAX_K {
        Some(search_next_call(query, std::cmp::min(k * 2, MAX_K), &opts))
    } else {
        None
    };

    match opts.format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&json!({
            "query": query,
            "mode": search_mode_name(opts.mode),
            "ranking": {
                "semantic_required": !matches!(opts.mode, SearchMode::Keyword),
                "vector": matches!(opts.mode, SearchMode::Hybrid | SearchMode::Vector),
                "lexical": matches!(opts.mode, SearchMode::Hybrid | SearchMode::Keyword),
                "embedding_model_id": get_meta(&conn, "embedding_model_id")?,
            },
            "filters": {
                "excluded_by_default": DEFAULT_EXCLUDED_TYPES,
                "old_content_cutoff": if opts.include_old { JsonValue::Null } else { json!(OLD_CONTENT_CUTOFF) },
            },
            "meta": {
                "returned": records.len(),
                "candidate_count": candidate_count,
                "truncated": candidate_count > records.len(),
                "returned_chars": records.iter().map(|hit| hit.snippet.len()).sum::<usize>(),
                "next_call": next_call,
            },
            "hits": records,
        }))?),
        OutputFormat::Markdown => Ok(format_hits_markdown(&records)),
    }
}

fn search_mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::Vector => "vector",
        SearchMode::Keyword => "keyword",
    }
}

fn sort_by_name(sort_by: SortBy) -> &'static str {
    match sort_by {
        SortBy::Relevance => "relevance",
        SortBy::Recency => "recency",
    }
}

fn search_next_call(query: &str, k: usize, opts: &SearchOptions<'_>) -> String {
    let mut args = vec![
        format!("query={}", mcp_string(query)),
        format!("k={k}"),
        format!("mode=\"{}\"", search_mode_name(opts.mode)),
    ];
    if let Some(types) = opts.types {
        let rendered = types
            .iter()
            .map(|value| mcp_string(value))
            .collect::<Vec<_>>()
            .join(", ");
        args.push(format!("types=[{rendered}]"));
    }
    if let Some(date_from) = opts.date_from {
        args.push(format!("date_from={}", mcp_string(date_from)));
    }
    if let Some(date_to) = opts.date_to {
        args.push(format!("date_to={}", mcp_string(date_to)));
    }
    if let Some(doc_scope) = opts.doc_scope {
        args.push(format!("doc_scope={}", mcp_string(doc_scope)));
    }
    if !matches!(opts.sort_by, SortBy::Relevance) {
        args.push(format!("sort_by=\"{}\"", sort_by_name(opts.sort_by)));
    }
    if opts.include_old {
        args.push("include_old=true".to_string());
    }
    format!("search({})", args.join(", "))
}

fn mcp_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    let exists: i64 = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?1)",
        [table],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}

fn ensure_vector_search_ready(conn: &Connection) -> Result<()> {
    let model_id = get_meta(conn, "embedding_model_id")?.ok_or_else(|| {
        anyhow!("semantic search unavailable: missing embedding_model_id metadata")
    })?;
    if !model_id.starts_with("embeddinggemma") {
        bail!(
            "semantic search unavailable: installed corpus uses unsupported embedding model `{model_id}`; install an EmbeddingGemma corpus"
        );
    }
    if !model_path()?.exists() {
        bail!(
            "semantic search unavailable: model file missing at {}",
            model_path()?.display()
        );
    }
    if !tokenizer_path()?.exists() {
        bail!(
            "semantic search unavailable: tokenizer missing at {}",
            tokenizer_path()?.display()
        );
    }
    if !table_exists(conn, "chunk_embeddings")? {
        bail!("semantic search unavailable: installed corpus has no chunk_embeddings table; run `ato-mcp update`");
    }
    let embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
        row.get(0)
    })?;
    if embeddings == 0 {
        bail!("semantic search unavailable: installed corpus has no chunk embeddings");
    }
    Ok(())
}

fn vector_search(
    conn: &Connection,
    query_embedding: &[i8; EMBEDDING_DIM],
    filter: &SqlFilter,
    limit: usize,
) -> Result<Vec<VectorHit>> {
    let where_filter = if filter.sql.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", filter.sql)
    };
    let sql = format!(
        r#"
        SELECT e.chunk_id, e.embedding
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        {where_filter}
        "#
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(filter.params.clone()), |row| {
        Ok((
            row.get::<_, i64>("chunk_id")?,
            row.get::<_, Vec<u8>>("embedding")?,
        ))
    })?;
    let mut hits = Vec::new();
    for row in rows {
        let (chunk_id, embedding) = row?;
        hits.push(VectorHit {
            chunk_id,
            score: dot_i8(query_embedding, &embedding)?,
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    Ok(hits)
}

fn lexical_search(
    conn: &Connection,
    query: &str,
    filter: &SqlFilter,
    limit: usize,
) -> Result<Vec<VectorHit>> {
    let where_filter = if filter.sql.is_empty() {
        String::new()
    } else {
        format!(" AND {}", filter.sql)
    };
    let sql = format!(
        r#"
        SELECT f.rowid AS chunk_id, bm25(chunks_fts) AS score
        FROM chunks_fts f
        JOIN chunks c ON c.chunk_id = f.rowid
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE chunks_fts MATCH ? {where_filter}
        ORDER BY score ASC
        LIMIT ?
        "#
    );
    let mut params_vec = vec![Value::Text(fts_query(query))];
    params_vec.extend(filter.params.clone());
    params_vec.push(Value::Integer(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let rows = match stmt.query_map(params_from_iter(params_vec), |row| {
        Ok(VectorHit {
            chunk_id: row.get::<_, i64>("chunk_id")?,
            score: row.get::<_, f64>("score")?,
        })
    }) {
        Ok(rows) => rows.collect::<rusqlite::Result<Vec<_>>>()?,
        Err(rusqlite::Error::SqliteFailure(_, _)) => Vec::new(),
        Err(err) => return Err(err.into()),
    };
    Ok(rows)
}

fn rrf_fuse(vector_hits: &[VectorHit], lexical_hits: &[VectorHit]) -> Vec<VectorHit> {
    const RRF_K: f64 = 60.0;
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for (rank, hit) in vector_hits.iter().enumerate() {
        scores
            .entry(hit.chunk_id)
            .and_modify(|score| *score += 1.0 / (RRF_K + rank as f64 + 1.0))
            .or_insert_with(|| 1.0 / (RRF_K + rank as f64 + 1.0));
    }
    for (rank, hit) in lexical_hits.iter().enumerate() {
        scores
            .entry(hit.chunk_id)
            .and_modify(|score| *score += 1.0 / (RRF_K + rank as f64 + 1.0))
            .or_insert_with(|| 1.0 / (RRF_K + rank as f64 + 1.0));
    }
    let mut out = scores
        .into_iter()
        .map(|(chunk_id, score)| VectorHit { chunk_id, score })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn rank_provenance(
    vector_hits: &[VectorHit],
    lexical_hits: &[VectorHit],
) -> HashMap<i64, RankingDetails> {
    let mut out: HashMap<i64, RankingDetails> = HashMap::new();
    for (idx, hit) in vector_hits.iter().enumerate() {
        let details = out.entry(hit.chunk_id).or_default();
        details.vector_rank = Some(idx + 1);
        details.vector_score = Some(hit.score);
    }
    for (idx, hit) in lexical_hits.iter().enumerate() {
        let details = out.entry(hit.chunk_id).or_default();
        details.lexical_rank = Some(idx + 1);
        details.lexical_score = Some(hit.score);
    }
    out
}

fn dot_i8(query: &[i8; EMBEDDING_DIM], document: &[u8]) -> Result<f64> {
    if document.len() != EMBEDDING_DIM {
        bail!(
            "invalid stored embedding length: got {}, expected {}",
            document.len(),
            EMBEDDING_DIM
        );
    }
    let mut dot = 0i32;
    for (q, d) in query.iter().zip(document.iter()) {
        dot += i32::from(*q) * i32::from(*d as i8);
    }
    Ok(dot as f64 / (127.0 * 127.0))
}

struct SemanticRuntime {
    tokenizer: Tokenizer,
    session: Session,
    has_token_type_ids: bool,
}

impl SemanticRuntime {
    fn load() -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path()?)
            .map_err(|err| anyhow!("loading tokenizer: {err}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..TruncationParams::default()
            }))
            .map_err(|err| anyhow!("configuring tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let session = Session::builder()
            .map_err(|err| anyhow!("creating ONNX Runtime session: {err}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|err| anyhow!("configuring ONNX Runtime session: {err}"))?
            .commit_from_file(model_path()?)
            .map_err(|err| anyhow!("loading ONNX model: {err}"))?;
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            tokenizer,
            session,
            has_token_type_ids,
        })
    }

    fn encode_query(&mut self, query: &str) -> Result<[i8; EMBEDDING_DIM]> {
        let prefixed = format!("{QUERY_PREFIX}{query}");
        let mut encodings = self
            .tokenizer
            .encode_batch(vec![prefixed], true)
            .map_err(|err| anyhow!("tokenizing query: {err}"))?;
        let encoding = encodings
            .pop()
            .ok_or_else(|| anyhow!("tokenizer returned no query encoding"))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|id| i64::from(*id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|mask| i64::from(*mask))
            .collect();
        let seq_len = input_ids.len();
        if seq_len == 0 {
            bail!("semantic search unavailable: query produced no tokens");
        }

        let input_ids_tensor =
            TensorRef::from_array_view(([1usize, seq_len], input_ids.as_slice()))?;
        let attention_mask_tensor =
            TensorRef::from_array_view(([1usize, seq_len], attention_mask.as_slice()))?;
        let outputs = if self.has_token_type_ids {
            let token_type_ids = vec![0i64; seq_len];
            let token_type_ids_tensor =
                TensorRef::from_array_view(([1usize, seq_len], token_type_ids.as_slice()))?;
            self.session.run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            })?
        } else {
            self.session.run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            })?
        };
        let output = outputs
            .get("sentence_embedding")
            .unwrap_or_else(|| &outputs[0]);
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let embedding = pooled_embedding(shape, data, &attention_mask)?;
        quantize_embedding(&embedding)
    }
}

#[derive(Default)]
struct ServerState {
    semantic_runtime: Option<SemanticRuntime>,
}

impl ServerState {
    fn encode_query_embedding(&mut self, query: &str) -> Result<[i8; EMBEDDING_DIM]> {
        if self.semantic_runtime.is_none() {
            self.semantic_runtime = Some(SemanticRuntime::load()?);
        }
        self.semantic_runtime
            .as_mut()
            .expect("semantic runtime was just initialized")
            .encode_query(query)
    }
}

fn encode_query_embedding(query: &str) -> Result<[i8; EMBEDDING_DIM]> {
    let mut runtime = SemanticRuntime::load()?;
    runtime.encode_query(query)
}

fn pooled_embedding(shape: &[i64], data: &[f32], attention_mask: &[i64]) -> Result<Vec<f32>> {
    match shape {
        [1, dims] => {
            let dims = *dims as usize;
            if data.len() < dims {
                bail!("model output too short for shape {:?}", shape);
            }
            Ok(data[..dims].to_vec())
        }
        [1, seq_len, dims] => {
            let seq_len = *seq_len as usize;
            let dims = *dims as usize;
            if data.len() < seq_len * dims {
                bail!("model output too short for shape {:?}", shape);
            }
            let mut pooled = vec![0.0f32; dims];
            let mut denom = 0.0f32;
            for token_idx in 0..seq_len {
                let mask = attention_mask.get(token_idx).copied().unwrap_or(0) as f32;
                denom += mask;
                let offset = token_idx * dims;
                for dim in 0..dims {
                    pooled[dim] += data[offset + dim] * mask;
                }
            }
            let denom = denom.max(1e-6);
            for value in &mut pooled {
                *value /= denom;
            }
            Ok(pooled)
        }
        _ => bail!("unsupported model output shape {:?}", shape),
    }
}

fn quantize_embedding(values: &[f32]) -> Result<[i8; EMBEDDING_DIM]> {
    if values.len() < EMBEDDING_DIM {
        bail!(
            "model output has {} dimensions, expected at least {}",
            values.len(),
            EMBEDDING_DIM
        );
    }
    let values = &values[..EMBEDDING_DIM];
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return Ok([0; EMBEDDING_DIM]);
    }
    let mut out = [0i8; EMBEDDING_DIM];
    for (idx, value) in values.iter().enumerate() {
        out[idx] = ((*value / norm).clamp(-1.0, 1.0) * 127.0).round() as i8;
    }
    Ok(out)
}

fn load_hit(conn: &Connection, chunk_id: i64, query: &str) -> Result<Option<Hit>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
               d.type, d.title, d.date
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id = ?
        "#,
    )?;
    let mut rows = stmt.query([chunk_id])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let doc_id: String = row.get("doc_id")?;
    let text = decompress_text(row.get("text")?)?;
    let chunk_id: i64 = row.get("chunk_id")?;
    let ord: i64 = row.get("ord")?;
    Ok(Some(Hit {
        doc_id: doc_id.clone(),
        title: row.get("title")?,
        doc_type: row.get("type")?,
        date: row.get("date")?,
        heading_path: row
            .get::<_, Option<String>>("heading_path")?
            .unwrap_or_default(),
        anchor: row.get("anchor")?,
        snippet: highlight_snippet(&text, query, SNIPPET_CHARS),
        canonical_url: canonical_url(&doc_id),
        score: None,
        chunk_id: Some(chunk_id),
        ord: Some(ord),
        next_call: Some(format!("get_chunks(chunk_ids=[{chunk_id}])")),
        ranking: None,
    }))
}

fn highlight_snippet(text: &str, query: &str, max_chars: usize) -> String {
    let word_re = Regex::new(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*").expect("valid regex");
    let words: HashSet<String> = word_re
        .find_iter(query)
        .map(|m| m.as_str().to_ascii_lowercase())
        .collect();
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        return String::new();
    }
    let lower = cleaned.to_ascii_lowercase();
    let best = words
        .iter()
        .filter_map(|w| lower.find(w))
        .min()
        .unwrap_or(0);
    let mut start = best.saturating_sub(max_chars / 3);
    while start > 0 && !cleaned.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = std::cmp::min(cleaned.len(), start + max_chars);
    while end < cleaned.len() && !cleaned.is_char_boundary(end) {
        end += 1;
    }
    let mut snippet = cleaned[start..end].to_string();
    if start > 0 {
        snippet.insert_str(0, "...");
    }
    snippet
}

fn format_hits_markdown(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "_No hits._".to_string();
    }
    let mut out = String::new();
    out.push_str("| # | Chunk | Ord | Type | Date | Title | Section | Snippet |\n");
    out.push_str("|---:|---:|---:|---|---|---|---|---|\n");
    for (idx, hit) in hits.iter().enumerate() {
        let chunk = hit.chunk_id.map(|id| id.to_string()).unwrap_or_default();
        let ord = hit.ord.map(|ord| ord.to_string()).unwrap_or_default();
        out.push_str(&format!(
            "| {} | {} | {} | `{}` | {} | [{}]({})<br><small>`{}`</small> | {} | {} |\n",
            idx + 1,
            chunk,
            ord,
            escape_md(&hit.doc_type),
            hit.date.as_deref().unwrap_or(""),
            escape_md(&hit.title),
            hit.canonical_url,
            escape_md(&hit.doc_id),
            escape_md(&hit.heading_path),
            escape_md(&hit.snippet)
        ));
    }
    out
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn search_titles(
    query: &str,
    k: usize,
    types: Option<&[String]>,
    include_old: bool,
    format: OutputFormat,
) -> Result<String> {
    let conn = open_read()?;
    let k = k.clamp(1, 100);
    let filter = build_doc_filter("d", types, None, None, None, include_old);
    let where_filter = if filter.sql.is_empty() {
        String::new()
    } else {
        format!(" AND {}", filter.sql)
    };
    let sql = format!(
        r#"
        SELECT t.doc_id AS doc_id, bm25(title_fts) AS score,
               d.type, d.title, d.date
        FROM title_fts t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE title_fts MATCH ? {where_filter}
        ORDER BY score ASC
        LIMIT ?
        "#
    );
    let mut params_vec = vec![Value::Text(fts_query(query))];
    params_vec.extend(filter.params);
    params_vec.push(Value::Integer(k as i64 + 1));

    let mut stmt = conn.prepare(&sql)?;
    let mut rows = match stmt.query_map(params_from_iter(params_vec), |row| {
        let doc_id: String = row.get("doc_id")?;
        let title: String = row.get("title")?;
        let mut score = row.get::<_, f64>("score").ok();
        if title_matches_normalized_query(&title, query) {
            score = score.map(|s| s - 1000.0);
        }
        Ok(Hit {
            canonical_url: canonical_url(&doc_id),
            doc_id: doc_id.clone(),
            title: title.clone(),
            doc_type: row.get("type")?,
            date: row.get("date")?,
            heading_path: String::new(),
            anchor: None,
            snippet: title,
            score,
            chunk_id: None,
            ord: None,
            next_call: Some(format!(
                "get_document(doc_id=\"{doc_id}\", format=\"card\")"
            )),
            ranking: None,
        })
    }) {
        Ok(rows) => rows.collect::<rusqlite::Result<Vec<_>>>()?,
        Err(rusqlite::Error::SqliteFailure(_, _)) => Vec::new(),
        Err(err) => return Err(err.into()),
    };
    rows.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truncated = rows.len() > k;
    rows.truncate(k);
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&json!({
            "query": query,
            "filters": {
                "excluded_by_default": DEFAULT_EXCLUDED_TYPES,
                "old_content_cutoff": if include_old { JsonValue::Null } else { json!(OLD_CONTENT_CUTOFF) },
            },
            "meta": {
                "returned": rows.len(),
                "truncated": truncated,
                "returned_chars": rows.iter().map(|hit| hit.snippet.len()).sum::<usize>(),
            },
            "hits": rows,
        }))?),
        OutputFormat::Markdown => Ok(format_hits_markdown(&rows)),
    }
}

fn title_matches_normalized_query(title: &str, query: &str) -> bool {
    let q = normalize_alnum(query);
    q.len() >= 4 && normalize_alnum(title).contains(&q)
}

fn normalize_alnum(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

#[derive(Serialize)]
struct ChunkOut {
    chunk_id: i64,
    ord: i64,
    heading_path: String,
    anchor: Option<String>,
    text: String,
}

struct GetDocumentOptions<'a> {
    format: DocumentFormat,
    anchor: Option<&'a str>,
    heading_path: Option<&'a str>,
    from_ord: Option<i64>,
    include_children: bool,
    count: Option<usize>,
    max_chars: Option<usize>,
}

fn get_document(doc_id: &str, opts: GetDocumentOptions<'_>) -> Result<String> {
    let conn = open_read()?;
    let doc = load_document_row(&conn, doc_id)?;
    let Some(doc) = doc else {
        return Ok(format!("_Document not found: `{}`_", doc_id));
    };
    if opts.format == DocumentFormat::Outline {
        let outline =
            outline_for_doc(&conn, doc_id, opts.anchor, opts.heading_path, opts.from_ord)?;
        return Ok(format_outline(&doc, &outline));
    }
    if opts.format == DocumentFormat::Card {
        let outline =
            outline_for_doc(&conn, doc_id, opts.anchor, opts.heading_path, opts.from_ord)?;
        return document_card(&conn, &doc, outline);
    }
    let selected = select_chunks(&conn, doc_id, &opts)?;
    let Some((chunks, continuation_ord)) = selected else {
        return Ok(format!(
            "_Section not found in {} (anchor={:?}, heading_path={:?}, from_ord={:?})._",
            doc_id, opts.anchor, opts.heading_path, opts.from_ord
        ));
    };
    if opts.format == DocumentFormat::Json {
        let returned_chars = chunks.iter().map(|chunk| chunk.text.len()).sum::<usize>();
        let next_call = continuation_ord.map(|ord| {
            format!(
                "get_document(doc_id=\"{}\", format=\"json\", from_ord={}, max_chars={})",
                doc.doc_id,
                ord,
                opts.max_chars.unwrap_or(20_000)
            )
        });
        return Ok(serde_json::to_string_pretty(&json!({
            "document": doc,
            "chunks": chunks,
            "continuation_ord": continuation_ord,
            "meta": {
                "returned": chunks.len(),
                "returned_chars": returned_chars,
                "truncated": continuation_ord.is_some(),
                "next_call": next_call,
            },
        }))?);
    }
    Ok(format_document_markdown(&doc, &chunks, continuation_ord))
}

#[derive(Debug, Serialize)]
struct DocumentRow {
    doc_id: String,
    #[serde(rename = "type")]
    doc_type: String,
    title: String,
    date: Option<String>,
    downloaded_at: String,
    canonical_url: String,
}

fn load_document_row(conn: &Connection, doc_id: &str) -> Result<Option<DocumentRow>> {
    let mut stmt = conn.prepare(
        "SELECT doc_id, type, title, date, downloaded_at FROM documents WHERE doc_id = ?",
    )?;
    let mut rows = stmt.query([doc_id])?;
    if let Some(row) = rows.next()? {
        let doc_id: String = row.get("doc_id")?;
        Ok(Some(DocumentRow {
            canonical_url: canonical_url(&doc_id),
            doc_id,
            doc_type: row.get("type")?,
            title: row.get("title")?,
            date: row.get("date")?,
            downloaded_at: row.get("downloaded_at")?,
        }))
    } else {
        Ok(None)
    }
}

fn document_card(
    conn: &Connection,
    doc: &DocumentRow,
    outline: Vec<OutlineEntry>,
) -> Result<String> {
    let (chunk_count, first_ord, last_ord, compressed_bytes): (i64, Option<i64>, Option<i64>, i64) =
        conn.query_row(
            r#"
            SELECT COUNT(*), MIN(ord), MAX(ord), COALESCE(SUM(length(text)), 0)
            FROM chunks
            WHERE doc_id = ?
            "#,
            [&doc.doc_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
    let payload = json!({
        "document": doc,
        "summary": {
            "chunk_count": chunk_count,
            "heading_count": outline.len(),
            "first_ord": first_ord,
            "last_ord": last_ord,
            "compressed_text_bytes": compressed_bytes,
        },
        "hydration": {
            "full_body_call": format!("get_document(doc_id=\"{}\", format=\"markdown\", max_chars=20000)", doc.doc_id),
            "first_page_call": format!("get_document(doc_id=\"{}\", format=\"json\", from_ord={}, max_chars=12000)", doc.doc_id, first_ord.unwrap_or(0)),
            "outline_call": format!("get_document(doc_id=\"{}\", format=\"outline\")", doc.doc_id),
            "chunk_range": {
                "from_ord": first_ord,
                "to_ord": last_ord,
            },
        },
        "outline": outline,
        "next_calls": [
            format!("get_document(doc_id=\"{}\", format=\"outline\")", doc.doc_id),
            format!("get_document(doc_id=\"{}\", format=\"markdown\", from_ord={})", doc.doc_id, first_ord.unwrap_or(0)),
        ],
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

#[derive(Serialize)]
struct OutlineEntry {
    heading_path: String,
    anchor: Option<String>,
    depth: usize,
    start_ord: i64,
    chunk_count: i64,
}

fn outline_for_doc(
    conn: &Connection,
    doc_id: &str,
    anchor: Option<&str>,
    heading_path: Option<&str>,
    from_ord: Option<i64>,
) -> Result<Vec<OutlineEntry>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT heading_path, anchor, MIN(ord) AS start_ord, COUNT(*) AS chunk_count
        FROM chunks
        WHERE doc_id = ?
        GROUP BY heading_path
        ORDER BY start_ord ASC
        "#,
    )?;
    let entries = stmt
        .query_map([doc_id], |row| {
            let hp: String = row
                .get::<_, Option<String>>("heading_path")?
                .unwrap_or_default();
            let depth = if hp.is_empty() {
                0
            } else if hp.contains(" > ") {
                hp.matches(" > ").count() + 1
            } else {
                hp.matches(" › ").count() + 1
            };
            Ok(OutlineEntry {
                heading_path: hp,
                anchor: row.get("anchor")?,
                depth,
                start_ord: row.get("start_ord")?,
                chunk_count: row.get("chunk_count")?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if anchor.is_none() && heading_path.is_none() && from_ord.is_none() {
        return Ok(entries);
    }
    let Some(start_idx) = entries.iter().position(|e| {
        anchor
            .map(|a| e.anchor.as_deref() == Some(a))
            .unwrap_or(false)
            || heading_path.map(|hp| e.heading_path == hp).unwrap_or(false)
            || from_ord.map(|ord| e.start_ord >= ord).unwrap_or(false)
    }) else {
        return Ok(Vec::new());
    };
    let start_path = entries[start_idx].heading_path.clone();
    let mut out = Vec::new();
    for e in entries.into_iter().skip(start_idx) {
        if start_path.is_empty()
            || e.heading_path == start_path
            || e.heading_path.starts_with(&(start_path.clone() + " › "))
            || e.heading_path.starts_with(&(start_path.clone() + " > "))
        {
            out.push(e);
        } else {
            break;
        }
    }
    Ok(out)
}

fn select_chunks(
    conn: &Connection,
    doc_id: &str,
    opts: &GetDocumentOptions<'_>,
) -> Result<Option<(Vec<ChunkOut>, Option<i64>)>> {
    let mut stmt = conn.prepare(
        "SELECT chunk_id, ord, heading_path, anchor, text FROM chunks WHERE doc_id = ? ORDER BY ord ASC",
    )?;
    let rows = stmt
        .query_map([doc_id], |row| {
            Ok((
                row.get::<_, i64>("chunk_id")?,
                row.get::<_, i64>("ord")?,
                row.get::<_, Option<String>>("heading_path")?
                    .unwrap_or_default(),
                row.get::<_, Option<String>>("anchor")?,
                row.get::<_, Vec<u8>>("text")?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Ok(Some((Vec::new(), None)));
    }

    let start_idx =
        if opts.anchor.is_none() && opts.heading_path.is_none() && opts.from_ord.is_none() {
            0
        } else {
            let Some(idx) = rows.iter().position(|(_, ord, hp, anchor, _)| {
                opts.anchor
                    .map(|a| anchor.as_deref() == Some(a))
                    .unwrap_or(false)
                    || opts
                        .heading_path
                        .map(|target| hp == target)
                        .unwrap_or(false)
                    || opts
                        .from_ord
                        .map(|from_ord| *ord >= from_ord)
                        .unwrap_or(false)
            }) else {
                return Ok(None);
            };
            idx
        };
    let start_path = rows[start_idx].2.clone();
    let mut candidates: Vec<_> = rows.into_iter().skip(start_idx).collect();
    if (opts.anchor.is_some() || opts.heading_path.is_some()) && !opts.include_children {
        if let Some(anchor) = opts.anchor {
            candidates.retain(|(_, _, _, a, _)| a.as_deref() == Some(anchor));
        } else if let Some(hp) = opts.heading_path {
            candidates.retain(|(_, _, h, _, _)| h == hp);
        }
    } else if (opts.anchor.is_some() || opts.heading_path.is_some()) && opts.include_children {
        candidates = candidates
            .into_iter()
            .take_while(|(_, _, hp, _, _)| {
                start_path.is_empty()
                    || hp == &start_path
                    || hp.starts_with(&(start_path.clone() + " › "))
                    || hp.starts_with(&(start_path.clone() + " > "))
            })
            .collect();
    }

    let mut out = Vec::new();
    let mut chars = 0usize;
    let mut continuation_ord = None;
    for idx in 0..candidates.len() {
        let (chunk_id, ord, heading_path, anchor, blob) = &candidates[idx];
        let text = decompress_text(blob.clone())?;
        if opts
            .max_chars
            .is_some_and(|max| !out.is_empty() && chars + text.len() > max)
        {
            continuation_ord = Some(*ord);
            break;
        }
        chars += text.len();
        out.push(ChunkOut {
            chunk_id: *chunk_id,
            ord: *ord,
            heading_path: heading_path.clone(),
            anchor: anchor.clone(),
            text,
        });
        if opts.count.is_some_and(|count| out.len() >= count) {
            if let Some((_, next_ord, _, _, _)) = candidates.get(idx + 1) {
                continuation_ord = Some(*next_ord);
            }
            break;
        }
    }
    Ok(Some((out, continuation_ord)))
}

fn format_outline(doc: &DocumentRow, entries: &[OutlineEntry]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", doc.title));
    out.push_str(&format!("`{}` | `{}`", doc.doc_id, doc.doc_type));
    if let Some(date) = &doc.date {
        out.push_str(&format!(" | Date: {}", date));
    }
    out.push_str(&format!("\nSource: {}\n\n", doc.canonical_url));
    if entries.is_empty() {
        out.push_str("_No outline entries._");
        return out;
    }
    out.push_str("| Ord | Chunks | Heading |\n|---:|---:|---|\n");
    for e in entries {
        let indent = "&nbsp;".repeat(e.depth.saturating_sub(1) * 2);
        let display = if e.heading_path.is_empty() {
            "(intro)".to_string()
        } else {
            escape_md(&e.heading_path)
        };
        out.push_str(&format!(
            "| {} | {} | {}{} |\n",
            e.start_ord, e.chunk_count, indent, display
        ));
    }
    out
}

fn format_document_markdown(
    doc: &DocumentRow,
    chunks: &[ChunkOut],
    continuation_ord: Option<i64>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", doc.title));
    out.push_str(&format!("`{}` | `{}`", doc.doc_id, doc.doc_type));
    if let Some(date) = &doc.date {
        out.push_str(&format!(" | Date: {}", date));
    }
    out.push_str(&format!("\nSource: {}\n\n", doc.canonical_url));
    for chunk in chunks {
        if !chunk.heading_path.is_empty() {
            out.push_str(&format!("## {}\n\n", chunk.heading_path));
        }
        out.push_str(&chunk.text);
        out.push_str("\n\n");
    }
    if let Some(ord) = continuation_ord {
        out.push_str(&format!(
            "_Truncated. Continue with `get_document(doc_id=\"{}\", format=\"markdown\", from_ord={})`._",
            doc.doc_id, ord
        ));
    }
    out
}

fn whats_new(
    since: Option<&str>,
    before: Option<&str>,
    limit: usize,
    types: Option<&[String]>,
    format: OutputFormat,
) -> Result<String> {
    let conn = open_read()?;
    let mut clauses = Vec::new();
    let mut params_out = Vec::new();
    let sort_expr = "COALESCE(date, downloaded_at)";
    if let Some(since) = since {
        clauses.push(format!("{sort_expr} >= ?"));
        params_out.push(Value::Text(since.to_string()));
    }
    if let Some(before) = before {
        clauses.push(format!("{sort_expr} < ?"));
        params_out.push(Value::Text(before.to_string()));
    }
    if let Some(types) = types {
        if !types.is_empty() {
            let mut ors = Vec::new();
            for t in types {
                if t.contains('*') {
                    ors.push("type LIKE ? ESCAPE '\\'".to_string());
                    params_out.push(Value::Text(glob_to_like(t)));
                } else {
                    ors.push("type = ?".to_string());
                    params_out.push(Value::Text(t.clone()));
                }
            }
            clauses.push(format!("({})", ors.join(" OR ")));
        }
    } else {
        let placeholders = vec!["?"; DEFAULT_EXCLUDED_TYPES.len()].join(",");
        clauses.push(format!("type NOT IN ({placeholders})"));
        for t in DEFAULT_EXCLUDED_TYPES {
            params_out.push(Value::Text((*t).to_string()));
        }
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    let limit = limit.clamp(1, 500);
    params_out.push(Value::Integer(limit as i64 + 1));
    let sql = format!(
        "SELECT doc_id, type, title, date, downloaded_at FROM documents {where_sql} ORDER BY {sort_expr} DESC LIMIT ?"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut hits = stmt
        .query_map(params_from_iter(params_out), |row| {
            let doc_id: String = row.get("doc_id")?;
            let date: Option<String> = row.get("date")?;
            let downloaded_at: String = row.get("downloaded_at")?;
            Ok(Hit {
                canonical_url: canonical_url(&doc_id),
                doc_id: doc_id.clone(),
                title: row.get("title")?,
                doc_type: row.get("type")?,
                date: date.clone(),
                heading_path: String::new(),
                anchor: None,
                snippet: if let Some(date) = date {
                    format!("published {}", date)
                } else {
                    format!("ingested {}", downloaded_at)
                },
                score: None,
                chunk_id: None,
                ord: None,
                next_call: Some(format!(
                    "get_document(doc_id=\"{doc_id}\", format=\"card\")"
                )),
                ranking: None,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let truncated = hits.len() > limit;
    if truncated {
        hits.truncate(limit);
    }
    let next_before = hits
        .last()
        .and_then(|hit| hit.date.as_deref())
        .map(|date| date.to_string());
    let next_call = if truncated {
        next_before.as_ref().map(|date| {
            let mut args = vec![
                format!("before={}", mcp_string(date)),
                format!("limit={limit}"),
            ];
            if let Some(since) = since {
                args.push(format!("since={}", mcp_string(since)));
            }
            if let Some(types) = types {
                let rendered = types
                    .iter()
                    .map(|value| mcp_string(value))
                    .collect::<Vec<_>>()
                    .join(", ");
                args.push(format!("types=[{rendered}]"));
            }
            format!("whats_new({})", args.join(", "))
        })
    } else {
        None
    };
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&json!({
            "since": since,
            "before": before,
            "hits": hits,
            "meta": {
                "returned": hits.len(),
                "truncated": truncated,
                "returned_chars": hits.iter().map(|hit| hit.snippet.len()).sum::<usize>(),
                "next_call": next_call,
            },
        }))?),
        OutputFormat::Markdown => Ok(format_hits_markdown(&hits)),
    }
}

fn stats(format: OutputFormat) -> Result<String> {
    let conn = open_read()?;
    let docs: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
    let embeddings: i64 = if table_exists(&conn, "chunk_embeddings")? {
        conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |r| r.get(0))?
    } else {
        0
    };
    let mut types = BTreeMap::new();
    let mut stmt =
        conn.prepare("SELECT type, COUNT(*) AS n FROM documents GROUP BY type ORDER BY n DESC")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (typ, n) = row?;
        types.insert(typ, n);
    }
    let payload = json!({
        "data_dir": data_dir()?.display().to_string(),
        "index_version": get_meta(&conn, "index_version")?,
        "last_update_at": get_meta(&conn, "last_update_at")?,
        "embedding_model_id": get_meta(&conn, "embedding_model_id")?,
        "search_modes": ["hybrid", "vector", "keyword"],
        "default_search_mode": "hybrid",
        "documents": docs,
        "chunks": chunks,
        "chunk_embeddings": embeddings,
        "types": types,
        "default_search_policy": {
            "excluded_types": DEFAULT_EXCLUDED_TYPES,
            "old_content_cutoff": OLD_CONTENT_CUTOFF,
            "old_content_exception_types": [LEGISLATION_TYPE],
        }
    });
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&payload)?),
        OutputFormat::Markdown => {
            let mut out = String::new();
            out.push_str(&format!("data_dir: `{}`\n", data_dir()?.display()));
            out.push_str(&format!(
                "index_version: `{}`\n",
                payload["index_version"].as_str().unwrap_or("?")
            ));
            out.push_str(&format!(
                "last_update_at: `{}`\n",
                payload["last_update_at"].as_str().unwrap_or("?")
            ));
            out.push_str(&format!(
                "embedding_model_id: `{}`\n",
                payload["embedding_model_id"].as_str().unwrap_or("?")
            ));
            out.push_str(&format!("documents: `{}`\n", docs));
            out.push_str(&format!("chunks: `{}`\n", chunks));
            out.push_str(&format!("chunk_embeddings: `{}`\n", embeddings));
            out.push_str("default_search_mode: `hybrid`\n");
            out.push_str(&format!(
                "default_search: excludes `{}` and dates before `{}` except `{}`\n",
                DEFAULT_EXCLUDED_TYPES.join(", "),
                OLD_CONTENT_CUTOFF,
                LEGISLATION_TYPE
            ));
            Ok(out)
        }
    }
}

fn doctor(rollback: bool) -> Result<()> {
    if rollback {
        let backup = backups_dir()?.join("ato.db.prev");
        if !backup.exists() {
            bail!("no backup found at {}", backup.display());
        }
        fs::copy(&backup, db_path()?)?;
        println!("rollback complete.");
        return Ok(());
    }
    let conn = open_read()?;
    let docs: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
    if docs == 0 || chunks == 0 {
        bail!("corpus is empty: documents={docs}, chunks={chunks}");
    }
    println!("documents: {docs}");
    println!("chunks: {chunks}");
    let model_id = get_meta(&conn, "embedding_model_id")?.unwrap_or_default();
    if model_id.starts_with("embeddinggemma") {
        ensure_vector_search_ready(&conn)?;
        let embeddings: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |r| r.get(0))?;
        println!("chunk_embeddings: {embeddings}");
        println!("semantic_search: ready");
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    schema_version: i64,
    index_version: String,
    created_at: String,
    #[serde(default)]
    min_client_version: String,
    model: ModelInfo,
    #[serde(default)]
    documents: Vec<DocRef>,
    #[serde(default)]
    packs: Vec<PackInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    id: String,
    sha256: String,
    size: u64,
    url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DocRef {
    doc_id: String,
    content_hash: String,
    pack_sha8: String,
    offset: u64,
    length: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PackInfo {
    sha8: String,
    sha256: String,
    size: u64,
    url: String,
}

struct HfModelFile {
    path: &'static str,
    output_name: &'static str,
    sha256: &'static str,
    size: u64,
}

const EMBEDDINGGEMMA_HF_FILES: &[HfModelFile] = &[
    HfModelFile {
        path: "onnx/model_quantized.onnx",
        output_name: "model_quantized.onnx",
        sha256: "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8",
        size: 567_874,
    },
    HfModelFile {
        path: "onnx/model_quantized.onnx_data",
        output_name: "model_quantized.onnx_data",
        sha256: "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0",
        size: 308_890_624,
    },
    HfModelFile {
        path: "tokenizer.json",
        output_name: "tokenizer.json",
        sha256: "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
        size: 20_323_312,
    },
];

#[derive(Default)]
struct UpdateStats {
    added: usize,
    changed: usize,
    removed: usize,
    bytes_downloaded: u64,
}

fn apply_update(manifest_url: &str) -> Result<UpdateStats> {
    let lock = lock_file()?;
    let result = apply_update_locked(manifest_url);
    lock.unlock()?;
    result
}

fn update_before_serve() -> Result<()> {
    let url = default_manifest_url();
    match apply_update(&url) {
        Ok(stats) => {
            eprintln!(
                "ato-mcp serve: update complete (+{} ~{} -{}, {:.2} MB downloaded)",
                stats.added,
                stats.changed,
                stats.removed,
                stats.bytes_downloaded as f64 / 1_000_000.0
            );
            Ok(())
        }
        Err(err) => {
            if db_path()?.exists() {
                eprintln!("ato-mcp serve: update failed; serving installed corpus: {err}");
                Ok(())
            } else {
                Err(err).context("ato-mcp serve could not install the corpus before startup")
            }
        }
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn ensure_installed_db() -> Result<()> {
    if db_path()?.exists() {
        Ok(())
    } else {
        bail!("no live DB found; run `ato-mcp init` before serving offline")
    }
}

fn apply_update_locked(manifest_url: &str) -> Result<UpdateStats> {
    let staging = staging_dir()?;
    let manifest_context = UrlContext::from_manifest_url(manifest_url);
    let manifest_bytes = fetch_bytes(manifest_url, &manifest_context)
        .with_context(|| format!("fetching manifest from {manifest_url}"))?;
    let new_manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let old_manifest = load_installed_manifest()?;

    ensure_model(&new_manifest, &manifest_context, &staging)?;

    let db = db_path()?;
    let had_existing_db = db.exists();
    let conn = open_write()?;
    init_db(&conn)?;
    let (added, mut changed, removed) = diff_manifests(old_manifest.as_ref(), &new_manifest);
    if semantic_backfill_required(&conn, &new_manifest)? {
        let added_ids = added
            .iter()
            .map(|doc| doc.doc_id.as_str())
            .collect::<HashSet<_>>();
        changed = new_manifest
            .documents
            .iter()
            .filter(|doc| !added_ids.contains(doc.doc_id.as_str()))
            .cloned()
            .collect();
    }

    let backup = backups_dir()?.join("ato.db.prev");
    if had_existing_db {
        fs::copy(&db, &backup)?;
    }

    let mut bytes_downloaded = manifest_bytes.len() as u64;
    let tx = conn.unchecked_transaction()?;
    let apply_result = (|| -> Result<()> {
        for doc_id in &removed {
            delete_doc(&tx, doc_id)?;
        }
        for doc in &changed {
            delete_doc(&tx, &doc.doc_id)?;
        }

        let mut pack_to_refs: HashMap<String, Vec<DocRef>> = HashMap::new();
        for doc in added.iter().chain(changed.iter()) {
            pack_to_refs
                .entry(doc.pack_sha8.clone())
                .or_default()
                .push(doc.clone());
        }
        let pack_index: HashMap<String, PackInfo> = new_manifest
            .packs
            .iter()
            .map(|p| (p.sha8.clone(), p.clone()))
            .collect();
        for (sha8, refs) in pack_to_refs {
            let info = pack_index
                .get(&sha8)
                .ok_or_else(|| anyhow!("manifest missing pack info for {sha8}"))?;
            let pack_url = resolve_manifest_asset(&info.url, &manifest_context);
            let pack_bytes = fetch_bytes(&pack_url, &manifest_context)
                .with_context(|| format!("fetching pack {}", info.url))?;
            if !info.sha256.is_empty() {
                verify_sha256_bytes(&pack_bytes, &info.sha256)
                    .with_context(|| format!("verifying {}", info.url))?;
            }
            bytes_downloaded += pack_bytes.len() as u64;
            for doc_ref in refs {
                let record =
                    read_record_from_pack_bytes(&pack_bytes, doc_ref.offset, doc_ref.length)?;
                insert_record(&tx, &record, &doc_ref)?;
            }
        }
        set_meta(&tx, "index_version", &new_manifest.index_version)?;
        set_meta(&tx, "embedding_model_id", &new_manifest.model.id)?;
        set_meta(&tx, "last_update_at", &Utc::now().to_rfc3339())?;
        verify_semantic_install(&tx, &new_manifest)?;
        Ok(())
    })();

    if let Err(err) = apply_result {
        tx.rollback()?;
        if backup.exists() {
            fs::copy(&backup, db_path()?)?;
        }
        return Err(err);
    }
    tx.commit()?;
    let manifest_json = serde_json::to_vec_pretty(&new_manifest)?;
    fs::write(installed_manifest_path()?, manifest_json)?;
    Ok(UpdateStats {
        added: added.len(),
        changed: changed.len(),
        removed: removed.len(),
        bytes_downloaded,
    })
}

fn semantic_backfill_required(conn: &Connection, manifest: &Manifest) -> Result<bool> {
    if !manifest.model.id.starts_with("embeddinggemma") {
        return Ok(false);
    }
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    if chunks == 0 {
        return Ok(false);
    }
    let embeddings = chunk_embedding_count(conn)?;
    Ok(embeddings < chunks)
}

fn verify_semantic_install(conn: &Connection, manifest: &Manifest) -> Result<()> {
    if !manifest.model.id.starts_with("embeddinggemma") {
        return Ok(());
    }
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    let embeddings = chunk_embedding_count(conn)?;
    if embeddings != chunks {
        bail!(
            "semantic corpus install incomplete: chunk_embeddings={embeddings}, chunks={chunks}; rebuild the release packs with embedding_b64"
        );
    }
    Ok(())
}

fn chunk_embedding_count(conn: &Connection) -> Result<i64> {
    if table_exists(conn, "chunk_embeddings")? {
        conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
            row.get(0)
        })
        .map_err(Into::into)
    } else {
        Ok(0)
    }
}

fn load_installed_manifest() -> Result<Option<Manifest>> {
    let path = installed_manifest_path()?;
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

fn diff_manifests(
    old: Option<&Manifest>,
    new: &Manifest,
) -> (Vec<DocRef>, Vec<DocRef>, Vec<String>) {
    let old_docs: HashMap<&str, &DocRef> = old
        .map(|m| m.documents.iter().map(|d| (d.doc_id.as_str(), d)).collect())
        .unwrap_or_default();
    let new_docs: HashMap<&str, &DocRef> = new
        .documents
        .iter()
        .map(|d| (d.doc_id.as_str(), d))
        .collect();
    let mut added = Vec::new();
    let mut changed = Vec::new();
    for doc in &new.documents {
        match old_docs.get(doc.doc_id.as_str()) {
            None => added.push(doc.clone()),
            Some(old_doc) if old_doc.content_hash != doc.content_hash => changed.push(doc.clone()),
            _ => {}
        }
    }
    let removed = old_docs
        .keys()
        .filter(|doc_id| !new_docs.contains_key(**doc_id))
        .map(|doc_id| (*doc_id).to_string())
        .collect();
    (added, changed, removed)
}

#[derive(Clone)]
struct UrlContext {
    manifest_dir: Option<PathBuf>,
    manifest_base_url: Option<String>,
}

impl UrlContext {
    fn from_manifest_url(manifest_url: &str) -> Self {
        if let Some(path) = local_path_from_urlish(manifest_url) {
            return Self {
                manifest_dir: path.parent().map(Path::to_path_buf),
                manifest_base_url: None,
            };
        }
        let manifest_base_url = manifest_url
            .rsplit_once('/')
            .map(|(base, _)| base.to_string());
        Self {
            manifest_dir: None,
            manifest_base_url,
        }
    }
}

fn resolve_manifest_asset(asset_url: &str, context: &UrlContext) -> String {
    if asset_url.starts_with("http://")
        || asset_url.starts_with("https://")
        || asset_url.starts_with("file://")
    {
        return asset_url.to_string();
    }
    if let Some(dir) = &context.manifest_dir {
        return dir.join(asset_url).display().to_string();
    }
    if let Some(base) = &context.manifest_base_url {
        return format!(
            "{}/{}",
            base.trim_end_matches('/'),
            asset_url.trim_start_matches('/')
        );
    }
    asset_url.to_string()
}

fn local_path_from_urlish(value: &str) -> Option<PathBuf> {
    if let Ok(url) = Url::parse(value) {
        if url.scheme() == "file" {
            return url.to_file_path().ok();
        }
        return None;
    }
    let path = PathBuf::from(value);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn fetch_bytes(url_or_path: &str, context: &UrlContext) -> Result<Vec<u8>> {
    if let Some(path) = local_path_from_urlish(url_or_path) {
        return Ok(fs::read(path)?);
    }
    if let Some(dir) = &context.manifest_dir {
        if let Some(name) = url_or_path.rsplit('/').next() {
            for candidate in [dir.join(name), dir.join("packs").join(name)] {
                if candidate.exists() {
                    return Ok(fs::read(candidate)?);
                }
            }
        }
    }
    let client = http_client()?;
    let mut resp = client.get(url_or_path).send()?.error_for_status().with_context(|| {
        format!(
            "download failed for {url_or_path}. This Rust client does not read GitHub tokens or invoke gh; use a public release, an authenticated mirror, or a file URL."
        )
    })?;
    let mut out = Vec::new();
    resp.copy_to(&mut out)?;
    Ok(out)
}

fn fetch_to_file(url_or_path: &str, context: &UrlContext, dest: &Path) -> Result<u64> {
    if let Some(path) = local_path_from_urlish(url_or_path) {
        fs::copy(path, dest).map_err(Into::into)
    } else if let Some(dir) = &context.manifest_dir {
        if let Some(name) = url_or_path.rsplit('/').next() {
            for candidate in [dir.join(name), dir.join("packs").join(name)] {
                if candidate.exists() {
                    return fs::copy(candidate, dest).map_err(Into::into);
                }
            }
        }
        fetch_http_to_file(url_or_path, dest)
    } else {
        fetch_http_to_file(url_or_path, dest)
    }
}

fn fetch_http_to_file(url: &str, dest: &Path) -> Result<u64> {
    let client = http_client()?;
    let mut resp = client.get(url).send()?.error_for_status().with_context(|| {
        format!(
            "download failed for {url}. This Rust client does not read GitHub tokens or invoke gh; use a public release, an authenticated mirror, or a file URL."
        )
    })?;
    let mut file = File::create(dest)?;
    Ok(resp.copy_to(&mut file)?)
}

fn parse_hf_model_url(value: &str) -> Option<(&str, &str)> {
    let spec = value.strip_prefix("hf://")?;
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    if repo.is_empty() || revision.is_empty() {
        None
    } else {
        Some((repo, revision))
    }
}

fn hf_resolve_url(repo: &str, revision: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{path}")
}

fn http_client() -> Result<Client> {
    Ok(Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .build()?)
}

fn verify_sha256_bytes(bytes: &[u8], expected: &str) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        bail!("sha256 mismatch: got {actual}, expected {expected}");
    }
    Ok(())
}

fn verify_sha256_file(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        bail!(
            "sha256 mismatch for {}: got {actual}, expected {expected}",
            path.display()
        );
    }
    Ok(())
}

fn ensure_model(manifest: &Manifest, context: &UrlContext, staging: &Path) -> Result<()> {
    if !manifest.model.id.starts_with("embeddinggemma") {
        bail!(
            "semantic search requires an EmbeddingGemma model bundle; manifest uses `{}`",
            manifest.model.id
        );
    }
    let live_model = model_path()?;
    let tokenizer = tokenizer_path()?;
    let marker = live_dir()?.join(".model.sha256");
    let marker_value =
        if manifest.model.sha256.is_empty() && parse_hf_model_url(&manifest.model.url).is_some() {
            EMBEDDINGGEMMA_HF_FINGERPRINT
        } else {
            manifest.model.sha256.as_str()
        };
    if live_model.exists()
        && tokenizer.exists()
        && marker.exists()
        && fs::read_to_string(&marker)?.trim() == marker_value
    {
        return Ok(());
    }

    if let Some((repo, revision)) = parse_hf_model_url(&manifest.model.url) {
        install_hf_embedding_model(repo, revision, staging)?;
        fs::write(marker, marker_value)?;
        return Ok(());
    }

    let bundle_url = resolve_manifest_asset(&manifest.model.url, context);
    let bundle = staging.join("model-bundle.tar.zst.part");
    fetch_to_file(&bundle_url, context, &bundle)?;
    if !manifest.model.sha256.is_empty() {
        verify_sha256_file(&bundle, &manifest.model.sha256)?;
    }
    let extract_dir = staging.join("model-bundle-extracted");
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::create_dir_all(&extract_dir)?;
    let bundle_file = File::open(&bundle)?;
    let decoder = zstd::stream::read::Decoder::new(bundle_file)?;
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(&extract_dir)?;

    for entry in fs::read_dir(&extract_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            fs::rename(entry.path(), live_dir()?.join(entry.file_name()))?;
        }
    }
    ensure_model_alias()?;
    fs::write(marker, marker_value)?;
    let _ = fs::remove_file(bundle);
    let _ = fs::remove_dir_all(extract_dir);
    Ok(())
}

fn install_hf_embedding_model(repo: &str, revision: &str, staging: &Path) -> Result<()> {
    fs::create_dir_all(staging)?;
    for file in EMBEDDINGGEMMA_HF_FILES {
        let url = hf_resolve_url(repo, revision, file.path);
        let part = staging.join(format!("{}.part", file.output_name));
        fetch_http_to_file(&url, &part)
            .with_context(|| format!("downloading Hugging Face model file {}", file.path))?;
        verify_sha256_file(&part, file.sha256)
            .with_context(|| format!("verifying Hugging Face model file {}", file.path))?;
        let size = part.metadata()?.len();
        if size != file.size {
            bail!(
                "size mismatch for Hugging Face model file {}: got {}, expected {}",
                file.path,
                size,
                file.size
            );
        }
        let dest = live_dir()?.join(file.output_name);
        if dest.exists() {
            fs::remove_file(&dest)?;
        }
        fs::rename(&part, dest)?;
    }
    ensure_model_alias()
}

fn ensure_model_alias() -> Result<()> {
    let model_link = live_dir()?.join("model.onnx");
    let quantized = live_dir()?.join("model_quantized.onnx");
    if !quantized.exists() {
        bail!("model_quantized.onnx missing after model install");
    }
    if model_link.exists() {
        fs::remove_file(&model_link)?;
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink("model_quantized.onnx", &model_link)?;
    #[cfg(not(unix))]
    fs::copy(&quantized, &model_link)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct PackRecord {
    doc_id: String,
    #[serde(default, rename = "type")]
    doc_type: String,
    title: String,
    date: Option<String>,
    downloaded_at: String,
    content_hash: String,
    #[serde(default)]
    chunks: Vec<PackChunk>,
}

#[derive(Debug, Deserialize)]
struct PackChunk {
    ord: i64,
    #[serde(default)]
    heading_path: Option<String>,
    #[serde(default)]
    anchor: Option<String>,
    text: String,
    #[serde(default)]
    embedding_b64: Option<String>,
}

fn read_record_from_pack_bytes(pack: &[u8], offset: u64, length: u64) -> Result<PackRecord> {
    let start = offset as usize;
    let end = start + length as usize;
    if end > pack.len() || length < 4 {
        bail!(
            "pack range out of bounds: offset={offset}, length={length}, pack_len={}",
            pack.len()
        );
    }
    let blob = &pack[start..end];
    let payload_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    if payload_len + 4 != blob.len() {
        bail!(
            "pack record length mismatch: header says {}, range says {}",
            payload_len + 4,
            blob.len()
        );
    }
    let decoded = zstd::stream::decode_all(Cursor::new(&blob[4..]))?;
    Ok(serde_json::from_slice(&decoded)?)
}

fn insert_record(conn: &Connection, record: &PackRecord, doc_ref: &DocRef) -> Result<()> {
    let doc_type = if record.doc_type.is_empty() {
        "Unknown"
    } else {
        &record.doc_type
    };
    conn.execute(
        r#"
        INSERT OR REPLACE INTO documents
            (doc_id, type, title, date, downloaded_at, content_hash, pack_sha8)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
        params![
            record.doc_id,
            doc_type,
            record.title,
            record.date,
            record.downloaded_at,
            record.content_hash,
            doc_ref.pack_sha8,
        ],
    )?;
    let headings = record
        .chunks
        .iter()
        .filter_map(|c| c.heading_path.as_deref())
        .collect::<Vec<_>>()
        .join(" ");
    conn.execute(
        "INSERT INTO title_fts (doc_id, title, headings) VALUES (?, ?, ?)",
        params![record.doc_id, record.title, headings],
    )?;
    for chunk in &record.chunks {
        let blob = compress_text(&chunk.text)?;
        conn.execute(
            "INSERT INTO chunks (doc_id, ord, heading_path, anchor, text) VALUES (?, ?, ?, ?, ?)",
            params![
                record.doc_id,
                chunk.ord,
                chunk.heading_path,
                chunk.anchor,
                blob,
            ],
        )?;
        let rowid = conn.last_insert_rowid();
        if let Some(embedding_b64) = &chunk.embedding_b64 {
            let embedding = decode_embedding_b64(embedding_b64)?;
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                params![rowid, embedding],
            )?;
        }
        conn.execute(
            "INSERT INTO chunks_fts (rowid, text, heading_path) VALUES (?, ?, ?)",
            params![
                rowid,
                chunk.text,
                chunk.heading_path.as_deref().unwrap_or("")
            ],
        )?;
    }
    Ok(())
}

fn decode_embedding_b64(value: &str) -> Result<Vec<u8>> {
    let embedding = base64::engine::general_purpose::STANDARD
        .decode(value)
        .context("decoding chunk embedding")?;
    if embedding.len() != EMBEDDING_DIM {
        bail!(
            "invalid chunk embedding length: got {}, expected {}",
            embedding.len(),
            EMBEDDING_DIM
        );
    }
    Ok(embedding)
}

fn delete_doc(conn: &Connection, doc_id: &str) -> Result<()> {
    let chunk_ids = {
        let mut stmt = conn.prepare("SELECT chunk_id FROM chunks WHERE doc_id = ?")?;
        let rows = stmt
            .query_map([doc_id], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for chunk_id in chunk_ids {
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
            [chunk_id],
        )?;
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", [chunk_id])?;
    }
    conn.execute("DELETE FROM title_fts WHERE doc_id = ?", [doc_id])?;
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])?;
    conn.execute("DELETE FROM documents WHERE doc_id = ?", [doc_id])?;
    Ok(())
}

fn serve() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = ServerState::default();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Result<JsonValue> = serde_json::from_str(&line);
        let response = match parsed {
            Ok(message) => handle_rpc(message, &mut state),
            Err(err) => Some(json_rpc_error(
                JsonValue::Null,
                -32700,
                &format!("parse error: {err}"),
            )),
        };
        if let Some(response) = response {
            serde_json::to_writer(&mut stdout, &response)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn handle_rpc(message: JsonValue, state: &mut ServerState) -> Option<JsonValue> {
    if message.is_array() {
        let responses: Vec<JsonValue> = message
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| handle_single_rpc(m.clone(), state))
            .collect();
        if responses.is_empty() {
            None
        } else {
            Some(JsonValue::Array(responses))
        }
    } else {
        handle_single_rpc(message, state)
    }
}

fn handle_single_rpc(message: JsonValue, state: &mut ServerState) -> Option<JsonValue> {
    let id = message.get("id").cloned();
    let Some(method) = message.get("method").and_then(|m| m.as_str()) else {
        return id.map(|id| json_rpc_error(id, -32600, "invalid request"));
    };
    let id = id?;
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "ato-mcp", "version": env!("CARGO_PKG_VERSION") },
            "instructions": server_instructions(),
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_descriptors() })),
        "tools/call" => {
            let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
            call_tool(params, state)
        }
        _ => Err(anyhow!("method not found: {method}")),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err(err) => json_rpc_error(id, -32000, &err.to_string()),
    })
}

fn json_rpc_error(id: JsonValue, code: i64, message: &str) -> JsonValue {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn call_tool(params: JsonValue, state: &mut ServerState) -> Result<JsonValue> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("tools/call missing params.name"))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let text = match name {
        "search" => {
            let query = required_str(&args, "query")?;
            let types = optional_string_array(&args, "types")?;
            let mode = match args
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("hybrid")
            {
                "hybrid" => SearchMode::Hybrid,
                "vector" => SearchMode::Vector,
                "keyword" => SearchMode::Keyword,
                other => bail!("mode must be one of hybrid, vector, keyword; got `{other}`"),
            };
            let sort_by = match args
                .get("sort_by")
                .and_then(|v| v.as_str())
                .unwrap_or("relevance")
            {
                "recency" => SortBy::Recency,
                _ => SortBy::Relevance,
            };
            let format = output_format_arg(&args);
            search(
                query,
                SearchOptions {
                    k: optional_usize(&args, "k").unwrap_or(DEFAULT_K),
                    types: types.as_deref(),
                    date_from: args.get("date_from").and_then(|v| v.as_str()),
                    date_to: args.get("date_to").and_then(|v| v.as_str()),
                    doc_scope: args.get("doc_scope").and_then(|v| v.as_str()),
                    mode,
                    sort_by,
                    include_old: optional_bool(&args, "include_old").unwrap_or(false),
                    format,
                },
                Some(state),
            )?
        }
        "search_titles" => {
            let query = required_str(&args, "query")?;
            let types = optional_string_array(&args, "types")?;
            search_titles(
                query,
                optional_usize(&args, "k").unwrap_or(20),
                types.as_deref(),
                optional_bool(&args, "include_old").unwrap_or(false),
                output_format_arg(&args),
            )?
        }
        "get_document" => {
            let doc_id = required_str(&args, "doc_id")?;
            let format = match args
                .get("format")
                .and_then(|v| v.as_str())
                .unwrap_or("outline")
            {
                "json" => DocumentFormat::Json,
                "markdown" => DocumentFormat::Markdown,
                "card" => DocumentFormat::Card,
                "outline" => DocumentFormat::Outline,
                other => {
                    bail!("format must be one of outline, card, markdown, json; got `{other}`")
                }
            };
            get_document(
                doc_id,
                GetDocumentOptions {
                    format,
                    anchor: args.get("anchor").and_then(|v| v.as_str()),
                    heading_path: args.get("heading_path").and_then(|v| v.as_str()),
                    from_ord: args.get("from_ord").and_then(|v| v.as_i64()),
                    include_children: optional_bool(&args, "include_children").unwrap_or(false),
                    count: optional_usize(&args, "count"),
                    max_chars: optional_usize(&args, "max_chars"),
                },
            )?
        }
        "get_chunks" => get_chunks_mcp(&args)?,
        "whats_new" => {
            let types = optional_string_array(&args, "types")?;
            whats_new(
                args.get("since").and_then(|v| v.as_str()),
                args.get("before").and_then(|v| v.as_str()),
                optional_usize(&args, "limit").unwrap_or(50),
                types.as_deref(),
                output_format_arg(&args),
            )?
        }
        "stats" => stats(output_format_arg(&args))?,
        _ => bail!("unknown tool: {name}"),
    };
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false,
    }))
}

fn required_str<'a>(args: &'a JsonValue, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing required string argument `{name}`"))
}

fn optional_usize(args: &JsonValue, name: &str) -> Option<usize> {
    args.get(name).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn optional_bool(args: &JsonValue, name: &str) -> Option<bool> {
    args.get(name).and_then(|v| v.as_bool())
}

fn optional_string_array(args: &JsonValue, name: &str) -> Result<Option<Vec<String>>> {
    let Some(value) = args.get(name) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("`{name}` must be an array of strings"))?;
    let mut out = Vec::new();
    for item in array {
        out.push(
            item.as_str()
                .ok_or_else(|| anyhow!("`{name}` must be an array of strings"))?
                .to_string(),
        );
    }
    Ok(Some(out))
}

fn output_format_arg(args: &JsonValue) -> OutputFormat {
    match args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("markdown")
    {
        "json" => OutputFormat::Json,
        _ => OutputFormat::Markdown,
    }
}

fn get_chunks_mcp(args: &JsonValue) -> Result<String> {
    let ids = args
        .get("chunk_ids")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing chunk_ids array"))?;
    let chunk_ids = ids
        .iter()
        .map(|v| {
            v.as_i64()
                .ok_or_else(|| anyhow!("chunk_ids must contain integers"))
        })
        .collect::<Result<Vec<_>>>()?;
    get_chunks(
        &chunk_ids,
        GetChunksOptions {
            before: optional_usize(args, "before").unwrap_or(0).min(20),
            after: optional_usize(args, "after").unwrap_or(0).min(20),
            max_chars: optional_usize(args, "max_chars"),
            format: output_format_arg(args),
        },
    )
}

struct GetChunksOptions {
    before: usize,
    after: usize,
    max_chars: Option<usize>,
    format: OutputFormat,
}

#[derive(Debug, Clone, Serialize)]
struct HydratedChunk {
    chunk_id: i64,
    requested: bool,
    doc_id: String,
    #[serde(rename = "type")]
    doc_type: String,
    title: String,
    date: Option<String>,
    ord: i64,
    heading_path: String,
    anchor: Option<String>,
    canonical_url: String,
    text: String,
}

#[derive(Debug, Clone)]
struct ChunkPointer {
    chunk_id: i64,
    doc_id: String,
    ord: i64,
}

fn get_chunks(chunk_ids: &[i64], opts: GetChunksOptions) -> Result<String> {
    if chunk_ids.is_empty() {
        return Ok("_No chunk ids provided._".to_string());
    }
    let conn = open_read()?;
    let placeholders = vec!["?"; chunk_ids.len()].join(",");
    let sql =
        format!("SELECT chunk_id, doc_id, ord FROM chunks WHERE chunk_id IN ({placeholders})");
    let params_vec: Vec<Value> = chunk_ids.iter().map(|id| Value::Integer(*id)).collect();
    let mut stmt = conn.prepare(&sql)?;
    let pointers = stmt
        .query_map(params_from_iter(params_vec), |row| {
            Ok(ChunkPointer {
                chunk_id: row.get("chunk_id")?,
                doc_id: row.get("doc_id")?,
                ord: row.get("ord")?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .map(|pointer| (pointer.chunk_id, pointer))
        .collect::<HashMap<_, _>>();

    let mut seen = HashSet::new();
    let requested_set: HashSet<i64> = chunk_ids.iter().copied().collect();
    let mut out = Vec::new();
    let mut returned_chars = 0usize;
    let mut truncated_at: Option<HydratedChunk> = None;
    for requested_id in chunk_ids {
        let Some(pointer) = pointers.get(requested_id) else {
            continue;
        };
        let from_ord = pointer.ord.saturating_sub(opts.before as i64);
        let to_ord = pointer.ord.saturating_add(opts.after as i64);
        for mut chunk in load_chunks_by_ord_range(&conn, &pointer.doc_id, from_ord, to_ord)? {
            chunk.requested = requested_set.contains(&chunk.chunk_id);
            if !seen.insert(chunk.chunk_id) {
                continue;
            }
            let projected_chars = returned_chars + chunk.text.len();
            if opts
                .max_chars
                .is_some_and(|max| !out.is_empty() && projected_chars > max)
            {
                truncated_at = Some(chunk);
                break;
            }
            returned_chars = projected_chars;
            out.push(chunk);
        }
        if truncated_at.is_some() {
            break;
        }
    }
    let next_call = truncated_at.as_ref().map(|chunk| {
        format!(
            "get_document(doc_id=\"{}\", format=\"json\", from_ord={}, max_chars={})",
            chunk.doc_id,
            chunk.ord,
            opts.max_chars.unwrap_or(20_000)
        )
    });
    let returned = out.len();
    if matches!(opts.format, OutputFormat::Json) {
        return Ok(serde_json::to_string_pretty(&json!({
            "requested_chunk_ids": chunk_ids,
            "context": {
                "before": opts.before,
                "after": opts.after,
            },
            "chunks": out,
            "meta": {
                "returned": returned,
                "returned_chars": returned_chars,
                "truncated": truncated_at.is_some(),
                "truncated_at": truncated_at.as_ref().map(|chunk| json!({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "ord": chunk.ord,
                })),
                "next_call": next_call,
            },
        }))?);
    }
    let mut text = String::new();
    for chunk in out {
        text.push_str(&format!(
            "**{}** ([{}]({})) - chunk `{}` / ord `{}` - {}\n\n{}\n\n---\n",
            chunk.title,
            chunk.doc_id,
            chunk.canonical_url,
            chunk.chunk_id,
            chunk.ord,
            chunk.heading_path,
            chunk.text
        ));
    }
    if let Some(next_call) = next_call {
        text.push_str(&format!("_Truncated. Continue with `{next_call}`._\n"));
    }
    Ok(text)
}

fn load_chunks_by_ord_range(
    conn: &Connection,
    doc_id: &str,
    from_ord: i64,
    to_ord: i64,
) -> Result<Vec<HydratedChunk>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
               d.type, d.title, d.date
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.doc_id = ? AND c.ord BETWEEN ? AND ?
        ORDER BY c.ord ASC
        "#,
    )?;
    let rows = stmt.query_map(params![doc_id, from_ord, to_ord], |row| {
        let doc_id: String = row.get("doc_id")?;
        Ok((
            row.get::<_, i64>("chunk_id")?,
            doc_id,
            row.get::<_, String>("type")?,
            row.get::<_, String>("title")?,
            row.get::<_, Option<String>>("date")?,
            row.get::<_, i64>("ord")?,
            row.get::<_, Option<String>>("heading_path")?
                .unwrap_or_default(),
            row.get::<_, Option<String>>("anchor")?,
            row.get::<_, Vec<u8>>("text")?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (chunk_id, doc_id, doc_type, title, date, ord, heading_path, anchor, text_blob) = row?;
        out.push(HydratedChunk {
            chunk_id,
            requested: false,
            doc_id: doc_id.clone(),
            doc_type,
            title,
            date,
            ord,
            heading_path,
            anchor,
            canonical_url: canonical_url(&doc_id),
            text: decompress_text(text_blob)?,
        });
    }
    Ok(out)
}

fn server_instructions() -> String {
    match stats(OutputFormat::Json)
        .ok()
        .and_then(|s| serde_json::from_str::<JsonValue>(&s).ok())
    {
        Some(s) => format!(
            "ATO legal corpus. Documents: {}, chunks: {}. Index: {}. Default search excludes Edited_private_advice and content dated before {} except legislation. Use include_old=true when older authorities are required.",
            s["documents"].as_i64().unwrap_or(0),
            s["chunks"].as_i64().unwrap_or(0),
            s["index_version"].as_str().unwrap_or("?"),
            OLD_CONTENT_CUTOFF,
        ),
        None => "ATO legal corpus. Run `ato-mcp init` before serving.".to_string(),
    }
}

fn tool_descriptors() -> JsonValue {
    json!([
        {
            "name": "search",
            "description": "Search ATO legal documents. Defaults to hybrid semantic-plus-lexical ranking. Explicit mode=keyword is allowed, but hybrid/vector never fall back to keyword when semantic search is unavailable.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "doc_scope": {"type": "string"},
                    "mode": {"type": "string", "enum": ["hybrid", "vector", "keyword"]},
                    "sort_by": {"type": "string", "enum": ["relevance", "recency"]},
                    "include_old": {"type": "boolean"},
                    "format": {"type": "string", "enum": ["markdown", "json"]}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_titles",
            "description": "Fast title-only search for citations, section numbers, and case names.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 100},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "include_old": {"type": "boolean"},
                    "format": {"type": "string", "enum": ["markdown", "json"]}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_document",
            "description": "Fetch a document outline, full body, section, or ordinal range.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["outline", "card", "markdown", "json"]},
                    "anchor": {"type": "string"},
                    "heading_path": {"type": "string"},
                    "from_ord": {"type": "integer"},
                    "include_children": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "max_chars": {"type": "integer"}
                },
                "required": ["doc_id"]
            }
        },
        {
            "name": "get_chunks",
            "description": "Fetch exact chunks by chunk id from search results, optionally with before/after neighbor context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}},
                    "before": {"type": "integer", "minimum": 0, "maximum": 20},
                    "after": {"type": "integer", "minimum": 0, "maximum": 20},
                    "max_chars": {"type": "integer", "minimum": 1},
                    "format": {"type": "string", "enum": ["markdown", "json"]}
                },
                "required": ["chunk_ids"]
            }
        },
        {
            "name": "whats_new",
            "description": "Most recently published documents by corpus date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since": {"type": "string"},
                    "before": {"type": "string"},
                    "limit": {"type": "integer"},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "format": {"type": "string", "enum": ["markdown", "json"]}
                }
            }
        },
        {
            "name": "stats",
            "description": "Index version, document counts, and default search policy.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["markdown", "json"]}
                }
            }
        }
    ])
}
