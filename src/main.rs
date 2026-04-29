use anyhow::{anyhow, bail, Context, Result};
use chrono::{Datelike, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
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
use url::Url;

const APP_NAME: &str = "ato-mcp";
const DEFAULT_RELEASES_URL: &str = "https://github.com/gunba/ato-mcp/releases/latest/download";
const DEFAULT_K: usize = 8;
const MAX_K: usize = 50;
const SNIPPET_CHARS: usize = 280;
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
    Serve,
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
        #[arg(long, default_value = "relevance")]
        sort_by: SortBy,
        #[arg(long)]
        include_old: bool,
        #[arg(long, default_value = "markdown")]
        format: OutputFormat,
    },
    /// Search document titles only.
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
enum DocumentFormat {
    Outline,
    Markdown,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve => serve(),
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
                        sort_by,
                        include_old,
                        format,
                    },
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
            limit,
            types,
            format,
        } => {
            let types = empty_vec_as_none(types);
            println!(
                "{}",
                whats_new(since.as_deref(), limit, types.as_deref(), format)?
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
}

struct SearchOptions<'a> {
    k: usize,
    types: Option<&'a [String]>,
    date_from: Option<&'a str>,
    date_to: Option<&'a str>,
    doc_scope: Option<&'a str>,
    sort_by: SortBy,
    include_old: bool,
    format: OutputFormat,
}

fn search(query: &str, opts: SearchOptions<'_>) -> Result<String> {
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
    let where_filter = if filter.sql.is_empty() {
        String::new()
    } else {
        format!(" AND {}", filter.sql)
    };
    let internal_k = std::cmp::max(k * 5, 50);
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
    params_vec.extend(filter.params);
    params_vec.push(Value::Integer(internal_k as i64));

    let mut records = Vec::new();
    let mut stmt = conn.prepare(&sql)?;
    let rows = match stmt.query_map(params_from_iter(params_vec), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
    }) {
        Ok(rows) => rows.collect::<rusqlite::Result<Vec<_>>>()?,
        Err(rusqlite::Error::SqliteFailure(_, _)) => Vec::new(),
        Err(err) => return Err(err.into()),
    };

    let frontier = match opts.sort_by {
        SortBy::Relevance => std::cmp::max(k * 2, 20),
        SortBy::Recency => k * 3,
    };
    for (chunk_id, bm25_score) in rows.into_iter().take(frontier) {
        if let Some(mut hit) = load_hit(&conn, chunk_id, query)? {
            hit.score = Some((-bm25_score) * type_weight(&hit.doc_type));
            records.push(hit);
        }
    }

    match opts.sort_by {
        SortBy::Relevance => {
            apply_recency_boost(&mut records);
            records.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        SortBy::Recency => {
            records.sort_by(|a, b| b.date.cmp(&a.date));
        }
    }
    records.truncate(k);

    match opts.format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&json!({
            "query": query,
            "mode": "keyword",
            "filters": {
                "excluded_by_default": DEFAULT_EXCLUDED_TYPES,
                "old_content_cutoff": if opts.include_old { JsonValue::Null } else { json!(OLD_CONTENT_CUTOFF) },
            },
            "hits": records,
        }))?),
        OutputFormat::Markdown => Ok(format_hits_markdown(&records)),
    }
}

fn type_weight(doc_type: &str) -> f64 {
    match doc_type {
        "Practical_compliance_guidelines" => 1.8,
        "Taxpayer_alerts" => 1.7,
        "Public_rulings" => 1.6,
        "Law_administration_practice_statements" => 1.4,
        "Decision_impact_statements" => 1.2,
        "Cases" => 1.0,
        "ATO_interpretative_decisions" => 0.9,
        "Legislation_and_supporting_material" => 0.45,
        _ => 0.8,
    }
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
        chunk_id: Some(row.get("chunk_id")?),
    }))
}

fn apply_recency_boost(records: &mut [Hit]) {
    let now_year = Utc::now().year();
    let decay = std::f64::consts::LN_2 / 5.0;
    for hit in records {
        let Some(date) = &hit.date else {
            continue;
        };
        let Some(year) = date.get(0..4).and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let age = std::cmp::max(0, now_year - year) as f64;
        if let Some(score) = hit.score {
            hit.score = Some(score * (0.5 + (-age * decay).exp()));
        }
    }
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
    out.push_str("| # | Type | Date | Title | Section | Snippet |\n");
    out.push_str("|---:|---|---|---|---|---|\n");
    for (idx, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "| {} | `{}` | {} | [{}]({})<br><small>`{}`</small> | {} | {} |\n",
            idx + 1,
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
    params_vec.push(Value::Integer(k as i64));

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
            doc_id,
            title: title.clone(),
            doc_type: row.get("type")?,
            date: row.get("date")?,
            heading_path: String::new(),
            anchor: None,
            snippet: title,
            score,
            chunk_id: None,
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
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&json!({
            "query": query,
            "filters": {
                "excluded_by_default": DEFAULT_EXCLUDED_TYPES,
                "old_content_cutoff": if include_old { JsonValue::Null } else { json!(OLD_CONTENT_CUTOFF) },
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
    let selected = select_chunks(&conn, doc_id, &opts)?;
    let Some((chunks, continuation_ord)) = selected else {
        return Ok(format!(
            "_Section not found in {} (anchor={:?}, heading_path={:?}, from_ord={:?})._",
            doc_id, opts.anchor, opts.heading_path, opts.from_ord
        ));
    };
    if opts.format == DocumentFormat::Json {
        return Ok(serde_json::to_string_pretty(&json!({
            "document": doc,
            "chunks": chunks,
            "continuation_ord": continuation_ord,
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
    for (idx, (chunk_id, ord, heading_path, anchor, blob)) in candidates.into_iter().enumerate() {
        let text = decompress_text(blob)?;
        if opts
            .max_chars
            .is_some_and(|max| !out.is_empty() && chars + text.len() > max)
        {
            continuation_ord = Some(ord);
            break;
        }
        chars += text.len();
        out.push(ChunkOut {
            chunk_id,
            ord,
            heading_path,
            anchor,
            text,
        });
        if opts.count.is_some_and(|count| out.len() >= count) {
            if let Some(next) = out.get(idx + 1) {
                continuation_ord = Some(next.ord);
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
        out.push_str(&format!("_Continues at `from_ord={}`._", ord));
    }
    out
}

fn whats_new(
    since: Option<&str>,
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
    params_out.push(Value::Integer(limit.clamp(1, 500) as i64));
    let sql = format!(
        "SELECT doc_id, type, title, date, downloaded_at FROM documents {where_sql} ORDER BY {sort_expr} DESC LIMIT ?"
    );
    let mut stmt = conn.prepare(&sql)?;
    let hits = stmt
        .query_map(params_from_iter(params_out), |row| {
            let doc_id: String = row.get("doc_id")?;
            let date: Option<String> = row.get("date")?;
            let downloaded_at: String = row.get("downloaded_at")?;
            Ok(Hit {
                canonical_url: canonical_url(&doc_id),
                doc_id,
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
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(
            &json!({"since": since, "hits": hits}),
        )?),
        OutputFormat::Markdown => Ok(format_hits_markdown(&hits)),
    }
}

fn stats(format: OutputFormat) -> Result<String> {
    let conn = open_read()?;
    let docs: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
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
        "documents": docs,
        "chunks": chunks,
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

fn apply_update_locked(manifest_url: &str) -> Result<UpdateStats> {
    let staging = staging_dir()?;
    let manifest_context = UrlContext::from_manifest_url(manifest_url);
    let manifest_bytes = fetch_bytes(manifest_url, &manifest_context)
        .with_context(|| format!("fetching manifest from {manifest_url}"))?;
    let new_manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let old_manifest = load_installed_manifest()?;

    ensure_model(&new_manifest, &manifest_context, &staging)?;

    let conn = open_write()?;
    init_db(&conn)?;
    let (added, changed, removed) = diff_manifests(old_manifest.as_ref(), &new_manifest);

    let backup = backups_dir()?.join("ato.db.prev");
    let db = db_path()?;
    if db.exists() {
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
    let live_model = model_path()?;
    let tokenizer = tokenizer_path()?;
    let marker = live_dir()?.join(".model.sha256");
    if live_model.exists()
        && tokenizer.exists()
        && marker.exists()
        && fs::read_to_string(&marker)?.trim() == manifest.model.sha256
    {
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
    let model_link = live_dir()?.join("model.onnx");
    if !model_link.exists() {
        let quantized = live_dir()?.join("model_quantized.onnx");
        if quantized.exists() {
            #[cfg(unix)]
            std::os::unix::fs::symlink("model_quantized.onnx", &model_link)?;
            #[cfg(not(unix))]
            fs::copy(&quantized, &model_link)?;
        }
    }
    fs::write(marker, &manifest.model.sha256)?;
    let _ = fs::remove_file(bundle);
    let _ = fs::remove_dir_all(extract_dir);
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

fn delete_doc(conn: &Connection, doc_id: &str) -> Result<()> {
    let chunk_ids = {
        let mut stmt = conn.prepare("SELECT chunk_id FROM chunks WHERE doc_id = ?")?;
        let rows = stmt
            .query_map([doc_id], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    for chunk_id in chunk_ids {
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
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Result<JsonValue> = serde_json::from_str(&line);
        let response = match parsed {
            Ok(message) => handle_rpc(message),
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

fn handle_rpc(message: JsonValue) -> Option<JsonValue> {
    if message.is_array() {
        let responses: Vec<JsonValue> = message
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| handle_single_rpc(m.clone()))
            .collect();
        if responses.is_empty() {
            None
        } else {
            Some(JsonValue::Array(responses))
        }
    } else {
        handle_single_rpc(message)
    }
}

fn handle_single_rpc(message: JsonValue) -> Option<JsonValue> {
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
            call_tool(params)
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

fn call_tool(params: JsonValue) -> Result<JsonValue> {
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
                    sort_by,
                    include_old: optional_bool(&args, "include_old").unwrap_or(false),
                    format,
                },
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
                _ => DocumentFormat::Outline,
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
    let format = output_format_arg(args);
    get_chunks(&chunk_ids, format)
}

fn get_chunks(chunk_ids: &[i64], format: OutputFormat) -> Result<String> {
    if chunk_ids.is_empty() {
        return Ok("_No chunk ids provided._".to_string());
    }
    let conn = open_read()?;
    let placeholders = vec!["?"; chunk_ids.len()].join(",");
    let sql = format!(
        r#"
        SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
               d.type, d.title, d.date
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({placeholders})
        "#
    );
    let params_vec: Vec<Value> = chunk_ids.iter().map(|id| Value::Integer(*id)).collect();
    let mut stmt = conn.prepare(&sql)?;
    let mut out = Vec::new();
    let rows = stmt.query_map(params_from_iter(params_vec), |row| {
        Ok((
            row.get::<_, i64>("chunk_id")?,
            row.get::<_, String>("doc_id")?,
            row.get::<_, String>("type")?,
            row.get::<_, String>("title")?,
            row.get::<_, Option<String>>("date")?,
            row.get::<_, Option<String>>("heading_path")?
                .unwrap_or_default(),
            row.get::<_, Option<String>>("anchor")?,
            row.get::<_, Vec<u8>>("text")?,
        ))
    })?;
    for row in rows {
        let (chunk_id, doc_id, doc_type, title, date, heading_path, anchor, text_blob) = row?;
        out.push(json!({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "type": doc_type,
            "title": title,
            "date": date,
            "heading_path": heading_path,
            "anchor": anchor,
            "canonical_url": canonical_url(&doc_id),
            "text": decompress_text(text_blob)?,
        }));
    }
    if matches!(format, OutputFormat::Json) {
        return Ok(serde_json::to_string_pretty(&json!({ "chunks": out }))?);
    }
    let mut text = String::new();
    for chunk in out {
        text.push_str(&format!(
            "**{}** ([{}]({})) - {}\n\n{}\n\n---\n",
            chunk["title"].as_str().unwrap_or(""),
            chunk["doc_id"].as_str().unwrap_or(""),
            chunk["canonical_url"].as_str().unwrap_or(""),
            chunk["heading_path"].as_str().unwrap_or(""),
            chunk["text"].as_str().unwrap_or("")
        ));
    }
    Ok(text)
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
            "description": "Search ATO legal documents with BM25 over a GPU-built corpus. Defaults exclude Edited Private Advice and very old non-legislation content; set include_old=true to opt in.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "doc_scope": {"type": "string"},
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
                    "format": {"type": "string", "enum": ["outline", "markdown", "json"]},
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
            "description": "Fetch specific chunks by chunk id from search results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}},
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
