//! Splitting documents into retrievable chunks.
//!
//! Chunks follow the document's structure instead of fixed token windows:
//!
//! 1. Markdown headings start a new **section**. A chunk never spans two
//!    sections, and every chunk records its heading path
//!    (`["Setup", "Windows"]`), which helps both ranking and citations.
//! 2. Inside a section, text is split into **blocks**: paragraphs separated by
//!    blank lines, and fenced code blocks, which are never cut in the middle
//!    unless they alone exceed the size limit.
//! 3. Blocks are packed greedily into chunks of at most
//!    [`ChunkOpts::max_chars`] characters. Oversized blocks fall back to line
//!    boundaries, and oversized lines to word boundaries.
//!
//! Every chunk is an exact slice of the source (`&src[byte_start..byte_end]`)
//! with 1-based line numbers, so callers can cite `path:line` precisely.

use serde::{Deserialize, Serialize};

/// Options controlling how a document is chunked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOpts {
    /// Upper bound on the length of a chunk, in Unicode scalar values.
    /// About 1200 characters is roughly 250–300 tokens for English text.
    pub max_chars: usize,
    /// Treat `#` headings and ```` ``` ```` fences as Markdown structure.
    /// When `false`, only blank lines separate blocks.
    pub markdown: bool,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        Self {
            max_chars: 1200,
            markdown: true,
        }
    }
}

/// A contiguous piece of a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// The chunk text: exactly `src[byte_start..byte_end]`.
    pub text: String,
    /// Headings enclosing this chunk, outermost first. Empty before the first heading.
    pub heading_path: Vec<String>,
    /// Byte offset of the first byte of the chunk in the source.
    pub byte_start: usize,
    /// Byte offset one past the last byte of the chunk in the source.
    pub byte_end: usize,
    /// 1-based line number of the first line of the chunk.
    pub line_start: u32,
    /// 1-based line number of the last line of the chunk (inclusive).
    pub line_end: u32,
}

/// Split `src` into chunks according to `opts`.
///
/// Whitespace-only input produces no chunks. Every non-whitespace character
/// of `src` ends up in exactly one chunk, and chunks are returned in document
/// order without overlapping.
///
/// # Panics
/// Panics if `opts.max_chars` is zero.
#[must_use]
pub fn chunk_text(src: &str, opts: &ChunkOpts) -> Vec<Chunk> {
    assert!(opts.max_chars > 0, "max_chars must be positive");
    let lines = split_lines(src);
    let blocks = parse_blocks(src, &lines, opts.markdown);

    // Turn blocks into units no longer than max_chars, then pack units.
    let mut units: Vec<Unit> = Vec::new();
    for block in &blocks {
        let first = &lines[block.first_line];
        let last = &lines[block.last_line];
        let unit = Unit {
            start: first.start,
            end: last.end,
            line_start: block.first_line,
            line_end: block.last_line,
            section: block.section,
        };
        if char_len(src, unit.start, unit.end) <= opts.max_chars {
            units.push(unit);
        } else {
            split_block(src, &lines, block, opts.max_chars, &mut units);
        }
    }

    pack_units(src, &lines, &units, &section_paths(&blocks), opts.max_chars)
}

/// One line of the source. `end` excludes the line terminator (`\n` or `\r\n`).
#[derive(Debug, Clone, Copy)]
struct Line {
    start: usize,
    end: usize,
}

fn split_lines(src: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    let mut offset = 0;
    for raw in src.split_inclusive('\n') {
        let content = raw.strip_suffix('\n').unwrap_or(raw);
        let content = content.strip_suffix('\r').unwrap_or(content);
        lines.push(Line {
            start: offset,
            end: offset + content.len(),
        });
        offset += raw.len();
    }
    lines
}

/// A run of lines that should stay together when possible.
#[derive(Debug, Clone)]
struct Block {
    first_line: usize,
    last_line: usize,
    /// Index into the list of sections; each section has one heading path.
    section: usize,
    heading_path: Vec<String>,
}

fn parse_blocks(src: &str, lines: &[Line], markdown: bool) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut headings: Vec<(usize, String)> = Vec::new(); // (level, text)
    let mut section = 0usize;
    let mut i = 0;

    let path = |headings: &[(usize, String)]| headings.iter().map(|(_, t)| t.clone()).collect();

    while i < lines.len() {
        let text = &src[lines[i].start..lines[i].end];
        if text.trim().is_empty() {
            i += 1;
            continue;
        }

        if markdown && let Some((level, title)) = parse_heading(text) {
            // A heading opens a new section; the heading line is its first block.
            while headings.last().is_some_and(|(l, _)| *l >= level) {
                headings.pop();
            }
            headings.push((level, title));
            section += 1;
            blocks.push(Block {
                first_line: i,
                last_line: i,
                section,
                heading_path: path(&headings),
            });
            i += 1;
            continue;
        }

        if markdown && let Some(fence) = parse_fence_open(text) {
            // A fenced code block runs to its closing fence (or end of input).
            let mut j = i + 1;
            while j < lines.len() && !is_fence_close(&src[lines[j].start..lines[j].end], fence) {
                j += 1;
            }
            let last = j.min(lines.len() - 1);
            blocks.push(Block {
                first_line: i,
                last_line: last,
                section,
                heading_path: path(&headings),
            });
            i = last + 1;
            continue;
        }

        // A paragraph: consecutive non-blank lines up to a heading or fence.
        let mut j = i;
        while j + 1 < lines.len() {
            let next = &src[lines[j + 1].start..lines[j + 1].end];
            let breaks = next.trim().is_empty()
                || (markdown
                    && (parse_heading(next).is_some() || parse_fence_open(next).is_some()));
            if breaks {
                break;
            }
            j += 1;
        }
        blocks.push(Block {
            first_line: i,
            last_line: j,
            section,
            heading_path: path(&headings),
        });
        i = j + 1;
    }
    blocks
}

/// Parse an ATX heading (`## Title ##`) indented by at most three spaces.
fn parse_heading(line: &str) -> Option<(usize, String)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.len() - rest.trim_start_matches('#').len();
    if !(1..=6).contains(&level) {
        return None;
    }
    let after = &rest[level..];
    if !(after.is_empty() || after.starts_with(' ') || after.starts_with('\t')) {
        return None; // "#hashtag" is not a heading
    }
    // A closing run of '#' is only syntax when preceded by a space or tab
    // (CommonMark), so "# Learning C#" keeps its '#'.
    let content = after.trim();
    let without_closing = content.trim_end_matches('#');
    let title = if without_closing.is_empty() {
        ""
    } else if without_closing.ends_with([' ', '\t']) {
        without_closing.trim_end()
    } else {
        content
    };
    Some((level, title.to_string()))
}

/// A fence opener: three or more backticks or tildes. Returns (char, length).
fn parse_fence_open(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start_matches(' ');
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let ch = trimmed.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let n = trimmed.len() - trimmed.trim_start_matches(ch).len();
    (n >= 3).then_some((ch, n))
}

fn is_fence_close(line: &str, (ch, len): (char, usize)) -> bool {
    let trimmed = line.trim();
    let n = trimmed.len() - trimmed.trim_start_matches(ch).len();
    n >= len && trimmed.len() == n
}

/// A piece of text no longer than `max_chars`, ready to be packed.
#[derive(Debug, Clone, Copy)]
struct Unit {
    start: usize,
    end: usize,
    line_start: usize, // 0-based line indices
    line_end: usize,
    section: usize,
}

/// Split an oversized block into line-sized (or smaller) units.
fn split_block(src: &str, lines: &[Line], block: &Block, max: usize, out: &mut Vec<Unit>) {
    for (idx, &line) in lines
        .iter()
        .enumerate()
        .take(block.last_line + 1)
        .skip(block.first_line)
    {
        if src[line.start..line.end].trim().is_empty() {
            continue;
        }
        if char_len(src, line.start, line.end) <= max {
            out.push(Unit {
                start: line.start,
                end: line.end,
                line_start: idx,
                line_end: idx,
                section: block.section,
            });
            continue;
        }
        // Very long line: cut at the last whitespace inside each window.
        let mut pos = line.start;
        while pos < line.end {
            // Skip leading whitespace so pieces start on content.
            let rest = &src[pos..line.end];
            pos += rest.len() - rest.trim_start().len();
            if pos >= line.end {
                break;
            }
            let window_end = byte_after_chars(src, pos, line.end, max);
            let cut = if window_end < line.end {
                src[pos..window_end]
                    .rfind(char::is_whitespace)
                    .filter(|&k| k > 0)
                    .map_or(window_end, |k| pos + k)
            } else {
                window_end
            };
            let piece = &src[pos..cut];
            let end = pos + piece.trim_end().len();
            out.push(Unit {
                start: pos,
                end,
                line_start: idx,
                line_end: idx,
                section: block.section,
            });
            pos = cut;
        }
    }
}

/// Heading path for each section id. Section ids are dense (one per heading),
/// so a `Vec` indexed by id replaces a map.
fn section_paths(blocks: &[Block]) -> Vec<Vec<String>> {
    let count = blocks.last().map_or(0, |b| b.section + 1);
    let mut paths = vec![Vec::new(); count];
    for b in blocks {
        if paths[b.section].is_empty() {
            paths[b.section].clone_from(&b.heading_path);
        }
    }
    paths
}

fn pack_units(
    src: &str,
    lines: &[Line],
    units: &[Unit],
    section_paths: &[Vec<String>],
    max: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < units.len() {
        let first = units[i];
        let mut last = first;
        let mut j = i + 1;
        while j < units.len()
            && units[j].section == first.section
            && char_len(src, first.start, units[j].end) <= max
        {
            last = units[j];
            j += 1;
        }
        // Line numbers come from the byte offsets so sub-line pieces stay exact.
        let line_start = line_of(lines, first.start).max(first.line_start);
        let line_end =
            line_of(lines, last.end.saturating_sub(1).max(last.start)).min(last.line_end);
        chunks.push(Chunk {
            text: src[first.start..last.end].to_string(),
            heading_path: section_paths[first.section].clone(),
            byte_start: first.start,
            byte_end: last.end,
            line_start: u32::try_from(line_start + 1).unwrap_or(u32::MAX),
            line_end: u32::try_from(line_end.max(line_start) + 1).unwrap_or(u32::MAX),
        });
        i = j;
    }
    chunks
}

/// 0-based index of the line containing byte `pos`.
fn line_of(lines: &[Line], pos: usize) -> usize {
    lines.partition_point(|l| l.start <= pos).saturating_sub(1)
}

fn char_len(src: &str, start: usize, end: usize) -> usize {
    src[start..end].chars().count()
}

/// Byte offset reached after advancing `n` characters from `start`, capped at `limit`.
fn byte_after_chars(src: &str, start: usize, limit: usize, n: usize) -> usize {
    src[start..limit]
        .char_indices()
        .nth(n)
        .map_or(limit, |(k, _)| start + k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(src: &str, max_chars: usize) -> Vec<Chunk> {
        chunk_text(
            src,
            &ChunkOpts {
                max_chars,
                markdown: true,
            },
        )
    }

    #[test]
    fn empty_and_blank_input_produce_no_chunks() {
        assert!(chunks("", 100).is_empty());
        assert!(chunks("\n  \n\t\n", 100).is_empty());
    }

    #[test]
    fn small_document_is_one_chunk_with_lines() {
        let c = chunks("hello\nworld\n", 100);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].text, "hello\nworld");
        assert_eq!((c[0].line_start, c[0].line_end), (1, 2));
        assert!(c[0].heading_path.is_empty());
    }

    #[test]
    fn headings_start_sections_and_build_paths() {
        let src = "# Setup\nintro\n## Windows\nuse winget\n## Linux\nuse apt\n# Usage\nrun it\n";
        let c = chunks(src, 1000);
        let paths: Vec<_> = c.iter().map(|c| c.heading_path.join(" > ")).collect();
        assert_eq!(
            paths,
            ["Setup", "Setup > Windows", "Setup > Linux", "Usage"]
        );
        assert_eq!(c[1].text, "## Windows\nuse winget");
        assert_eq!((c[1].line_start, c[1].line_end), (3, 4));
    }

    #[test]
    fn hashtags_and_deep_indent_are_not_headings() {
        assert!(parse_heading("#tag").is_none());
        assert!(parse_heading("    # code").is_none());
        assert_eq!(parse_heading("### Title ###"), Some((3, "Title".into())));
        assert_eq!(parse_heading("#"), Some((1, String::new())));
        assert_eq!(
            parse_heading("# Learning C#"),
            Some((1, "Learning C#".into()))
        );
        assert_eq!(parse_heading("## F# ##"), Some((2, "F#".into())));
        assert_eq!(parse_heading("# ###"), Some((1, String::new())));
    }

    #[test]
    fn headings_inside_code_fences_are_ignored() {
        let src = "# Real\n```bash\n# not a heading\necho hi\n```\nafter\n";
        let c = chunks(src, 1000);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].heading_path, ["Real"]);
    }

    #[test]
    fn paragraphs_are_packed_up_to_the_limit() {
        let src = "aaaa\n\nbbbb\n\ncccc\n";
        let c = chunks(src, 10); // "aaaa\n\nbbbb" is exactly 10 chars
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].text, "aaaa\n\nbbbb");
        assert_eq!(c[1].text, "cccc");
        assert_eq!((c[1].line_start, c[1].line_end), (5, 5));
    }

    #[test]
    fn long_lines_split_at_whitespace() {
        let src = "alpha beta gamma delta";
        let c = chunks(src, 11);
        let texts: Vec<_> = c.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["alpha beta", "gamma delta"]);
        assert!(c.iter().all(|c| c.line_start == 1 && c.line_end == 1));
    }

    #[test]
    fn long_words_are_hard_split_on_char_boundaries() {
        let src = "привет".repeat(3); // 18 two-byte chars, no whitespace
        let c = chunks(&src, 7);
        assert!(c.iter().all(|c| c.text.chars().count() <= 7));
        assert_eq!(c.iter().map(|c| c.text.as_str()).collect::<String>(), src);
    }

    #[test]
    fn crlf_line_endings_are_handled() {
        let c = chunks("one\r\ntwo\r\n\r\nthree\r\n", 5);
        let texts: Vec<_> = c.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["one", "two", "three"]);
        assert_eq!((c[2].line_start, c[2].line_end), (4, 4));
    }

    #[test]
    fn plain_mode_ignores_markdown_syntax() {
        let opts = ChunkOpts {
            max_chars: 1000,
            markdown: false,
        };
        let c = chunk_text("# not special\ntext\n", &opts);
        assert_eq!(c.len(), 1);
        assert!(c[0].heading_path.is_empty());
    }
}
