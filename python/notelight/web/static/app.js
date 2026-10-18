// notelight web UI: plain JavaScript, no dependencies, no build step.
// All text from the server is inserted with textContent, never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("search-form");
const input = $("q");
const folderSelect = $("folder");
const results = $("results");
const message = $("message");
const sourcesBox = $("sources");
const template = $("result-template");

let searchSeq = 0;      // drops responses that arrive after a newer search
let currentTerms = [];  // query terms, for highlighting

// -- theme ---------------------------------------------------------------------
function storedTheme() {
  try { return localStorage.getItem("notelight-theme"); } catch { return null; }
}

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme;
  }
}

$("theme").addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("notelight-theme", next); } catch { /* private window */ }
});
applyTheme(storedTheme());

// -- API -----------------------------------------------------------------------
async function api(path, params) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== "" && value !== null && value !== undefined) query.set(key, String(value));
  }
  const response = await fetch(`${path}?${query}`, { headers: { Accept: "application/json" } });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return body;
}

// -- highlighting --------------------------------------------------------------
function termsOf(query) {
  const words = query.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  return [...new Set(words.filter((w) => w.length > 1 || /\p{N}/u.test(w)))];
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Fills `element` with `text`, wrapping every query term in <mark>.
function highlight(element, text, terms) {
  element.textContent = "";
  if (!terms.length) {
    element.textContent = text;
    return;
  }
  const pattern = new RegExp(terms.map(escapeRegExp).join("|"), "giu");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) element.append(text.slice(last, match.index));
    const mark = document.createElement("mark");
    mark.textContent = match[0];
    element.append(mark);
    last = match.index + match[0].length;
  }
  if (last < text.length) element.append(text.slice(last));
}

// -- sources and status --------------------------------------------------------
async function loadSources() {
  let sources;
  try {
    sources = await api("/api/sources");
  } catch (e) {
    showMessage(`Cannot list folders: ${e.message}`, true);
    return;
  }
  const wanted = new URLSearchParams(location.search).get("folder") || "";
  sourcesBox.textContent = "";
  for (const s of sources) {
    const option = document.createElement("option");
    option.value = s.folder;
    option.textContent = s.folder;
    folderSelect.append(option);

    const chip = document.createElement("span");
    chip.className = "source";
    chip.title = s.folder;
    const name = document.createElement("span");
    name.className = "source-name";
    name.textContent = s.folder;
    const badge = document.createElement("span");
    badge.className = "badge";
    chip.append(name, badge);
    sourcesBox.append(chip);

    if (s.error) {
      badge.classList.add("error");
      badge.textContent = s.error;
      continue;
    }
    badge.textContent = `${s.files} files`;
    loadStatus(s.folder, badge, chip);
  }
  if (!sources.length) {
    showMessage("Nothing is indexed yet. Run `notelight index <folder>` first.", true);
  }
  folderSelect.value = sources.some((s) => s.folder === wanted) ? wanted : "";
}

async function loadStatus(folder, badge, chip) {
  try {
    const st = await api("/api/status", { folder });
    if (st.stale) {
      badge.classList.add("stale");
      const changes = st.added + st.modified + st.removed;
      badge.textContent = `stale: ${changes} changed`;
      chip.title = st.hint || folder;
    } else {
      badge.textContent = `${st.files} files · up to date`;
    }
  } catch (e) {
    badge.classList.add("error");
    badge.textContent = e.message;
  }
}

// -- search --------------------------------------------------------------------
function showMessage(text, isError) {
  message.textContent = text;
  message.classList.toggle("error", Boolean(isError));
}

async function runSearch() {
  const q = input.value.trim();
  const folder = folderSelect.value;
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (folder) params.set("folder", folder);
  const search = params.toString();
  history.replaceState(null, "", search ? `?${search}` : location.pathname);

  const seq = ++searchSeq;
  if (!q) {
    results.textContent = "";
    showMessage("");
    return;
  }
  showMessage("Searching…");
  let response;
  try {
    response = await api("/api/search", { q, folder, k: 20 });
  } catch (e) {
    if (seq === searchSeq) showMessage(e.message, true);
    return;
  }
  if (seq !== searchSeq) return;
  currentTerms = termsOf(q);
  renderResults(response);
}

function renderResults(response) {
  results.textContent = "";
  const skipped = response.skipped.map((s) => `${s.folder} skipped: ${s.error}`).join("; ");
  const count = response.results.length;
  // Same words as the CLI (search.NO_CLOSE_MATCH).
  const far = response.close_match === false && count
    ? " No close match: nothing indexed is closely related to this query. These are the nearest chunks."
    : "";
  showMessage(
    (count ? `${count} result${count === 1 ? "" : "s"}.` : "No results.")
      + far
      + (skipped ? ` (${skipped})` : ""),
    Boolean(skipped) && !count,
  );
  const several = folderSelect.value === "" && folderSelect.options.length > 2;
  for (const hit of response.results) {
    const item = template.content.firstElementChild.cloneNode(true);
    item.querySelector(".location").textContent = hit.location;
    item.querySelector(".headings").textContent = hit.heading_path.join(" › ");
    highlight(item.querySelector(".snippet"), hit.snippet, currentTerms);
    if (several) item.querySelector(".folder-name").textContent = hit.folder;
    const head = item.querySelector(".result-head");
    head.addEventListener("click", () => toggleContext(item, hit));
    results.append(item);
  }
}

async function toggleContext(item, hit) {
  const head = item.querySelector(".result-head");
  const box = item.querySelector(".context");
  const open = head.getAttribute("aria-expanded") === "true";
  head.setAttribute("aria-expanded", String(!open));
  box.hidden = open;
  if (open || box.dataset.loaded) return;
  box.textContent = "Loading…";
  try {
    const ctx = await api("/api/context", {
      location: hit.location, folder: hit.folder, context: 8,
    });
    renderContext(box, ctx, hit);
    box.dataset.loaded = "1";
  } catch (e) {
    box.textContent = e.message;
  }
}

function renderContext(box, ctx, hit) {
  box.textContent = "";
  const meta = document.createElement("div");
  meta.className = "context-meta";
  meta.textContent = `${ctx.path}, lines ${ctx.line_start}-${ctx.line_end} of ${ctx.total_lines}`
    + (ctx.truncated ? " (truncated)" : "");
  box.append(meta);
  ctx.text.split("\n").forEach((text, i) => {
    const n = ctx.line_start + i;
    const line = document.createElement("div");
    line.className = "line";
    if (n >= hit.line_start && n <= hit.line_end) line.classList.add("hit");
    const ln = document.createElement("span");
    ln.className = "ln";
    ln.textContent = String(n);
    const code = document.createElement("span");
    code.className = "code";
    highlight(code, text || " ", currentTerms);
    line.append(ln, code);
    box.append(line);
  });
}

// -- keyboard ------------------------------------------------------------------
function resultButtons() {
  return [...results.querySelectorAll(".result-head")];
}

function moveFocus(step) {
  const buttons = resultButtons();
  if (!buttons.length) return;
  const index = buttons.indexOf(document.activeElement);
  const next = index < 0 ? (step > 0 ? 0 : buttons.length - 1)
    : Math.min(buttons.length - 1, Math.max(0, index + step));
  buttons[next].focus();
  buttons[next].scrollIntoView({ block: "nearest" });
}

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const inField = event.target === input || event.target === folderSelect;
  if (event.key === "/" && !inField) {
    event.preventDefault();
    input.focus();
    input.select();
  } else if (event.key === "Escape") {
    input.focus();
  } else if (event.key === "ArrowDown" || (event.key === "j" && !inField)) {
    if (event.target === folderSelect) return;
    event.preventDefault();
    moveFocus(1);
  } else if (event.key === "ArrowUp" || (event.key === "k" && !inField)) {
    if (inField) return;
    event.preventDefault();
    moveFocus(-1);
  }
});

let debounce = 0;
input.addEventListener("input", () => {
  clearTimeout(debounce);
  debounce = setTimeout(runSearch, 250);
});
folderSelect.addEventListener("change", runSearch);
form.addEventListener("submit", (event) => {
  event.preventDefault();
  clearTimeout(debounce);
  runSearch();
});

// -- start ---------------------------------------------------------------------
(async () => {
  input.value = new URLSearchParams(location.search).get("q") || "";
  await loadSources();
  if (input.value) runSearch();
})();
