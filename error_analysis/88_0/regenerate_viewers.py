#!/usr/bin/env python3
"""
Workflow (curated CoT vs error base viewer):
  - problem_ids_matched_v12.csv holds generated_cot per id (edit there for new CoT).
  - cot_viewer_v12_new_ids.txt lists ids shown in cot_viewer_v12_new.html (one id per line).
  - This script builds cot_viewer_v12_new.html from v12 CSV + ids file,
    cot_viewer_v12_changed.html from bit_manipulation_87_0.csv beside cot_viewer_v11_missing248.html
    (LoRA validation vs your v11 CoT; order = cot_viewer_v12_new_ids.txt),
    and bit_manipulation_87_0.html from bit_manipulation_87_0.csv excluding those ids.

Run from anywhere:
  python3 error_analysis/88_0/regenerate_viewers.py
"""
from __future__ import annotations

import csv
import html
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
IDS_FILE = DIR / "cot_viewer_v12_new_ids.txt"
V12_CSV = DIR / "problem_ids_matched_v12.csv"
BM_CSV = DIR / "bit_manipulation_87_0.csv"
V11_HTML = DIR / "cot_viewer_v11_missing248.html"
COT_HTML = DIR / "cot_viewer_v12_new.html"
CHANGED_HTML = DIR / "cot_viewer_v12_changed.html"
BM_HTML = DIR / "bit_manipulation_87_0.html"


def load_viewer_ids() -> list[str]:
    if not IDS_FILE.is_file():
        print(f"Missing {IDS_FILE.name}; create it with one 8-char id per line.", file=sys.stderr)
        sys.exit(1)
    ids: list[str] = []
    for line in IDS_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not re.fullmatch(r"[0-9a-f]{8}", s):
            print(f"Bad id line (expected 8 hex chars): {line!r}", file=sys.stderr)
            sys.exit(1)
        ids.append(s)
    if not ids:
        print(f"{IDS_FILE.name} has no ids.", file=sys.stderr)
        sys.exit(1)
    return ids


def esc(s: str | None) -> str:
    return html.escape(s or "", quote=True)


def parse_v11_cot_inner_by_id(html: str) -> dict[str, str]:
    """Extract inner HTML of each .cot-box from cot_viewer_v11_missing248.html by problem id."""
    chunks = re.split(r'(?=<div class="card(?:\s+hidden)?")', html)
    out: dict[str, str] = {}
    cot_close = re.compile(
        r'<div class="cot-box">([\s\S]*?)</div>\s*</div>\s*</div>',
        re.MULTILINE,
    )
    for ch in chunks[1:]:
        mid = re.search(r'class="badge badge-id">([0-9a-f]{8})<', ch)
        mbox = cot_close.search(ch)
        if mid and mbox:
            out[mid.group(1)] = mbox.group(1)
    return out


def build_cot_viewer_changed_side_by_side(
    rows: list[dict],
    v11_by_id: dict[str, str] | None,
) -> str:
    """Validation trace (bit_manipulation CSV) vs your v11 CoT HTML, same nav/search as v12 new."""
    total = len(rows)
    ids_js = ",".join(f'"{r["id"]}"' for r in rows)

    def card(i: int, row: dict, hidden: bool) -> str:
        hid = " hidden" if hidden else ""
        pid = esc(row["id"])
        ans = esc(row["answer"])
        typ = esc(row["type"])
        prompt = esc(row["prompt"])
        val_cot = esc(row["generated_cot"])
        raw_id = row["id"]
        if v11_by_id is None:
            v11_block = (
                f'<div class="cot-missing">Missing <code>{V11_HTML.name}</code> '
                f"next to <code>regenerate_viewers.py</code>; re-run after adding that file.</div>"
            )
        else:
            v11_inner = v11_by_id.get(raw_id)
            if v11_inner is None:
                v11_block = (
                    '<div class="cot-missing">This id is not in <code>cot_viewer_v11_missing248.html</code> '
                    "(only 237 of the 248 missing-CoT ids appear there).</div>"
                )
            else:
                v11_block = v11_inner
        return f"""<div class="card{hid}" id="card-{i}" data-problem-id="{raw_id}">
  <div class="card-header">
    <span class="badge badge-id">{pid}</span>
    <span class="badge badge-ans">answer: {ans}</span>
    <span class="badge badge-type">{typ}</span>
    <span class="badge badge-val">LoRA validation</span>
    <span class="badge badge-v11">v11 CoT (yours)</span>
  </div>
  <div class="section">
    <div class="section-label">Prompt</div>
    <div class="prompt-box">{prompt}</div>
  </div>
  <hr class="div">
  <div class="section">
    <div class="section-label">Compare</div>
    <div class="cot-split">
      <div class="cot-col">
        <div class="section-head">
          <div class="section-label" style="margin-bottom:0">Validation output (LoRA SFT)</div>
          <button type="button" class="copy-btn" onclick="copyPane(this, &quot;val&quot;)">Copy</button>
        </div>
        <div class="cot-box cot-val">{val_cot}</div>
      </div>
      <div class="cot-col">
        <div class="section-head">
          <div class="section-label" style="margin-bottom:0">Your generated CoT (<code>v11 missing248</code>)</div>
          <button type="button" class="copy-btn" onclick="copyPane(this, &quot;v11&quot;)">Copy</button>
        </div>
        <div class="cot-box cot-v11">{v11_block}</div>
      </div>
    </div>
  </div>
</div>"""

    cards_inner = "".join(card(i, rows[i], i > 0) for i in range(total))
    id_list = ", ".join(r["id"] for r in rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoT viewer (v12) — changed vs v11</title>
<style>
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  margin: 0;
  padding: 20px;
}}
h1 {{ color: #7eb8f7; margin-bottom: 4px; font-size: 1.35rem; }}
.subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
.nav {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
.nav button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 18px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}}
.nav button:hover {{ background: #3a3d4a; color: #fff; }}
.counter {{
  background: #1e2130;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 5px 14px;
  font-size: 13px;
  color: #7eb8f7;
}}
.search-bar {{ display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.search-bar input[type=text] {{
  background: #2a2d3a;
  color: #e0e0e0;
  border: 1px solid #444;
  padding: 7px 12px;
  border-radius: 6px;
  width: 200px;
  font-size: 14px;
  font-family: monospace;
  outline: none;
}}
.search-bar input[type=text]:focus {{ border-color: #7eb8f7; }}
.search-bar button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 16px;
  border-radius: 6px;
  cursor: pointer;
}}
.search-bar button:hover {{ background: #3a3d4a; }}
.notfound {{ color: #eb5757; font-size: 13px; display: none; }}
input[type=number] {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 5px 10px;
  border-radius: 6px;
  width: 70px;
  font-size: 14px;
}}
.card {{
  background: #1a1d2e;
  border: 1px solid #2e3250;
  border-radius: 10px;
  padding: 24px;
  max-width: 1280px;
}}
.card.hidden {{ display: none; }}
.card-header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }}
.badge {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: monospace;
}}
.badge-id {{ background: #2a2d4a; color: #9ab4f7; }}
.badge-ans {{ background: #1a3a2a; color: #6fcf97; }}
.badge-type {{ background: #2a2d3a; color: #bbb; }}
.badge-val {{ background: #1a2a3a; color: #56ccf2; font-size: 11px; }}
.badge-v11 {{ background: #2a2a1a; color: #f2c94c; font-size: 11px; }}
.section {{ margin-bottom: 18px; }}
.section-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 8px;
}}
.section-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}}
.section-head .section-label {{ margin-bottom: 0; }}
.copy-btn {{
  background: #2a4d6a;
  color: #b8d4f0;
  border: 1px solid #4a6a88;
  padding: 4px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  flex-shrink: 0;
}}
.copy-btn:hover {{ background: #3a5d7a; color: #fff; }}
.copy-btn.copied {{ background: #1a3a2a; border-color: #3d6a4a; color: #6fcf97; }}
.prompt-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 14px 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 13px;
  white-space: pre-wrap;
  color: #b0b8d0;
  max-height: 220px;
  overflow-y: auto;
}}
.cot-split {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  align-items: start;
}}
@media (max-width: 1100px) {{
  .cot-split {{ grid-template-columns: 1fr; }}
}}
.cot-col {{ min-width: 0; }}
.cot-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
  white-space: pre-wrap;
  line-height: 1.6;
  color: #c8d0e8;
  max-height: 65vh;
  overflow-y: auto;
}}
.cot-missing {{
  background: #2a1f1f;
  border: 1px solid #5a3030;
  border-radius: 6px;
  padding: 16px;
  color: #eb5757;
  font-size: 13px;
  white-space: normal;
}}
.div {{ border: none; border-top: 1px solid #2e3250; margin: 14px 0; }}
.source {{ color: #666; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>CoT viewer (v12) — validation vs v11</h1>
<div class="subtitle">Left: LoRA SFT validation trace from <code>bit_manipulation_87_0.csv</code> · Right: your CoT from <code>cot_viewer_v11_missing248.html</code> · Order: <code>{IDS_FILE.name}</code></div>
<div class="nav">
  <button type="button" onclick="prev()">Prev</button>
  <button type="button" onclick="next()">Next</button>
  <span>Go to #:</span>
  <input type="number" id="gi" min="1" max="{total}" value="1" onchange="goTo(this.value)">
  <div class="counter" id="ctr">1 / {total}</div>
</div>
<div class="search-bar">
  <span style="color:#888;font-size:14px">Search by ID:</span>
  <input type="text" id="si" placeholder="8-char id" maxlength="8"
    oninput="document.getElementById('nf').style.display='none'"
    onkeydown="if(event.key==='Enter')doSearch()">
  <button type="button" onclick="doSearch()">Find</button>
  <span class="notfound" id="nf">ID not found</span>
</div>
<div id="cards">
{cards_inner}
</div>
<p class="source">Ids (registry <code>data-problem-id</code>): {id_list}</p>
<script>
const total = {total};
const ids = [{ids_js}];
let cur = 0;
function copyPane(btn, pane) {{
  if (total === 0) return;
  const card = document.getElementById('card-' + cur);
  const box = card.querySelector(pane === 'val' ? '.cot-val' : '.cot-v11');
  if (!box) return;
  const text = box.innerText;
  const doFlash = () => {{
    if (!btn) return;
    btn.classList.add('copied');
    const t = btn.textContent;
    btn.textContent = 'Copied';
    setTimeout(() => {{
      btn.textContent = t;
      btn.classList.remove('copied');
    }}, 1400);
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(doFlash).catch(() => fallbackCopy(text, btn, doFlash));
  }} else {{
    fallbackCopy(text, btn, doFlash);
  }}
}}
function fallbackCopy(text, btn, flash) {{
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {{
    document.execCommand('copy');
    if (flash) flash();
  }} catch (e) {{
    alert('Copy not supported in this browser.');
  }}
  document.body.removeChild(ta);
}}
function show(n) {{
  if (total === 0) return;
  document.getElementById('card-' + cur).classList.add('hidden');
  cur = Math.max(0, Math.min(total - 1, n));
  document.getElementById('card-' + cur).classList.remove('hidden');
  document.getElementById('ctr').textContent = (cur + 1) + ' / ' + total;
  document.getElementById('gi').value = cur + 1;
  window.scrollTo(0, 0);
}}
function prev() {{ show(cur - 1); }}
function next() {{ show(cur + 1); }}
function goTo(v) {{ show(parseInt(v, 10) - 1); }}
function doSearch() {{
  if (total === 0) return;
  const q = document.getElementById('si').value.trim().toLowerCase();
  if (!q) return;
  let i = ids.findIndex(x => x.toLowerCase() === q);
  if (i === -1) i = ids.findIndex(x => x.toLowerCase().startsWith(q));
  if (i !== -1) {{
    document.getElementById('nf').style.display = 'none';
    show(i);
  }} else {{
    document.getElementById('nf').style.display = 'inline';
  }}
}}
document.addEventListener('keydown', e => {{
  if (total === 0) return;
  if (document.activeElement === document.getElementById('si')) return;
  if (e.key === 'ArrowRight') next();
  if (e.key === 'ArrowLeft') prev();
}});
</script>
</body>
</html>
"""


def build_cot_viewer(
    rows: list[dict],
    *,
    title: str = "CoT viewer (v12) — new",
    h1: str | None = None,
    subtitle: str | None = None,
) -> str:
    if h1 is None:
        h1 = title
    if subtitle is None:
        subtitle = (
            f'One problem per view · rows from <code>problem_ids_matched_v12.csv</code> '
            f'· id list: <code>{IDS_FILE.name}</code>'
        )
    total = len(rows)
    ids_js = ",".join(f'"{r["id"]}"' for r in rows)

    def card_cot(i: int, row: dict, hidden: bool) -> str:
        hid = " hidden" if hidden else ""
        pid = esc(row["id"])
        ans = esc(row["answer"])
        typ = esc(row["type"])
        prompt = esc(row["prompt"])
        cot = esc(row["generated_cot"])
        raw_id = row["id"]
        return f"""<div class="card{hid}" id="card-{i}" data-problem-id="{raw_id}">
  <div class="card-header">
    <span class="badge badge-id">{pid}</span>
    <span class="badge badge-ans">answer: {ans}</span>
    <span class="badge badge-type">{typ}</span>
  </div>
  <div class="section">
    <div class="section-label">Prompt</div>
    <div class="prompt-box">{prompt}</div>
  </div>
  <hr class="div">
  <div class="section">
    <div class="section-head">
      <div class="section-label">Generated CoT</div>
      <button type="button" class="copy-btn" onclick="copyCurrentOutput(this)">Copy</button>
    </div>
    <div class="cot-box">{cot}</div>
  </div>
</div>"""

    cards_inner = "".join(card_cot(i, rows[i], i > 0) for i in range(total))
    id_list = ", ".join(r["id"] for r in rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  margin: 0;
  padding: 20px;
}}
h1 {{ color: #7eb8f7; margin-bottom: 4px; font-size: 1.35rem; }}
.subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
.nav {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
.nav button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 18px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}}
.nav button:hover {{ background: #3a3d4a; color: #fff; }}
.counter {{
  background: #1e2130;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 5px 14px;
  font-size: 13px;
  color: #7eb8f7;
}}
.search-bar {{ display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.search-bar input[type=text] {{
  background: #2a2d3a;
  color: #e0e0e0;
  border: 1px solid #444;
  padding: 7px 12px;
  border-radius: 6px;
  width: 200px;
  font-size: 14px;
  font-family: monospace;
  outline: none;
}}
.search-bar input[type=text]:focus {{ border-color: #7eb8f7; }}
.search-bar button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 16px;
  border-radius: 6px;
  cursor: pointer;
}}
.search-bar button:hover {{ background: #3a3d4a; }}
.notfound {{ color: #eb5757; font-size: 13px; display: none; }}
input[type=number] {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 5px 10px;
  border-radius: 6px;
  width: 70px;
  font-size: 14px;
}}
.card {{
  background: #1a1d2e;
  border: 1px solid #2e3250;
  border-radius: 10px;
  padding: 24px;
  max-width: 960px;
}}
.card.hidden {{ display: none; }}
.card-header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }}
.badge {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: monospace;
}}
.badge-id {{ background: #2a2d4a; color: #9ab4f7; }}
.badge-ans {{ background: #1a3a2a; color: #6fcf97; }}
.badge-type {{ background: #2a2d3a; color: #bbb; }}
.section {{ margin-bottom: 18px; }}
.section-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 8px;
}}
.section-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}}
.section-head .section-label {{ margin-bottom: 0; }}
.copy-btn {{
  background: #2a4d6a;
  color: #b8d4f0;
  border: 1px solid #4a6a88;
  padding: 4px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  flex-shrink: 0;
}}
.copy-btn:hover {{ background: #3a5d7a; color: #fff; }}
.copy-btn.copied {{ background: #1a3a2a; border-color: #3d6a4a; color: #6fcf97; }}
.prompt-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 14px 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 13px;
  white-space: pre-wrap;
  color: #b0b8d0;
  max-height: 220px;
  overflow-y: auto;
}}
.cot-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
  white-space: pre-wrap;
  line-height: 1.6;
  color: #c8d0e8;
  max-height: 70vh;
  overflow-y: auto;
}}
.div {{ border: none; border-top: 1px solid #2e3250; margin: 14px 0; }}
.source {{ color: #666; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>{h1}</h1>
<div class="subtitle">{subtitle}</div>
<div class="nav">
  <button type="button" onclick="prev()">Prev</button>
  <button type="button" onclick="next()">Next</button>
  <span>Go to #:</span>
  <input type="number" id="gi" min="1" max="{total}" value="1" onchange="goTo(this.value)">
  <div class="counter" id="ctr">1 / {total}</div>
</div>
<div class="search-bar">
  <span style="color:#888;font-size:14px">Search by ID:</span>
  <input type="text" id="si" placeholder="8-char id" maxlength="8"
    oninput="document.getElementById('nf').style.display='none'"
    onkeydown="if(event.key==='Enter')doSearch()">
  <button type="button" onclick="doSearch()">Find</button>
  <span class="notfound" id="nf">ID not found</span>
</div>
<div id="cards">
{cards_inner}
</div>
<p class="source">Ids (registry <code>data-problem-id</code>): {id_list}</p>
<script>
const total = {total};
const ids = [{ids_js}];
let cur = 0;
function copyCurrentOutput(btn) {{
  if (total === 0) return;
  const card = document.getElementById('card-' + cur);
  const box = card.querySelector('.cot-box');
  if (!box) return;
  const text = box.innerText;
  const doFlash = () => {{
    if (!btn) return;
    btn.classList.add('copied');
    const t = btn.textContent;
    btn.textContent = 'Copied';
    setTimeout(() => {{
      btn.textContent = t;
      btn.classList.remove('copied');
    }}, 1400);
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(doFlash).catch(() => fallbackCopy(text, btn, doFlash));
  }} else {{
    fallbackCopy(text, btn, doFlash);
  }}
}}
function fallbackCopy(text, btn, flash) {{
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {{
    document.execCommand('copy');
    if (flash) flash();
  }} catch (e) {{
    alert('Copy not supported in this browser.');
  }}
  document.body.removeChild(ta);
}}
function show(n) {{
  if (total === 0) return;
  document.getElementById('card-' + cur).classList.add('hidden');
  cur = Math.max(0, Math.min(total - 1, n));
  document.getElementById('card-' + cur).classList.remove('hidden');
  document.getElementById('ctr').textContent = (cur + 1) + ' / ' + total;
  document.getElementById('gi').value = cur + 1;
  window.scrollTo(0, 0);
}}
function prev() {{ show(cur - 1); }}
function next() {{ show(cur + 1); }}
function goTo(v) {{ show(parseInt(v, 10) - 1); }}
function doSearch() {{
  if (total === 0) return;
  const q = document.getElementById('si').value.trim().toLowerCase();
  if (!q) return;
  let i = ids.findIndex(x => x.toLowerCase() === q);
  if (i === -1) i = ids.findIndex(x => x.toLowerCase().startsWith(q));
  if (i !== -1) {{
    document.getElementById('nf').style.display = 'none';
    show(i);
  }} else {{
    document.getElementById('nf').style.display = 'inline';
  }}
}}
document.addEventListener('keydown', e => {{
  if (total === 0) return;
  if (document.activeElement === document.getElementById('si')) return;
  if (e.key === 'ArrowRight') next();
  if (e.key === 'ArrowLeft') prev();
}});
</script>
</body>
</html>
"""


def build_bm_viewer(rows: list[dict], exclude: set[str]) -> str:
    bm_total = len(rows)
    ids_bm_js = ",".join(f'"{r["id"]}"' for r in rows)

    def card_bm(i: int, row: dict, hidden: bool) -> str:
        hid = " hidden" if hidden else ""
        pid = esc(row["id"])
        ans = esc(row["answer"])
        pred = esc(row["predicted"])
        cat = esc(row["category"])
        corr = esc(row["correct"])
        mlp = esc(row["minlogprob"])
        prompt = esc(row["prompt"])
        output = esc(row["output"])
        wrong = (row.get("predicted") or "").strip() != (row.get("answer") or "").strip()
        pred_cls = "badge-pred-wrong" if wrong else "badge-pred-ok"
        return f"""<div class="card{hid}" id="card-{i}">
  <div class="card-header">
    <span class="badge badge-id">{pid}</span>
    <span class="badge badge-ans">answer: {ans}</span>
    <span class="badge {pred_cls}">predicted: {pred}</span>
    <span class="badge badge-cat">{cat}</span>
    <span class="badge badge-corr">correct: {corr}</span>
    <span class="badge badge-mlp">min log prob: {mlp}</span>
  </div>
  <div class="section">
    <div class="section-label">Prompt</div>
    <div class="prompt-box">{prompt}</div>
  </div>
  <hr class="div">
  <div class="section">
    <div class="section-head">
      <div class="section-label">Model output (full trace)</div>
      <button type="button" class="copy-btn" onclick="copyCurrentOutput(this)">Copy</button>
    </div>
    <div class="cot-box">{output}</div>
  </div>
</div>"""

    cards_bm = (
        "".join(card_bm(i, rows[i], i > 0) for i in range(bm_total))
        if bm_total
        else '<p style="color:#888">No rows left after exclusion.</p>'
    )
    ex = ", ".join(sorted(exclude))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bit_manipulation_87_0 — viewer</title>
<style>
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  margin: 0;
  padding: 20px;
}}
h1 {{ color: #7eb8f7; margin-bottom: 4px; font-size: 1.35rem; }}
.subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
.nav {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
.nav button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 18px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}}
.nav button:hover {{ background: #3a3d4a; color: #fff; }}
.counter {{
  background: #1e2130;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 5px 14px;
  font-size: 13px;
  color: #7eb8f7;
}}
.search-bar {{ display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.search-bar input[type=text] {{
  background: #2a2d3a;
  color: #e0e0e0;
  border: 1px solid #444;
  padding: 7px 12px;
  border-radius: 6px;
  width: 200px;
  font-size: 14px;
  font-family: monospace;
  outline: none;
}}
.search-bar input[type=text]:focus {{ border-color: #7eb8f7; }}
.search-bar button {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 7px 16px;
  border-radius: 6px;
  cursor: pointer;
}}
.search-bar button:hover {{ background: #3a3d4a; }}
.notfound {{ color: #eb5757; font-size: 13px; display: none; }}
input[type=number] {{
  background: #2a2d3a;
  color: #ccc;
  border: 1px solid #444;
  padding: 5px 10px;
  border-radius: 6px;
  width: 70px;
  font-size: 14px;
}}
.card {{
  background: #1a1d2e;
  border: 1px solid #2e3250;
  border-radius: 10px;
  padding: 24px;
  max-width: 960px;
}}
.card.hidden {{ display: none; }}
.card-header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }}
.badge {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: monospace;
}}
.badge-id {{ background: #2a2d4a; color: #9ab4f7; }}
.badge-ans {{ background: #1a3a2a; color: #6fcf97; }}
.badge-pred-wrong {{ background: #3a1a1a; color: #eb5757; }}
.badge-pred-ok {{ background: #1a2a3a; color: #56ccf2; }}
.badge-cat {{ background: #2a2d3a; color: #bbb; }}
.badge-corr {{ background: #2d2a1a; color: #f2994a; }}
.badge-mlp {{ background: #1a1d2e; color: #888; font-size: 11px; }}
.section {{ margin-bottom: 18px; }}
.section-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 8px;
}}
.section-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}}
.section-head .section-label {{ margin-bottom: 0; }}
.copy-btn {{
  background: #2a4d6a;
  color: #b8d4f0;
  border: 1px solid #4a6a88;
  padding: 4px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  flex-shrink: 0;
}}
.copy-btn:hover {{ background: #3a5d7a; color: #fff; }}
.copy-btn.copied {{ background: #1a3a2a; border-color: #3d6a4a; color: #6fcf97; }}
.prompt-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 14px 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 13px;
  white-space: pre-wrap;
  color: #b0b8d0;
  max-height: 220px;
  overflow-y: auto;
}}
.cot-box {{
  background: #12141f;
  border: 1px solid #2a2d40;
  border-radius: 6px;
  padding: 16px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
  white-space: pre-wrap;
  line-height: 1.6;
  color: #c8d0e8;
  max-height: 70vh;
  overflow-y: auto;
}}
.div {{ border: none; border-top: 1px solid #2e3250; margin: 14px 0; }}
</style>
</head>
<body>
<h1>bit_manipulation_87_0</h1>
<div class="subtitle"><code>bit_manipulation_87_0.csv</code> — {bm_total} rows (excluding ids in <code>{IDS_FILE.name}</code>: {ex})</div>
<div class="nav">
  <button type="button" onclick="prev()">Prev</button>
  <button type="button" onclick="next()">Next</button>
  <span>Go to #:</span>
  <input type="number" id="gi" min="1" max="{max(bm_total, 1)}" value="1" onchange="goTo(this.value)">
  <div class="counter" id="ctr">1 / {max(bm_total, 1)}</div>
</div>
<div class="search-bar">
  <span style="color:#888;font-size:14px">Search by ID:</span>
  <input type="text" id="si" placeholder="8-char id" maxlength="8"
    oninput="document.getElementById('nf').style.display='none'"
    onkeydown="if(event.key==='Enter')doSearch()">
  <button type="button" onclick="doSearch()">Find</button>
  <span class="notfound" id="nf">ID not found</span>
</div>
<div id="cards">
{cards_bm}
</div>
<script>
const total = {bm_total};
const ids = [{ids_bm_js}];
let cur = 0;
function copyCurrentOutput(btn) {{
  if (total === 0) return;
  const card = document.getElementById('card-' + cur);
  const box = card.querySelector('.cot-box');
  if (!box) return;
  const text = box.innerText;
  const doFlash = () => {{
    if (!btn) return;
    btn.classList.add('copied');
    const t = btn.textContent;
    btn.textContent = 'Copied';
    setTimeout(() => {{
      btn.textContent = t;
      btn.classList.remove('copied');
    }}, 1400);
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(doFlash).catch(() => fallbackCopy(text, btn, doFlash));
  }} else {{
    fallbackCopy(text, btn, doFlash);
  }}
}}
function fallbackCopy(text, btn, flash) {{
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {{
    document.execCommand('copy');
    if (flash) flash();
  }} catch (e) {{
    alert('Copy not supported in this browser.');
  }}
  document.body.removeChild(ta);
}}
function show(n) {{
  if (total === 0) return;
  document.getElementById('card-' + cur).classList.add('hidden');
  cur = Math.max(0, Math.min(total - 1, n));
  document.getElementById('card-' + cur).classList.remove('hidden');
  document.getElementById('ctr').textContent = (cur + 1) + ' / ' + total;
  document.getElementById('gi').value = cur + 1;
  window.scrollTo(0, 0);
}}
function prev() {{ show(cur - 1); }}
function next() {{ show(cur + 1); }}
function goTo(v) {{ show(parseInt(v, 10) - 1); }}
function doSearch() {{
  if (total === 0) return;
  const q = document.getElementById('si').value.trim().toLowerCase();
  if (!q) return;
  let i = ids.findIndex(x => x.toLowerCase() === q);
  if (i === -1) i = ids.findIndex(x => x.toLowerCase().startsWith(q));
  if (i !== -1) {{
    document.getElementById('nf').style.display = 'none';
    show(i);
  }} else {{
    document.getElementById('nf').style.display = 'inline';
  }}
}}
document.addEventListener('keydown', e => {{
  if (total === 0) return;
  if (document.activeElement === document.getElementById('si')) return;
  if (e.key === 'ArrowRight') next();
  if (e.key === 'ArrowLeft') prev();
}});
</script>
</body>
</html>
"""


def main() -> None:
    viewer_ids = load_viewer_ids()
    exclude = set(viewer_ids)

    with V12_CSV.open(newline="", encoding="utf-8") as f:
        by_id = {r["id"]: r for r in csv.DictReader(f)}

    cot_rows: list[dict] = []
    for pid in viewer_ids:
        if pid not in by_id:
            print(f"Id {pid} not in {V12_CSV.name}", file=sys.stderr)
            sys.exit(1)
        cot_rows.append(by_id[pid])

    with BM_CSV.open(newline="", encoding="utf-8") as f:
        bm_all = list(csv.DictReader(f))
    bm_rows = [r for r in bm_all if r["id"] not in exclude]

    bm_by_id = {r["id"]: r for r in bm_all}
    changed_rows: list[dict] = []
    for pid in viewer_ids:
        if pid not in bm_by_id:
            print(
                f"Warning: id {pid} from {IDS_FILE.name} not in {BM_CSV.name}; "
                f"omitted from {CHANGED_HTML.name}.",
                file=sys.stderr,
            )
            continue
        r = bm_by_id[pid]
        changed_rows.append(
            {
                "id": r["id"],
                "answer": r["answer"],
                "type": r.get("category") or "",
                "prompt": r["prompt"],
                "generated_cot": r["output"],
            }
        )

    if V11_HTML.is_file():
        v11_by_id = parse_v11_cot_inner_by_id(V11_HTML.read_text(encoding="utf-8"))
    else:
        print(f"Warning: {V11_HTML.name} not found; v11 pane will show a placeholder.", file=sys.stderr)
        v11_by_id = None

    COT_HTML.write_text(build_cot_viewer(cot_rows), encoding="utf-8")
    CHANGED_HTML.write_text(
        build_cot_viewer_changed_side_by_side(changed_rows, v11_by_id),
        encoding="utf-8",
    )
    BM_HTML.write_text(build_bm_viewer(bm_rows, exclude), encoding="utf-8")

    print(f"Wrote {COT_HTML.name}: {len(cot_rows)} curated id(s)")
    n_v11 = sum(1 for r in changed_rows if v11_by_id and r["id"] in v11_by_id)
    print(
        f"Wrote {CHANGED_HTML.name}: {len(changed_rows)} row(s) from {BM_CSV.name} "
        f"(side-by-side with v11: {n_v11} id(s) have v11 CoT)"
    )
    print(f"Wrote {BM_HTML.name}: {len(bm_rows)} row(s) (excluded {len(exclude)} id(s))")


if __name__ == "__main__":
    main()
