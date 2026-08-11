#!/usr/bin/env python3
"""book.json を1枚の自己完結 HTML 絵本ビューアに変換する。

  python3 tools/book_to_html.py pb-en-0700
  -> preview/pb-en-0700.html (画像は base64 埋め込み。単体で開ける)

操作: ← → キー / 画面の左右クリック / 下部のボタン
"""
import base64
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: #efe5d0;
    font-family: "Comic Sans MS", "Segoe Print", "Noto Sans", sans-serif;
    padding: 16px;
  }
  .book {
    background: #fffdf6;
    border-radius: 18px;
    box-shadow: 0 12px 40px rgba(90, 60, 20, .25);
    max-width: 640px; width: 100%;
    overflow: hidden;
    display: flex; flex-direction: column;
  }
  .art { position: relative; background: #fffdf6; user-select: none; }
  .art img { display: block; width: 100%; height: auto; }
  .art .zone { position: absolute; top: 0; bottom: 0; width: 50%; cursor: pointer; }
  .art .zone.prev { left: 0; }
  .art .zone.next { right: 0; }
  .caption {
    padding: 22px 30px 14px;
    font-size: 1.45rem; line-height: 1.55; color: #4a3623;
    min-height: 5.2em; text-align: center; white-space: pre-line;
  }
  .caption.title-page { min-height: 1em; padding: 8px; }
  .nav {
    display: flex; align-items: center; justify-content: center; gap: 18px;
    padding: 6px 16px 18px;
  }
  .nav button {
    border: none; background: #f0b429; color: #4a3623;
    font: inherit; font-size: 1.3rem; font-weight: bold;
    width: 54px; height: 44px; border-radius: 12px; cursor: pointer;
  }
  .nav button:disabled { opacity: .25; cursor: default; }
  .nav .counter { font-size: .95rem; color: #8a7358; min-width: 6em; text-align: center; }
  .colophon {
    margin-top: 14px; font-size: .8rem; color: #9a8770; text-align: center;
  }
</style>
</head>
<body>
  <div class="book">
    <div class="art">
      <img id="art" alt="">
      <div class="zone prev" onclick="go(-1)"></div>
      <div class="zone next" onclick="go(1)"></div>
    </div>
    <div class="caption" id="caption"></div>
    <div class="nav">
      <button id="prev" onclick="go(-1)">&#8592;</button>
      <span class="counter" id="counter"></span>
      <button id="next" onclick="go(1)">&#8594;</button>
    </div>
  </div>
  <div class="colophon">__COLOPHON__</div>
<script>
const pages = __PAGES__;
let i = 0;
function render() {
  document.getElementById('art').src = pages[i].img;
  const cap = document.getElementById('caption');
  cap.textContent = pages[i].text;
  cap.classList.toggle('title-page', i === 0);
  document.getElementById('counter').textContent =
    i === 0 ? 'cover' : i + ' / ' + (pages.length - 1);
  document.getElementById('prev').disabled = i === 0;
  document.getElementById('next').disabled = i === pages.length - 1;
}
function go(d) { i = Math.min(pages.length - 1, Math.max(0, i + d)); render(); }
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ') go(1);
  if (e.key === 'ArrowLeft') go(-1);
});
render();
</script>
</body>
</html>
"""


def data_uri(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def main(book_id):
    book_dir = REPO / "books" / book_id
    book = json.loads((book_dir / "book.json").read_text(encoding="utf-8"))
    attr = book["attribution"]

    pages = [{"img": data_uri(book_dir / book["coverImagePath"]), "text": ""}]
    for page in book["pages"]:
        img = book_dir / page["imagePath"]
        if not img.is_file():  # 既存書には欠損画像の warnings があるため
            continue
        pages.append({"img": data_uri(img), "text": page["text"]})

    colophon = " · ".join(filter(None, [
        html.escape(book["title"]),
        html.escape(attr["author"] or ""),
        html.escape(attr["license"] or ""),
        f'Level {book["level"]} · {book["wordCount"]} words',
    ]))

    out_dir = REPO / "preview"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{book_id}.html"
    out.write_text(
        TEMPLATE
        .replace("__TITLE__", html.escape(book["title"]))
        .replace("__COLOPHON__", colophon)
        .replace("__PAGES__", json.dumps(pages, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(pages)} screens)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pb-en-0700")
