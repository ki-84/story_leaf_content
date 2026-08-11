#!/usr/bin/env python3
"""生成した絵本が既存スキーマ・慣習と整合しているか検証する。

  python3 tools/validate_book.py pb-en-0700
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEVEL_BANDS = {1: (1, 149), 2: (150, 350), 3: (351, 700), 4: (701, 10**6)}
REFERENCE = "pb-en-0001"  # スキーマ比較の基準書


def main(book_id):
    errors = []
    ok = lambda msg: print(f"  OK: {msg}")

    book_dir = REPO / "books" / book_id
    book = json.loads((book_dir / "book.json").read_text(encoding="utf-8"))
    ref = json.loads(
        (REPO / "books" / REFERENCE / "book.json").read_text(encoding="utf-8")
    )

    # キー集合 (slug / bundled は初期20冊のみの任意キーなので除外して比較)
    optional = {"slug", "bundled"}
    ref_keys = set(ref) - optional
    if set(book) == ref_keys:
        ok(f"book.json のキー集合が {REFERENCE} と一致")
    else:
        errors.append(f"キー差分: +{set(book) - ref_keys} -{ref_keys - set(book)}")
    if set(book["attribution"]) == set(ref["attribution"]):
        ok("attribution の9キーが一致")
    else:
        errors.append("attribution キー不一致")
    for i, page in enumerate(book["pages"]):
        if set(page) != {"index", "imagePath", "text"} or page["index"] != i:
            errors.append(f"pages[{i}] のキー/index が不正")

    # ページ数・画像
    if book["pageCount"] == len(book["pages"]):
        ok(f"pageCount == len(pages) == {book['pageCount']}")
    else:
        errors.append("pageCount と pages 数が不一致")
    images = sorted((book_dir / "images").glob("*.jpg"))
    if len(images) == book["pageCount"] + 1:
        ok(f"画像枚数 {len(images)} = pageCount+1")
    else:
        errors.append(f"画像枚数 {len(images)} != pageCount+1")
    for page in book["pages"]:
        if not (book_dir / page["imagePath"]).is_file():
            errors.append(f"実ファイルなし: {page['imagePath']}")
    if (book_dir / book["coverImagePath"]).is_file():
        ok("coverImagePath 実在")
    else:
        errors.append("表紙画像なし")

    # 画像サイズ
    from PIL import Image

    sizes = {Image.open(p).size for p in images}
    if sizes == {(546, 546)}:
        ok("全画像 546x546 JPEG")
    else:
        errors.append(f"画像サイズ不一致: {sizes}")

    # 語数とレベル
    wc = sum(len(p["text"].split()) for p in book["pages"])
    lo, hi = LEVEL_BANDS[book["level"]]
    if wc == book["wordCount"]:
        ok(f"wordCount 実測一致 ({wc})")
    else:
        errors.append(f"wordCount {book['wordCount']} != 実測 {wc}")
    if lo <= wc <= hi:
        ok(f"level {book['level']} の語数帯 [{lo},{hi}] 内")
    else:
        errors.append(f"語数 {wc} が level {book['level']} の帯域外")

    # covers/ と catalog.json
    cover = REPO / "covers" / f"{book_id}.jpg"
    if cover.is_file() and cover.read_bytes() == (book_dir / "images/01.jpg").read_bytes():
        ok("covers/{id}.jpg が images/01.jpg と同一")
    else:
        errors.append("covers のコピーが無いか内容不一致")

    catalog = json.loads((REPO / "catalog.json").read_text(encoding="utf-8"))
    ids = [b["bookId"] for b in catalog["books"]]
    entry = next((b for b in catalog["books"] if b["bookId"] == book_id), None)
    if entry is None:
        errors.append("catalog.json にエントリなし")
    else:
        ok("catalog.json にエントリあり")
        ref_entry_keys = set(catalog["books"][0])
        if set(entry) != ref_entry_keys:
            errors.append(f"catalog エントリのキー不一致: {set(entry) ^ ref_entry_keys}")
        for key in ("title", "level", "genres", "wordCount", "pageCount"):
            if entry[key] != book[key]:
                errors.append(f"catalog.{key} が book.json と不一致")
    if ids == sorted(ids):
        ok(f"catalog.json は bookId 昇順を維持 (全{len(ids)}冊)")
    else:
        errors.append("catalog.json の順序が崩れた")

    print()
    if errors:
        print(f"NG ({len(errors)} 件):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"{book_id}: 全チェック合格")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pb-en-0700"))
