#!/usr/bin/env python3
"""絵本スペックの決定的 Lint (生成前ゲート)。

  python3 tools/spec_lint.py tools/specs/pb-en-0703.json

チェック内容:
- 必須フィールド (bookId/title/level/genres/backend/cover/pages)
- ページ必須: text/scene/cast/seed/location
- location は spec の locations 辞書に定義されていること
- 場所が変わるページで、直前ページの text/scene に移動の手がかりがあるか (警告)
- レベル別の総語数帯 (L1<150 / L2 150-350 / L3 351-700 / L4 701+)
- seed の重複禁止
- ジャンルは既存8種のみ
- lora バックエンド: cast が castSpec に存在すること
- story_ledger があれば、各項目の satisfied_by が実在ページを指すこと
終了コード: エラーあり=1 / 警告のみ=0
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEVEL_BANDS = {1: (1, 149), 2: (150, 350), 3: (351, 700), 4: (701, 10**6)}
GENRES = {"にちじょう", "かぞく・ともだち", "どうぶつ", "しぜん・かがく",
          "むかしばなし", "ファンタジー", "ぼうけん", "ゆかい・ユーモア"}
MOVE_HINTS = re.compile(
    r"\b(walk\w*|run\w*|go|goes|went|came|come\w*|arriv\w*|knock\w*|door\w*|"
    r"outside|inside|to the|off to|head\w*|visit\w*|enter\w*|leav\w*|left|"
    r"climb\w*|fly|flew|sail\w*|hop\w*|step\w*)\b", re.I
)


def lint(spec_path):
    errors, warnings = [], []
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))

    for key in ("bookId", "title", "level", "genres", "cover", "pages"):
        if key not in spec:
            errors.append(f"必須フィールドなし: {key}")
    if errors:
        return spec, errors, warnings

    if not set(spec["genres"]) <= GENRES:
        errors.append(f"未知のジャンル: {set(spec['genres']) - GENRES}")

    locations = spec.get("locations", {})
    if not locations:
        errors.append("locations 辞書がありません (v2 スペックでは必須)")

    cast_chars = {}
    if spec.get("castSpec"):
        cast_chars = json.loads((REPO / spec["castSpec"]).read_text(encoding="utf-8"))[
            "characters"
        ]

    seeds = {}
    prev_loc, prev_entry = None, None
    entries = [("cover", spec["cover"])] + [
        (f"p{i + 1}", p) for i, p in enumerate(spec["pages"])
    ]
    for name, e in entries:
        for key in ("scene", "cast", "seed"):
            if key not in e:
                errors.append(f"{name}: {key} なし")
        if name != "cover" and "text" not in e:
            errors.append(f"{name}: text なし")
        loc = e.get("location")
        if not loc:
            errors.append(f"{name}: location なし")
        elif locations and loc not in locations:
            errors.append(f"{name}: 未定義の location '{loc}'")
        if "seed" in e:
            if e["seed"] in seeds:
                errors.append(f"{name}: seed {e['seed']} が {seeds[e['seed']]} と重複")
            seeds[e["seed"]] = name
        if cast_chars:
            for c in e.get("cast", []):
                if c not in cast_chars:
                    errors.append(f"{name}: castSpec に無いキャラ '{c}'")
        # 場所遷移の妥当性 (表紙は対象外)
        if name != "cover" and prev_loc and loc and loc != prev_loc:
            hint_src = f"{prev_entry.get('text', '')} {prev_entry.get('scene', '')} {e.get('text', '')} {e.get('scene', '')}"
            if not MOVE_HINTS.search(hint_src):
                warnings.append(
                    f"{name}: 場所が {prev_loc} → {loc} に変わるが、移動の手がかりが本文/シーンに見当たらない"
                )
        if name != "cover":
            prev_loc, prev_entry = loc, e

    wc = sum(len(p["text"].split()) for p in spec["pages"] if "text" in p)
    lo, hi = LEVEL_BANDS[spec["level"]]
    if not (lo <= wc <= hi):
        errors.append(f"総語数 {wc} が level {spec['level']} の帯域 [{lo},{hi}] 外")

    for i, item in enumerate(spec.get("story_ledger", [])):
        ref = item.get("satisfied_by", "")
        m = re.match(r"(cover|p(\d+))", str(ref))
        if not m or (m.group(2) and int(m.group(2)) > len(spec["pages"])):
            errors.append(f"story_ledger[{i}] '{item.get('item')}': satisfied_by '{ref}' が不正")

    return spec, errors, warnings


def main():
    spec_path = sys.argv[1]
    spec, errors, warnings = lint(spec_path)
    wc = sum(len(p["text"].split()) for p in spec.get("pages", []) if "text" in p)
    print(f"{spec.get('bookId', '?')}: 総語数 {wc}, ページ {len(spec.get('pages', []))}")
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print(f"NG ({len(errors)} errors, {len(warnings)} warnings)")
        return 1
    print(f"OK ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
