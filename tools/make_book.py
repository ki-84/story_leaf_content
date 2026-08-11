#!/usr/bin/env python3
"""絵本制作オーケストレータ: 全品質ゲートを直列実行する唯一の入口。

  python3 tools/make_book.py tools/specs/pb-en-07XX.json

ステージ:
  1. spec_lint     決定的チェック (エラーなら中断)
  2. generate      画像生成 (spec の backend に従う)
  3. vlm audit     VLM 構造監査 → FAIL ページを seed+1000 で自動再ロール (最大3周)
  4. validate      既存699冊とのスキーマ整合
  5. viewer        preview/{bookId}.html 生成
  6. 残タスク表示   目視確認と人間承認 (機械ゲートでは代替しない)

--skip-generate で 3 以降のみ再実行できる。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
MAX_REROLL_ROUNDS = 3


def run(desc, cmd, **kw):
    print(f"\n=== {desc} ===", flush=True)
    return subprocess.run(cmd, cwd=REPO, **kw).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    book_id = spec["bookId"]

    # 1. Lint (エラーで中断)
    if run("1/6 spec lint", [PY, "tools/spec_lint.py", str(spec_path)]) != 0:
        print("Lint エラー。スペックを修正してください。")
        return 1

    # 2. 生成
    if not args.skip_generate:
        if run("2/6 画像生成", [PY, "tools/gen_original_book.py", str(spec_path)]) != 0:
            print("生成失敗。ログを確認してください。")
            return 1

    # 3. VLM 監査 + 自動再ロール
    for round_no in range(1, MAX_REROLL_ROUNDS + 1):
        audit_json = REPO / "preview" / f"audit-{book_id}.json"
        run(f"3/6 VLM 構造監査 (round {round_no})",
            [PY, "tools/review_book.py", str(spec_path), "--json", str(audit_json)])
        result = json.loads(audit_json.read_text(encoding="utf-8"))
        fail_pages = result["fail_pages"]
        if not fail_pages:
            print("構造監査クリア")
            break
        if round_no == MAX_REROLL_ROUNDS:
            print(f"自動再ロール上限。残 FAIL ページ {fail_pages} はシーン記述の見直しが必要です。")
            print("対処ヒント: 頭数の明示 (exactly N animals)・構図の単独化・場所の明示。")
            return 1
        # seed を +1000 して該当ページのみ再生成 (スペックに記録され再現可能)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        entries = {"01": spec["cover"]}
        for i, p in enumerate(spec["pages"]):
            entries[f"{i + 2:02d}"] = p
        for num in fail_pages:
            entries[num]["seed"] += 1000
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"FAIL ページ {fail_pages} を seed+1000 で再ロールします")
        if run(f"再ロール {','.join(fail_pages)}",
               [PY, "tools/gen_original_book.py", str(spec_path), "--only", ",".join(fail_pages)]) != 0:
            return 1

    # 4. スキーマ検証
    if run("4/6 スキーマ検証", [PY, "tools/validate_book.py", book_id]) != 0:
        return 1

    # 5. ビューア
    if run("5/6 ビューア生成", [PY, "tools/book_to_html.py", book_id]) != 0:
        return 1

    print(f"""
=== 6/6 残タスク (機械ゲートでは代替不可) ===
  [ ] preview/{book_id}.html を開き全ページ目視 (美的判定: 画風の揺れ・微妙な特徴混線・かわいさ)
  [ ] 本文を音読して英語表現とレベル感を確認
  [ ] 人間 (発行責任者) の承認
  [ ] 承認後: git add books/{book_id} covers/{book_id}.jpg catalog.json tools/specs/ && git commit && git push
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
