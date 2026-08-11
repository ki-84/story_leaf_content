# Story Leaf コンテンツリポジトリ

制作の標準手順・品質ゲート・失敗パターン辞書はすべて **AGENTS.md** にあります。
絵本制作のタスクでは必ず AGENTS.md を読んでから作業してください。

- 1冊作る: `python3 tools/make_book.py tools/specs/pb-en-07XX.json` (venv: ~/venvs/torch)
- 機械ゲートを通らない本は発行しない。通っても目視と人間承認を省略しない。
