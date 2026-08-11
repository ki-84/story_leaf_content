# Story Leaf オリジナル絵本 制作標準手順書 (SOP)

このリポジトリは英語多読アプリ「Story Leaf」の配信コンテンツです。既存絵本 (CC BY) に加え、
**Nobuaki Kuwabara 著作のオリジナル絵本 (pb-en-0700〜, All Rights Reserved)** を制作・発行します。
この文書は、どの AI エージェント (Claude / Codex / その他) が作業しても
**お客様向け品質**の絵本を再現できるようにするための唯一の正典です。

## 大原則

1. **機械ゲートを通らない本は発行しない。** ゲートは `tools/make_book.py` に直列化されている。
2. **ゲートが通っても目視と人間承認を省略しない。** 美的判定 (画風の揺れ・微妙な特徴混線) は機械化されていない。
3. **すべての生成はスペック JSON から再現可能に保つ。** 手動レタッチ禁止。直したければスペックを直して再生成する。
4. 内容は子供向けに安全なもの (暴力・恐怖・差別表現なし)。実在作家・作品のキャラクターを模倣しない。

## 制作フロー (1冊)

```bash
source /home/kuwabara/venvs/torch/bin/activate

# 1. スペックを書く: tools/specs/pb-en-07XX.json (下記スキーマ)。物語・シーン英文はエージェントが執筆
# 2. オーケストレータ実行 (lint → 生成 → VLM監査+自動再ロール → 検証 → ビューア)
python3 tools/make_book.py tools/specs/pb-en-07XX.json
# 3. preview/pb-en-07XX.html を目視、問題ページはスペック修正して
python3 tools/gen_original_book.py tools/specs/pb-en-07XX.json --only 04,07   # 部分再ロール
python3 tools/make_book.py tools/specs/pb-en-07XX.json --skip-generate        # ゲート再実行
# 4. 人間の承認後にコミット・プッシュ
```

## スペック v2 スキーマ (tools/specs/pb-en-07XX.json)

```jsonc
{
  "bookId": "pb-en-07XX",         // 連番。既存最大値+1
  "title": "...",                  // 英語タイトル (表紙に焼き込まれる)
  "level": 2,                      // 1..4。総語数帯: L1<150 / L2 150-350 / L3 351-700 / L4 701+
  "genres": ["かぞく・ともだち"],   // 既存8種のみ: にちじょう/かぞく・ともだち/どうぶつ/しぜん・かがく/むかしばなし/ファンタジー/ぼうけん/ゆかい・ユーモア
  "backend": "lora",              // lora (推奨・Skyportキャスト) / qwen (参照画像方式) / flux
  "castSpec": "tools/specs/cast/skyport.json",
  "lora": "tools/lora/output/skyport-cast-v1-bf16.safetensors",
  "loraMultiplier": 1.2,
  "style_suffix": "simple pale hand-drawn kawaii children's illustration",
  "locations": {                   // 舞台の辞書 (必須)。ページはここの key を参照
    "bruno-kitchen": "Bruno's simple cozy kitchen"
  },
  "story_ledger": [                // 話の成立条件の台帳 (任意だが強く推奨)
    {"item": "flour", "satisfied_by": "p2"}   // 「必要なもの」が「どのページで満たされるか」
  ],
  "cover":  {"scene": "...", "cast": [...], "seed": N, "location": "..."},
  "pages": [
    {"text": "本文英語",           // 曲がり引用符 “ ” を使用。改行は \n
     "scene": "絵の英語記述",       // キャラは "the bear cub" のように種で参照 (名前は通じない)
     "cast": ["bruno"],           // castSpec のキー。プロンプトには trigger+prompt_desc が自動併記される
     "seed": N,                   // 全ページ一意。再ロールは +1000 して記録
     "location": "bruno-kitchen", // locations のキー
     "must_show": ["milk jug"]}   // VLM監査で存在確認したい小道具 (任意)
  ]
}
```

## レベル別文体ガイド

| レベル | 総語数 | 文体 |
|---|---|---|
| 1 | <150 | 現在形・1文/頁・基礎語彙・反復構文 ("Is it...? No...") |
| 2 | 150-350 | 過去形・2〜3文/頁・単文中心・会話少し |
| 3 | 351-700 | 複文・会話多め・段落なし |
| 4 | 701+ | 段落構成・描写豊か |

## 表情バイブル (シリーズ共通の言い回し。学習データと同一のためこのまま使うと安定)

- 喜: `smiling joyfully with closed eyes curved like happy arcs and a small open triangle mouth`
- 怒: `angry with short slanted eyebrows drawn above the dot eyes, puffed round cheeks and a tiny pout`
- 哀: `sad with one big round teardrop at the corner of one eye and a small wavy downturned mouth`
- 驚: `surprised with slightly bigger round dot eyes, a tiny round open mouth and one small sweat drop beside the head`
- 困: `worried with tilted eyebrows and a small wavy mouth`
- 平常: `with a calm gentle smile`

## 失敗パターン辞書 (シーン英文の書き方)

生成AIで繰り返し起きた不良と、実証済みの対処:

| 症状 | 対処 |
|---|---|
| キャラの頭数が減る/増える | `exactly five animal friends` のように**頭数を数詞で明示**。それでも駄目なら**構図を単独/2キャラに簡素化** |
| キャラ同士の特徴混線 (くちばし移植等) | 同席キャラを減らす。`do not mix` は効かない。**否定形は基本無視される** — 肯定表現で書く (bareheaded 等) |
| 場所・時刻の食い違い | location を必須化済み。**夜は "at night" をシーンに明示** (書かないと昼になる) |
| 建物・物のデザイン揺れ | locations の記述に色・形を書き込み、全ページ同文で参照させる |
| 表紙にモデルが勝手に文字を描く | `Absolutely no letters, no words, no writing anywhere` をシーン末尾に |
| 人間が混入する | VLM監査が検出。再ロールで直らなければ `two mice` のように種+数を明示 |

## 生成環境の注意 (RTX 5090 / 32GB)

- venv: 生成系 `~/venvs/torch` / 学習系 `~/venvs/musubi` (musubi-tuner)
- **bitsandbytes NF4 は Blackwell でノイズ画像になる。量子化は必ず int8**
- Qwen 系 20B は transformer と TE を**同時に VRAM に載せない** (2パス方式。既存スクリプトは対応済み)
- musubi 推論: `--attn_mode torch` (sdpa は KeyError)、`--negative_prompt " "` 必須、LoRA は bf16 版を使う
- シェルで `| tail` だけにパイプすると失敗が exit 0 に見える。**`set -o pipefail` + `tee` でログを残す**
- 長時間生成の前に `sudo nvidia-smi -pl 450` 推奨 (温度余裕。速度-1割)

## キャスト運用 (Skyport Village)

- 正典: `tools/specs/cast/skyport.json` (トリガー語・prompt_desc・確定デザイン画像パス)
- キャラ追加手順: デザイン画像を確定 → `tools/gen_lora_dataset.py` で60枚生成 → 目視選別 →
  `tools/gen_lora_pairs.py` でペア追加 → `tools/lora/train_skyport.sh all` で再学習 (約70分) →
  bf16 変換 → テスト生成で検証
- LoRA 出力と学習データは配信対象外 (.gitignore 済み)。**学習済み LoRA はリポジトリ外にもバックアップすること**

## 発行チェックリスト

- [ ] `tools/make_book.py` が全ゲート green
- [ ] ビューアで全ページ目視 (キャラ・画風・話の流れ・絵と本文の一致)
- [ ] 本文音読チェック (レベル感・自然な英語)
- [ ] story_ledger の全項目が絵でも確認できる
- [ ] 人間 (発行責任者) の承認
- [ ] git commit (作者: Nobuaki Kuwabara) → push
