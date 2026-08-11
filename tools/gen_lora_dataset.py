#!/usr/bin/env python3
"""キャスト LoRA の学習データを Qwen-Image-Edit-2511 で量産する。

確定デザイン画像を参照に、ポーズ・表情・背景を変えたソロ画像を
キャラごとに生成し、musubi-tuner 形式 (画像 + 同名 .txt キャプション) で
dataset/{castId}/{char}/ に保存する。

  python3 tools/gen_lora_dataset.py tools/specs/cast/skyport.json --per-char 60
  python3 tools/gen_lora_dataset.py ... --chars pip,nana --start 30   # 追い足し
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
GEN_SIZE = 832

# 表情バイブル: シリーズ全体で共通の「描き方の決まり」。
# 学習キャプションと量産プロンプトで同じ言い回しを使うことで表情を標準化する。
EXPRESSIONS = [
    ("joyful", "smiling joyfully with closed eyes curved like happy arcs and a small open triangle mouth"),
    ("angry", "angry with short slanted eyebrows drawn above the dot eyes, puffed round cheeks and a tiny pout"),
    ("sad", "sad with one big round teardrop at the corner of one eye and a small wavy downturned mouth"),
    ("surprised", "surprised with slightly bigger round dot eyes, a tiny round open mouth and one small sweat drop beside the head"),
    ("worried", "worried with tilted eyebrows and a small wavy mouth"),
    ("normal", "with a calm gentle smile"),
]
POSES = [
    "standing and waving one paw",
    "sitting on the ground",
    "walking",
    "running",
    "jumping with both arms up",
    "lying on the tummy",
    "looking up at the sky",
    "seen from the side, standing",
    "seen from behind, looking back over the shoulder",
    "crouching and looking at something on the ground",
]
BACKGROUNDS = [
    "plain pure white background",
    "plain pale-cream background",
    "simple grassy meadow with two small flowers",
    "simple cozy room with a small round window",
    "simple pale-blue sky with one small cloud",
]


def build_jobs(per_char, start):
    jobs = []
    n_e, n_p, n_b = len(EXPRESSIONS), len(POSES), len(BACKGROUNDS)
    for i in range(start, start + per_char):
        expr = EXPRESSIONS[i % n_e]
        pose = POSES[(i // n_e) % n_p]
        bg = BACKGROUNDS[(i * 3 + i // (n_e * n_p)) % n_b]
        if "lying" in pose and expr[0] in ("surprised", "angry"):
            expr = EXPRESSIONS[5]  # 寝転びは穏やか顔に固定
        jobs.append((i, expr, pose, bg))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("--per-char", type=int, default=60)
    ap.add_argument("--start", type=int, default=0, help="連番の開始 (追い足し用)")
    ap.add_argument("--chars", help="対象キャラのカンマ区切り (省略時全員)")
    args = ap.parse_args()

    spec = json.loads(Path(args.cast).read_text(encoding="utf-8"))
    chars = dict(spec["characters"])
    if args.chars:
        chars = {k: chars[k] for k in args.chars.split(",")}

    import gc
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    import gen_original_book as g

    g._qwen_env()
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    # ジョブ一覧を確定 (未生成分のみ)
    work = []
    for name, ch in chars.items():
        ref = Image.open(REPO / ch["ref"]).convert("RGB")
        ref.thumbnail((640, 854), Image.LANCZOS)
        out_dir = REPO / "dataset" / spec["castId"] / name
        out_dir.mkdir(parents=True, exist_ok=True)
        char_base = sum(ord(c) for c in name) * 31  # 安定シード (hash() は毎回変わる)
        for i, expr, pose, bg in build_jobs(args.per_char, args.start):
            ekey, etext = expr
            out = out_dir / f"{name}_{i:03d}.png"
            if out.exists():
                continue
            prompt = (
                f"Using the character in the reference image ({ch['desc']}), draw "
                f"exactly the same character, alone, {pose}, {etext}, on a {bg}. "
                "Keep the character's design, colors, proportions, face and eye "
                "style exactly identical to the reference image. "
                f"{spec['style_suffix']}"
            )
            caption = (
                f"{ch['trigger']}, a {ch['species']}, {pose}, {etext}, {bg}, "
                "simple pale hand-drawn kawaii children's illustration"
            )
            work.append(
                {"ref": ref, "out": out, "prompt": prompt, "caption": caption,
                 "seed": 50000 + char_base + i * 7, "label": f"{name} {i:03d} {ekey}"}
            )
    if not work:
        print("生成対象なし")
        return
    print(f"{len(work)} 枚を生成します", flush=True)

    # pass 1: TE のみで埋め込みを事前計算 (transformer と同居すると 32GB で OOM)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        g.QWEN_MODEL, transformer=None, torch_dtype=torch.bfloat16,
        quantization_config=g._quant_config(["text_encoder"]),
    )
    pipe.to("cuda")
    with torch.inference_mode():
        for n, job in enumerate(work):
            cw, ch_ = calculate_dimensions(
                CONDITION_IMAGE_SIZE, job["ref"].size[0] / job["ref"].size[1]
            )
            cond = [pipe.image_processor.resize(job["ref"], ch_, cw)]
            pe, pm = pipe.encode_prompt(prompt=[job["prompt"]], image=cond, device="cuda")
            job["pe"] = pe.cpu()
            job["pm"] = pm.cpu() if pm is not None else None
            if n % 25 == 0:
                print(f"  embed {n}/{len(work)}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # pass 2: transformer のみで生成
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        g.QWEN_MODEL, text_encoder=None, torch_dtype=torch.bfloat16,
        quantization_config=g._quant_config(["transformer"]),
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    steps, cfg = g.apply_lightning(pipe)
    for n, job in enumerate(work):
        print(f"[{n}/{len(work)}] {job['label']} seed={job['seed']}", flush=True)
        img = pipe(
            image=[job["ref"]],
            prompt_embeds=job["pe"].to("cuda"),
            prompt_embeds_mask=job["pm"].to("cuda") if job["pm"] is not None else None,
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            height=GEN_SIZE,
            width=GEN_SIZE,
            generator=torch.Generator("cpu").manual_seed(job["seed"]),
        ).images[0]
        img.save(job["out"])
        job["out"].with_suffix(".txt").write_text(job["caption"], encoding="utf-8")
        job["pe"] = job["pm"] = None
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
