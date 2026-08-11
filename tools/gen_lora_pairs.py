#!/usr/bin/env python3
"""2キャラ同席の学習画像を生成する (特徴混線対策の要)。全ペア×2枚。"""
import gc
import itertools
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import gen_original_book as g

g._qwen_env()
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)

spec = json.loads((REPO / "tools/specs/cast/skyport.json").read_text(encoding="utf-8"))
chars = spec["characters"]
SCENES = [
    ("standing side by side and waving", "in a simple grassy meadow with two small flowers"),
    ("walking together and talking happily", "on a plain pale-cream background"),
]
out_dir = REPO / "dataset" / "skyport" / "pairs"
out_dir.mkdir(parents=True, exist_ok=True)

refs = {}
for name, ch in chars.items():
    img = Image.open(REPO / ch["ref"]).convert("RGB")
    img.thumbnail((512, 683), Image.LANCZOS)
    refs[name] = img

work = []
for k, (a, b) in enumerate(itertools.combinations(chars, 2)):
    for s, (pose, bg) in enumerate(SCENES):
        out = out_dir / f"pair_{a}_{b}_{s}.png"
        if out.exists():
            continue
        ca, cb = chars[a], chars[b]
        prompt = (
            f"Using the two characters from the reference images (image 1: {ca['desc']} "
            f"image 2: {cb['desc']}), draw exactly these two characters together, "
            f"{pose}, {bg}. Only these two characters: one {ca['species']} and one "
            f"{cb['species']}. Keep each character's design, colors, proportions, "
            "face and eye style exactly identical to its reference image; do not "
            "mix the two characters' features. "
            f"{spec['style_suffix']}"
        )
        caption = (
            f"{ca['trigger']} and {cb['trigger']}, a {ca['species']} and a "
            f"{cb['species']}, {pose}, {bg}, simple pale hand-drawn kawaii "
            "children's illustration"
        )
        work.append({"a": a, "b": b, "out": out, "prompt": prompt,
                     "caption": caption, "seed": 70000 + k * 10 + s})

print(f"{len(work)} 枚のペア画像を生成", flush=True)
if not work:
    sys.exit(0)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    g.QWEN_MODEL, transformer=None, torch_dtype=torch.bfloat16,
    quantization_config=g._quant_config(["text_encoder"]),
)
pipe.to("cuda")
with torch.inference_mode():
    for job in work:
        cond = []
        for n in (job["a"], job["b"]):
            cw, ch_ = calculate_dimensions(
                CONDITION_IMAGE_SIZE, refs[n].size[0] / refs[n].size[1]
            )
            cond.append(pipe.image_processor.resize(refs[n], ch_, cw))
        pe, pm = pipe.encode_prompt(prompt=[job["prompt"]], image=cond, device="cuda")
        job["pe"] = pe.cpu()
        job["pm"] = pm.cpu() if pm is not None else None
del pipe
gc.collect()
torch.cuda.empty_cache()

pipe = QwenImageEditPlusPipeline.from_pretrained(
    g.QWEN_MODEL, text_encoder=None, torch_dtype=torch.bfloat16,
    quantization_config=g._quant_config(["transformer"]),
)
pipe.to("cuda")
pipe.vae.enable_tiling()
steps, cfg = g.apply_lightning(pipe)
for n, job in enumerate(work):
    print(f"[{n + 1}/{len(work)}] {job['out'].name}", flush=True)
    img = pipe(
        image=[refs[job["a"]], refs[job["b"]]],
        prompt_embeds=job["pe"].to("cuda"),
        prompt_embeds_mask=job["pm"].to("cuda") if job["pm"] is not None else None,
        true_cfg_scale=cfg,
        num_inference_steps=steps,
        height=g.QWEN_GEN_SIZE,
        width=g.QWEN_GEN_SIZE,
        generator=torch.Generator("cpu").manual_seed(job["seed"]),
    ).images[0]
    img.save(job["out"])
    job["out"].with_suffix(".txt").write_text(job["caption"], encoding="utf-8")
    torch.cuda.empty_cache()
print("done")
