#!/usr/bin/env python3
"""新キャスト「Skyport Village」のデザイン案を FLUX.1-schnell で生成して合成する。"""
import sys
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/home/kuwabara/devwork/story_leaf_content/preview")
SUFFIX = (
    " Extremely simple and loose Japanese hand-drawn kawaii manga mascot style: "
    "thin soft slightly wobbly hand-drawn outline like a gentle pencil sketch, "
    "squat pear-shaped soft body, short nubby stub arms, tiny feet, "
    "small round black dot eyes, a tiny simple open mouth like a small triangle, "
    "faint pink cheek blush, milky pale colors with lots of white and cream, "
    "flat minimal shading, plain pure white background, full body, standing "
    "front view, centered, no text, no letters."
)
CAST = [
    ("pip", 9701, "A small pale-cream hedgehog child with a very round blob body. His back is covered with a soft scalloped shape of short round gentle bumps, drawn very simply, pale beige. Cream face and belly, empty little paws."),
    ("nana", 9802, "A soft white rabbit child with a very round chubby blob body, short stubby limbs, long upright ears, and a compact well-balanced face: the tiny dot eyes, tiny pink nose and small mouth are gathered close together near the center of the round face. No clothes."),
    ("coco", 9603, "A tiny round pale-yellow duckling with a very small orange beak, tiny orange webbed feet, a little tuft of feathers on top of the head, no clothes."),
    ("bruno", 9604, "A small bear cub child with a perfectly round circle-shaped face, small round ears, a tiny pale tan muzzle, light-brown fur, a round pale-tan tummy patch, empty little paws, no clothes."),
    ("olive", 9705, "A round pale-cream owl with very thin round wire glasses over tiny dot eyes, drawn very simply with minimal feather detail, small folded pale-beige wings, tiny simple beak."),
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

REGEN = {"nana"}  # 全員 (目のハイライト標準化)
tiles = []
for name, seed, desc in CAST:
    cached = OUT / f"cast-{name}.png"
    if name not in REGEN and cached.is_file():
        tiles.append((name.capitalize(), Image.open(cached).convert("RGB")))
        continue
    print(f"[{name}] seed={seed}", flush=True)
    img = pipe(
        desc + SUFFIX,
        width=768,
        height=1024,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    img.save(OUT / f"cast-{name}.png")
    tiles.append((name.capitalize(), img))

# ほぼ均一サイズ (Coco だけ少し小さい)。地面ラインを揃えて合成する
RATIO = {"Pip": 0.90, "Nana": 1.00, "Coco": 0.78, "Bruno": 1.00, "Olive": 0.88}
BASE_H = 900
pad, label_h = 40, 56
scaled = []
for label, tile in tiles:
    h = int(BASE_H * RATIO[label])
    w = int(tile.width * h / tile.height)
    scaled.append((label, tile.resize((w, h), Image.LANCZOS)))
tw = sum(t.width for _, t in scaled) + pad * (len(scaled) + 1)
sheet = Image.new("RGB", (tw, BASE_H + label_h + pad * 2), "white")
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
x = pad
for label, tile in scaled:
    sheet.paste(tile, (x, pad + BASE_H - tile.height))
    lw = draw.textlength(label, font=font)
    draw.text((x + (tile.width - lw) / 2, BASE_H + pad + 8), label, font=font, fill="#4a3623")
    x += tile.width + pad
sheet.save(OUT / "cast-proposal-skyport.png")
print("saved: preview/cast-proposal-skyport.png")
