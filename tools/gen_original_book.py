#!/usr/bin/env python3
"""Story Leaf オリジナル絵本生成パイプライン。

スペック JSON (tools/specs/{bookId}.json) から:
  1. FLUX.1-schnell (ローカル GPU) で表紙 + 全ページの絵を生成
  2. 表紙にタイトルと著者名を焼き込み
  3. 546x546 JPEG に縮小して books/{bookId}/images/NN.jpg へ配置
  4. 既存スキーマ互換の book.json を出力
  5. covers/{bookId}.jpg へ表紙をコピーし catalog.json を更新

バックエンドは2種:
  flux : FLUX.1-schnell の txt2img (キャラはプロンプト文で固定)
  qwen : Qwen-Image-Edit-2509 (Apache 2.0)。キャラシート画像を毎ページ参照して
         一貫性を保つ。spec の "backend": "qwen" か --backend qwen で有効。
         事前に --make-cast-sheet でシートを作り目視確認しておくこと。

実行例 (venv: /home/kuwabara/venvs/torch):
  python3 tools/gen_original_book.py tools/specs/pb-en-0700.json
  python3 tools/gen_original_book.py tools/specs/pb-en-0700.json --only 01,05 --seed-shift 100
  python3 tools/gen_original_book.py tools/specs/pb-en-0702.json --make-cast-sheet
  python3 tools/gen_original_book.py tools/specs/pb-en-0702.json --backend qwen
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
QWEN_GEN_SIZE = 832  # 1024 だと int8 常駐 30GB + 複数参照で 32GB VRAM が溢れる
_HUB = Path.home() / ".cache/huggingface/hub"
QWEN_VERSIONS = {
    "2509": (
        "Qwen/Qwen-Image-Edit-2509",
        str(_HUB / "models--lightx2v--Qwen-Image-Lightning/snapshots/*/**/*Edit-2509*8step*.safetensors"),
    ),
    # 2511: キャラ一貫性 (特に複数人物の特徴混線) が改善された後継。既定はこちら
    "2511": (
        "Qwen/Qwen-Image-Edit-2511",
        str(_HUB / "models--lightx2v--Qwen-Image-Edit-2511-Lightning/snapshots/*/**/*8steps*bf16*.safetensors"),
    ),
}
QWEN_MODEL, LIGHTNING_GLOB = QWEN_VERSIONS["2511"]


def set_qwen_version(version):
    global QWEN_MODEL, LIGHTNING_GLOB
    QWEN_MODEL, LIGHTNING_GLOB = QWEN_VERSIONS[version]
GEN_SIZE = 1024          # 生成解像度
OUT_SIZE = 546           # 既存絵本と同じ配信解像度
JPEG_QUALITY = 87
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

AUTHOR = "Nobuaki Kuwabara"
BASE_URL = "https://ki-84.github.io/story_leaf_content/"
LICENSE = "All Rights Reserved"
LICENSE_URL = BASE_URL + "LICENSE-original.txt"
MUSUBI = Path.home() / "devwork/musubi-tuner"
MUSUBI_PY = Path.home() / "venvs/musubi/bin/python"


def build_prompt(spec, entry):
    parts = [spec["style_prefix"], entry["scene"]]
    parts += [spec["characters"][name] for name in entry.get("cast", [])]
    parts.append(spec["style_suffix"])
    return " ".join(parts)


def draw_cover_text(img, title):
    """表紙にタイトル (上部) を焼き込む。フォントは画像幅に比例させる。"""
    draw = ImageDraw.Draw(img)
    w = img.width
    scale = w / 1024

    def fit_lines(text, size):
        font = ImageFont.truetype(FONT_PATH, int(size * scale))
        words, lines, cur = text.split(), [], ""
        for word in words:
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= w * 0.88:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        return font, lines

    font, lines = fit_lines(title, 96)
    if len(lines) > 2:
        font, lines = fit_lines(title, 76)
    y = int(48 * scale)
    for line in lines:
        x = (w - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, font=font, fill="#ffffff",
                  stroke_width=max(4, int(10 * scale)), stroke_fill="#5b3a1e")
        y += int(font.size * 1.18)
    return img


def page_entries(spec, only):
    entries = [("01", spec["cover"])] + [
        (f"{i + 2:02d}", page) for i, page in enumerate(spec["pages"])
    ]
    if only:
        entries = [(num, e) for num, e in entries if num in only]
    if not entries:
        raise SystemExit(f"--only に一致する画像がありません: {only}")
    return entries


def save_page_image(image, spec, num, img_dir):
    if num == "01":
        image = draw_cover_text(image, spec["title"])
    image = image.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    out = img_dir / f"{num}.jpg"
    image.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    print(f"  -> {out}", flush=True)


def load_flux():
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    return pipe


def cast_sheet_path(spec):
    return REPO / "tools" / "specs" / "cast" / f"{spec['bookId']}.png"


def make_cast_sheet(spec):
    """キャラシート (全キャラ横並び 1 枚) を FLUX.1-schnell で生成する。"""
    import torch

    sheet = spec["castSheet"]
    prompt = f'{sheet["scene"]} {spec["style_suffix"]}'
    pipe = load_flux()
    print(f"[cast sheet] seed={sheet['seed']} prompt={prompt[:100]}...", flush=True)
    image = pipe(
        prompt,
        width=1344,
        height=768,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(sheet["seed"]),
    ).images[0]
    out = cast_sheet_path(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"  -> {out}", flush=True)


def generate_images(spec, book_dir, only, seed_shift):
    import torch

    pipe = load_flux()
    img_dir = book_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for num, entry in page_entries(spec, only):
        seed = entry["seed"] + seed_shift
        prompt = build_prompt(spec, entry)
        print(f"[{num}.jpg] seed={seed} prompt={prompt[:100]}...", flush=True)
        image = pipe(
            prompt,
            width=GEN_SIZE,
            height=GEN_SIZE,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        save_page_image(image, spec, num, img_dir)


def cast_card_path(spec, name):
    return REPO / "tools" / "specs" / "cast" / f"{spec['bookId']}-{name}.png"


def anchor_path(spec):
    return REPO / "tools" / "specs" / "cast" / f"{spec['bookId']}-anchor.png"


def _hcat(imgs, pad=32):
    h = max(i.height for i in imgs)
    canvas = Image.new(
        "RGB", (sum(i.width for i in imgs) + pad * (len(imgs) + 1), h), "white"
    )
    x = pad
    for im in imgs:
        canvas.paste(im, (x, (h - im.height) // 2))
        x += im.width + pad
    return canvas


def sheet_crop(spec, name):
    sheet = Image.open(cast_sheet_path(spec)).convert("RGB")
    w, h = sheet.size
    lo, hi = spec["castSheet"]["slots"][name]
    return sheet.crop((int(lo * w), 0, int(hi * w), h))


def char_fragment(spec, name):
    return spec["characters"][name].replace(" is ", ", ", 1).rstrip(".")


def build_page_reference(spec, entry):
    """ページの登場キャラの参照画像リストと凡例を作る。

    キャラカード (全身+顔アップ、--make-cast-cards で生成) があればそれを
    1キャラ1枚で渡す。無ければキャラシートから切り出して1枚に合成する。
    シート全体を渡すと未登場キャラまで描いてしまうため必要分だけに絞る。
    """
    slots = spec["castSheet"].get("slots", {})
    cast = [c for c in entry.get("cast", []) if c in slots]
    if not cast:
        return (
            [Image.open(cast_sheet_path(spec)).convert("RGB")],
            spec["castSheet"]["legend"],
            False,
        )

    if all(cast_card_path(spec, c).is_file() for c in cast):
        cards = []
        for c in cast:
            img = Image.open(cast_card_path(spec, c)).convert("RGB")
            # 参照が大きいと latent トークンが増えて 32GB VRAM で OOM するため縮小
            img.thumbnail((640, 480), Image.LANCZOS)
            cards.append(img)
        anchor_file = anchor_path(spec)
        anchor = (
            Image.open(anchor_file).convert("RGB") if anchor_file.is_file() else None
        )

        # 参照は3枚まで。アンカーを入れる枠が無ければカードを1枚に合成する
        if anchor is not None and len(cards) + 1 > 3:
            merged = _hcat(cards)
            merged.thumbnail((960, 480), Image.LANCZOS)
            labels = ["left", "middle", "right"][: len(cast)]
            inner = "; ".join(
                f"{l}: {char_fragment(spec, c)}" for l, c in zip(labels, cast)
            )
            imgs = [merged, anchor]
            legend = (
                f"image 1 shows the characters side by side ({inner}); "
                "image 2: a finished page of this book, the style reference"
            )
            return imgs, legend, True

        imgs = cards
        legend = "; ".join(
            f"image {i + 1}: {char_fragment(spec, c)}" for i, c in enumerate(cast)
        )
        if anchor is not None:
            imgs = cards + [anchor]
            legend += (
                f"; image {len(imgs)}: a finished page of this book, "
                "the style reference"
            )
        return imgs, legend, anchor is not None

    canvas = _hcat([sheet_crop(spec, c) for c in cast])
    positions = {1: ["the character"], 2: ["left", "right"], 3: ["left", "middle", "right"]}
    labels = positions.get(len(cast), [f"#{i + 1}" for i in range(len(cast))])
    legend = "; ".join(
        f"{label}: {char_fragment(spec, c)}" for label, c in zip(labels, cast)
    )
    return [canvas], legend, False


def build_qwen_prompt(spec, entry, legend, n_cast, has_anchor=False):
    imgs = "images" if n_cast > 1 else "image"
    style_match = (
        " Match the art style, line weight, coloring and character rendering "
        "of the style reference image exactly, while keeping this scene's own "
        "lighting and time of day."
        if has_anchor
        else ""
    )
    count = (
        f" This scene contains exactly {n_cast} main character{'s' if n_cast > 1 else ''} "
        f"from the reference {imgs}, and no other reference characters."
        if n_cast
        else ""
    )
    sizes = spec["castSheet"].get("sizes", "")
    if sizes and n_cast > 1:
        sizes = f" {sizes}"
    else:
        sizes = ""
    return (
        f"{spec['style_prefix']} "
        f"Using the characters from the reference {imgs} ({legend}), "
        f"draw a completely new scene: {entry['scene']}{count}{sizes} "
        "Keep every character's design, colors, proportions, face and the "
        f"exact eye style precisely the same as in the reference {imgs}."
        f"{style_match} "
        "Do not copy the layout or the white background of the reference "
        f"{imgs}. {spec['style_suffix']}"
    )


def _qwen_env():
    import os

    # bf16 断片化対策。torch の CUDA 初期化前に設定する必要がある
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _quant_config(components):
    import torch
    from diffusers import PipelineQuantizationConfig

    # NF4 は RTX 5090 (Blackwell) でノイズ画像を出す既知の相性問題があるため int8 を使う
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=components,
    )


def apply_lightning(pipe, no_lora=False):
    import math

    from diffusers import FlowMatchEulerDiscreteScheduler

    lora = sorted(glob.glob(LIGHTNING_GLOB, recursive=True))
    if lora and not no_lora:
        pipe.load_lora_weights(lora[0])
        # Lightning 蒸留は shift=log(3) の exponential スケジュール前提。
        # 素の設定のまま 8 steps にするとノイズが残る。
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        )
        print(f"Lightning LoRA 使用 (8 steps): {Path(lora[0]).name}", flush=True)
        return 8, 1.0
    print("Lightning LoRA なし (40 steps, CFG 4.0)", flush=True)
    return 40, 4.0


def load_qwen_pipe(no_lora=False):
    _qwen_env()
    import torch
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        quantization_config=_quant_config(["transformer", "text_encoder"]),
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()  # 32GB VRAM ではデコードが溢れるため必須
    steps, cfg = apply_lightning(pipe, no_lora)
    return pipe, steps, cfg


def make_cast_cards(spec, no_lora=False):
    """キャラごとの参照カード (全身 + 顔アップ) を Qwen 自身に描かせて正典化する。

    シート切り出しは顔が小さく目の描き方が伝わらないため、出力と同じ画風の
    大きな顔つきカードを作って参照させる。
    """
    import torch

    if not cast_sheet_path(spec).is_file():
        raise SystemExit("キャラシートがありません。先に --make-cast-sheet を実行")
    pipe, steps, cfg = load_qwen_pipe(no_lora)
    base_seed = spec["castSheet"]["seed"]
    for i, name in enumerate(spec["castSheet"]["slots"]):
        prompt = (
            "Character reference card on a plain pure white background: redraw "
            "the character from the reference image twice — on the left the "
            "full body standing in a front view, on the right a large close-up "
            "of the head and face. Keep exactly the same design, colors, "
            "proportions, face and eye style as the reference image. "
            f"The character: {char_fragment(spec, name)}. "
            "Children's picture book cartoon style, flat colors, thick clean "
            "outlines. No text, no letters, no words."
        )
        seed = base_seed + 10 + i
        print(f"[card:{name}] seed={seed}", flush=True)
        image = pipe(
            image=[sheet_crop(spec, name)],
            prompt=prompt,
            negative_prompt=" ",
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            height=768,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        out = cast_card_path(spec, name)
        image.save(out)
        print(f"  -> {out}", flush=True)


def generate_images_qwen(spec, book_dir, only, seed_shift, no_lora=False):
    """2パス生成: transformer (20.5GB) と text encoder (8.3GB) は 32GB VRAM に
    同居できないため、先に TE だけで全ページの埋め込みを計算してから解放し、
    transformer だけで生成する。"""
    _qwen_env()
    import gc

    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    cast_file = cast_sheet_path(spec)
    if not cast_file.is_file():
        raise SystemExit(f"キャラシートがありません。先に --make-cast-sheet を実行: {cast_file}")

    jobs = []
    for num, entry in page_entries(spec, only):
        ref_imgs, legend, has_anchor = build_page_reference(spec, entry)
        n_cast = len(
            [c for c in entry.get("cast", []) if c in spec["castSheet"].get("slots", {})]
        )
        prompt = build_qwen_prompt(spec, entry, legend, n_cast, has_anchor)
        jobs.append((num, entry, ref_imgs, prompt))

    # ---- pass 1: text encoder のみロードして埋め込みを事前計算 ----
    print("pass 1/2: プロンプト埋め込みを計算中...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_MODEL,
        transformer=None,
        torch_dtype=torch.bfloat16,
        quantization_config=_quant_config(["text_encoder"]),
    )
    pipe.to("cuda")
    need_neg = no_lora  # CFG を使うのは LoRA なしの時だけ
    embeds = {}
    with torch.inference_mode():
        for num, entry, ref_imgs, prompt in jobs:
            cond = []
            for img in ref_imgs:
                cw, ch = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, img.size[0] / img.size[1]
                )
                cond.append(pipe.image_processor.resize(img, ch, cw))
            # batch=1 ではマスクが全1になり encode_prompt は None を返す (正常)
            cpu = lambda t: t.cpu() if t is not None else None
            pe, pm = pipe.encode_prompt(prompt=[prompt], image=cond, device="cuda")
            entry_embeds = [cpu(pe), cpu(pm)]
            if need_neg:
                ne, nm = pipe.encode_prompt(prompt=[" "], image=cond, device="cuda")
                entry_embeds += [cpu(ne), cpu(nm)]
            embeds[num] = entry_embeds
            print(f"  embed {num} (tokens={pe.shape[1]})", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ---- pass 2: transformer のみロードして生成 ----
    print("pass 2/2: 画像を生成中...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_MODEL,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
        quantization_config=_quant_config(["transformer"]),
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    steps, cfg = apply_lightning(pipe, no_lora)

    img_dir = book_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    gpu = lambda t: t.to("cuda") if t is not None else None
    for num, entry, ref_imgs, prompt in jobs:
        seed = entry["seed"] + seed_shift
        emb = embeds[num]
        kwargs = {}
        if need_neg:
            kwargs = {
                "negative_prompt_embeds": gpu(emb[2]),
                "negative_prompt_embeds_mask": gpu(emb[3]),
            }
        print(f"[{num}.jpg] seed={seed} refs={len(ref_imgs)}", flush=True)
        image = pipe(
            image=ref_imgs,
            prompt_embeds=gpu(emb[0]),
            prompt_embeds_mask=gpu(emb[1]),
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            height=QWEN_GEN_SIZE,
            width=QWEN_GEN_SIZE,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        save_page_image(image, spec, num, img_dir)
        torch.cuda.empty_cache()


def write_metadata(spec, book_dir):
    book_id = spec["bookId"]
    pages = [
        {"index": i, "imagePath": f"images/{i + 2:02d}.jpg", "text": p["text"]}
        for i, p in enumerate(spec["pages"])
    ]
    word_count = sum(len(p["text"].split()) for p in spec["pages"])
    book = {
        "bookId": book_id,
        "title": spec["title"],
        "level": spec["level"],
        "levelSource": "original",
        "genres": spec["genres"],
        "genreSource": "original",
        "wordCount": word_count,
        "pageCount": len(pages),
        "coverImagePath": "images/01.jpg",
        "source": "original",
        "attribution": {
            "title": spec["title"],
            "author": AUTHOR,
            "illustrator": f"AI generated (FLUX.1-schnell), directed by {AUTHOR}",
            "translator": None,
            "publisher": AUTHOR,
            "sourceURL": BASE_URL,
            "license": LICENSE,
            "licenseURL": LICENSE_URL,
            "modified": "生成AI (FLUX.1-schnell) による画像生成・オリジナルテキスト",
        },
        "warnings": [],
        "pages": pages,
    }
    book_path = book_dir / "book.json"
    book_path.write_text(
        json.dumps(book, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {book_path} (wordCount={word_count}, pageCount={len(pages)})")

    cover_src = book_dir / "images" / "01.jpg"
    cover_dst = REPO / "covers" / f"{book_id}.jpg"
    shutil.copyfile(cover_src, cover_dst)
    print(f"wrote {cover_dst}")

    catalog_path = REPO / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["books"] = [b for b in catalog["books"] if b["bookId"] != book_id]
    catalog["books"].append(
        {
            "bookId": book_id,
            "title": spec["title"],
            "level": spec["level"],
            "genres": spec["genres"],
            "wordCount": word_count,
            "pageCount": len(pages),
            "source": "original",
            "license": LICENSE,
            "cover": f"covers/{book_id}.jpg",
        }
    )
    catalog["books"].sort(key=lambda b: b["bookId"])
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"updated {catalog_path} ({len(catalog['books'])} books)")


def generate_images_lora(spec, book_dir, only, seed_shift):
    """キャスト LoRA + Qwen-Image T2I でページを生成する (musubi-tuner 推論)。

    レシピ: トリガー語 + 正典の色記述をプロンプトに常時併記、LoRA 強度 1.2。
    """
    import subprocess
    import tempfile

    cast = json.loads((REPO / spec["castSpec"]).read_text(encoding="utf-8"))
    chars = cast["characters"]
    entries = page_entries(spec, only)

    lines = []
    for num, entry in entries:
        who = ", and ".join(
            f"{chars[c]['trigger']}, {chars[c]['prompt_desc']}"
            for c in entry.get("cast", [])
        )
        prompt = f"{who}, {entry['scene']}, {spec['style_suffix']}"
        lines.append(f"{prompt} --d {entry['seed'] + seed_shift}")

    hub = Path.home() / ".cache/huggingface/hub"
    model = lambda pat: str(sorted(hub.glob(pat))[0])
    tmp = Path(tempfile.mkdtemp(prefix="lora_book_"))
    (tmp / "prompts.txt").write_text("\n".join(lines), encoding="utf-8")
    out_tmp = tmp / "out"
    out_tmp.mkdir()
    cmd = [
        str(MUSUBI_PY), "src/musubi_tuner/qwen_image_generate_image.py",
        "--dit", model("models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/diffusion_models/qwen_image_bf16.safetensors"),
        "--vae", model("models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/vae/qwen_image_vae.safetensors"),
        "--text_encoder", model("models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
        "--lora_weight", str(REPO / spec["lora"]),
        "--lora_multiplier", str(spec.get("loraMultiplier", 1.2)),
        "--negative_prompt", " ",
        "--fp8", "--fp8_scaled", "--text_encoder_cpu", "--vae_enable_tiling",
        "--attn_mode", "torch",
        "--image_size", str(QWEN_GEN_SIZE), str(QWEN_GEN_SIZE),
        "--infer_steps", "25",
        "--from_file", str(tmp / "prompts.txt"),
        "--save_path", str(out_tmp),
    ]
    print(f"musubi 推論: {len(lines)} 枚", flush=True)
    subprocess.run(cmd, cwd=MUSUBI, check=True)

    outs = sorted(out_tmp.glob("*.png"))
    if len(outs) != len(entries):
        raise SystemExit(f"生成枚数不一致: {len(outs)} != {len(entries)} (ログ確認)")
    img_dir = book_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for (num, entry), f in zip(entries, outs):
        save_page_image(Image.open(f).convert("RGB"), spec, num, img_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="tools/specs/{bookId}.json")
    parser.add_argument(
        "--only",
        help="再生成する画像番号のカンマ区切り (例: 01,05)。01=表紙",
    )
    parser.add_argument(
        "--seed-shift",
        type=int,
        default=0,
        help="全対象画像の seed に加算する値 (再ロール用)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="画像生成を飛ばしメタデータのみ再出力",
    )
    parser.add_argument(
        "--backend",
        choices=["flux", "qwen", "lora"],
        help="生成バックエンド (省略時は spec の backend、無ければ flux)",
    )
    parser.add_argument(
        "--make-cast-sheet",
        action="store_true",
        help="キャラシートのみ生成して終了 (qwen バックエンドの事前準備)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="qwen: Lightning LoRA を使わず 40 steps + CFG で生成",
    )
    parser.add_argument(
        "--make-cast-cards",
        action="store_true",
        help="キャラ別参照カード (全身+顔アップ) のみ生成して終了",
    )
    parser.add_argument(
        "--model",
        choices=sorted(QWEN_VERSIONS),
        default="2511",
        help="qwen バックエンドのモデル版 (既定 2511)",
    )
    args = parser.parse_args()
    set_qwen_version(args.model)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    book_dir = REPO / "books" / spec["bookId"]
    only = set(args.only.split(",")) if args.only else None

    if args.make_cast_sheet:
        make_cast_sheet(spec)
        return
    if args.make_cast_cards:
        make_cast_cards(spec, args.no_lora)
        return

    if not args.skip_generate:
        backend = args.backend or spec.get("backend", "flux")
        if backend == "qwen":
            generate_images_qwen(spec, book_dir, only, args.seed_shift, args.no_lora)
        elif backend == "lora":
            generate_images_lora(spec, book_dir, only, args.seed_shift)
        else:
            generate_images(spec, book_dir, only, args.seed_shift)
    write_metadata(spec, book_dir)


if __name__ == "__main__":
    main()
