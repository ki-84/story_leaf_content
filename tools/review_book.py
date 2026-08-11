#!/usr/bin/env python3
"""絵本の画像を VLM (Qwen2.5-VL-7B-Instruct) で構造監査する (生成後ゲート)。

  python3 tools/review_book.py tools/specs/pb-en-0703.json            # 監査のみ
  python3 tools/review_book.py tools/specs/pb-en-0703.json --json out.json

ページごとに VLM へ画像と質問を渡し、スペックと突き合わせる:
- 動物キャラの頭数と種 (増殖・消失・混線種の検出)
- 人間の有無 (混入検出)
- 屋内/屋外 (location と照合)
- ページ指定の必須小道具 (spec pages[].must_show があれば)
判定は JSON で出力。FAIL があれば終了コード 1。
美的判定 (画風・微妙な特徴混線) は対象外 — 作業エージェントの目視と人間の最終確認で行う。
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

SPECIES_WORDS = {
    "hedgehog": "hedgehog",
    "porcupine": "hedgehog",  # VLM はハリネズミをヤマアラシと観測しがち
    "bunny": "rabbit",
    "rabbit": "rabbit",
    "duck": "duck",
    "duckling": "duck",
    "chick": "duck",  # VLM は Coco をヒヨコと観測しがち
    "chicken": "duck",
    "bear": "bear",
    "owl": "owl",
    "fox": "fox",
    "mouse": "mouse",
    "turtle": "turtle",
    "otter": "otter",
    "bird": "bird",
}

AUDIT_PROMPT = """Look at this children's book illustration and answer in strict JSON only:
{"animal_count": <number of animal characters>, "species": [<species name of each animal character, e.g. "rabbit","bear">], "humans": <number of human figures>, "setting": "indoor" or "outdoor", "objects": [<up to 8 prominent objects>]}
Count each animal character once. Do not explain."""


def load_vlm():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor


def ask(model, processor, image_path, prompt=AUDIT_PROMPT):
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user",
         "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    reply = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def expected_species(spec, cast_chars, entry):
    out = []
    for c in entry.get("cast", []):
        sp = cast_chars.get(c, {}).get("species", c)
        for w, canon in SPECIES_WORDS.items():
            if w in sp.lower():
                out.append(canon)
                break
        else:
            out.append(sp.lower())
    return sorted(out)


def norm_species(lst):
    out = []
    for s in lst or []:
        s = str(s).lower()
        for w, canon in SPECIES_WORDS.items():
            if w in s:
                out.append(canon)
                break
        else:
            out.append(s)
    return sorted(out)


def audit_book(spec_path, json_out=None):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    book_dir = REPO / "books" / spec["bookId"]
    cast_chars = {}
    if spec.get("castSpec"):
        cast_chars = json.loads(
            (REPO / spec["castSpec"]).read_text(encoding="utf-8")
        )["characters"]
    outdoor_locs = {
        k for k, v in spec.get("locations", {}).items()
        if re.search(r"meadow|path|sea|sky|garden|forest|outside|hill|pond|harbor|beach|in front of", v, re.I)
    }
    plain_locs = {
        k for k, v in spec.get("locations", {}).items()
        if re.search(r"plain|background", v, re.I)
    }

    model, processor = load_vlm()
    entries = [("01", spec["cover"])] + [
        (f"{i + 2:02d}", p) for i, p in enumerate(spec["pages"])
    ]
    findings = []
    for num, entry in entries:
        img = book_dir / "images" / f"{num}.jpg"
        obs = ask(model, processor, img)
        page_findings = []
        if obs is None:
            page_findings.append({"level": "WARN", "issue": "VLM 応答を解析できず"})
        else:
            exp_sp = expected_species(spec, cast_chars, entry)
            got_sp = norm_species(obs.get("species"))
            if len(got_sp) != len(exp_sp):
                page_findings.append({
                    "level": "FAIL",
                    "issue": f"キャラ数不一致: 期待 {len(exp_sp)} ({exp_sp}) / 観測 {len(got_sp)} ({got_sp})",
                })
            elif got_sp != exp_sp:
                page_findings.append({
                    "level": "FAIL",
                    "issue": f"種の不一致: 期待 {exp_sp} / 観測 {got_sp}",
                })
            if obs.get("humans", 0):
                page_findings.append({"level": "FAIL", "issue": f"人間が {obs['humans']} 人写っている"})
            loc = entry.get("location")
            if loc and loc not in plain_locs:
                expected_setting = "outdoor" if loc in outdoor_locs else "indoor"
                if obs.get("setting") and obs["setting"] != expected_setting:
                    page_findings.append({
                        "level": "WARN",
                        "issue": f"屋内外の不一致: location '{loc}' は {expected_setting} 想定 / 観測 {obs['setting']}",
                    })
            for prop in entry.get("must_show", []):
                objs = " ".join(str(o).lower() for o in obs.get("objects", []))
                if not all(w in objs for w in prop.lower().split()[-1:]):
                    page_findings.append({
                        "level": "WARN",
                        "issue": f"必須小道具 '{prop}' が objects に見当たらない (観測: {obs.get('objects')})",
                    })
        for f in page_findings:
            f["page"] = num
            findings.append(f)
        status = "NG" if any(f["level"] == "FAIL" for f in page_findings) else "ok"
        print(f"  [{num}] {status} {'; '.join(f['issue'] for f in page_findings) if page_findings else ''}",
              flush=True)

    fails = [f for f in findings if f["level"] == "FAIL"]
    result = {"bookId": spec["bookId"], "findings": findings,
              "fail_pages": sorted({f["page"] for f in fails})}
    if json_out:
        Path(json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"監査結果: FAIL {len(fails)} 件 / WARN {len(findings) - len(fails)} 件")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--json", help="結果 JSON の出力先")
    args = ap.parse_args()
    result = audit_book(args.spec, args.json)
    return 1 if result["fail_pages"] else 0


if __name__ == "__main__":
    sys.exit(main())
