import json
import math
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"flux-krea-beauty-test-{uuid.uuid4()}"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "flux_krea_beauty_test"

MODEL = "flux1-krea-dev_fp8_scaled.safetensors"
CLIP_L = "clip_l.safetensors"
T5XXL = "t5xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "ae.safetensors"

WIDTH = 832
HEIGHT = 1216

BASE_PROMPT = (
    "photorealistic full body editorial portrait of a beautiful East Asian adult woman, 22 years old, "
    "refined symmetrical face, bright clear eyes, long straight glossy black hair, slim graceful figure, "
    "modern tasteful fashion, elegant fitted outfit, natural makeup, realistic skin texture, "
    "high-end magazine photography, clean composition, soft natural light, 85mm lens, sharp focus"
)

NEGATIVE_HINT = (
    "cartoon, anime, illustration, plastic skin, distorted eyes, crossed eyes, asymmetrical eyes, "
    "bad hands, extra fingers, deformed face, old-fashioned clothing, messy hair, low quality, blurry"
)

CASES = [
    (
        "001_modern_white",
        "white silk blouse, tailored light gray trousers, minimalist studio with warm window light",
        92041001,
    ),
    (
        "002_black_dress",
        "sleek black midi dress, refined urban evening street, subtle neon and wet pavement reflections",
        92041002,
    ),
    (
        "003_denim_jacket",
        "cropped ivory designer jacket, fitted white top, high waisted blue jeans, clean modern cafe background",
        92041003,
    ),
    (
        "004_luxury_casual",
        "luxury casual outfit, beige trench coat over fitted black top, clean city boulevard with glass storefronts",
        92041004,
    ),
]


def post_json(path, payload):
    req = Request(
        f"{SERVER}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path):
    with urlopen(f"{SERVER}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait(prompt_id):
    while True:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(3)


def build_prompt(name, scene, seed):
    prompt = f"{BASE_PROMPT}, {scene}. Avoid: {NEGATIVE_HINT}."
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["1", 0], "max_shift": 1.15, "base_shift": 0.5, "width": WIDTH, "height": HEIGHT},
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": CLIP_L, "clip_name2": T5XXL, "type": "flux"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["3", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 3.5},
        },
        "6": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["3", 0], "clip_l": "", "t5xxl": "", "guidance": 3.5},
        },
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": 28,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"flux_krea_beauty_test/{name}"}},
    }


def make_grid():
    files = sorted(OUTPUT_DIR.glob("*.png"))
    if not files:
        return
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        pass

    tile_w, tile_h = 260, 380
    label_h = 34
    pad = 12
    cols = 4
    rows = math.ceil(len(files) / cols)
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad), (246, 246, 244))
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, (tile_w, tile_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), (230, 230, 228))
        tile.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
        x = pad + (i % cols) * (tile_w + pad)
        y = pad + (i // cols) * (tile_h + label_h + pad)
        canvas.paste(tile, (x, y))
        label = path.stem.split("_0000", 1)[0]
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + max(0, (tile_w - (bbox[2] - bbox[0])) // 2), y + tile_h + 8), label, fill=(25, 25, 25), font=font)
    out = OUTPUT_DIR / "flux_krea_beauty_grid.jpg"
    canvas.save(out, quality=94)
    print("GRID", out, flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, scene, seed in CASES:
        print(f"SUBMIT {name} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(name, scene, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("10", {}).get("images", [])
        print(json.dumps(images, ensure_ascii=False), flush=True)
    make_grid()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
