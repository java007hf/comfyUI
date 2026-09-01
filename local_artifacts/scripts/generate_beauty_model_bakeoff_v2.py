import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"beauty-model-bakeoff-v2-{uuid.uuid4()}"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "beauty_model_bakeoff_v2"

FACE_REFERENCE = "user_face_style_reference.jpg"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
IP_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"

MODELS = [
    {"short": "RealVisXL", "ckpt": "RealVisXL_V5.0_fp16.safetensors", "kind": "normal"},
    {"short": "JuggernautXI", "ckpt": "Juggernaut-XI-byRunDiffusion.safetensors", "kind": "normal"},
    {"short": "DreamShaperXL_Turbo", "ckpt": "DreamShaperXL_Turbo_v2_1.safetensors", "kind": "fast"},
]

STYLES = [
    {
        "short": "black_dress",
        "text": (
            "sleek fitted black square-neck midi dress, cropped ivory designer jacket, "
            "black ankle boots, elegant minimal jewelry, modern luxury fashion styling"
        ),
    },
    {
        "short": "white_denim",
        "text": (
            "fitted white ribbed short sleeve top, high-waisted straight light blue jeans, "
            "cropped cream jacket, clean white sneakers, fresh upscale city style"
        ),
    },
    {
        "short": "skirt_boots",
        "text": (
            "tailored white satin blouse with modest neckline, high-waisted black A-line mini skirt, "
            "short structured leather jacket, black knee boots, chic contemporary fashion"
        ),
    },
]

BASE_PROMPT = (
    "photorealistic full-length fashion reference photo of a beautiful East Asian adult woman, age 20, "
    "adult facial features, delicate refined oval face, symmetrical large clear eyes, straight nose, "
    "soft natural lips, long straight glossy black hair, neat natural center part, slim graceful figure, "
    "balanced model proportions, solo, standing centered, looking at camera, confident calm expression, "
    "tasteful fully clothed outfit, plain light gray studio background, professional DSLR photography, "
    "natural softbox lighting, 85mm lens, realistic skin texture, realistic eyes, sharp focus"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, child, underage, "
    "teenager, childlike face, mature woman, old woman, male, multiple people, duplicate person, twin, "
    "asymmetrical face, asymmetrical eyes, uneven eyes, lazy eye, crossed eyes, squint, deformed eyes, "
    "distorted face, bad teeth, wavy hair, curly hair, short hair, messy hair, old fashioned outfit, "
    "frumpy outfit, bulky clothes, deep neckline, exposed cleavage, lingerie, nude, nsfw, low quality, "
    "blurry, noisy, watermark, text, logo, bad anatomy, extra fingers, missing fingers, deformed hands, "
    "cropped head, cropped feet, oversized head"
)

PARAMS = {
    "normal": {"steps": 26, "cfg": 5.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "width": 768, "height": 1152},
    "fast": {"steps": 10, "cfg": 2.0, "sampler_name": "euler", "scheduler": "sgm_uniform", "width": 768, "height": 1152},
}


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


def build_sdxl(model, style, seed):
    p = PARAMS[model["kind"]]
    prompt = f"{BASE_PROMPT}, {style['text']}"
    prefix = f"beauty_model_bakeoff_v2/{model['short']}/{model['short']}_{style['short']}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model["ckpt"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": FACE_REFERENCE}},
        "3": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "4": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IP_FACE}},
        "5": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["4", 0],
                "image": ["2", 0],
                "clip_vision": ["3", 0],
                "weight": 0.38,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.62,
                "embeds_scaling": "V only",
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": p["steps"],
                "cfg": p["cfg"],
                "sampler_name": p["sampler_name"],
                "scheduler": p["scheduler"],
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def make_grids():
    items = []
    for model in MODELS:
        for style in STYLES:
            folder = OUTPUT_DIR / model["short"]
            files = sorted(folder.glob(f"{model['short']}_{style['short']}_*.png"))
            if files:
                items.append((f"{model['short']} / {style['short']}", files[-1]))

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except Exception:
        pass

    def draw_label(draw, x, y, w, text):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + max(0, (w - tw) // 2), y + 8), text, fill=(20, 20, 20), font=font)

    def grid(mode, tile_w, tile_h, output_path):
        cols = 3
        label_h = 34
        pad = 14
        rows = (len(items) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad), (245, 245, 242))
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(items):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if mode == "face":
                img = img.crop((int(w * 0.23), int(h * 0.02), int(w * 0.77), int(h * 0.43)))
                img = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))
            else:
                img = ImageOps.contain(img, (tile_w, tile_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), (230, 230, 226))
            tile.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
            c = i % cols
            r = i // cols
            x = pad + c * (tile_w + pad)
            y = pad + r * (tile_h + label_h + pad)
            canvas.paste(tile, (x, y))
            draw_label(draw, x, y + tile_h, tile_w, name)
        canvas.save(output_path, quality=94)
        return str(output_path)

    print("GRID_FULL", grid("full", 330, 495, OUTPUT_DIR / "beauty_model_bakeoff_v2_full_grid.jpg"), flush=True)
    print("GRID_FACE", grid("face", 330, 330, OUTPUT_DIR / "beauty_model_bakeoff_v2_face_grid.jpg"), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for model_index, model in enumerate(MODELS):
        for style_index, style in enumerate(STYLES):
            seed = 270533000 + model_index * 100 + style_index
            print(f"SUBMIT {model['short']} {style['short']} seed={seed}", flush=True)
            result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_sdxl(model, style, seed)})
            item = wait(result["prompt_id"])
            images = item.get("outputs", {}).get("11", {}).get("images", [])
            outputs.extend(images)
            print(json.dumps(images, ensure_ascii=False), flush=True)

    make_grids()
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
