import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"faceid-beauty-bakeoff-{uuid.uuid4()}"
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "faceid_beauty_bakeoff"

REFERENCE_IMAGE = "user_face_style_reference.jpg"

TESTS = [
    {
        "short": "RealVisXL_faceid_plusv2",
        "ckpt": "RealVisXL_V5.0_fp16.safetensors",
        "preset": "FACEID PLUS V2",
        "lora_strength": 0.55,
        "weight": 0.78,
        "weight_faceidv2": 1.15,
    },
    {
        "short": "RealVisXL_portrait_unnorm",
        "ckpt": "RealVisXL_V5.0_fp16.safetensors",
        "preset": "FACEID PORTRAIT UNNORM - SDXL only (strong)",
        "lora_strength": 0.0,
        "weight": 0.62,
        "weight_faceidv2": 1.0,
    },
    {
        "short": "JuggernautXI_faceid_plusv2",
        "ckpt": "Juggernaut-XI-byRunDiffusion.safetensors",
        "preset": "FACEID PLUS V2",
        "lora_strength": 0.55,
        "weight": 0.78,
        "weight_faceidv2": 1.15,
    },
    {
        "short": "JuggernautXI_portrait_unnorm",
        "ckpt": "Juggernaut-XI-byRunDiffusion.safetensors",
        "preset": "FACEID PORTRAIT UNNORM - SDXL only (strong)",
        "lora_strength": 0.0,
        "weight": 0.62,
        "weight_faceidv2": 1.0,
    },
]

PROMPT = (
    "photorealistic full body editorial fashion photo of a stunning East Asian adult woman, 22 years old, "
    "refined delicate face, symmetrical clear eyes, straight nose, soft natural lips, long straight glossy black hair, "
    "slim graceful hourglass figure, elegant posture, looking at camera, confident calm expression, "
    "modern luxury outfit, fitted black sleeveless midi dress, cropped ivory tailored jacket, black ankle boots, "
    "minimal jewelry, tasteful fully clothed styling, plain light gray studio background, high-end fashion photography, "
    "professional softbox lighting, 85mm lens, realistic skin texture, realistic eyes, sharp focus"
)

NEGATIVE = (
    "anime, cartoon, illustration, painting, CGI, 3d render, doll, plastic skin, child, underage, teenager, "
    "childlike face, old woman, mature woman, male, multiple people, duplicate person, twin, asymmetrical face, "
    "asymmetrical eyes, uneven eyes, lazy eye, crossed eyes, squint, distorted face, bad teeth, deformed eyes, "
    "wavy hair, curly hair, short hair, messy hair, frumpy outfit, old fashioned outfit, bulky clothes, lingerie, "
    "nude, nsfw, exposed cleavage, low quality, blurry, noisy, watermark, text, logo, bad anatomy, extra fingers, "
    "missing fingers, deformed hands, cropped head, cropped feet, oversized head"
)

PARAMS = {
    "steps": 26,
    "cfg": 5.5,
    "sampler_name": "dpmpp_2m",
    "scheduler": "karras",
    "width": 768,
    "height": 1152,
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


def build_prompt(test, seed):
    prefix = f"faceid_beauty_bakeoff/{test['short']}/{test['short']}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": test["ckpt"]}},
        "2": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": test["preset"],
                "lora_strength": test["lora_strength"],
                "provider": "CPU",
            },
        },
        "3": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}},
        "4": {
            "class_type": "IPAdapterFaceID",
            "inputs": {
                "model": ["2", 0],
                "ipadapter": ["2", 1],
                "image": ["3", 0],
                "weight": test["weight"],
                "weight_faceidv2": test["weight_faceidv2"],
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.72,
                "embeds_scaling": "V only",
            },
        },
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": PARAMS["width"], "height": PARAMS["height"], "batch_size": 1}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": PARAMS["steps"],
                "cfg": PARAMS["cfg"],
                "sampler_name": PARAMS["sampler_name"],
                "scheduler": PARAMS["scheduler"],
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def make_grids():
    items = []
    for test in TESTS:
        files = sorted((OUTPUT_DIR / test["short"]).glob(f"{test['short']}_*.png"))
        if files:
            items.append((test["short"], files[-1]))

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
        cols = 2
        label_h = 34
        pad = 14
        rows = (len(items) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad), (245, 245, 242))
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(items):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if mode == "face":
                img = img.crop((int(w * 0.18), int(h * 0.02), int(w * 0.82), int(h * 0.46)))
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

    print("GRID_FULL", grid("full", 390, 585, OUTPUT_DIR / "faceid_beauty_bakeoff_full_grid.jpg"), flush=True)
    print("GRID_FACE", grid("face", 390, 390, OUTPUT_DIR / "faceid_beauty_bakeoff_face_grid.jpg"), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, test in enumerate(TESTS):
        seed = 280533000 + index
        print(f"SUBMIT {test['short']} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(test, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("10", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    make_grids()
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
