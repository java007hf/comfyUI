import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"reference-img2img-faceid-{uuid.uuid4()}"
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "reference_img2img_faceid"

REFERENCE_IMAGE = "user_face_style_reference.jpg"
MODEL = "RealVisXL_V5.0_fp16.safetensors"

TESTS = [
    {
        "short": "low_repaint_original_style",
        "denoise": 0.32,
        "prompt": (
            "photorealistic polished full body fashion photo of the same East Asian adult woman from the reference, "
            "22 years old, beautiful refined face, symmetrical clear eyes, long straight glossy black hair, "
            "slim graceful figure, modern clean white fitted t-shirt, high waisted blue jeans, holding a book, "
            "fresh upscale casual fashion, natural pose, realistic skin texture, high-end DSLR photo, sharp focus"
        ),
    },
    {
        "short": "medium_repaint_original_style",
        "denoise": 0.42,
        "prompt": (
            "photorealistic polished full body fashion photo of the same East Asian adult woman from the reference, "
            "22 years old, beautiful refined face, symmetrical clear eyes, long straight glossy black hair, "
            "slim graceful figure, modern clean white fitted t-shirt, high waisted blue jeans, holding a book, "
            "fresh upscale casual fashion, natural pose, realistic skin texture, high-end DSLR photo, sharp focus"
        ),
    },
    {
        "short": "medium_repaint_luxury_casual",
        "denoise": 0.44,
        "prompt": (
            "photorealistic polished full body fashion photo of the same East Asian adult woman from the reference, "
            "22 years old, beautiful refined oval face, symmetrical clear eyes, long straight glossy black hair, "
            "slim graceful figure, modern luxury casual outfit, fitted white short sleeve top, high waisted blue jeans, "
            "cropped ivory designer jacket, clean elegant styling, natural pose, realistic skin texture, high-end DSLR photo, sharp focus"
        ),
    },
    {
        "short": "higher_repaint_luxury_casual",
        "denoise": 0.55,
        "prompt": (
            "photorealistic full body editorial fashion photo of the same East Asian adult woman from the reference, "
            "22 years old, beautiful refined oval face, symmetrical clear eyes, long straight glossy black hair, "
            "slim graceful figure, modern luxury casual outfit, fitted white short sleeve top, high waisted blue jeans, "
            "cropped ivory designer jacket, clean elegant styling, natural pose, realistic skin texture, high-end DSLR photo, sharp focus"
        ),
    },
]

NEGATIVE = (
    "anime, cartoon, illustration, painting, CGI, 3d render, doll, plastic skin, child, underage, teenager, "
    "childlike face, old woman, mature woman, male, multiple people, duplicate person, twin, asymmetrical face, "
    "asymmetrical eyes, uneven eyes, lazy eye, crossed eyes, squint, distorted face, bad teeth, deformed eyes, "
    "curly hair, short hair, messy hair, frumpy outfit, old fashioned outfit, bulky clothes, lingerie, nude, nsfw, "
    "exposed cleavage, low quality, blurry, noisy, watermark, text, logo, bad anatomy, extra fingers, missing fingers, "
    "deformed hands, cropped head, cropped feet, oversized head"
)

PARAMS = {
    "steps": 24,
    "cfg": 4.8,
    "sampler_name": "dpmpp_2m",
    "scheduler": "karras",
    "width": 768,
    "height": 1360,
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
    prefix = f"reference_img2img_faceid/{test['short']}/{test['short']}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL}},
        "2": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": 0.52,
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
                "weight": 0.82,
                "weight_faceidv2": 1.18,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.68,
                "embeds_scaling": "V only",
            },
        },
        "5": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": PARAMS["width"],
                "height": PARAMS["height"],
                "crop": "disabled",
            },
        },
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": test["prompt"], "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": PARAMS["steps"],
                "cfg": PARAMS["cfg"],
                "sampler_name": PARAMS["sampler_name"],
                "scheduler": PARAMS["scheduler"],
                "denoise": test["denoise"],
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def make_grids():
    items = []
    for test in TESTS:
        files = sorted((OUTPUT_DIR / test["short"]).glob(f"{test['short']}_*.png"))
        if files:
            items.append((test["short"], files[-1]))

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 16)
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
                img = img.crop((int(w * 0.18), int(h * 0.06), int(w * 0.72), int(h * 0.42)))
                img = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))
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

    print("GRID_FULL", grid("full", 360, 640, OUTPUT_DIR / "reference_img2img_faceid_full_grid.jpg"), flush=True)
    print("GRID_FACE", grid("face", 360, 360, OUTPUT_DIR / "reference_img2img_faceid_face_grid.jpg"), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, test in enumerate(TESTS):
        seed = 290533000 + index
        print(f"SUBMIT {test['short']} denoise={test['denoise']} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(test, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("11", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    make_grids()
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
