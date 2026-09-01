import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"young-beauty-reference-{uuid.uuid4()}"

PROMPT = (
    "photorealistic full body studio fashion reference photo of a beautiful East Asian adult woman, "
    "20 years old, youthful fresh look, solo, standing centered, looking straight at camera, "
    "perfectly symmetrical face, balanced eyes, clear almond-shaped eyes, refined nose, soft natural lips, "
    "long straight silky black hair, smooth straight hair, neat center part, hair falling naturally over shoulders, "
    "slim graceful figure, elegant proportions, tasteful fully clothed outfit, fitted tan leather jacket, "
    "ivory blouse, teal blue silk scarf, high waisted khaki trousers, brown ankle boots, outfit clearly visible, "
    "plain light gray studio background, professional DSLR fashion photography, realistic skin texture, "
    "natural softbox lighting, 85mm lens, sharp focus, realistic hands, realistic eyes"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, mature woman, old woman, "
    "child, teenager, male, multiple people, duplicate person, twin, asymmetrical face, asymmetrical eyes, uneven eyes, "
    "lazy eye, crossed eyes, squint, deformed eyes, distorted face, bad teeth, wavy hair, curly hair, short hair, messy hair, "
    "low quality, blurry, noisy, watermark, text, logo, bad anatomy, extra fingers, missing fingers, deformed hands, "
    "cropped head, cropped feet, oversized head, nude, lingerie, nsfw"
)


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


def build_prompt(seed):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 1152, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 26,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "consistent_story/young_beauty_reference_candidates/east_asian_young_beauty_ref",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=260531000)
    args = parser.parse_args()

    outputs = []
    for index in range(args.count):
        seed = args.seed + index
        print(f"SUBMIT seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("7", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
