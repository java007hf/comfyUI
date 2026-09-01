import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"beauty-reference-{uuid.uuid4()}"

PROMPT = (
    "photorealistic full body editorial character reference photo of a stunning East Asian adult woman, "
    "around 26 years old, solo, standing, long glossy black hair flowing over shoulders, exquisite delicate face, "
    "symmetrical facial features, elegant eyes, refined nose, soft natural lips, clear realistic skin texture, "
    "graceful slim hourglass figure, tall model proportions, confident relaxed posture, tasteful fully clothed outfit, "
    "fitted tan leather jacket, elegant ivory blouse, teal silk scarf, high waisted khaki trousers, brown leather ankle boots, "
    "outfit clearly visible, plain light gray studio background, professional DSLR fashion photography, "
    "natural softbox lighting, 85mm lens, sharp focus, realistic proportions"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, duplicate person, twin, "
    "multiple people, child, teenager, old woman, male, low quality, blurry, noisy, watermark, text, logo, "
    "bad anatomy, distorted face, asymmetrical eyes, extra fingers, missing fingers, deformed hands, cropped head, "
    "cropped feet, oversized head, short hair, green jacket, bulky body, nude, lingerie, nsfw"
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
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": PROMPT, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 1152, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 24,
                "cfg": 6.2,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "consistent_story/beauty_reference_candidates/east_asian_beauty_ref",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260530100)
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
