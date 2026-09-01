import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"realistic-reference-{uuid.uuid4()}"

PROMPT = (
    "photorealistic full body character reference photo of a beautiful East Asian adult woman, "
    "around 28 years old, solo, standing, centered, short black bob haircut, natural makeup, "
    "warm brown eyes, calm confident expression, vivid crimson red leather field jacket, teal blue shirt, "
    "thick cream white scarf around her neck, khaki cargo pants, brown leather satchel, "
    "brown practical boots, outfit clearly visible, plain light gray studio background, "
    "professional DSLR photography, realistic skin texture, natural softbox lighting, 50mm lens"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, "
    "duplicate person, twin, multiple people, child, teenager, old woman, male, low quality, blurry, "
    "noisy, watermark, text, logo, bad anatomy, distorted face, extra fingers, missing fingers, "
    "cropped head, cropped feet, green jacket, olive jacket, gray jacket, black jacket, nude, nsfw"
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


prompt = {
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
            "seed": 260529101,
            "steps": 20,
            "cfg": 6.0,
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
            "filename_prefix": "consistent_story/realistic_reference_candidates/east_asian_real_ref",
        },
    },
}

result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": prompt})
prompt_id = result["prompt_id"]
print(f"SUBMIT {prompt_id}", flush=True)
item = wait(prompt_id)
images = item.get("outputs", {}).get("7", {}).get("images", [])
print(json.dumps(images, ensure_ascii=False, indent=2), flush=True)
