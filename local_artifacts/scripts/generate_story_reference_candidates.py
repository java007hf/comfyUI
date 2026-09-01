import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"story-reference-{uuid.uuid4()}"


PROMPT = (
    "single consistent main character reference portrait, adult woman explorer, "
    "short black bob haircut with a small silver hairpin, warm brown eyes, "
    "calm determined expression, crimson utility jacket, cream scarf, dark cargo pants, "
    "brown leather satchel strap, practical boots, three-quarter body view, "
    "standing pose, plain light gray studio background, clean graphic novel realism, "
    "high detail face, outfit clearly visible, professional character design sheet"
)

NEGATIVE = (
    "low quality, blurry, noisy, text, watermark, logo, duplicate person, extra people, "
    "bad anatomy, bad hands, extra fingers, missing fingers, distorted face, cropped head, "
    "full nudity, nsfw"
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
        "inputs": {"width": 1024, "height": 1024, "batch_size": 4},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 260525900,
            "steps": 24,
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
            "filename_prefix": "consistent_story/reference_candidates/character_ref",
        },
    },
}

result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": prompt})
prompt_id = result["prompt_id"]
print(f"SUBMIT {prompt_id}", flush=True)
item = wait(prompt_id)
images = item.get("outputs", {}).get("7", {}).get("images", [])
print(json.dumps(images, ensure_ascii=False, indent=2), flush=True)
