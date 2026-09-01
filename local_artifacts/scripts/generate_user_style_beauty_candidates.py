import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"user-style-beauty-{uuid.uuid4()}"

FACE_REFERENCE = "user_face_style_reference.jpg"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
IP_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"

PROMPT = (
    "photorealistic full body fashion reference photo of a beautiful East Asian adult woman, 20 years old, "
    "inspired by the reference face, delicate oval face, large clear eyes, straight nose, soft natural lips, "
    "long straight glossy black hair, youthful fresh beauty, slim graceful figure, elegant proportions, "
    "solo, standing centered, looking at camera, modern clean stylish outfit, fitted white short sleeve top, "
    "high waisted light blue straight jeans, slim black cropped blazer, white sneakers, minimal jewelry, "
    "outfit clearly visible, contemporary urban fashion, plain light gray studio background, professional DSLR photography, "
    "natural softbox lighting, 85mm lens, realistic skin texture, realistic eyes, sharp focus"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, mature woman, old woman, "
    "child, teenager, male, multiple people, duplicate person, twin, asymmetrical face, asymmetrical eyes, uneven eyes, "
    "lazy eye, crossed eyes, squint, deformed eyes, distorted face, bad teeth, wavy hair, curly hair, short hair, messy hair, "
    "old fashioned outfit, scarf, leather field jacket, khaki trousers, boots, bulky clothes, low quality, blurry, noisy, "
    "watermark, text, logo, bad anatomy, extra fingers, missing fingers, deformed hands, cropped head, cropped feet, "
    "oversized head, nude, lingerie, nsfw"
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
                "weight": 0.58,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.7,
                "embeds_scaling": "V only",
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 1152, "batch_size": 1}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": 26,
                "cfg": 6.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": "consistent_story/user_style_beauty_candidates/user_style_beauty_ref",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260532000)
    args = parser.parse_args()

    outputs = []
    for index in range(args.count):
        seed = args.seed + index
        print(f"SUBMIT seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("11", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
