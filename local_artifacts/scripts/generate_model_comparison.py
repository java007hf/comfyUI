import json
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"model-compare-{uuid.uuid4()}"


MODELS = [
    {
        "name": "Juggernaut-XI-byRunDiffusion.safetensors",
        "kind": "normal",
    },
    {
        "name": "Juggernaut-XI-byRunDiffusion-Lightning.safetensors",
        "kind": "fast",
    },
    {
        "name": "RealVisXL_V5.0_fp16.safetensors",
        "kind": "normal",
    },
    {
        "name": "Illustrious-XL-v1.1.safetensors",
        "kind": "normal",
    },
    {
        "name": "DreamShaperXL_Turbo_v2_1.safetensors",
        "kind": "fast",
    },
]


CATEGORIES = [
    {
        "index": "01",
        "slug": "person",
        "width": 832,
        "height": 1216,
        "prompt": (
            "cinematic portrait of a fully clothed adult woman, modern studio, "
            "natural skin texture, expressive eyes, soft key light, 85mm lens, "
            "neutral background, detailed fabric, professional photography"
        ),
    },
    {
        "index": "02",
        "slug": "landscape",
        "width": 1216,
        "height": 832,
        "prompt": (
            "wide angle landscape photo of an alpine lake at sunrise, snow capped "
            "mountains, pine forest, clean reflections on water, gentle mist, "
            "golden hour light, high detail, natural colors"
        ),
    },
    {
        "index": "03",
        "slug": "scifi",
        "width": 1024,
        "height": 1024,
        "prompt": (
            "cinematic science fiction scene, futuristic city street at night, "
            "neon signs, reflective wet pavement, flying vehicles, towering "
            "architecture, dramatic lighting, highly detailed concept art"
        ),
    },
]


NEGATIVE = (
    "low quality, blurry, noisy, jpeg artifacts, watermark, text, logo, "
    "deformed, distorted face, bad anatomy, extra fingers, missing fingers, "
    "cropped, out of frame, nsfw, nude"
)


PARAMS = {
    "normal": {
        "steps": 22,
        "cfg": 6.0,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
    },
    "fast": {
        "steps": 6,
        "cfg": 1.8,
        "sampler_name": "euler",
        "scheduler": "sgm_uniform",
    },
}


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{SERVER}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path):
    with urlopen(f"{SERVER}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def model_stem(model_name):
    return model_name.removesuffix(".safetensors")


def build_prompt(model, category, seed):
    params = PARAMS[model["kind"]]
    prefix = f"model_compare/{model_stem(model['name'])}_{category['index']}_{category['slug']}"

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model["name"]},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": category["prompt"], "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": category["width"],
                "height": category["height"],
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": params["steps"],
                "cfg": params["cfg"],
                "sampler_name": params["sampler_name"],
                "scheduler": params["scheduler"],
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": prefix},
        },
    }


def wait_for_prompt(prompt_id):
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


def main():
    outputs = []
    base_seed = 260525001

    for model_index, model in enumerate(MODELS):
        for category_index, category in enumerate(CATEGORIES):
            seed = base_seed + model_index * 100 + category_index
            prompt = build_prompt(model, category, seed)
            print(
                f"SUBMIT {model['name']} {category['index']}_{category['slug']} seed={seed}",
                flush=True,
            )
            try:
                result = post_json(
                    "/prompt",
                    {"client_id": CLIENT_ID, "prompt": prompt},
                )
            except (HTTPError, URLError) as exc:
                raise RuntimeError(f"Failed to submit prompt: {exc}") from exc

            prompt_id = result["prompt_id"]
            item = wait_for_prompt(prompt_id)
            images = item.get("outputs", {}).get("7", {}).get("images", [])
            for image in images:
                outputs.append(image)
                print(
                    "DONE "
                    f"{model['name']} {category['index']}_{category['slug']} "
                    f"{image.get('subfolder', '')}/{image.get('filename', '')}",
                    flush=True,
                )

    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
