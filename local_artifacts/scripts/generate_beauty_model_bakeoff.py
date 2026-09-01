import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"beauty-model-bakeoff-{uuid.uuid4()}"

FACE_REFERENCE = "user_face_style_reference.jpg"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
IP_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"

MODELS = [
    {"short": "RealVisXL", "ckpt": "RealVisXL_V5.0_fp16.safetensors", "type": "sdxl", "kind": "normal"},
    {"short": "JuggernautXI", "ckpt": "Juggernaut-XI-byRunDiffusion.safetensors", "type": "sdxl", "kind": "normal"},
    {"short": "JuggernautXI_Lightning", "ckpt": "Juggernaut-XI-byRunDiffusion-Lightning.safetensors", "type": "sdxl", "kind": "fast"},
    {"short": "DreamShaperXL_Turbo", "ckpt": "DreamShaperXL_Turbo_v2_1.safetensors", "type": "sdxl", "kind": "fast"},
    {"short": "IllustriousXL", "ckpt": "Illustrious-XL-v1.1.safetensors", "type": "sdxl", "kind": "normal"},
    {"short": "SD35M_Turbo_prompt_only", "ckpt": "sd3.5m_turbo.safetensors", "type": "sd3", "kind": "sd3_turbo"},
]

PROMPT = (
    "photorealistic full body fashion reference photo of a stunning East Asian adult woman, 20 years old, "
    "youthful fresh beauty, delicate small oval face, large clear balanced eyes, straight nose, soft natural lips, "
    "long straight glossy black hair, neat natural center part, slim graceful figure, elegant model proportions, "
    "solo, standing centered, looking straight at camera, modern stylish outfit, fitted white crop length top, "
    "high waisted light blue straight jeans, short black tailored blazer, clean white sneakers, minimal jewelry, "
    "tasteful fully clothed outfit, contemporary urban fashion, plain light gray studio background, "
    "professional DSLR photography, natural softbox lighting, 85mm lens, realistic skin texture, realistic eyes, sharp focus"
)

NEGATIVE = (
    "anime, cartoon, illustration, comic, painting, 3d render, CGI, doll, plastic skin, mature woman, old woman, "
    "child, teenager, male, multiple people, duplicate person, twin, asymmetrical face, asymmetrical eyes, uneven eyes, "
    "lazy eye, crossed eyes, squint, deformed eyes, distorted face, bad teeth, wavy hair, curly hair, short hair, messy hair, "
    "old fashioned outfit, scarf, leather field jacket, khaki trousers, boots, bulky clothes, low quality, blurry, noisy, "
    "watermark, text, logo, bad anatomy, extra fingers, missing fingers, deformed hands, cropped head, cropped feet, "
    "oversized head, nude, lingerie, nsfw"
)

PARAMS = {
    "normal": {"steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "width": 768, "height": 1152},
    "fast": {"steps": 8, "cfg": 2.0, "sampler_name": "euler", "scheduler": "sgm_uniform", "width": 768, "height": 1152},
    "sd3_turbo": {"steps": 4, "cfg": 1.2, "sampler_name": "euler", "scheduler": "sgm_uniform", "width": 768, "height": 1024},
}

SD3_CLIP = {
    "clip_name1": "clip_l.safetensors",
    "clip_name2": "clip_g.safetensors",
    "clip_name3": "t5xxl_fp8_e4m3fn_scaled.safetensors",
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


def build_sdxl(model, seed):
    p = PARAMS[model["kind"]]
    prefix = f"beauty_model_bakeoff/{model['short']}/{model['short']}_beauty_ref"
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
                "weight": 0.42,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.65,
                "embeds_scaling": "V only",
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
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


def build_sd3(model, seed):
    p = PARAMS[model["kind"]]
    prefix = f"beauty_model_bakeoff/{model['short']}/{model['short']}_beauty_ref"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model["ckpt"]}},
        "2": {"class_type": "TripleCLIPLoader", "inputs": SD3_CLIP},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": p["steps"],
                "cfg": p["cfg"],
                "sampler_name": p["sampler_name"],
                "scheduler": p["scheduler"],
                "denoise": 1.0,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def build_prompt(model, seed):
    if model["type"] == "sd3":
        return build_sd3(model, seed)
    return build_sdxl(model, seed)


def output_node(model):
    return "8" if model["type"] == "sd3" else "11"


def main():
    outputs = []
    for index, model in enumerate(MODELS):
        seed = 260533000 + index
        print(f"SUBMIT {model['short']} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(model, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get(output_node(model), {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
