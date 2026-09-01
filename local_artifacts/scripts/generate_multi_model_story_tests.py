import argparse
import json
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"multi-model-story-{uuid.uuid4()}"

REFERENCE_FULL = "story_character_reference.png"
REFERENCE_FACE = "story_character_face_reference.png"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
IP_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"

SD3_CLIP = {
    "clip_name1": "clip_l.safetensors",
    "clip_name2": "clip_g.safetensors",
    "clip_name3": "t5xxl_fp8_e4m3fn_scaled.safetensors",
}

MODELS = [
    {"short": "JuggernautXI", "ckpt": "Juggernaut-XI-byRunDiffusion.safetensors", "type": "sdxl", "kind": "normal"},
    {"short": "JuggernautXI_Lightning", "ckpt": "Juggernaut-XI-byRunDiffusion-Lightning.safetensors", "type": "sdxl", "kind": "fast"},
    {"short": "DreamShaperXL_Turbo", "ckpt": "DreamShaperXL_Turbo_v2_1.safetensors", "type": "sdxl", "kind": "fast"},
    {"short": "IllustriousXL", "ckpt": "Illustrious-XL-v1.1.safetensors", "type": "sdxl", "kind": "normal"},
    {"short": "SD35M_Turbo_prompt_only", "ckpt": "sd3.5m_turbo.safetensors", "type": "sd3", "kind": "sd3_turbo"},
    {"short": "SD35Large_Turbo_prompt_only", "ckpt": "sd3.5_large_turbo.safetensors", "type": "sd3", "kind": "sd3_turbo"},
]

BACKGROUND_MODELS = [
    {"short": "RealVisXL", "ckpt": "RealVisXL_V5.0_fp16.safetensors", "type": "sdxl", "kind": "normal"},
    *MODELS,
]

STORY_SCENES = [
    ("01_wakeup", "morning light in a small mountain inn room, she studies an old brass compass on a wooden table"),
    ("03_market", "busy old town market at sunrise, she walks through stalls carrying her satchel, vendors in background"),
    ("07_bridge", "crossing a narrow rope bridge above a deep green valley, wind moving her cream scarf"),
    ("10_ruins", "discovering moss covered stone ruins at dawn, carved pillars half buried in vines"),
    ("25_airship", "boarding a small brass airship at sunrise, propellers turning, clouds below the dock"),
    ("28_guardian", "facing a huge ancient stone guardian, she stands calmly holding the glowing artifact"),
]

BACKGROUND_SCENES = [
    (
        "01_mountain_vehicle",
        "wide cinematic panel, same solo woman explorer standing beside a rugged expedition jeep on a mountain road, "
        "snow capped peaks, river valley, pine forest, dramatic clouds, full environment visible",
    ),
    (
        "02_street_vehicle",
        "wide cinematic panel, same solo woman explorer walking on an old city street, tram and motorcycles nearby, "
        "storefronts, pedestrians in background, distant mountains visible between buildings",
    ),
    (
        "03_airship_dock",
        "wide cinematic panel, same solo woman explorer at a cliffside airship dock, brass airship and small aircraft, "
        "stone street, market stalls, waterfalls and mountains behind, rich background detail",
    ),
]

CHARACTER = (
    "solo, one woman only, same beautiful East Asian adult woman explorer, long glossy black hair, "
    "exquisite delicate face, elegant eyes, slim graceful figure, warm brown eyes, calm confident expression, "
    "tan brown leather field jacket, ivory blouse, bright teal blue scarf around her neck, "
    "khaki trousers, brown leather satchel backpack, brown boots"
)

STYLE = (
    "cinematic graphic novel story panel, realistic comic illustration, varied medium shot or wide shot, "
    "rich detailed environment, visible background, dynamic storyboard composition, subtle inked edges, detailed lighting"
)

NEGATIVE = (
    "different person, short hair, changed hairstyle, missing scarf, wrong scarf color, duplicate main character, twin, clone, extra people as main character, "
    "low quality, blurry, noisy, text, caption, speech bubble, watermark, logo, bad anatomy, distorted face, extra fingers, "
    "missing fingers, cropped head, plain gray background, studio background, character sheet, portrait only, empty background, nude, nsfw"
)

PARAMS = {
    "normal": {"steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "width": 1024, "height": 1024},
    "fast": {"steps": 8, "cfg": 2.0, "sampler_name": "euler", "scheduler": "sgm_uniform", "width": 1024, "height": 1024},
    "sd3_turbo": {"steps": 4, "cfg": 1.2, "sampler_name": "euler", "scheduler": "sgm_uniform", "width": 768, "height": 768},
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


def build_sdxl_prompt(model, prompt_text, prefix):
    params = PARAMS[model["kind"]]
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model["ckpt"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_FULL}},
        "3": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_FACE}},
        "4": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "5": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IP_FACE}},
        "6": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["5", 0],
                "image": ["3", 0],
                "clip_vision": ["4", 0],
                "weight": 0.68,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.75,
                "embeds_scaling": "V only",
            },
        },
        "7": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IP_PLUS}},
        "8": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["6", 0],
                "ipadapter": ["7", 0],
                "image": ["2", 0],
                "clip_vision": ["4", 0],
                "weight": 0.12,
                "weight_type": "style transfer",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.35,
                "embeds_scaling": "V only",
            },
        },
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": params["width"], "height": params["height"], "batch_size": 1},
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["11", 0],
                "seed": 0,
                "steps": params["steps"],
                "cfg": params["cfg"],
                "sampler_name": params["sampler_name"],
                "scheduler": params["scheduler"],
                "denoise": 1.0,
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }


def build_sd3_prompt(model, prompt_text, prefix):
    params = PARAMS[model["kind"]]
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model["ckpt"]}},
        "2": {"class_type": "TripleCLIPLoader", "inputs": SD3_CLIP},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": params["width"], "height": params["height"], "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": 0,
                "steps": params["steps"],
                "cfg": params["cfg"],
                "sampler_name": params["sampler_name"],
                "scheduler": params["scheduler"],
                "denoise": 1.0,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def build_prompt(model, prompt_text, prefix):
    if model["type"] == "sd3":
        return build_sd3_prompt(model, prompt_text, prefix)
    return build_sdxl_prompt(model, prompt_text, prefix)


def set_seed(prompt, seed):
    for node in prompt.values():
        if node.get("class_type") == "KSampler":
            node["inputs"]["seed"] = seed


def output_node_id(model):
    return "8" if model["type"] == "sd3" else "14"


def run_job(model, slug, prompt_text, prefix, seed):
    prompt = build_prompt(model, prompt_text, prefix)
    set_seed(prompt, seed)
    print(f"SUBMIT {model['short']} {slug} seed={seed}", flush=True)
    try:
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": prompt})
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"submit failed: {model['short']} {slug}: {exc}") from exc
    item = wait_for_prompt(result["prompt_id"])
    images = item.get("outputs", {}).get(output_node_id(model), {}).get("images", [])
    for image in images:
        print(f"DONE {model['short']} {slug} {image['subfolder']}/{image['filename']}", flush=True)
    return images


def selected_models(models, wanted):
    if not wanted:
        return models
    names = {name.strip() for name in wanted.split(",") if name.strip()}
    return [model for model in models if model["short"] in names]


def run_consistency(wanted=None, scene_limit=None):
    outputs = []
    for model_index, model in enumerate(selected_models(MODELS, wanted)):
        for scene_index, (slug, scene) in enumerate(STORY_SCENES[:scene_limit]):
            prompt_text = f"{CHARACTER}, {scene}, {STYLE}"
            prefix = f"multi_model_story_test/{model['short']}/{model['short']}_{slug}"
            seed = 260527000 + model_index * 100 + scene_index
            outputs.extend(run_job(model, slug, prompt_text, prefix, seed))
    return outputs


def run_background(wanted=None, scene_limit=None):
    outputs = []
    for model_index, model in enumerate(selected_models(BACKGROUND_MODELS, wanted)):
        for scene_index, (slug, scene) in enumerate(BACKGROUND_SCENES[:scene_limit]):
            prompt_text = f"{CHARACTER}, {scene}, {STYLE}, sharp foreground character and clear background"
            prefix = f"person_background_test/{model['short']}/{model['short']}_{slug}"
            seed = 260528000 + model_index * 100 + scene_index
            outputs.extend(run_job(model, slug, prompt_text, prefix, seed))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["consistency", "background", "all"], default="all")
    parser.add_argument("--models", default=None, help="Comma-separated model short names.")
    parser.add_argument("--scene-limit", type=int, default=None)
    args = parser.parse_args()

    all_outputs = []
    if args.mode in ("consistency", "all"):
        all_outputs.extend(run_consistency(args.models, args.scene_limit))
    if args.mode in ("background", "all"):
        all_outputs.extend(run_background(args.models, args.scene_limit))
    print("ALL_DONE", flush=True)
    print(json.dumps(all_outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
