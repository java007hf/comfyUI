import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"consistent-comic-{uuid.uuid4()}"

MODEL = "RealVisXL_V5.0_fp16.safetensors"
FULL_REFERENCE_IMAGE = "story_character_reference.png"
FACE_REFERENCE_IMAGE = "story_character_face_reference.png"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
IP_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"

BASE_CHARACTER = (
    "solo, one woman only, same adult woman explorer, short black bob haircut with a small silver hairpin, "
    "warm brown eyes, calm confident expression, tan brown leather field jacket, "
    "gray shirt, bright teal blue scarf around her neck, khaki cargo pants, "
    "brown leather satchel backpack, practical brown boots, same outfit in every panel"
)

STYLE = (
    "cinematic graphic novel story panel, realistic comic illustration, varied medium shot or wide shot, "
    "rich detailed environment, visible background, dynamic storyboard composition, subtle inked edges, "
    "detailed lighting, no speech bubbles, no captions, no text"
)

NEGATIVE = (
    "different person, changed hairstyle, changed outfit, missing scarf, wrong scarf color, "
    "extra people as main character, duplicate character, low quality, blurry, noisy, text, "
    "caption, speech bubble, watermark, logo, bad anatomy, distorted face, extra fingers, "
    "missing fingers, cropped head, plain gray background, studio background, character sheet, "
    "portrait only, empty background, companion, twin, clone, nude, nsfw"
)

SCENES = [
    ("wakeup", "morning light in a small mountain inn room, she studies an old brass compass on a wooden table"),
    ("map", "close view at a cluttered desk, she opens a weathered map with red route marks and ancient symbols"),
    ("market", "busy old town market at sunrise, she walks through stalls carrying her satchel, vendors in background"),
    ("train", "standing on an old train platform, steam drifting around her boots, distant mountains behind"),
    ("window", "inside a moving train, she looks out the window at forests and cliffs passing by"),
    ("station", "arriving at a remote mountain station, fog rolls over the tracks, she steps onto the platform"),
    ("bridge", "crossing a narrow rope bridge above a deep green valley, wind moving her cream scarf"),
    ("forest", "dense pine forest trail, shafts of sunlight through trees, she follows compass directions"),
    ("campfire", "night camp beside a small fire, she checks the map while blue shadows fill the forest"),
    ("ruins", "discovering moss covered stone ruins at dawn, carved pillars half buried in vines"),
    ("gate", "standing before a sealed ancient gate with geometric patterns glowing faintly"),
    ("puzzle", "carefully turning stone rings on a wall puzzle, dust floating in a thin beam of light"),
    ("chamber", "inside a hidden underground chamber, she raises a lantern toward painted ceiling murals"),
    ("artifact", "finding a small crystal artifact on a stone pedestal, soft blue light reflecting on her face"),
    ("escape", "the chamber begins to shake, she runs down a corridor as stones fall behind her"),
    ("river", "emerging beside an underground river, she balances on wet rocks with the glowing artifact"),
    ("boat", "small wooden boat in a misty canyon river, she rows through towering stone walls"),
    ("village", "remote mountain village at twilight, warm windows, she asks an elder for directions"),
    ("workshop", "inside a clockmaker workshop, gears and brass tools around her as she repairs the compass"),
    ("snow", "high snowy pass under pale sky, her red jacket stands out against white mountains"),
    ("storm", "caught in a dramatic snowstorm, she shields her face and keeps moving forward"),
    ("cave", "sheltering in an ice cave, blue crystalline walls sparkle around her lantern"),
    ("observatory", "abandoned mountaintop observatory, giant telescope pointing at a star filled sky"),
    ("star_map", "projected star map fills the dome, she compares it to the glowing crystal artifact"),
    ("airship", "boarding a small brass airship at sunrise, propellers turning, clouds below the dock"),
    ("sky", "on an airship deck above clouds, wind pulls her scarf as she looks toward a floating island"),
    ("island", "floating island with waterfalls and ancient towers, she walks across a stone causeway"),
    ("guardian", "facing a huge ancient stone guardian, she stands calmly holding the glowing artifact"),
    ("choice", "quiet temple interior, she places the crystal artifact into a circular mechanism"),
    ("return", "sunset on a hill above the old town, she closes the restored compass, journey complete"),
]


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


def build_prompt(index, slug, scene, seed):
    positive = f"solo, one woman only, {BASE_CHARACTER}, {scene}, {STYLE}"
    prefix = f"consistent_story/panels_final/RealVisXL_IPAdapter_final_story_{index:02d}_{slug}"

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": MODEL},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": FULL_REFERENCE_IMAGE},
        },
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": FACE_REFERENCE_IMAGE},
        },
        "3": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION},
        },
        "4": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": IP_FACE},
        },
        "5": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["4", 0],
                "image": ["14", 0],
                "clip_vision": ["3", 0],
                "weight": 0.5,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.65,
                "embeds_scaling": "V only",
            },
        },
        "6": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": IP_PLUS},
        },
        "7": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["5", 0],
                "ipadapter": ["6", 0],
                "image": ["2", 0],
                "clip_vision": ["3", 0],
                "weight": 0.05,
                "weight_type": "style transfer",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.25,
                "embeds_scaling": "V only",
            },
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["1", 1]},
        },
        "10": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["7", 0],
                "positive": ["8", 0],
                "negative": ["9", 0],
                "latent_image": ["10", 0],
                "seed": seed,
                "steps": 24,
                "cfg": 6.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["1", 2]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": prefix},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(SCENES))
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    outputs = []
    base_seed = 260526000
    selected = SCENES[args.start - 1 : args.start - 1 + args.limit]

    for offset, (slug, scene) in enumerate(selected):
        index = args.start + offset
        seed = base_seed + index
        prompt = build_prompt(index, slug, scene, seed)
        print(f"SUBMIT {index:02d}_{slug} seed={seed}", flush=True)
        try:
            result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": prompt})
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"submit failed for {index:02d}_{slug}: {exc}") from exc
        item = wait_for_prompt(result["prompt_id"])
        images = item.get("outputs", {}).get("13", {}).get("images", [])
        for image in images:
            outputs.append(image)
            print(f"DONE {index:02d}_{slug} {image['subfolder']}/{image['filename']}", flush=True)

    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
