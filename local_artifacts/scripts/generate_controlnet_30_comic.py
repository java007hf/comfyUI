import json
import math
import shutil
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps

import generate_story_controlnet_pose_tests as pose_tools


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"controlnet-30-comic-{uuid.uuid4()}"

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output" / "consistent_story_controlnet_30"
POSE_DIR = INPUT_DIR / "controlnet_30_comic_poses"

MODEL = "RealVisXL_V5.0_fp16.safetensors"
CONTROLNET = "xinsir-controlnet-openpose-sdxl-1.0.safetensors"
REFERENCE_SOURCE = ROOT / "output" / "reference_img2img_faceid" / "medium_repaint_luxury_casual" / "medium_repaint_luxury_casual_00001_.png"
REFERENCE_IMAGE = "story_character_reference.png"
WIDTH = 768
HEIGHT = 1152

BASE_CHARACTER = (
    "same East Asian adult woman from the reference, age 22, beautiful refined face, symmetrical clear eyes, "
    "long straight glossy black hair, slim graceful figure, modern fitted white short sleeve top, high waisted blue jeans, "
    "cropped ivory designer jacket, tasteful fully clothed fashion, natural makeup, realistic skin texture"
)

NEGATIVE = pose_tools.NEGATIVE

PANELS = [
    ("001_morning_street", "walk", "walking through a clean modern city street at morning, glass storefronts, soft sunlight, story frame"),
    ("002_bookstore_window", "book", "standing by a warm bookstore window and reading a blue book, shelves and sunlight behind her"),
    ("003_subway_entrance", "step", "stepping into a modern subway entrance, city commuters blurred in the background"),
    ("004_train_platform", "walk", "walking along a bright train platform beside a sleek blue train, travel story frame"),
    ("005_train_window", "stand", "standing near a large train window, countryside scenery outside, quiet reflective mood"),
    ("006_mountain_arrival", "stand", "standing at a mountain station, green hills and cloudy sky in the background"),
    ("007_forest_path", "walk", "walking on a forest path with tall trees and soft diffused light"),
    ("008_mountain_overlook", "stand", "standing at a mountain overlook, layered green mountains behind her"),
    ("009_old_bridge", "turn", "turning slightly on an old stone bridge over a clear river, travel documentary style"),
    ("010_riverside_map", "book", "holding an open travel notebook beside a quiet riverside, soft afternoon light"),
    ("011_village_lane", "walk", "walking through a clean mountain village lane, traditional houses and flowers behind her"),
    ("012_white_car", "lean", "leaning near a modern white electric car on a tree lined road, lifestyle photo"),
    ("013_highway_stop", "stand", "standing at a scenic highway rest stop, car parked nearby, distant hills behind her"),
    ("014_coastal_road", "walk", "walking along a coastal road with a modern car in the background, bright ocean air"),
    ("015_ferry_terminal", "step", "standing at a ferry terminal, water and city skyline behind her"),
    ("016_evening_market", "walk", "walking through an evening market street with warm lantern lights and soft crowds"),
    ("017_cafe_corner", "book", "holding a book in a modern cafe corner, window light, quiet story moment"),
    ("018_rooftop_sunset", "stand", "standing on a city rooftop at sunset, skyline behind her, cinematic warm light"),
    ("019_neon_crosswalk", "turn", "turning on a neon lit crosswalk at night, wet pavement reflections"),
    ("020_hotel_lobby", "stand", "standing in a stylish hotel lobby, marble floor and soft ambient lighting"),
    ("021_gallery_hall", "walk", "walking through a minimalist art gallery hall, framed photographs on white walls"),
    ("022_library_stack", "book", "holding a book between tall library shelves, realistic indoor lighting"),
    ("023_rainy_window", "stand", "standing beside a rainy window, city lights blurred outside, cinematic quiet mood"),
    ("024_morning_car", "lean", "leaning near the white electric car in early morning light, preparing to leave"),
    ("025_train_departure", "step", "stepping onto a train carriage, platform lights and motion blur behind her"),
    ("026_mountain_return", "stand", "standing again at the mountain overlook with clearer sky, confident expression"),
    ("027_city_arrival", "walk", "walking out of a modern station into the city, evening blue hour"),
    ("028_bookstore_return", "book", "returning to the bookstore holding the same blue book, warm shelves behind her"),
    ("029_quiet_alley", "turn", "turning in a clean quiet city alley with soft neon signs, cinematic frame"),
    ("030_final_rooftop", "stand", "standing on a rooftop at dawn with the city skyline behind her, calm satisfying ending"),
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


def setup_reference():
    if not REFERENCE_SOURCE.exists():
        raise FileNotFoundError(REFERENCE_SOURCE)
    shutil.copy2(REFERENCE_SOURCE, INPUT_DIR / REFERENCE_IMAGE)


def prepare_poses():
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for short, pose, _ in PANELS:
        filename = f"{short}_pose.png"
        path = POSE_DIR / filename
        pose_tools.make_pose_image(pose, path)
        paths[short] = f"controlnet_30_comic_poses/{filename}"
    return paths


def build_prompt(short, pose_image, scene, seed):
    prefix = f"consistent_story_controlnet_30/panels/{short}"
    prompt = (
        f"{BASE_CHARACTER}, full body, {scene}, photorealistic editorial story still, "
        "realistic background, natural cinematic light, 85mm lens, sharp focus, coherent single character"
    )
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL}},
        "2": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": 0.5,
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
                "weight": 0.78,
                "weight_faceidv2": 1.12,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.68,
                "embeds_scaling": "V only",
            },
        },
        "5": {"class_type": "LoadImage", "inputs": {"image": pose_image}},
        "6": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "9": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "control_net": ["6", 0],
                "image": ["5", 0],
                "strength": 0.7,
                "start_percent": 0.0,
                "end_percent": 0.82,
            },
        },
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["9", 0],
                "negative": ["9", 1],
                "latent_image": ["10", 0],
                "seed": seed,
                "steps": 24,
                "cfg": 5.2,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def make_grids():
    items = []
    for short, _, _ in PANELS:
        files = sorted((OUTPUT_DIR / "panels").glob(f"{short}_*.png"))
        if files:
            items.append((short, files[-1]))

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        pass

    def label(draw, x, y, w, text):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + max(0, (w - tw) // 2), y + 8), text, fill=(20, 20, 20), font=font)

    def grid(mode, tile_w, tile_h, cols, output_path):
        label_h = 30
        pad = 10
        rows = math.ceil(len(items) / cols)
        canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad), (245, 245, 242))
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(items):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if mode == "face":
                img = img.crop((int(w * 0.18), int(h * 0.02), int(w * 0.82), int(h * 0.42)))
                img = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))
            else:
                img = ImageOps.contain(img, (tile_w, tile_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), (230, 230, 226))
            tile.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
            c = i % cols
            r = i // cols
            x = pad + c * (tile_w + pad)
            y = pad + r * (tile_h + label_h + pad)
            canvas.paste(tile, (x, y))
            label(draw, x, y + tile_h, tile_w, name)
        canvas.save(output_path, quality=94)
        return str(output_path)

    print("GRID_FULL", grid("full", 220, 330, 5, OUTPUT_DIR / "comic_30_full_grid.jpg"), flush=True)
    print("GRID_FACE", grid("face", 220, 220, 5, OUTPUT_DIR / "comic_30_face_grid.jpg"), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_reference()
    poses = prepare_poses()
    outputs = []
    for index, (short, _, scene) in enumerate(PANELS):
        seed = 310533000 + index
        print(f"SUBMIT {short} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(short, poses[short], scene, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("13", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    make_grids()
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
