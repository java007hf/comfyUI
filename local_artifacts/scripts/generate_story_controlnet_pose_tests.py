import json
import math
import shutil
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"story-controlnet-pose-tests-{uuid.uuid4()}"

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output" / "story_controlnet_pose_tests"
POSE_DIR = INPUT_DIR / "story_pose_tests"

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

NEGATIVE = (
    "anime, cartoon, illustration, painting, CGI, 3d render, doll, plastic skin, child, underage, teenager, "
    "old woman, male, multiple people, duplicate person, twin, different face, asymmetrical face, asymmetrical eyes, "
    "uneven eyes, lazy eye, crossed eyes, squint, distorted face, deformed eyes, curly hair, short hair, messy hair, "
    "frumpy outfit, old fashioned outfit, bulky clothes, lingerie, nude, nsfw, exposed cleavage, low quality, blurry, "
    "noisy, watermark, text, logo, bad anatomy, extra fingers, missing fingers, deformed hands, cropped head, cropped feet"
)

TESTS = [
    {
        "short": "01_city_street",
        "pose": "walk",
        "prompt": (
            f"{BASE_CHARACTER}, full body walking on a clean modern city street, glass storefronts, soft morning light, "
            "editorial street fashion photography, realistic background, 85mm lens, sharp focus"
        ),
    },
    {
        "short": "02_mountain_view",
        "pose": "stand",
        "prompt": (
            f"{BASE_CHARACTER}, full body standing at a mountain overlook, layered green mountains and cloudy sky behind her, "
            "travel editorial photography, natural daylight, realistic background, sharp focus"
        ),
    },
    {
        "short": "03_car_side",
        "pose": "lean",
        "prompt": (
            f"{BASE_CHARACTER}, full body leaning near a modern white electric car on a quiet tree lined road, "
            "clean lifestyle photography, realistic vehicle and background, soft daylight, sharp focus"
        ),
    },
    {
        "short": "04_train_platform",
        "pose": "step",
        "prompt": (
            f"{BASE_CHARACTER}, full body standing on a bright train platform with a sleek train in the background, "
            "cinematic travel story frame, realistic background, soft daylight, sharp focus"
        ),
    },
    {
        "short": "05_bookstore",
        "pose": "book",
        "prompt": (
            f"{BASE_CHARACTER}, full body holding a book inside a warm modern bookstore, shelves in the background, "
            "cozy realistic lifestyle photography, natural window light, sharp focus"
        ),
    },
    {
        "short": "06_night_street",
        "pose": "turn",
        "prompt": (
            f"{BASE_CHARACTER}, full body turning slightly on a neon lit evening street, wet pavement reflections, "
            "cinematic urban story frame, realistic background, shallow depth of field, sharp focus"
        ),
    },
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


def transform(points, sx=1.0, sy=1.0, dx=0, dy=0):
    return {k: (int(x * sx + dx), int(y * sy + dy)) for k, (x, y) in points.items()}


def base_pose():
    return {
        "nose": (384, 160),
        "neck": (384, 245),
        "r_shoulder": (315, 260),
        "r_elbow": (280, 430),
        "r_wrist": (310, 585),
        "l_shoulder": (455, 260),
        "l_elbow": (500, 430),
        "l_wrist": (470, 585),
        "r_hip": (335, 520),
        "l_hip": (435, 520),
        "r_knee": (330, 760),
        "r_ankle": (320, 1010),
        "l_knee": (445, 760),
        "l_ankle": (455, 1010),
        "r_eye": (360, 145),
        "l_eye": (410, 145),
        "r_ear": (335, 155),
        "l_ear": (435, 155),
    }


def pose_variant(name):
    p = base_pose()
    if name == "walk":
        p.update({"r_elbow": (270, 405), "r_wrist": (250, 540), "l_elbow": (525, 405), "l_wrist": (548, 540), "r_knee": (288, 760), "r_ankle": (250, 1008), "l_knee": (472, 760), "l_ankle": (520, 1008)})
    elif name == "lean":
        p = transform(p, sx=1.0, sy=1.0, dx=28, dy=0)
        p.update({"neck": (405, 245), "r_shoulder": (340, 265), "l_shoulder": (470, 250), "r_wrist": (290, 610), "l_wrist": (520, 590), "r_ankle": (330, 1010), "l_ankle": (490, 1005)})
    elif name == "step":
        p.update({"r_knee": (340, 730), "r_ankle": (355, 1005), "l_knee": (460, 730), "l_ankle": (420, 1005), "r_wrist": (300, 570), "l_wrist": (500, 560)})
    elif name == "book":
        p.update({"r_elbow": (315, 405), "r_wrist": (350, 500), "l_elbow": (455, 405), "l_wrist": (420, 500), "r_knee": (335, 770), "l_knee": (430, 770)})
    elif name == "turn":
        p.update({"nose": (395, 160), "neck": (400, 245), "r_shoulder": (342, 262), "l_shoulder": (475, 255), "r_elbow": (285, 440), "r_wrist": (260, 610), "l_elbow": (510, 420), "l_wrist": (500, 570), "r_hip": (360, 525), "l_hip": (445, 518), "r_knee": (340, 765), "r_ankle": (325, 1008), "l_knee": (470, 760), "l_ankle": (500, 1005)})
    return p


POSE_EDGES = [
    ("neck", "r_shoulder", (255, 0, 0)),
    ("r_shoulder", "r_elbow", (255, 85, 0)),
    ("r_elbow", "r_wrist", (255, 170, 0)),
    ("neck", "l_shoulder", (255, 255, 0)),
    ("l_shoulder", "l_elbow", (170, 255, 0)),
    ("l_elbow", "l_wrist", (85, 255, 0)),
    ("neck", "r_hip", (0, 255, 0)),
    ("r_hip", "r_knee", (0, 255, 85)),
    ("r_knee", "r_ankle", (0, 255, 170)),
    ("neck", "l_hip", (0, 255, 255)),
    ("l_hip", "l_knee", (0, 170, 255)),
    ("l_knee", "l_ankle", (0, 85, 255)),
    ("neck", "nose", (0, 0, 255)),
    ("nose", "r_eye", (85, 0, 255)),
    ("r_eye", "r_ear", (170, 0, 255)),
    ("nose", "l_eye", (255, 0, 255)),
    ("l_eye", "l_ear", (255, 0, 170)),
]


def make_pose_image(name, out_path):
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    points = pose_variant(name)
    for a, b, color in POSE_EDGES:
        draw.line([points[a], points[b]], fill=color, width=12)
    for point in points.values():
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=(255, 255, 255))
    img.save(out_path)


def prepare_poses():
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for test in TESTS:
        filename = f"{test['short']}_pose.png"
        out = POSE_DIR / filename
        make_pose_image(test["pose"], out)
        paths[test["short"]] = f"story_pose_tests/{filename}"
    return paths


def build_prompt(test, pose_image, seed):
    prefix = f"story_controlnet_pose_tests/{test['short']}/{test['short']}"
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
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": test["prompt"], "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "9": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "control_net": ["6", 0],
                "image": ["5", 0],
                "strength": 0.72,
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
    for test in TESTS:
        files = sorted((OUTPUT_DIR / test["short"]).glob(f"{test['short']}_*.png"))
        if files:
            items.append((test["short"], files[-1]))

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        pass

    def label(draw, x, y, w, text):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + max(0, (w - tw) // 2), y + 8), text, fill=(20, 20, 20), font=font)

    def grid(mode, tile_w, tile_h, output_path):
        cols = 3
        label_h = 34
        pad = 14
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

    print("GRID_FULL", grid("full", 300, 450, OUTPUT_DIR / "story_controlnet_pose_tests_full_grid.jpg"), flush=True)
    print("GRID_FACE", grid("face", 300, 300, OUTPUT_DIR / "story_controlnet_pose_tests_face_grid.jpg"), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_reference()
    poses = prepare_poses()
    outputs = []
    for index, test in enumerate(TESTS):
        seed = 300533000 + index
        print(f"SUBMIT {test['short']} seed={seed}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(test, poses[test["short"]], seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("13", {}).get("images", [])
        outputs.extend(images)
        print(json.dumps(images, ensure_ascii=False), flush=True)

    make_grids()
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
