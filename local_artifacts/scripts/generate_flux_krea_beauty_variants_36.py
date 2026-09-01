import argparse
import json
import math
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"flux-krea-beauty-variants-{uuid.uuid4()}"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "flux_krea_beauty_variants"

MODEL = "flux1-krea-dev_fp8_scaled.safetensors"
CLIP_L = "clip_l.safetensors"
T5XXL = "t5xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "ae.safetensors"

WIDTH = 640
HEIGHT = 896
STEPS = 16

BASE_PROMPT = (
    "photorealistic high-end three-quarter to full body fashion photo of a unique beautiful East Asian adult woman, age 24, "
    "not a celebrity, unique identity, refined natural face, clear symmetrical eyes, healthy realistic skin, "
    "distinct facial structure, tasteful fully clothed styling, modern premium fashion, "
    "natural makeup, realistic body proportions, standing pose, visible waist and body silhouette, "
    "head to mid-thigh visible, the specified face shape and body type must be obvious and distinct, "
    "elegant pose, soft cinematic light, sharp focus, 85mm lens"
)

QUALITY_HINT = (
    "real photo, editorial photography, natural skin texture, coherent anatomy, detailed clothing fabric, "
    "clean background separation, professional color grading"
)

AVOID_HINT = (
    "cartoon, anime, illustration, CGI, plastic skin, distorted eyes, crossed eyes, asymmetrical eyes, "
    "bad hands, extra fingers, deformed face, duplicated person, old-fashioned clothing, cheap outfit, "
    "nudity, lingerie, explicit, low quality, blurry, watermark, text"
)

GROUPS = [
    {
        "key": "pure_oval_slim",
        "label": "清纯-鹅蛋脸-纤细",
        "face": "clearly soft oval face, smooth jawline, gentle clear eyes",
        "body": "noticeably slim graceful figure, narrow waist, delicate shoulders",
        "hair": "long straight glossy black hair with a clean middle part",
        "style": "pure fresh beauty, white cotton midi dress, delicate cardigan",
        "scene": "bright minimal apartment with morning window light",
    },
    {
        "key": "pure_round_petite",
        "label": "清纯-圆脸-小巧",
        "face": "clearly round face, fuller cheeks, small soft chin, sweet adult features",
        "body": "noticeably petite shorter frame, compact balanced figure",
        "hair": "shoulder-length straight dark hair with airy side bangs",
        "style": "pure clean look, pale blue knit top, cream long skirt",
        "scene": "sunlit botanical greenhouse, soft green background",
    },
    {
        "key": "pure_heart_tall",
        "label": "清纯-心形脸-高挑",
        "face": "clearly heart-shaped face, wider forehead, delicate pointed chin, bright calm eyes",
        "body": "tall slender figure, long legs, narrow elegant waist",
        "hair": "long black hair in a low polished ponytail",
        "style": "clean elegant style, ivory blouse, wide-leg beige trousers",
        "scene": "quiet modern library with warm natural light",
    },
    {
        "key": "charm_oval_soft_curvy",
        "label": "有韵味-鹅蛋脸-柔美曲线",
        "face": "classic oval face, softly defined cheekbones, composed expressive eyes",
        "body": "soft curvy graceful figure, visible waist and hips, elegant mature proportions",
        "hair": "long side-parted dark hair with very soft natural waves",
        "style": "charming mature elegance, champagne satin blouse, long black skirt",
        "scene": "minimalist art gallery, warm indirect light",
    },
    {
        "key": "charm_diamond_toned",
        "label": "有韵味-菱形脸-紧致",
        "face": "clearly diamond-shaped face, high defined cheekbones, narrower forehead and chin, refined gaze",
        "body": "toned balanced figure, fit arms, firm waist, athletic elegance",
        "hair": "sleek collarbone-length straight black hair tucked behind one ear",
        "style": "elegant oriental modern fashion, dark green silk wrap dress",
        "scene": "quiet tea room with wood and stone textures",
    },
    {
        "key": "charm_square_tall",
        "label": "有韵味-方脸-高挑",
        "face": "clearly soft square face, broader jawline, confident warm eyes",
        "body": "tall statuesque figure, long torso, strong elegant posture",
        "hair": "smooth low bun with a few clean face-framing strands",
        "style": "intellectual elegant look, tailored navy suit, white silk inner top",
        "scene": "premium bookstore lounge, warm shelves behind her",
    },
    {
        "key": "sexy_heart_hourglass",
        "label": "性感-心形脸-沙漏",
        "face": "glamorous heart-shaped face, wide upper face, delicate chin, refined lips",
        "body": "obvious hourglass figure with realistic proportions, narrow waist, visible hips",
        "hair": "long glossy black hair swept over one shoulder",
        "style": "sexy elegant high fashion, red off-shoulder evening dress, fully clothed",
        "scene": "luxury hotel corridor with cinematic warm light",
    },
    {
        "key": "sexy_oval_athletic",
        "label": "性感-鹅蛋脸-运动紧致",
        "face": "sharp oval face, confident eyes, clean jawline",
        "body": "athletic toned figure, defined shoulders, slim waist, long legs",
        "hair": "high sleek ponytail with straight dark hair",
        "style": "sexy modern fashion, black fitted jumpsuit, fully clothed",
        "scene": "modern penthouse with city skyline at night",
    },
    {
        "key": "sexy_round_curvy",
        "label": "性感-圆脸-丰盈曲线",
        "face": "soft round face, fuller cheeks, charming expressive eyes",
        "body": "curvy balanced figure, visible hips and soft waist curve, realistic proportions",
        "hair": "long dark hair with soft face-framing layers",
        "style": "sensual elegant style, wine-colored bodycon midi dress, fully clothed",
        "scene": "upscale cocktail lounge, soft bokeh background",
    },
    {
        "key": "sweet_long_petite",
        "label": "甜美-长脸-小巧",
        "face": "clearly slender long face, longer facial proportion, sweet bright eyes",
        "body": "petite slim figure, small frame, narrow shoulders",
        "hair": "straight black bob just below the jawline",
        "style": "sweet modern casual, cropped ivory jacket, fitted white top, light denim",
        "scene": "clean cafe terrace in afternoon sunlight",
    },
    {
        "key": "cool_diamond_slim",
        "label": "冷艳-菱形脸-纤瘦",
        "face": "clearly diamond-shaped face, cool distant eyes, high cheekbones, narrow chin",
        "body": "tall slim figure, long legs, straight elegant posture",
        "hair": "very long straight black hair with a sharp side part",
        "style": "cool sophisticated fashion, black turtleneck, long beige trench coat",
        "scene": "glass storefront street in winter daylight",
    },
    {
        "key": "retro_soft_square_curvy",
        "label": "复古-柔方脸-曲线",
        "face": "soft square face, fuller jawline, graceful vintage expression",
        "body": "soft curvy elegant figure, fuller hips, visible waistline",
        "hair": "retro shoulder-length dark hair with soft polished waves",
        "style": "retro Hong Kong cinema inspired fashion, burgundy blouse, high-waisted black skirt",
        "scene": "warm evening street with old neon signs and film grain mood",
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
        try:
            history = get_json(f"/history/{prompt_id}")
        except Exception as exc:
            print(f"WAIT_RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(15)
            continue
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(15)


def image_exists(prefix):
    return bool(list(OUTPUT_DIR.glob(f"{prefix}_*.png")))


def build_prompt(group, variant, seed):
    prompt = (
        f"{BASE_PROMPT}, {group['face']}, {group['body']}, {group['hair']}, {group['style']}, "
        f"{group['scene']}, variant {variant}, {QUALITY_HINT}. Avoid: {AVOID_HINT}."
    )
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["1", 0], "max_shift": 1.15, "base_shift": 0.5, "width": WIDTH, "height": HEIGHT},
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": CLIP_L, "clip_name2": T5XXL, "type": "flux"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["3", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 3.5},
        },
        "6": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["3", 0], "clip_l": "", "t5xxl": "", "guidance": 3.5},
        },
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": STEPS,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": f"flux_krea_beauty_variants/{group['index']:02d}_{group['key']}_{variant:02d}"},
        },
    }


def make_grids():
    files = sorted(OUTPUT_DIR.glob("*.png"))
    if not files:
        return

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        pass

    def render(mode, tile_w, tile_h, cols, output_name):
        label_h = 30
        pad = 10
        rows = math.ceil(len(files) / cols)
        canvas = Image.new(
            "RGB",
            (cols * tile_w + (cols + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad),
            (246, 246, 244),
        )
        draw = ImageDraw.Draw(canvas)
        for i, path in enumerate(files):
            img = Image.open(path).convert("RGB")
            if mode == "face":
                w, h = img.size
                img = img.crop((int(w * 0.2), int(h * 0.03), int(w * 0.8), int(h * 0.5)))
                img = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.38))
            else:
                img = ImageOps.contain(img, (tile_w, tile_h), Image.Resampling.LANCZOS)

            tile = Image.new("RGB", (tile_w, tile_h), (232, 232, 230))
            tile.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
            x = pad + (i % cols) * (tile_w + pad)
            y = pad + (i // cols) * (tile_h + label_h + pad)
            canvas.paste(tile, (x, y))

            label = path.stem.split("_0000", 1)[0]
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x + max(0, (tile_w - (bbox[2] - bbox[0])) // 2), y + tile_h + 7), label, fill=(25, 25, 25), font=font)

        out = OUTPUT_DIR / output_name
        canvas.save(out, quality=94)
        print(f"GRID {out}", flush=True)

    render("full", 180, 252, 6, "beauty_variants_full_grid.jpg")
    render("face", 180, 180, 6, "beauty_variants_face_grid.jpg")


def build_jobs():
    jobs = []
    for group_index, group in enumerate(GROUPS, start=1):
        group = dict(group)
        group["index"] = group_index
        for variant in range(1, 4):
            seed = 526240000 + group_index * 100 + variant
            prefix = f"{group_index:02d}_{group['key']}_{variant:02d}"
            jobs.append((group, variant, seed, prefix))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only submit the first N missing jobs.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    submitted = 0
    for index, (group, variant, seed, prefix) in enumerate(jobs, start=1):
        if args.skip_existing and image_exists(prefix):
            print(f"SKIP {index:02d}/36 {prefix}", flush=True)
            continue
        if args.limit and submitted >= args.limit:
            break

        print(f"SUBMIT {index:02d}/36 {prefix} seed={seed} label={group['label']}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(group, variant, seed)})
        item = wait(result["prompt_id"])
        images = item.get("outputs", {}).get("10", {}).get("images", [])
        print(json.dumps(images, ensure_ascii=False), flush=True)
        submitted += 1
        make_grids()

    make_grids()
    print(f"ALL_DONE submitted={submitted}", flush=True)


if __name__ == "__main__":
    main()
