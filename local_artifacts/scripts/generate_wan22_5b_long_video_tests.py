import json
import time
import uuid
from urllib.request import Request, urlopen


SERVER = "http://127.0.0.1:8188"
CLIENT_ID = f"wan22-5b-long-tests-{uuid.uuid4()}"

OUTPUT_SUBFOLDER = "wan22_5b_long_tests"

MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan2.2_vae.safetensors"

WIDTH = 640
HEIGHT = 352
LENGTH = 65
FPS = 12.0
STEPS = 20
CFG = 5.0

NEGATIVE = (
    "low quality, worst quality, blurry, distorted, oversaturated, text, watermark, "
    "static, frozen frame, jitter, flicker, deformed body, bad hands, bad face, extra limbs, messy background"
)

CASES = [
    {
        "name": "01_cinematic_woman_walk",
        "seed": 52702001,
        "prompt": (
            "A cinematic realistic video of a stylish East Asian adult woman walking slowly through a rainy neon city street at night. "
            "Fully clothed premium black coat, long dark hair, elegant confident posture. "
            "The camera tracks beside her with smooth motion, wet pavement reflections, soft neon bokeh, moving pedestrians, realistic body movement."
        ),
    },
    {
        "name": "02_mountain_road_car",
        "seed": 52702002,
        "prompt": (
            "A cinematic realistic video of a white electric car driving along a winding mountain road at sunset. "
            "The camera follows behind and slowly pans to the side, orange sky, distant mountains, smooth vehicle motion, natural reflections, filmic color grading."
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
        try:
            history = get_json(f"/history/{prompt_id}")
        except Exception as exc:
            print(f"WAIT_RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
            continue
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(30)


def build_prompt(case):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "cpu"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": case["prompt"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEGATIVE}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {"vae": ["6", 0], "width": WIDTH, "height": HEIGHT, "length": LENGTH, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": case["seed"],
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": FPS}},
        "11": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["10", 0],
                "filename_prefix": f"{OUTPUT_SUBFOLDER}/{case['name']}",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def main():
    outputs = []
    for index, case in enumerate(CASES, start=1):
        print(f"SUBMIT {index}/{len(CASES)} {case['name']} seed={case['seed']}", flush=True)
        result = post_json("/prompt", {"client_id": CLIENT_ID, "prompt": build_prompt(case)})
        item = wait(result["prompt_id"])
        node_outputs = item.get("outputs", {}).get("11", {})
        print(json.dumps(node_outputs, ensure_ascii=False), flush=True)
        outputs.append({"case": case["name"], "outputs": node_outputs})
    print("ALL_DONE", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
