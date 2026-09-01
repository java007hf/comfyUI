import argparse
import json
import time
import uuid
from pathlib import Path

from call_flux_api import DEFAULT_SERVER, build_view_url, get_json, post_json, wait_for_history


DEFAULT_WORKFLOW = Path(__file__).with_name("flux_img2img_ref_api.json")
DEFAULT_PROMPT = (
    "photorealistic original high-end glamour portrait of a different adult Chinese woman, "
    "not the same person as the input image, similar beauty type only: delicate refined V-shaped Chinese face, "
    "large calm almond eyes, refined nose bridge, small elegant lips, long black half-up hair, "
    "cool alluring gaze, slim waist and elegant hourglass figure, black lace corset evening dress, "
    "luxury indoor botanical lounge, moody cinematic window light, changed pose, changed background details, "
    "changed dress details, vertical fashion portrait from head to hips, realistic skin texture, sharp eyes, high detail"
)
DEFAULT_NEGATIVE = (
    "child, teen, underage, nude, naked, explicit nudity, nipples, genitals, pornographic, sex act, "
    "low quality, blurry, bad anatomy, deformed hands, extra fingers, distorted face, cross-eyed, "
    "baby face, idol face, anime face, big smile, toothy smile, short bob haircut, flat body, "
    "shapeless torso, bulky waist, bad body proportions, identical copy, same exact person, "
    "same exact face, duplicate image, copied pose, unchanged reference image"
)


def main():
    parser = argparse.ArgumentParser(description="Submit a FLUX img2img reference job to local ComfyUI.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--image", default="ref_20260528_142309.jpg", help="Image file name in ComfyUI input dir.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--denoise", type=float, default=0.74)
    parser.add_argument("--seed", type=int, default=0, help="0 means use current time.")
    parser.add_argument("--prefix", default="api_flux_img2img/ref142309")
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    seed = args.seed or int(time.time() * 1000) % 1_000_000_000_000

    prompt["2"]["inputs"]["width"] = args.width
    prompt["2"]["inputs"]["height"] = args.height
    prompt["5"]["inputs"]["clip_l"] = args.prompt
    prompt["5"]["inputs"]["t5xxl"] = args.prompt
    prompt["6"]["inputs"]["clip_l"] = args.negative
    prompt["6"]["inputs"]["t5xxl"] = args.negative
    prompt["11"]["inputs"]["image"] = args.image
    prompt["12"]["inputs"]["width"] = args.width
    prompt["12"]["inputs"]["height"] = args.height
    prompt["8"]["inputs"]["seed"] = seed
    prompt["8"]["inputs"]["steps"] = args.steps
    prompt["8"]["inputs"]["denoise"] = args.denoise
    prompt["10"]["inputs"]["filename_prefix"] = args.prefix

    client_id = f"local-flux-img2img-{uuid.uuid4()}"
    result = post_json(args.server, "/prompt", {"client_id": client_id, "prompt": prompt})
    prompt_id = result["prompt_id"]
    print(f"PROMPT_ID {prompt_id}", flush=True)

    item = wait_for_history(args.server, prompt_id)
    outputs = item.get("outputs", {})
    images = outputs.get("10", {}).get("images", [])
    print(json.dumps(images, ensure_ascii=False, indent=2), flush=True)
    for image in images:
        print(build_view_url(args.server, image), flush=True)


if __name__ == "__main__":
    main()
