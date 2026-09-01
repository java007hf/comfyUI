import argparse
import json
import time
import uuid
from pathlib import Path

from call_flux_api import DEFAULT_SERVER, build_view_url, post_json, wait_for_history


DEFAULT_WORKFLOW = Path(__file__).with_name("flux_inpaint_api.json")


def configure_workflow(template, image, mask, prompt, negative, width, height, steps, denoise, seed, filename_prefix):
    workflow = json.loads(json.dumps(template))
    workflow["2"]["inputs"]["width"] = width
    workflow["2"]["inputs"]["height"] = height
    workflow["5"]["inputs"]["clip_l"] = prompt
    workflow["5"]["inputs"]["t5xxl"] = prompt
    workflow["6"]["inputs"]["clip_l"] = negative
    workflow["6"]["inputs"]["t5xxl"] = negative
    workflow["11"]["inputs"]["image"] = image
    workflow["12"]["inputs"]["width"] = width
    workflow["12"]["inputs"]["height"] = height
    workflow["14"]["inputs"]["image"] = mask
    workflow["8"]["inputs"]["seed"] = seed or int(time.time() * 1000) % 1_000_000_000_000
    workflow["8"]["inputs"]["steps"] = steps
    workflow["8"]["inputs"]["denoise"] = denoise
    workflow["10"]["inputs"]["filename_prefix"] = filename_prefix
    return workflow


def main():
    parser = argparse.ArgumentParser(description="Submit a FLUX inpaint job to local ComfyUI.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--denoise", type=float, default=0.86)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="flux_inpaint_ref142309/ref142309_inpaint")
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    template = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow = configure_workflow(
        template,
        image=args.image,
        mask=args.mask,
        prompt=args.prompt,
        negative=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        denoise=args.denoise,
        seed=args.seed,
        filename_prefix=args.prefix,
    )

    client_id = f"local-flux-inpaint-{uuid.uuid4()}"
    result = post_json(args.server.rstrip("/"), "/prompt", {"client_id": client_id, "prompt": workflow})
    prompt_id = result["prompt_id"]
    print(f"PROMPT_ID {prompt_id}", flush=True)

    item = wait_for_history(args.server.rstrip("/"), prompt_id)
    images = item.get("outputs", {}).get("10", {}).get("images", [])
    print(json.dumps(images, ensure_ascii=False, indent=2), flush=True)
    for image in images:
        print(build_view_url(args.server.rstrip("/"), image), flush=True)


if __name__ == "__main__":
    main()
