import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = Path(__file__).with_name("flux_text2img_api.json")


def post_json(server, path, payload):
    req = Request(
        f"{server}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(server, path):
    with urlopen(f"{server}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_history(server, prompt_id):
    while True:
        try:
            history = get_json(server, f"/history/{prompt_id}")
        except Exception as exc:
            print(f"WAIT_RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10)
            continue

        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(5)


def build_view_url(server, image):
    query = urlencode(
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    return f"{server}/view?{query}"


def main():
    parser = argparse.ArgumentParser(description="Submit a FLUX text-to-image job to local ComfyUI.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="0 means use current time.")
    parser.add_argument("--prefix", default="api_flux/output")
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    seed = args.seed or int(time.time() * 1000) % 1000000000000

    prompt["2"]["inputs"]["width"] = args.width
    prompt["2"]["inputs"]["height"] = args.height
    prompt["5"]["inputs"]["clip_l"] = args.prompt
    prompt["5"]["inputs"]["t5xxl"] = args.prompt
    prompt["6"]["inputs"]["clip_l"] = args.negative
    prompt["6"]["inputs"]["t5xxl"] = args.negative
    prompt["7"]["inputs"]["width"] = args.width
    prompt["7"]["inputs"]["height"] = args.height
    prompt["8"]["inputs"]["seed"] = seed
    prompt["8"]["inputs"]["steps"] = args.steps
    prompt["10"]["inputs"]["filename_prefix"] = args.prefix

    client_id = f"local-flux-api-{uuid.uuid4()}"
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
