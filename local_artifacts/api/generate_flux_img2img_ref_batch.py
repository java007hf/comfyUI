import argparse
import json
import random
import time
import uuid
from pathlib import Path

from call_flux_api import DEFAULT_SERVER, build_view_url, get_json, post_json, wait_for_history
from call_flux_img2img_ref_api import DEFAULT_NEGATIVE, DEFAULT_PROMPT, DEFAULT_WORKFLOW


STYLE_CORE = (
    "photorealistic original high-end glamour portrait of one different adult Chinese woman, "
    "single woman only, solo portrait, "
    "not the same person as the input image, similar beauty type only: delicate refined V-shaped Chinese face, "
    "large calm almond eyes, refined nose bridge, small elegant lips, cool alluring gaze, "
    "realistic skin texture, sharp eyes, high detail"
)

BODY_CORE = (
    "slim waist and elegant hourglass figure, natural full bust under a structured bustier, "
    "G-cup-like fullness, round uplifting balconette cups, rounded full bust shape, "
    "smooth convex cup contour, lifted rounded cleavage, softly curved bust silhouette, "
    "molded supportive cups, sweetheart deep neckline, defined collarbones, graceful shoulders, "
    "glamorous waist-to-bust proportion"
)

BODY_CORE_STRONG = (
    "slim waist and dramatic curvy hourglass figure, visibly very large full bust under a structured push-up bustier, "
    "oversized but natural rounded bust, generous round uplifting balconette cups, prominent smooth rounded cleavage, "
    "convex satin cup contour, lifted full bust shape, deep open sweetheart neckline, narrow waist, soft curvy hips, "
    "defined collarbones, graceful shoulders, dramatic waist-to-bust proportion"
)

BODY_CORE_BALANCED = (
    "slim waist and elegant hourglass figure, natural bust, defined collarbones, graceful shoulders"
)

WARDROBE_VARIANTS = [
    "black lace push-up corset evening dress with rounded molded cups, thin straps and velvet waist panel, fully clothed",
    "black satin bustier gown with rounded structured cups, lace trim, deep neckline, fitted waist and side slit, fully clothed",
    "black sheer lace robe over a structured push-up corset dress with smooth rounded bust shape, fully clothed",
    "dark emerald velvet corset dress with black lace details, rounded supportive bust and narrow waist, fully clothed",
    "burgundy satin bustier evening dress with rounded cup silhouette, black lace gloves and deep neckline, fully clothed",
    "black off-shoulder lace gown with a fitted waist and sculpted rounded bust, fully clothed",
    "black halter corset dress with floral lace overlay, rounded bust contour and deep neckline, fully clothed",
    "charcoal silk slip dress with rounded lace bodice, thin straps and fur stole, fully clothed",
    "black lace bodysuit-style fashion top with high-waisted satin skirt and rounded structured cups, fully clothed",
    "deep navy velvet bustier dress with rounded black lace neckline and narrow waist, fully clothed",
]

THIGH_GLAM_WARDROBE_VARIANTS = [
    "black satin bustier gown with round balconette cups, sweetheart deep neckline, high side slit revealing a full thigh, fully clothed",
    "black satin corset evening dress with molded rounded cups, narrow waist, high slit showing one elegant thigh, fully clothed",
    "black velvet bustier dress with smooth structured cups, sweetheart neckline and dramatic side slit, fully clothed",
    "black satin-and-lace bustier gown with rounded supportive cups, fitted waist and exposed thigh through a high slit, fully clothed",
    "dark burgundy satin bustier gown with round molded cups, narrow waist and high slit revealing one thigh, fully clothed",
    "black off-shoulder bustier dress with rounded balconette support and high side slit, fully clothed",
]

THIGH_GLAM_FOCUS_WARDROBE_VARIANTS = [
    "black structured push-up bustier gown with large smooth satin cups, deep sweetheart neckline and high side slit showing one full thigh, fully clothed",
    "black velvet bustier dress with large rounded molded cups, sweetheart deep neckline and dramatic high slit revealing one full thigh, fully clothed",
    "black satin corset evening dress with generous round balconette cups, narrow waist and high slit exposing one full thigh, fully clothed",
    "black velvet-and-lace bustier dress with prominent rounded cup silhouette, fitted waist and high slit revealing one full thigh, fully clothed",
]

POSE_VARIANTS = [
    "standing three-quarter pose with one shoulder angled toward camera",
    "seated elegant pose on a velvet chair, torso turned slightly",
    "walking pose through a luxury botanical lounge, head turned back",
    "leaning near a tall window, one hand lightly holding the fur wrap",
    "front-facing half-body pose with arms relaxed at the sides",
    "side-profile fashion pose, looking back over one shoulder",
    "waist-up pose with one hand near the collarbone",
    "full-body standing pose with crossed legs and relaxed shoulders",
    "kneeling-on-sofa fashion editorial pose, upper body upright",
    "standing pose beside indoor plants, one arm behind the waist",
]

THIGH_GLAM_POSE_VARIANTS = [
    "seated elegant pose on a velvet chair, torso turned slightly, one thigh visible through the slit",
    "reclining seated fashion pose, one leg extended forward through a high slit",
    "side-seated pose with one hand near the collarbone and one thigh exposed by the dress slit",
    "standing three-quarter pose with one leg forward, high slit revealing one thigh",
    "leaning beside a tall window, one knee slightly bent, thigh visible through the slit",
    "front-facing half-body-to-thigh portrait, one hand on waist, high slit showing one thigh",
]

THIGH_GLAM_FOCUS_POSE_VARIANTS = [
    "seated editorial pose with one hand on waist, high slit clearly showing one rounded thigh",
    "seated elegant fashion pose, one knee forward, one full thigh clearly visible through the high slit",
    "front-facing seated glamour pose on a chair, torso upright, one full thigh exposed by the dress slit",
    "side-seated fashion pose, waist turned slightly, one full thigh visible and dress slit open",
]

SCENE_VARIANTS = [
    "luxury indoor botanical lounge with dark wall panels",
    "moody hotel suite beside a tall window and tropical plants",
    "velvet dressing room with warm side lighting",
    "greenhouse-style fashion studio with cinematic shadows",
    "dark luxury bedroom with silk curtains and indoor plants",
    "private lounge with black marble wall and soft floral decor",
    "evening fashion studio with a large window and leaf shadows",
    "luxury villa interior with botanical decor and low-key light",
]

CAMERA_VARIANTS = [
    "vertical fashion portrait from head to hips, 85mm lens",
    "full-body vertical portrait, natural perspective",
    "waist-up editorial portrait, shallow depth of field",
    "three-quarter body composition, cinematic framing",
    "upper-body portrait with visible waist curve, sharp eyes",
]

FOCUS_CAMERA_VARIANTS = [
    "three-quarter body composition from head to thigh, cinematic framing",
    "vertical portrait from head to upper thigh, 85mm lens, natural perspective",
    "front-facing head-to-thigh fashion portrait, sharp eyes, natural perspective",
]


def build_varied_prompt(index, rng, body_mode, variation_mode):
    if variation_mode == "thigh_glam_focus":
        wardrobe_pool = THIGH_GLAM_FOCUS_WARDROBE_VARIANTS
        pose_pool = THIGH_GLAM_FOCUS_POSE_VARIANTS
        camera_pool = FOCUS_CAMERA_VARIANTS
    elif variation_mode == "thigh_glam":
        wardrobe_pool = THIGH_GLAM_WARDROBE_VARIANTS
        pose_pool = THIGH_GLAM_POSE_VARIANTS
        camera_pool = CAMERA_VARIANTS
    else:
        wardrobe_pool = WARDROBE_VARIANTS
        pose_pool = POSE_VARIANTS
        camera_pool = CAMERA_VARIANTS
    wardrobe = wardrobe_pool[(index - 1) % len(wardrobe_pool)]
    pose = pose_pool[(index - 1 + rng.randrange(len(pose_pool))) % len(pose_pool)]
    scene = SCENE_VARIANTS[(index - 1 + rng.randrange(len(SCENE_VARIANTS))) % len(SCENE_VARIANTS)]
    camera = camera_pool[(index - 1 + rng.randrange(len(camera_pool))) % len(camera_pool)]
    hair = rng.choice(
        [
            "long black half-up hair with loose strands",
            "long glossy black waves over one shoulder",
            "long straight black hair swept back from the face",
            "long dark layered hair with volume at the crown",
        ]
    )
    if variation_mode == "thigh_glam_focus" and body_mode == "voluptuous":
        body = BODY_CORE_STRONG
    else:
        body = BODY_CORE if body_mode == "voluptuous" else BODY_CORE_BALANCED
    thigh_detail = (
        "rounded full thigh clearly visible through the high slit, elegant leg line, sensual seated glamour, "
        if variation_mode in {"thigh_glam", "thigh_glam_focus"}
        else ""
    )
    face_detail = (
        "striking beautiful face, refined symmetrical almond eyes, bright healthy expression, "
        "soft confident lips, elegant modern makeup, no sad expression, "
        if variation_mode == "thigh_glam_focus"
        else ""
    )
    return (
        f"{STYLE_CORE}, {face_detail}{body}, {thigh_detail}{hair}, {wardrobe}, {pose}, in a {scene}, {camera}, "
        "moody cinematic window light, changed clothing design, changed pose, changed scene details, "
        "sexy but non-explicit luxury black-lace glamour aesthetic, not a copy of the reference image"
    )


def configure_workflow(template, image, prompt, negative, width, height, steps, denoise, seed, filename_prefix):
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
    workflow["8"]["inputs"]["seed"] = seed
    workflow["8"]["inputs"]["steps"] = steps
    workflow["8"]["inputs"]["denoise"] = denoise
    workflow["10"]["inputs"]["filename_prefix"] = filename_prefix
    return workflow


def main():
    parser = argparse.ArgumentParser(description="Generate a FLUX img2img batch from one local reference image.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--image", default="ref_20260528_142309.jpg")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--denoise", type=float, default=0.78)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument(
        "--variation-mode",
        choices=["single", "wardrobe_pose", "thigh_glam", "thigh_glam_focus"],
        default="wardrobe_pose",
    )
    parser.add_argument("--body-mode", choices=["voluptuous", "balanced"], default="voluptuous")
    parser.add_argument("--prefix-dir", default="flux_img2img_ref142309_batch")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("flux_img2img_ref142309_batch_manifest.jsonl")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    server = args.server.rstrip("/")
    get_json(server, "/queue")

    workflow_path = Path(args.workflow)
    template = json.loads(workflow_path.read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seed_rng = random.Random(args.base_seed or time.time_ns())
    mode = "DRY_RUN" if args.dry_run else "SUBMIT"
    print(
        f"{mode} count={args.count} image={args.image} size={args.width}x{args.height} "
        f"steps={args.steps} denoise={args.denoise} variation_mode={args.variation_mode} body_mode={args.body_mode}",
        flush=True,
    )
    print(f"SERVER {server}", flush=True)
    print(f"WORKFLOW {workflow_path}", flush=True)
    print(f"MANIFEST {manifest_path}", flush=True)

    for offset in range(args.count):
        index = args.start_index + offset
        seed = args.base_seed + offset if args.base_seed else seed_rng.randrange(1, 1_000_000_000_000)
        filename_prefix = f"{args.prefix_dir}/ref142309_{index:03d}"
        prompt = (
            build_varied_prompt(index, seed_rng, args.body_mode, args.variation_mode)
            if args.variation_mode in {"wardrobe_pose", "thigh_glam", "thigh_glam_focus"}
            else args.prompt
        )
        negative = (
            f"{args.negative}, identical copy, same exact outfit, same exact pose, duplicate composition, "
            "unchanged clothing, unchanged background, two women, second woman, extra person, duplicate person, "
            "duplicate face, duplicate head, extra head, extra torso, double body, mirror duplicate, "
            "plain face, unattractive face, dull eyes, tired eyes, sad face, crying expression, drooping mouth, "
            "older-looking face, western woman, western features, conservative outfit, modest neckline, high neckline, "
            "loose bodice, low slit, closed slit, hidden thigh, covered legs, "
            "small chest, medium bust, average bust, flat chest, tiny bust, narrow bust, compressed bust, "
            "shapeless body, straight waist, "
            "pointy bust, conical bust, "
            "boxy bust, angular chest shape, flattened cleavage, sagging bust, asymmetrical bust, poorly supported bust, "
            "thin legs, skinny thighs, awkward leg pose, hidden legs"
        )
        record = {
            "index": index,
            "seed": seed,
            "image": args.image,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "denoise": args.denoise,
            "variation_mode": args.variation_mode,
            "body_mode": args.body_mode,
            "filename_prefix": filename_prefix,
            "prompt": prompt,
            "negative": negative,
        }

        if args.dry_run:
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue

        started_at = time.time()
        workflow = configure_workflow(
            template,
            image=args.image,
            prompt=prompt,
            negative=negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            denoise=args.denoise,
            seed=seed,
            filename_prefix=filename_prefix,
        )

        client_id = f"local-flux-img2img-batch-{uuid.uuid4()}"
        result = post_json(server, "/prompt", {"client_id": client_id, "prompt": workflow})
        prompt_id = result["prompt_id"]
        print(f"[{offset + 1}/{args.count}] submit index={index} seed={seed}", flush=True)

        item = wait_for_history(server, prompt_id)
        images = item.get("outputs", {}).get("10", {}).get("images", [])
        elapsed_sec = round(time.time() - started_at, 2)
        record.update(
            {
                "prompt_id": prompt_id,
                "elapsed_sec": elapsed_sec,
                "images": images,
                "view_urls": [build_view_url(server, image) for image in images],
            }
        )
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{offset + 1}/{args.count}] done index={index} images={len(images)} elapsed={elapsed_sec}s", flush=True)
        for url in record["view_urls"]:
            print(url, flush=True)


if __name__ == "__main__":
    main()
