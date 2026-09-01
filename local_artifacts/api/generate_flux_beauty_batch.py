import argparse
import json
import random
import time
import uuid
from pathlib import Path

from call_flux_api import DEFAULT_SERVER, DEFAULT_WORKFLOW, build_view_url, get_json, post_json, wait_for_history


DEFAULT_NEGATIVE = (
    "child, teen, underage, schoolgirl, loli, nude, naked, explicit nudity, nipples, genitals, "
    "pornographic, sex act, vulgar pose, low quality, blurry, bad anatomy, deformed hands, extra fingers, "
    "extra limbs, distorted face, asymmetrical eyes, cross-eyed, sad expression, crying face, gloomy face, "
    "tired eyes, emotionless expression, bad teeth, plastic skin, overprocessed, old-fashioned outfit, "
    "outdated styling, office uniform, generic face, same face, western-looking, European woman, Latina woman, "
    "Japanese woman, Korean woman, K-pop idol style, childish face, baby face, idol face, anime face"
)

FACE_TYPES = [
    "delicate oval Chinese face, refined facial features, bright almond eyes",
    "heart-shaped Chinese face, soft cheekbones, clear lively eyes",
    "small round Chinese face, sweet smile, elegant nose and lips",
    "slender Chinese face, graceful jawline, fresh youthful beauty",
    "high-fashion Chinese face, sharp cheekbones, confident gaze",
    "classic Chinese beauty face, balanced features, warm expressive eyes",
    "soft square Chinese face, clean jawline, sunny healthy charm",
    "petite Chinese face, delicate nose bridge, bright natural smile",
]

FACE_DETAILS = [
    "soft bright eyes, natural double eyelids, small beauty mark near the cheek",
    "round lively eyes, subtle dimples, fresh youthful smile",
    "slender almond eyes, defined cupid's bow lips, elegant nose bridge",
    "clear sparkling eyes, soft full lips, gentle cheek contour",
    "slightly arched eyebrows, refined nose, radiant smile",
    "large expressive eyes, smooth jawline, confident charming smile",
    "calm intelligent eyes, delicate lips, clean natural facial structure",
    "warm smiling eyes, graceful cheekbones, healthy glowing face",
]

FACE_ARCHETYPES = [
    {"age": 24, "face": "round Chinese face, soft cheeks, short nose, cheerful dimples", "hair": "short black bob haircut", "skin": "warm natural skin tone", "makeup": "fresh peach makeup", "expression": "bright friendly smile"},
    {"age": 27, "face": "oval Chinese face, high nose bridge, calm almond eyes", "hair": "long straight black hair with center part", "skin": "fair skin with realistic texture", "makeup": "clean modern makeup", "expression": "confident gentle smile"},
    {"age": 30, "face": "soft square Chinese face, defined jawline, hooded eyes", "hair": "shoulder-length chestnut hair", "skin": "healthy sun-kissed skin", "makeup": "sun-kissed healthy makeup", "expression": "relaxed sunny smile"},
    {"age": 25, "face": "heart-shaped Chinese face, small chin, large lively eyes", "hair": "long wavy dark hair", "skin": "clear porcelain skin", "makeup": "glossy fashion makeup", "expression": "playful warm smile"},
    {"age": 32, "face": "mature elegant Chinese face, sharp cheekbones, narrow eyes", "hair": "sleek black side-part hair", "skin": "warm ivory skin", "makeup": "elegant evening makeup", "expression": "composed confident smile"},
    {"age": 26, "face": "petite Chinese face, round eyes, delicate lips", "hair": "long black hair with airy bangs", "skin": "fresh natural skin", "makeup": "soft peach makeup", "expression": "open cheerful smile"},
    {"age": 29, "face": "athletic Chinese face, strong cheekbones, bright eyes", "hair": "high ponytail with loose strands", "skin": "tan healthy skin", "makeup": "minimal dewy makeup", "expression": "energetic radiant smile"},
    {"age": 23, "face": "slender Chinese face, soft jawline, gentle monolid eyes", "hair": "straight shoulder-length black hair", "skin": "fair matte skin", "makeup": "natural clean makeup", "expression": "shy but happy smile"},
    {"age": 34, "face": "sophisticated Chinese face, long nose bridge, full lips", "hair": "long glossy chestnut waves", "skin": "warm golden skin", "makeup": "luxury editorial makeup", "expression": "alluring elegant smile"},
    {"age": 28, "face": "cute round Chinese face, wide-set eyes, small beauty mark under one eye", "hair": "black hair with soft bangs", "skin": "soft rosy skin", "makeup": "fresh dewy makeup", "expression": "sweet bright smile"},
    {"age": 31, "face": "angular Chinese face, high cheekbones, phoenix eyes", "hair": "sleek low bun with side strands", "skin": "cool fair skin", "makeup": "sharp fashion makeup", "expression": "confident subtle smile"},
    {"age": 25, "face": "oval Chinese face, natural double eyelids, rounded nose tip", "hair": "long straight dark brown hair", "skin": "healthy glowing skin", "makeup": "clean modern Chinese beauty makeup", "expression": "warm natural smile"},
    {"age": 27, "face": "soft diamond Chinese face, refined cheekbones, expressive eyes", "hair": "medium-length wavy brown hair", "skin": "light honey skin", "makeup": "sunset-toned makeup", "expression": "lively confident smile"},
    {"age": 35, "face": "mature glamorous Chinese face, defined jawline, deep-set eyes", "hair": "long black Hollywood waves", "skin": "smooth warm skin", "makeup": "elegant smoky makeup", "expression": "bold charming smile"},
    {"age": 24, "face": "small oval Chinese face, soft brow, clear smiling eyes", "hair": "long straight black hair with curtain bangs", "skin": "fresh fair skin", "makeup": "light peach makeup", "expression": "sunny open smile"},
    {"age": 33, "face": "tall-model Chinese face, strong nose bridge, narrow jaw", "hair": "short sleek bob haircut", "skin": "natural beige skin", "makeup": "minimal high-fashion makeup", "expression": "cool confident smile"},
    {"age": 26, "face": "round-cheeked Chinese face, curved eyebrows, bright teeth", "hair": "long layered black hair", "skin": "warm healthy skin", "makeup": "fresh coral makeup", "expression": "big joyful smile"},
    {"age": 29, "face": "classic Chinese beauty face, balanced features, soft almond eyes", "hair": "long dark hair swept to one side", "skin": "ivory skin with natural pores", "makeup": "soft elegant makeup", "expression": "gentle radiant smile"},
    {"age": 22, "face": "youthful adult Chinese face, round eyes, small chin", "hair": "shoulder-length straight hair with bangs", "skin": "clear natural skin", "makeup": "light fresh makeup", "expression": "bright healthy smile"},
    {"age": 36, "face": "elegant mature Chinese face, sculpted cheekbones, full lips", "hair": "long loose black waves", "skin": "warm bronze skin", "makeup": "glamour evening makeup", "expression": "self-assured smile"},
    {"age": 28, "face": "soft square Chinese face, slightly wider nose bridge, sincere eyes", "hair": "black bob with side part", "skin": "healthy natural complexion", "makeup": "dewy natural makeup", "expression": "relaxed friendly smile"},
    {"age": 30, "face": "slim Chinese face, long eyelashes, defined cupid's bow", "hair": "long silky brown hair", "skin": "fair luminous skin", "makeup": "glossy editorial makeup", "expression": "alluring warm smile"},
    {"age": 25, "face": "heart-shaped Chinese face, soft cheeks, small dimples", "hair": "high ponytail with airy bangs", "skin": "sunlit fair skin", "makeup": "sporty fresh makeup", "expression": "healthy energetic smile"},
    {"age": 32, "face": "high-fashion Chinese face, angular brows, elegant nose", "hair": "straight black lob haircut", "skin": "cool light skin", "makeup": "modern runway makeup", "expression": "polished confident smile"},
    {"age": 27, "face": "gentle Chinese face, monolid eyes, soft lips", "hair": "long natural black hair", "skin": "warm clear skin", "makeup": "simple clean makeup", "expression": "quiet sunny smile"},
    {"age": 34, "face": "glamorous Chinese face, sharp cheekbones, bright expressive eyes", "hair": "long voluminous brown waves", "skin": "golden healthy skin", "makeup": "bronzed beauty makeup", "expression": "bold radiant smile"},
    {"age": 23, "face": "petite Chinese face, round nose tip, lively eyes", "hair": "short shoulder-length hair with curled ends", "skin": "soft natural skin", "makeup": "fresh pink makeup", "expression": "playful bright smile"},
    {"age": 29, "face": "oval Chinese face, graceful jawline, arched eyebrows", "hair": "sleek long black hair", "skin": "light warm skin", "makeup": "clean luxury makeup", "expression": "elegant warm smile"},
    {"age": 31, "face": "confident Chinese face, broad cheekbones, calm eyes", "hair": "low ponytail with face-framing strands", "skin": "healthy tan skin", "makeup": "natural sunlit makeup", "expression": "grounded confident smile"},
    {"age": 26, "face": "delicate Chinese face, bright almond eyes, subtle beauty mark near lips", "hair": "long wavy dark hair with side part", "skin": "fair glowing skin", "makeup": "soft glossy makeup", "expression": "sweet alluring smile"},
]

REF_142309_FACE_ARCHETYPES = [
    {"age": 26, "face": "stunning delicate V-shaped Chinese face, refined high nose bridge, large calm almond eyes, small elegant lips", "hair": "long black half-up hair with loose strands framing the face", "skin": "cool fair skin with realistic texture", "makeup": "soft smoky eye makeup, long eyelashes, muted red lips", "expression": "calm alluring gaze, subtle closed-mouth expression"},
    {"age": 28, "face": "exquisite oval Chinese face, sharp cheekbones, elegant narrow jawline, refined nose and lips", "hair": "long straight black hair swept back from the face", "skin": "porcelain fair skin, natural pores", "makeup": "luxury editorial makeup, defined eyeliner", "expression": "cool composed gaze, faint smile without teeth"},
    {"age": 30, "face": "mature high-fashion Chinese beauty face, high nose bridge, full soft lips, sculpted cheeks", "hair": "long dark layered hair with volume at the crown", "skin": "warm ivory skin with realistic texture", "makeup": "subtle smoky makeup, sculpted cheeks", "expression": "confident cold glamour expression"},
    {"age": 25, "face": "heart-shaped Chinese face, delicate chin, clear expressive eyes, refined symmetrical features", "hair": "long silky black hair, half tied, face-framing strands", "skin": "fair luminous skin", "makeup": "clean glamour makeup with soft eyeliner", "expression": "gentle serious gaze, lips softly closed"},
    {"age": 32, "face": "slender Chinese goddess face, refined jawline, deep-set elegant eyes, graceful nose bridge", "hair": "long glossy black waves over one shoulder", "skin": "cool light skin, realistic skin texture", "makeup": "elegant evening makeup, muted rose lips", "expression": "poised alluring expression, no big smile"},
    {"age": 27, "face": "classic Chinese top-model face, balanced refined features, long eyelashes, soft defined lips", "hair": "long black hair with a lifted crown and loose side strands", "skin": "smooth fair skin with natural facial asymmetry", "makeup": "high-end magazine makeup, soft contouring", "expression": "quiet intense gaze, subtle lips"},
]

BODY_TYPES = [
    "slim graceful figure",
    "tall elegant figure",
    "athletic toned figure",
    "curvy hourglass figure",
    "healthy fit figure",
    "soft feminine figure",
    "long legs and graceful posture",
    "toned waist and healthy proportions",
]

REF_142309_BODIES = [
    "slim waist, elegant hourglass silhouette, full bust under structured corset dress, long neck, defined collarbones, graceful shoulders",
    "tall curvy figure, narrow waist, glamorous bust and waist proportion, elegant long arms, refined posture",
    "fit feminine body, sculpted waistline, graceful collarbones, balanced curvy proportions, luxury glamour posture",
    "slender yet curvy figure, narrow waist and smooth hip line, elegant shoulders, confident model posture",
]

HAIR_STYLES = [
    "long straight black hair",
    "long silky brown hair",
    "shoulder-length straight hair",
    "long wavy dark hair",
    "sleek black hair with side part",
    "long glossy hair",
    "long straight dark hair with airy bangs",
    "long black hair with soft layers",
    "glossy chestnut brown straight hair",
]

MAKEUP_STYLES = [
    "fresh dewy makeup",
    "clean modern Chinese beauty makeup",
    "sun-kissed healthy makeup",
    "glossy fashion editorial makeup",
    "soft peach makeup with bright eyes",
    "elegant evening makeup, not heavy",
]

OUTFITS = [
    "bright fitted summer dress",
    "modern white mini dress with elegant styling",
    "silk camisole top and high-waisted skirt, fully clothed",
    "stylish off-shoulder dress",
    "modern designer dress with clean lines",
    "luxury resort swimsuit with a sheer beach cover-up, no nudity",
    "satin slip dress, fully clothed",
    "fashionable crop top and long skirt",
    "fitted knit dress with elegant neckline",
    "red evening dress with tasteful neckline",
]

BOLD_OUTFITS = [
    "deep V black evening dress, fully clothed, no nudity",
    "red high-slit evening dress, elegant and revealing, no nudity",
    "luxury one-piece swimsuit with sheer cover-up, tasteful glamour, no nudity",
    "satin backless dress, fully clothed, no nudity",
    "white bodycon dress with open shoulders, no nudity",
    "silk slip dress with low neckline, fully clothed, no nudity",
    "fashion bikini with beach cover-up, tasteful resort editorial, no nudity",
    "black lace evening dress over opaque lining, fully clothed, no nudity",
    "glamour editorial dress with side cutouts, no nudity",
    "wet-look fashion dress at beach sunset, fully clothed, no nudity",
]

LIKED_BOLD_OUTFITS = [
    "silk slip dress with low neckline, fully clothed, no nudity",
    "champagne satin slip dress with thin straps and low neckline, fully clothed, no nudity",
    "white bodycon mini dress with open shoulders, no nudity",
    "white off-shoulder fitted mini dress, elegant and revealing, no nudity",
    "fashion bikini with sheer beach cover-up, tasteful resort editorial, no nudity",
    "white bikini top with loose sheer cover-up, tasteful poolside fashion, no nudity",
    "satin backless dress, fully clothed, no nudity",
    "open-back satin camisole dress, fully clothed, no nudity",
    "deep V silk evening dress, fully clothed, no nudity",
    "soft beige bralette-style fashion top with lace cover-up, tasteful street editorial, no nudity",
]

REF_142309_OUTFITS = [
    "black velvet corset dress with lace bodice and deep neckline, fully clothed, no nudity",
    "black lace corset gown with sheer side lace panels, luxury glamour styling, no nudity",
    "black satin bustier dress with lace trim and fur stole, fully clothed, no nudity",
    "black velvet evening dress with structured corset top, elegant and revealing, no nudity",
    "black lace-and-satin fitted dress with thin straps, high-end editorial styling, no nudity",
    "black corset mini dress with lace details and soft fur wrap, fully clothed, no nudity",
]

SCENES = [
    "luxury hotel suite",
    "modern Shanghai city street at sunset",
    "seaside resort terrace",
    "high-end fashion studio",
    "neon-lit Shanghai night street",
    "elegant apartment interior",
    "rooftop lounge with skyline background",
    "art gallery with soft ambient light",
    "Sanya beach boardwalk during golden hour",
    "classic car parked on a clean city avenue",
    "private resort poolside",
    "bright sunlit villa bedroom",
]

REF_142309_SCENES = [
    "luxury indoor botanical lounge with plants and soft shadows",
    "moody private dressing room with plants and dark wall panels",
    "high-end hotel suite with indoor greenery and soft window light",
    "elegant studio corner with tropical plants and cinematic shadows",
    "luxury fashion room with dark velvet textures and botanical decor",
]

POSES = [
    "confident standing pose",
    "natural walking pose",
    "bright over-the-shoulder pose",
    "elegant seated pose",
    "fashion editorial pose",
    "relaxed cinematic pose",
    "sunny confident smile with relaxed shoulders",
    "healthy energetic pose with warm smile",
]

REF_142309_POSES = [
    "centered waist-up pose, straight elegant posture",
    "front-facing half-body pose with relaxed shoulders",
    "slight three-quarter pose, one shoulder angled toward camera",
    "standing portrait pose with fur wrap draped over arms",
    "elegant still pose, calm eye contact with camera",
]

EXPRESSIONS = [
    "bright open smile",
    "healthy natural smile",
    "confident radiant smile",
    "playful warm smile",
    "sunny relaxed expression",
    "lively eyes and gentle smile",
]

CAMERAS = [
    "85mm lens, shallow depth of field",
    "50mm lens, natural perspective",
    "full-body composition, professional fashion photography",
    "three-quarter body composition, cinematic framing",
    "portrait composition, sharp eyes",
]

LIGHTING = [
    "golden hour light",
    "soft studio lighting",
    "cinematic rim light",
    "natural window light",
    "bright clean commercial fashion lighting",
    "moody neon reflections",
    "warm sunny daylight",
    "fresh outdoor sunlight",
]

REF_142309_LIGHTING = [
    "dramatic low-key window light with soft plant shadows",
    "moody cinematic side lighting, gentle highlights on the face",
    "soft directional indoor light, deep shadows, luxury magazine look",
    "dim elegant evening light with realistic skin texture",
]

REF_142309_CAMERAS = [
    "vertical half-body portrait, 85mm lens, shallow depth of field",
    "upper-body portrait composition, 85mm lens, sharp eyes",
    "waist-up fashion portrait, natural perspective, cinematic framing",
]


def compose_prompt(rng, style, index, face_mode, outfit_mode):
    persona_pool = REF_142309_FACE_ARCHETYPES if outfit_mode == "ref142309" else FACE_ARCHETYPES
    persona = persona_pool[(index - 1) % len(persona_pool)]
    age = persona["age"] if face_mode == "archetype" else rng.randint(22, 28)
    if outfit_mode == "liked":
        outfit_pool = LIKED_BOLD_OUTFITS
    elif outfit_mode == "ref142309":
        outfit_pool = REF_142309_OUTFITS
    else:
        outfit_pool = BOLD_OUTFITS if style == "bold" else OUTFITS
    scene_pool = REF_142309_SCENES if outfit_mode == "ref142309" else SCENES
    lighting_pool = REF_142309_LIGHTING if outfit_mode == "ref142309" else LIGHTING
    camera_pool = REF_142309_CAMERAS if outfit_mode == "ref142309" else CAMERAS
    pose_pool = REF_142309_POSES if outfit_mode == "ref142309" else POSES
    face_type = persona["face"] if face_mode == "archetype" else rng.choice(FACE_TYPES)
    face_detail = persona["skin"] if face_mode == "archetype" else FACE_DETAILS[(index - 1 + rng.randrange(len(FACE_DETAILS))) % len(FACE_DETAILS)]
    hair_style = persona["hair"] if face_mode == "archetype" else rng.choice(HAIR_STYLES)
    makeup_style = persona["makeup"] if face_mode == "archetype" else rng.choice(MAKEUP_STYLES)
    expression = persona["expression"] if face_mode == "archetype" else EXPRESSIONS[(index - 1 + rng.randrange(len(EXPRESSIONS))) % len(EXPRESSIONS)]
    body_desc = rng.choice(REF_142309_BODIES) if outfit_mode == "ref142309" else rng.choice(BODY_TYPES)
    if style == "sunny":
        mood = "bright sunny healthy expression, confident warm smile, positive energetic mood"
    elif style == "dark_glam":
        mood = "cool elegant glamour, calm alluring gaze, high-end luxury magazine mood"
    else:
        mood = "bold sensual glamour, confident warm smile, alluring but elegant expression"
    parts = [
        f"photorealistic image of a beautiful {age}-year-old adult Chinese woman",
        "mainland Chinese glamour model, Chinese facial features, refined high-end beauty aesthetic",
        face_type,
        face_detail,
        body_desc,
        hair_style,
        makeup_style,
        rng.choice(outfit_pool),
        rng.choice(pose_pool),
        expression,
        f"in a {rng.choice(scene_pool)}",
        rng.choice(lighting_pool),
        rng.choice(camera_pool),
        mood,
        "tasteful sexy Chinese glamour, elegant fashion editorial style, revealing outfit but fully clothed, non-explicit",
        "premium black lace glamour portrait aesthetic, original fictional person",
        f"distinctive individual face identity {index:03d}, different person from every other image, realistic skin texture, natural facial asymmetry, detailed face, high detail, sharp focus",
    ]
    return ", ".join(parts)


def configure_workflow(template, prompt, negative, width, height, steps, seed, filename_prefix):
    workflow = json.loads(json.dumps(template))
    workflow["2"]["inputs"]["width"] = width
    workflow["2"]["inputs"]["height"] = height
    workflow["5"]["inputs"]["clip_l"] = prompt
    workflow["5"]["inputs"]["t5xxl"] = prompt
    workflow["6"]["inputs"]["clip_l"] = negative
    workflow["6"]["inputs"]["t5xxl"] = negative
    workflow["7"]["inputs"]["width"] = width
    workflow["7"]["inputs"]["height"] = height
    workflow["8"]["inputs"]["seed"] = seed
    workflow["8"]["inputs"]["steps"] = steps
    workflow["10"]["inputs"]["filename_prefix"] = filename_prefix
    return workflow


def submit_one(server, workflow):
    client_id = f"local-flux-batch-{uuid.uuid4()}"
    result = post_json(server, "/prompt", {"client_id": client_id, "prompt": workflow})
    return result["prompt_id"]


def main():
    parser = argparse.ArgumentParser(description="Generate a batch of tasteful adult glamour images with local FLUX.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=0, help="0 means random image seeds.")
    parser.add_argument("--prompt-seed", type=int, default=0, help="0 means random prompt combinations.")
    parser.add_argument("--style", choices=["bold", "sunny", "dark_glam"], default="bold")
    parser.add_argument("--face-mode", choices=["archetype", "random"], default="archetype")
    parser.add_argument("--outfit-mode", choices=["liked", "all", "ref142309"], default="liked")
    parser.add_argument("--prefix-dir", default="flux_beauty_batch")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("flux_beauty_batch_manifest.jsonl")))
    parser.add_argument("--extra-negative", default="")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between finished jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned prompts without submitting jobs.")
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    server = args.server.rstrip("/")
    get_json(server, "/queue")

    workflow_path = Path(args.workflow)
    template = json.loads(workflow_path.read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_rng = random.Random(args.prompt_seed or time.time_ns())
    seed_rng = random.Random(args.base_seed or time.time_ns())
    negative = DEFAULT_NEGATIVE
    if args.style == "dark_glam" or args.outfit_mode == "ref142309":
        negative = f"{negative}, big smile, toothy smile, cute dimples, cheerful kawaii look, short bob haircut, flat body, shapeless torso, bulky waist, bad body proportions, short neck, awkward shoulders"
    if args.extra_negative.strip():
        negative = f"{negative}, {args.extra_negative.strip()}"

    mode = "DRY_RUN" if args.dry_run else "SUBMIT"
    print(
        f"{mode} count={args.count} size={args.width}x{args.height} steps={args.steps} "
        f"style={args.style} face_mode={args.face_mode} outfit_mode={args.outfit_mode}",
        flush=True,
    )
    print(f"SERVER {server}", flush=True)
    print(f"WORKFLOW {workflow_path}", flush=True)
    print(f"MANIFEST {manifest_path}", flush=True)

    for offset in range(args.count):
        index = args.start_index + offset
        prompt = compose_prompt(prompt_rng, args.style, index, args.face_mode, args.outfit_mode)
        seed = args.base_seed + offset if args.base_seed else seed_rng.randrange(1, 1_000_000_000_000)
        filename_prefix = f"{args.prefix_dir}/beauty_{index:03d}"

        record = {
            "index": index,
            "seed": seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "style": args.style,
            "face_mode": args.face_mode,
            "outfit_mode": args.outfit_mode,
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
            prompt=prompt,
            negative=negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=seed,
            filename_prefix=filename_prefix,
        )

        print(f"[{offset + 1}/{args.count}] submit index={index} seed={seed}", flush=True)
        prompt_id = submit_one(server, workflow)
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

        if args.pause > 0 and offset + 1 < args.count:
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
