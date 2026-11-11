# Local FLUX API

ComfyUI local API endpoint:

```text
POST http://127.0.0.1:8188/prompt
```

Ready-to-submit workflow:

```text
D:\workspace\ComfyUI\local_artifacts\api\flux_text2img_api.json
```

Python helper:

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\local_artifacts\api\call_flux_api.py `
  --prompt "photorealistic cinematic portrait of an adult woman in a modern city, natural light, realistic skin texture" `
  --width 640 --height 896 --steps 16 `
  --prefix "api_flux/demo"
```

Direct request format:

```json
{
  "client_id": "local-flux-api",
  "prompt": {
    "...": "contents of flux_text2img_api.json"
  }
}
```

Useful fields to edit in `flux_text2img_api.json`:

```text
5.inputs.clip_l / 5.inputs.t5xxl     prompt
6.inputs.clip_l / 6.inputs.t5xxl     negative prompt
2.inputs.width / 2.inputs.height     model sampling size
7.inputs.width / 7.inputs.height     latent size, keep same as node 2
8.inputs.seed                        seed
8.inputs.steps                       sampling steps
10.inputs.filename_prefix            output folder/name prefix
```

Result lookup:

```text
GET http://127.0.0.1:8188/history/{prompt_id}
GET http://127.0.0.1:8188/view?filename=...&subfolder=...&type=output
```

## Batch FLUX beauty generation

Generate 100 tasteful adult glamour/fashion images:

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\local_artifacts\api\generate_flux_beauty_batch.py `
  --count 100 `
  --width 640 `
  --height 896 `
  --steps 16 `
  --style bold `
  --face-mode archetype `
  --outfit-mode liked `
  --prefix-dir "flux_beauty_batch"
```

`--style bold` focuses on adult Chinese glamour portraits with more revealing but non-explicit outfits.
Use `--style sunny` for brighter, healthier, less seductive fashion portraits.
`--face-mode archetype` uses fixed varied face profiles to reduce same-face collapse.
`--outfit-mode liked` favors the strongest outfit styles from the reviewed batch.
Use `--style dark_glam --outfit-mode ref142309` for the black lace cinematic glamour direction from `Downloads\20260528-142309.jpg`.

## FLUX img2img reference

Use the downloaded reference image as an initial image:

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\local_artifacts\api\call_flux_img2img_ref_api.py `
  --image "ref_20260528_142309.jpg" `
  --width 576 `
  --height 1024 `
  --steps 25 `
  --denoise 0.74 `
  --prefix "api_flux_img2img/ref142309"
```

For this use case, `--denoise 0.68-0.80` gives similar face/body type without copying the original image. Lower values like `0.25` are near redraws of the reference.

Batch from the same reference:

```powershell
.\venv\Scripts\python.exe .\local_artifacts\api\generate_flux_img2img_ref_batch.py `
  --image "ref_20260528_142309.jpg" `
  --count 30 `
  --width 576 `
  --height 1024 `
  --steps 25 `
  --denoise 0.78 `
  --variation-mode wardrobe_pose `
  --body-mode voluptuous `
  --prefix-dir "flux_img2img_ref142309_batch"
```

Use `--variation-mode thigh_glam` to favor the `ref142309_002` direction: seated/high-slit black bustier dress, visible full thigh, rounded bust support.

Output images are saved by ComfyUI under:

```text
D:\workspace\ComfyUI\output\flux_beauty_batch
```

Each submitted job is appended to:

```text
D:\workspace\ComfyUI\local_artifacts\api\flux_beauty_batch_manifest.jsonl
```

Preview prompts without generating:

```powershell
.\venv\Scripts\python.exe .\local_artifacts\api\generate_flux_beauty_batch.py --count 5 --dry-run
```
