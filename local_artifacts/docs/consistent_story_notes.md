# Consistent Story Test

## 结论

这轮对比里，写实角色和故事分镜优先使用：

- 底模：`RealVisXL_V5.0_fp16.safetensors`
- 角色一致性：`IPAdapter Plus Face SDXL` + `IPAdapter Plus SDXL`
- 参考图：`input\story_character_reference.png`
- 脸部/上半身参考图：`input\story_character_face_reference.png`

原因：

- `RealVisXL` 的人物脸部稳定性最好，写实细节比 DreamShaper Turbo 更稳。
- `Juggernaut XI` 综合也强，但这轮人物脸型稳定性略弱于 RealVisXL。
- `Illustrious XL` 适合二次元/插画，不适合当前写实故事角色。
- 纯提示词无法可靠维持 30 张角色一致，IPAdapter 是当前本地方案里最稳的选择。

## 已生成结果

高一致性 30 张：

```text
D:\workspace\ComfyUI\output\consistent_story\panels
```

总览图：

```text
D:\workspace\ComfyUI\output\consistent_story\comic_overview_grid.jpg
```

PDF：

```text
D:\workspace\ComfyUI\output\consistent_story\consistent_story_high_consistency_comic.pdf
```

参考候选图：

```text
D:\workspace\ComfyUI\output\consistent_story\reference_candidates
```

## 已安装 IPAdapter 文件

节点：

```text
D:\workspace\ComfyUI\custom_nodes\comfyui-ipadapter
```

模型：

```text
D:\workspace\ComfyUI\models\ipadapter\ip-adapter-plus-face_sdxl_vit-h.safetensors
D:\workspace\ComfyUI\models\ipadapter\ip-adapter-plus_sdxl_vit-h.safetensors
D:\workspace\ComfyUI\models\clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
```

## 重跑命令

先启动 ComfyUI：

```powershell
cd D:\workspace\ComfyUI
.\run_comfyui_nvidia.bat
```

生成角色参考候选：

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\generate_story_reference_candidates.py
```

生成 30 张连环画。当前脚本是继续调试用的平衡版模板，默认输出到 `panels_final`：

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\generate_consistent_comic.py --start 1 --limit 30
```

只重跑某几张：

```powershell
.\venv\Scripts\python.exe .\generate_consistent_comic.py --start 10 --limit 3
```

## 参数取舍

角色一致性更强：

- 提高 `IPAdapterAdvanced` 的 `weight`
- 缺点是背景更容易变成参考图风格，场景不够丰富

场景更自由：

- 降低 `IPAdapterAdvanced` 的 `weight`
- 缺点是衣服颜色、脸型、发型更容易漂移

当前已经完成的 30 张主结果在 `panels`，采用偏“一致性优先”的方案。后面调试过 `panels_v2`、`panels_v3`、`panels_v4`、`panels_final` 的样张：场景会更自由，但服装和角色稳定性会下降。若要做正式故事，建议以 `panels` 为主，挑不满意的单张单独重跑，而不是整体降低参考权重。
