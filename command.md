# ComfyUI 本地使用命令

本文档记录当前机器上的 ComfyUI 本地服务启动方式、本地 API 协议、模型更新方式和常用排查命令。

## 基本信息

- ComfyUI 目录：`D:\workspace\ComfyUI`
- Web 地址：`http://127.0.0.1:8188`
- 局域网访问：`http://<本机IP>:8188`
- 启动脚本：`D:\workspace\ComfyUI\run_comfyui_nvidia.bat`
- 当前启动参数包含：`--disable-api-nodes`
- `--disable-api-nodes` 只禁用 ComfyUI 的线上付费 API 节点和前端联网能力，不会禁用本机的 `/prompt` 生图 API。

## 启动 ComfyUI 服务

推荐直接运行：

```bat
cd /d D:\workspace\ComfyUI
run_comfyui_nvidia.bat
```

等价的 PowerShell 启动命令：

```powershell
cd D:\workspace\ComfyUI
.\venv\Scripts\python.exe .\main.py --listen 0.0.0.0 --port 8188 --lowvram --preview-method auto --disable-auto-launch --disable-api-nodes
```

启动后打开：

```text
http://127.0.0.1:8188
```

检查服务是否正常：

```powershell
curl.exe -s -I http://127.0.0.1:8188/
curl.exe -s http://127.0.0.1:8188/system_stats
```

停止当前 8188 端口上的 ComfyUI：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*main.py*--port 8188*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

查看当前 GPU 占用：

```powershell
nvidia-smi
```

## 本地 API 协议

ComfyUI 的本地 API 默认地址：

```text
http://127.0.0.1:8188
```

常用接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/prompt` | 提交一个生图任务 |
| `GET` | `/history/{prompt_id}` | 查询任务结果 |
| `GET` | `/queue` | 查看队列 |
| `POST` | `/interrupt` | 中断当前任务 |
| `GET` | `/view?filename=...&type=output&subfolder=...` | 读取输出图片 |
| `GET` | `/object_info` | 查看可用节点和输入参数 |
| `GET` | `/object_info/CheckpointLoaderSimple` | 查看可用 checkpoint |

`POST /prompt` 请求体格式：

```json
{
  "client_id": "local-api",
  "prompt": {
    "1": {
      "class_type": "CheckpointLoaderSimple",
      "inputs": {
        "ckpt_name": "Juggernaut-XI-byRunDiffusion.safetensors"
      }
    }
  }
}
```

实际工作流会包含多个节点。节点 ID 是字符串，节点之间用 `["节点ID", 输出序号]` 连接，例如：

```json
"7": {
  "class_type": "VAEDecode",
  "inputs": {
    "samples": ["6", 0],
    "vae": ["1", 2]
  }
}
```

提交已有 API workflow 示例：

```powershell
cd D:\workspace\ComfyUI

$prompt = Get-Content .\local_sd35_turbo_api_prompt.json -Raw | ConvertFrom-Json
$prompt."3".inputs.text = "a realistic red apple on a wooden table, studio lighting"
$prompt."6".inputs.seed = Get-Random

$body = @{
  client_id = "local-api"
  prompt = $prompt
} | ConvertTo-Json -Depth 20

$res = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8188/prompt `
  -ContentType "application/json" `
  -Body $body

$res.prompt_id
```

查询结果：

```powershell
$id = $res.prompt_id
Invoke-RestMethod "http://127.0.0.1:8188/history/$id" | ConvertTo-Json -Depth 20
```

结果里通常会出现：

```json
"outputs": {
  "9": {
    "images": [
      {
        "filename": "local_sd35_turbo_00001_.png",
        "subfolder": "",
        "type": "output"
      }
    ]
  }
}
```

读取图片 URL：

```text
http://127.0.0.1:8188/view?filename=local_sd35_turbo_00001_.png&type=output&subfolder=
```

注意：

- `local_sd35_turbo_api_prompt.json` 是 API 用的 prompt graph，可以直接传给 `/prompt`。
- `local_sd35_turbo_workflow.json` 是 UI 工作流文件，适合在网页里加载，不一定能直接提交给 `/prompt`。
- 如果从网页导出给 API 用的 JSON，需要使用 ComfyUI 的 API 格式导出，而不是普通 workflow 格式。

## 用网页创建本地模型生图流程

最小文生图流程：

```text
CheckpointLoaderSimple
  -> CLIPTextEncode 正向提示词
  -> CLIPTextEncode 反向提示词
  -> EmptyLatentImage
  -> KSampler
  -> VAEDecode
  -> SaveImage
```

SD3.5 本地流程一般使用：

```text
CheckpointLoaderSimple
TripleCLIPLoader
CLIPTextEncode
EmptySD3LatentImage
KSampler
VAEDecode
SaveImage
```

当前已经有一个可用示例：

```text
D:\workspace\ComfyUI\local_sd35_turbo_workflow.json
```

网页中加载方式：

1. 打开 `http://127.0.0.1:8188`
2. 把 `local_sd35_turbo_workflow.json` 拖进页面
3. 确认模型名称存在
4. 修改提示词、尺寸、seed
5. 点击 Queue

## 模型目录

常见模型放置路径：

| 类型 | 目录 |
| --- | --- |
| SDXL/SD1.5/SD3.5 单文件 checkpoint | `D:\workspace\ComfyUI\models\checkpoints` |
| LoRA | `D:\workspace\ComfyUI\models\loras` |
| CLIP/T5 文本编码器 | `D:\workspace\ComfyUI\models\text_encoders` |
| VAE | `D:\workspace\ComfyUI\models\vae` |
| ControlNet | `D:\workspace\ComfyUI\models\controlnet` |
| FLUX/Qwen/Wan 等部分新版模型主体 | `D:\workspace\ComfyUI\models\diffusion_models` 或按 workflow 要求放置 |

更新模型后，如果网页里没有显示：

1. 先点击 ComfyUI 页面里的刷新模型按钮。
2. 还不显示就重启 ComfyUI。
3. 用下面命令确认文件是否真的在正确目录。

```powershell
Get-ChildItem D:\workspace\ComfyUI\models\checkpoints -File | Select-Object Name,Length
Get-ChildItem D:\workspace\ComfyUI\models\loras -File | Select-Object Name,Length
```

## 当前已安装模型

截至 2026-05-25，本机 `models\checkpoints` 里已有：

| 模型 | 大小 | 用途 |
| --- | ---: | --- |
| `sd3.5m_turbo.safetensors` | 4.76GB | SD3.5 Medium Turbo，本地快速出图 |
| `sd3.5_large_turbo.safetensors` | 15.33GB | SD3.5 Large Turbo，质量高但更吃显存 |
| `Juggernaut-XI-byRunDiffusion.safetensors` | 6.62GB | SDXL 写实/通用，推荐优先试 |
| `Juggernaut-XI-byRunDiffusion-Lightning.safetensors` | 6.46GB | SDXL Lightning，少步数快速出图 |
| `RealVisXL_V5.0_fp16.safetensors` | 6.46GB | SDXL 写实人像/产品图 |
| `Illustrious-XL-v1.1.safetensors` | 6.46GB | SDXL 插画/二次元/艺术风格 |
| `DreamShaperXL_Turbo_v2_1.safetensors` | 6.46GB | SDXL Turbo，适合快速试提示词 |

本机 `models\loras` 里已有：

| LoRA | 大小 | 用途 |
| --- | ---: | --- |
| `sdxl_lightning_4step_lora.safetensors` | 376MB | 给普通 SDXL checkpoint 加速到 4 步左右 |

本机已删除旧 SD1.5 模型：

```text
D:\workspace\ComfyUI\models\checkpoints\v1-5-pruned-emaonly.safetensors
```

## 下载和更新模型

进入 ComfyUI 目录：

```powershell
cd D:\workspace\ComfyUI
```

使用 Hugging Face 官方源下载：

```powershell
$env:HF_HUB_DISABLE_XET = "1"
.\venv\Scripts\hf.exe download RunDiffusion/Juggernaut-XI-v11 Juggernaut-XI-byRunDiffusion.safetensors --local-dir .\models\checkpoints
```

如果官方源慢，可以尝试镜像源：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
.\venv\Scripts\hf.exe download <repo> <file> --local-dir <target-dir>
```

切回官方源：

```powershell
Remove-Item Env:\HF_ENDPOINT -ErrorAction SilentlyContinue
```

推荐模型下载命令：

```powershell
cd D:\workspace\ComfyUI
$env:HF_HUB_DISABLE_XET = "1"

.\venv\Scripts\hf.exe download RunDiffusion/Juggernaut-XI-v11 Juggernaut-XI-byRunDiffusion.safetensors --local-dir .\models\checkpoints

.\venv\Scripts\hf.exe download RunDiffusion/Juggernaut-XI-Lightning Juggernaut-XI-byRunDiffusion-Lightning.safetensors --local-dir .\models\checkpoints

.\venv\Scripts\hf.exe download SG161222/RealVisXL_V5.0 RealVisXL_V5.0_fp16.safetensors --local-dir .\models\checkpoints

.\venv\Scripts\hf.exe download OnomaAIResearch/Illustrious-XL-v1.1 Illustrious-XL-v1.1.safetensors --local-dir .\models\checkpoints

.\venv\Scripts\hf.exe download Lykon/dreamshaper-xl-v2-turbo DreamShaperXL_Turbo_v2_1.safetensors --local-dir .\models\checkpoints

.\venv\Scripts\hf.exe download ByteDance/SDXL-Lightning sdxl_lightning_4step_lora.safetensors --local-dir .\models\loras
```

删除旧模型示例：

```powershell
Remove-Item -LiteralPath D:\workspace\ComfyUI\models\checkpoints\v1-5-pruned-emaonly.safetensors
```

本机已删除：

```text
D:\workspace\ComfyUI\models\checkpoints\v1-5-pruned-emaonly.safetensors
```

## 推荐参数

SDXL 普通 checkpoint，例如 Juggernaut XI、RealVisXL、Illustrious XL：

```text
分辨率：1024x1024，832x1216，1216x832
步数：20-30
CFG：5-7
Sampler：DPM++ 2M 或 DPM++ 2M SDE
Scheduler：karras 或 normal
```

SDXL Lightning checkpoint 或 Lightning LoRA：

```text
分辨率：1024x1024
步数：4-8
CFG：1-2
Sampler：Euler 或 DPM++ 2M
```

SD3.5 Medium Turbo：

```text
分辨率：768x768 或 1024x1024
步数：4
CFG：1.0-1.5
Sampler：euler
Scheduler：sgm_uniform
```

RTX 3060 12GB 建议：

- 优先使用 fp16 checkpoint。
- 普通 SDXL 从 1024x1024 开始。
- 显存不足时降低分辨率，或继续使用启动参数里的 `--lowvram`。
- 不要同时加载多个大模型；切模型后首次生成会慢一些。

## 更新 ComfyUI 程序

```powershell
cd D:\workspace\ComfyUI
git pull
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

更新后重启服务：

```powershell
run_comfyui_nvidia.bat
```

## 常用排查

查看 checkpoint 是否被 ComfyUI 识别：

```powershell
curl.exe -s http://127.0.0.1:8188/object_info/CheckpointLoaderSimple
```

查看 LoRA 是否被识别：

```powershell
curl.exe -s http://127.0.0.1:8188/object_info/LoraLoader
```

查看服务日志：

```powershell
Get-Content D:\workspace\ComfyUI\comfyui.err.log -Tail 100
Get-Content D:\workspace\ComfyUI\comfyui.out.log -Tail 100
```

输出图片目录：

```text
D:\workspace\ComfyUI\output
```
