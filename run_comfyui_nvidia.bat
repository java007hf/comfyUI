@echo off
cd /d "%~dp0"
venv\Scripts\python.exe main.py --listen 0.0.0.0 --port 8188 --lowvram --preview-method auto --disable-auto-launch --disable-api-nodes --enable-assets
