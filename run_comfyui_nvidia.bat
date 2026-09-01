@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python main.py --listen 0.0.0.0 --port 8188 --lowvram --preview-method auto --disable-auto-launch --disable-api-nodes --enable-assets
