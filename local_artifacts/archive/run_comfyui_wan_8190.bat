@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python main.py --listen 127.0.0.1 --port 8190 --user-directory "%~dp0user_wan_test" --database-url sqlite:///C:/workspace/ComfyUI/user_wan_test/comfyui.db --lowvram --preview-method none --disable-auto-launch --disable-api-nodes
