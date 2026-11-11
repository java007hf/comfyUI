@echo off
cd /d "%~dp0..\.." || exit /b 1
if not exist "user_wan_test\" mkdir "user_wan_test" || exit /b 1
venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8190 --user-directory "%CD%\user_wan_test" --database-url "sqlite:///user_wan_test/comfyui.db" --lowvram --preview-method none --disable-auto-launch --disable-api-nodes
