@echo off
cd /d "%~dp0"
echo Starting Fayetteville 2-Yr Stormwater 3D Viewer on http://127.0.0.1:8765 ...
start "" "http://127.0.0.1:8765"
python -m http.server 8765 --bind 127.0.0.1
