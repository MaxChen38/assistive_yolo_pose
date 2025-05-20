#!/bin/bash

echo "✅ 啟動 Assistive YOLO + OpenPose 系統"

# 啟動 FastAPI 後端
cd backend
echo "🚀 啟動後端 (FastAPI + OpenPose + YOLOv9 OpenVINO)"
gnome-terminal -- bash -c "uvicorn api_server:app --host 0.0.0.0 --port 8000; exec bash"
cd ..

# 啟動前端
cd frontend
echo "🖼️ 啟動前端 (Next.js)"
gnome-terminal -- bash -c "npm run dev; exec bash"
