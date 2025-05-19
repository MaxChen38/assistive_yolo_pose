# backend/api_server.py
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import JSONResponse
import base64, cv2, numpy as np
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from yolov9.assistive_yolo_pose_modified_v3_openvino import run_inference_from_numpy

app = FastAPI()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    print("🟢 HTTP 收到圖片，開始推論...")
    result_img, result_data = run_inference_from_numpy(image)
    _, buffer = cv2.imencode(".jpg", result_img)
    b64_img = base64.b64encode(buffer).decode("utf-8")
    alert_flag = any(d['label'] in ['wheelchair', 'cane', 'walker'] for d in result_data)

    return JSONResponse(content={
        "image": b64_img,
        "detections": result_data,
        "alert": alert_flag
    })

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket 已建立連線，等待 frame...")
    while True:
        try:
            data = await websocket.receive_text()
            if not data:
                print("⚠️ 收到空的 WebSocket 資料，跳過")
                continue

            print("🟢 WebSocket 收到圖片，開始解碼")
            img_data = base64.b64decode(data)
            np_img = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if frame is None:
                print("❌ 解碼失敗：cv2.imdecode 回傳 None")
                await websocket.send_json({"error": "Invalid image format"})
                continue

            print("🟢 成功解碼，開始推論")
            start_time = time.time()
            result_img, result_data = run_inference_from_numpy(frame)
            inference_time = time.time() - start_time

            _, buffer = cv2.imencode(".jpg", result_img)
            b64_result = base64.b64encode(buffer).decode("utf-8")
            alert_flag = any(d['label'] in ['wheelchair', 'cane', 'walker'] for d in result_data)

            await websocket.send_json({
                "image": b64_result,
                "detections": result_data,
                "alert": alert_flag,
                "inference_time": inference_time
            })

        except Exception as e:
            print("❌ WebSocket 錯誤：", str(e))
            await websocket.send_json({"error": str(e)})
            break