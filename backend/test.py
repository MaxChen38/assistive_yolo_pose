import websocket
import base64
import cv2
import time
import json

# 讀取本地圖片並轉 base64
img = cv2.imread("seq_1477481082.5172983330.jpg")  # ← 你要準備一張 test.jpg
if img is None:
    print("❌ test.jpg 不存在或讀取失敗")
    exit()

_, buffer = cv2.imencode(".png", img)
b64_img = base64.b64encode(buffer).decode("utf-8")

# 回傳訊息時執行
def on_message(ws, message):
    print("✅ 收到伺服器回應 JSON：")
    try:
        parsed = json.loads(message)
        print(json.dumps(parsed, indent=2))
    except:
        print("非 JSON 格式", message[:100])

# WebSocket 啟動時送圖片
def on_open(ws):
    print("🟢 WebSocket 已連線，傳送圖片中...")
    ws.send(b64_img)

ws = websocket.WebSocketApp("ws://localhost:8000/ws/detect", on_message=on_message, on_open=on_open)
ws.run_forever()
