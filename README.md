# assistive_yolo_pose

這是我整合 YOLOv9 與 OpenPose 的系統，應用於辨識公車站的行動輔助者與姿勢分析...


## 📦 安裝環境需求（Dependencies）

請使用 Python 3.10+ 環境，並執行以下指令安裝所有依賴套件：

```bash
pip install -r requirements.txt


🔒 本專案已排除所有大型模型與資料集（超過 100MB 檔案），請將下列檔案手動放置於本機：

- yolov9/yolov9_openvino/last.bin
- yolov9/datasets/open_dataset/Images_RGB.zip
- yolov9/runs/train/**/weights/*.pt

 或者可使用 Google Drive / HuggingFace 等載點提供補充資源。
