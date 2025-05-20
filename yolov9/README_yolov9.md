# 自訂 YOLOv9 模組說明 / Customized YOLOv9 Module

## 📌 中文說明

本資料夾為本專案專用的 YOLOv9 模型程式碼，原始版本來自：
[WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

在此基礎上，已針對專案需求進行下列修改：

- ✅ 整合 OpenPose 姿勢估測功能
- ✅ 新增 `assistive_yolo_pose.py`，支援同時進行物件偵測與姿態辨識
- ✅ 優化輸出格式與後處理流程，以適應 FastAPI 後端整合
- ✅ 調整模型參數與輸出畫面以符合公車站行動輔助者偵測需求

如需執行或修改 YOLOv9 模型，請從本資料夾進行，而非原始官方版本。

---

## 📌 English Description

This folder contains a **customized version of YOLOv9**, based on  
[WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9), adapted specifically for this project.

### 🔧 Modifications include:

- ✅ Integration of OpenPose for human pose estimation
- ✅ Added `assistive_yolo_pose.py` to enable simultaneous object and pose detection
- ✅ Customized output format for use with FastAPI backend
- ✅ Fine-tuned configurations for detecting mobility-assistive users at bus stops

Please use this version of YOLOv9 for any development or execution within this repository.
