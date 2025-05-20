# assistive_yolo_pose_modified_v3.py
# 功能：支援 webcam 修正、原圖儲存、YOLO 標註、即時顯示與姿態 + API 呼叫

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# ---------- OpenPose 設定 ----------
OPENPOSE_PATH = "/home/user/openpose"
sys.path.append(os.path.join(OPENPOSE_PATH, 'build/python'))
from openpose import pyopenpose as op

params = {
    "model_folder": os.path.join(OPENPOSE_PATH, "models"),
    "model_pose": "BODY_25",
    "hand": False,
    "face": False,
    "disable_multi_thread": True,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
# ---------- OpenPose 結束 ----------

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import (scale_boxes, xyxy2xywh, non_max_suppression, check_img_size)
from utils.torch_utils import select_device
from utils.plots import colors

# ✅ 初始化模型與裝置為全域變數
DEVICE = select_device(0)
MODEL = DetectMultiBackend("/home/user/yolo-openpose-app/yolov9/runs/train/final_ten/weights/last.pt", device=DEVICE)
MODEL.warmup(imgsz=(1, 3, 640, 640))
NAMES = MODEL.names


def run_inference_from_numpy(np_image):
    im0 = np_image.copy()
    im = cv2.resize(im0, (640, 640))
    im = im.transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(DEVICE).float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    pred = MODEL(im)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    result = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                label = NAMES[int(cls)]
                result.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [int(x.item()) for x in xyxy]
                })
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(im0, f"{label} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    datum = op.Datum()
    datum.cvInputData = im0
    datum_vector = op.VectorDatum()
    datum_vector.append(datum)
    opWrapper.emplaceAndPop(datum_vector)
    im0 = datum.cvOutputData  # 如果你需要回傳帶姿態的影像
    # ✅ debug 印出
    print(f"✅ 偵測結果共 {len(result)} 個物件")
    for r in result:
        print(" -", r)

    return im0, result
