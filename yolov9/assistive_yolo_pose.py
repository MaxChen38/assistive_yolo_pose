# assistive_yolo_pose_modified_v3.py
# 功能：支援 webcam 修正、原圖儲存、YOLO 標註、即時顯示與姿態

import argparse
import os
import platform
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
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / 'yolo.pt',
    source=ROOT / 'data/images',
    data=ROOT / 'data/coco.yaml',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=False,
    save_txt=True,
    save_conf=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / 'runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    use_openpose=False
):
    source = str(source)
    save_img = not nosave
    webcam = source.isnumeric()
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    raw_dir = save_dir / 'raw_images'
    label_dir = save_dir / 'labels'
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=augment, visualize=False)
            pred = pred[0][1]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if isinstance(path, list):
                p = Path(path[i])
                im0 = im0s[i].copy()
            else:
                p = Path(path)
                im0 = im0s.copy()

            filename = f"frame_{seen:06d}"
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            cv2.imwrite(str(raw_dir / f"{filename}.jpg"), im0)

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                with open(label_dir / f"{filename}.txt", 'w') as f:
                    for *xyxy, conf, cls in reversed(det):
                        if conf < 0.6:
                            continue
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        label = None if hide_labels else (names[int(cls)] if hide_conf else f'{names[int(cls)]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

            if use_openpose:
                datum = op.Datum()
                datum.cvInputData = im0
                datum_vector = op.VectorDatum()
                datum_vector.append(datum)
                opWrapper.emplaceAndPop(datum_vector)
                im0 = datum.cvOutputData

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    return

    LOGGER.info(f"結果已儲存至 {save_dir}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=ROOT / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=3, type=int)
    parser.add_argument('--hide-labels', action='store_true')
    parser.add_argument('--hide-conf', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    parser.add_argument('--use-openpose', action='store_true')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
