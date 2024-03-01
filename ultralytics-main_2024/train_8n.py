import torch
import pdb
from ultralytics import YOLO

# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train32/weights/best.pt')
# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/coco_v8l/weights/best.pt')
# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('yolov8l.pt')
# success = modelL.export(format="onnx",device="cpu")

data = "coco128.yaml"

# model_t.model.model[-1].set_Distillation = True

# model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)
# "sftp://192.168.67.16/home/cms/Distillation/ultralytics/models/v8/yolov8l.yaml"
model_s = YOLO('yolov8n-spd.yaml',task = "detect")        #.load('yolov8n.pt')
# model_s = YOLO('/home/ymm/v8/ultralytics-main_2024/runs/detect/train/weights/last.pt',task = "detect")

# success = modeln.export(format="onnx")
# modelL.val(data=data)

# model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)
model_s.train(data=data,
              epochs=300,
              device='cpu',
              workers=2,
              imgsz=640,
              batch=2,
              augment = True,
              resume = False
              )



