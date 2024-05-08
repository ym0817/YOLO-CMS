import torch
from ultralytics import YOLO



data = "score_data.yaml"

# model_t.model.model[-1].set_Distillation = True

# model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)
# "sftp://192.168.67.16/home/cms/Distillation/ultralytics/models/v8/yolov8l.yaml"
# model = YOLO('/home/ymm/v8/ultralytics_relu/weights/run_a_416/best.pt',task = "detect")        #.load('yolov8n.pt')
model = YOLO('/home/ymm/v8/ultralytics_relu/weights/cms_p2_480_640/best.pt',task = "detect")
# model.eval()
# success = modeln.export(format="onnx")
model.val(data=data,
        device='cpu',
        imgsz=[480,640],
        # imgsz=[416,416],
        batch=4
          )

# model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)
# model_s.train(data=data,
#               epochs=300,
#               device='cpu',
#               workers=2,
#               # imgsz=640,
#               imgsz=[640,480],
#               batch=2,
#               augment = True,
#               resume = False
#               )



