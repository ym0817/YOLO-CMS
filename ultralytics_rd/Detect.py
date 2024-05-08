import torch
import os,cv2
from ultralytics import YOLO



# data = "score_data.yaml"

img_dir = '/home/ymm/v8/ultralytics_relu/test_imgs'
model_path = '/home/ymm/v8/ultralytics_relu/weights/face_plate_480_640/best.pt'
model = YOLO(model_path,task = "detect")

test_files = [os.path.join(img_dir,d) for d in os.listdir(img_dir)]
input_ims = [cv2.imread(file) for file in test_files]

results = model.predict(source=input_ims, save=True, save_txt=True)  # 将预测保存为标签

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])


# success = modeln.export(format="onnx")
# model.val(data=data,
#         device='cpu',
#         imgsz=[480,640],
#         # imgsz=[416,416],
#         batch=4
#           )