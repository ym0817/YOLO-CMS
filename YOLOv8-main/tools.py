# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/last.pt") # 加载预训练的 YOLOv8n 模型
model.train(data='coco128.yaml') # 训练模型model.val() # 在验证集模型
model.predict(source='https://ultralytics.com/images/bus.jpg') # 对图像进行预测
model.export(format='onnx',opset=13) # 将模型导出为 ONNX 格式
