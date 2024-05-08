from ultralytics import YOLO

# Load a model
model = YOLO('/home/ymm/v8/ultralytics_relu/weights/cms_p2_480_640/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx',
             # imgsz=[640, 640],
             imgsz=[480,640],
             opset=12,
             simplify=True)