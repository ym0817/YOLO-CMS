from ultralytics import YOLO

# Load a model
model = YOLO('/home/ymm/v8/YOLOv8-main-2023/runs_coco/detect/train/weights/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx',opset=11, simplify=True)