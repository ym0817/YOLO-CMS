nohup yolo task=detect \
mode=train \
data=face_data.yaml \
device=0 \
epochs=350 \
batch=20 \
workers=4 \
imgsz=[480,640] \
model='yolov8n.yaml'  \
augment = True \
resume=False \
optimizer=SGD > nohup.out 2>&1 &
tail -f nohup.out

