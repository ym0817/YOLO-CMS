nohup yolo task=detect \
mode=train \
data='D.yaml'  \
device=0 \
epochs=400 \
batch=22 \
workers=4 \
imgsz=[480,640] \
model='/mnt2T/YM/Train/ultralytics_r/runs/detect/train6/weights/last.pt'  \
augment = True \
resume=True \
optimizer=SGD \
> nohup1.out 2>&1 &
tail -f nohup1.out

