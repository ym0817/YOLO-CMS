nohup yolo task=detect \
mode=train \
data=score_data.yaml \
device=0 \
epochs=400 \
batch=40 \
workers=8 \
imgsz=[480,640] \
model='/mnt2T/YM/Train/ultralytics_r/runs/detect/train4/weights/last.pt'  \
augment = True \
resume=True \
optimizer=SGD \
> nohup1.out 2>&1 &
tail -f nohup1.out

# pretrained = True \