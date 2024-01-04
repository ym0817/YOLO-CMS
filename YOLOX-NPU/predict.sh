#!/usr/bin/env bash

# predict images
#python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="../weights/yolox-s.pth" img_dir='/data/dataset/coco_dataset/images/val2017'
python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" img_dir='/data/dataset/coco_dataset/images/val2017'


python predict.py gpus='0' backbone="CSPDarknet-nano" vis_thresh=0.3 load_model="model_best_hs.pth" img_dir='D:\Workspace\CMS\ZG_A\YOLOX2\YOLOX-NPU\coco2017\images'



# predict video
#python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" video_dir='/path/to/your/video.mp4'

# fuse BN into Conv to speed up
#python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" img_dir='/data/dataset/coco_dataset/images/val2017' fuse=True
