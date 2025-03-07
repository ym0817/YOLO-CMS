# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 21:50
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import tqdm
import json
import numpy as np
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval

from config import opt
from utils.util import NpEncoder
from models.yolox import Detector
from data.dataset import Test_COCOeval

def evaluate():
    detector = Detector(opt)
    gt_ann = opt.val_ann if "test_ann" not in opt.keys() else opt.test_ann
    img_dir = opt.data_dir#opt.dataset_path + "/images/" + ("test2017" if "test" in os.path.basename(gt_ann) else "val2017")
    batch_size = opt.batch_size

    assert os.path.isfile(gt_ann), 'cannot find gt {}'.format(gt_ann)
    coco = coco_.COCO(gt_ann)
    images = coco.getImgIds()
    class_ids = sorted(coco.getCatIds())
    num_samples = len(images)
    
    print("==>> evaluating batch_size={}".format(batch_size))
    print('find {} samples in {}'.format(num_samples, gt_ann))

    result_file = "result_{}_{}.json".format(opt.backbone, opt.test_size[0])
    coco_res = []
    samples_idx = list(range(num_samples))
    iterations = int(np.ceil(num_samples / float(batch_size)))
    for its in tqdm.tqdm(range(iterations)):
        batch_index = samples_idx[its * batch_size: (its + 1) * batch_size]
        batch_images = []
        batch_img_ids = []
        for index in batch_index:
            img_id = images[index]
            file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
            image_path = img_dir + "/" + file_name
            # image_path = file_name#img_dir + "/" + file_name
            assert os.path.isfile(image_path), "cannot find img {}".format(image_path)
            img = cv2.imread(image_path)
            if img is not None:
                batch_images.append(img)
                batch_img_ids.append(img_id)

        batch_results = detector.run(batch_images, vis_thresh=0.01)

        for index in range(len(batch_images)):
            results = batch_results[index]
            img_id = batch_img_ids[index]
            for res in results:
                cls, conf, bbox = res[0], res[1], res[2]
                if len(res) > 3:
                    reid_feat = res[4]
                cls_index = opt.label_name.index(cls)
                coco_res.append(
                    {'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                     'category_id': class_ids[cls_index],
                     'image_id': int(img_id),
                     'score': conf})

    with open(result_file, 'w') as f_dump:
        json.dump(coco_res, f_dump, cls=NpEncoder)
    '''
    if "test" in os.path.basename(gt_ann):
        print("save result to {}, you can zip the result and upload it to COCO website"
              "(https://competitions.codalab.org/competitions/20794#participate)".format(result_file))
        try:
            zip_file = result_file.replace(".json", ".zip")
            os.system("zip {} {}".format(zip_file, result_file))
            print("--> create upload file done: {}".format(zip_file))
        except:
            print("please zip it before uploading")
        return
   '''
    coco_det = coco.loadRes(result_file)#('{}/results.json'.format(opt.save_dir))#
    coco_eval = Test_COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Each class is evaluated separately
    # classes = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    # for i, cat_id in enumerate(class_ids):
    #     print('-------- evaluate class: {} --------'.format(classes[i]))
    #     coco_eval.params.catIds = cat_id
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
    # os.remove(result_file)

def my_evaluate():
    # detector = Detector(opt)
    gt_ann = opt.val_ann if "test_ann" not in opt.keys() else opt.test_ann
    # img_dir = opt.data_dir#opt.dataset_path + "/images/" + ("test2017" if "test" in os.path.basename(gt_ann) else "val2017")
    # batch_size = opt.batch_size

    assert os.path.isfile(gt_ann), 'cannot find gt {}'.format(gt_ann)
    coco = coco_.COCO(gt_ann)
    images = coco.getImgIds()
    class_ids = sorted(coco.getCatIds())
    num_samples = len(images)

    result_file = "result_{}_{}.json".format(opt.backbone, opt.test_size[0])
    coco_det = coco.loadRes(result_file)  # ('{}/results.json'.format(opt.save_dir))#
    coco_eval = Test_COCOeval(coco, coco_det, 'bbox')
    # print(coco.anns)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def get_voc_eval():
    # opt.label_name = ['person', 'car', 'truck', 'bicycle', 'bus', 'motorcycle']
    _list0, _list1, _list2, _list3, _list4, _list5= [] ,[], [],[],[], []

    imagespath = r"D:\Workspace\CMS\ZG_A\YOLOX2\YOLOX-NPU\generate_COCO\images\Val2017"
    file_list = [os.path.join(imagespath, file) for file in os.listdir(imagespath)]
    num_samples = len(file_list)
    result_file = "voc_result_{}_{}.json".format(opt.backbone, opt.test_size[0])
    coco_res = []
    detector = Detector(opt)
    batch_size = opt.batch_size
    samples_idx = list(range(num_samples))
    iterations = int(np.ceil(num_samples / float(batch_size)))
    for its in tqdm.tqdm(range(iterations)):
        batch_index = samples_idx[its * batch_size: (its + 1) * batch_size]
        batch_images = []
        batch_img_ids = []
        for index in batch_index:
            # file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
            # image_path = img_dir + "/" + file_name
            image_path = file_list[index]
            assert os.path.isfile(image_path), "cannot find img {}".format(image_path)
            img = cv2.imread(image_path)
            if img is not None:
                batch_images.append(img)
                batch_img_ids.append(index)

        batch_results = detector.run(batch_images, vis_thresh=0.01)

        for bi in range(len(batch_images)):
            results = batch_results[bi]
            img_id = batch_img_ids[bi]
            for res in results:
                cls, conf, bbox = res[0], res[1], res[2]
                if len(res) > 3:
                    reid_feat = res[4]
                cls_index = opt.label_name.index(cls)
                file_name = os.path.splitext(os.path.basename(file_list[img_id]))[0]
                temp = [file_name, conf, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                if cls_index == 0:
                    _list0.append(temp)
                if cls_index == 1:
                    _list1.append(temp)
                if cls_index == 2:
                    _list2.append(temp)
                if cls_index == 3:
                    _list3.append(temp)
                if cls_index == 4:
                    _list4.append(temp)
                if cls_index == 5:
                    _list5.append(temp)


    voc_res ={ opt.label_name[0]: _list0,
               opt.label_name[1]: _list1,
               opt.label_name[2]: _list2,
               opt.label_name[3]: _list3,
               opt.label_name[4]: _list4,
               opt.label_name[5]: _list5 }

    with open(result_file, 'w') as f_dump:
        json.dump(voc_res, f_dump, cls=NpEncoder)




if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    # evaluate()
    # my_evaluate()

    get_voc_eval()
