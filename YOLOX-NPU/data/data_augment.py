#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random
import cv2
import numpy as np


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def random_perspective(
        img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, rgb_means=None, std=None, tracking=False, max_labels=50, augment=True):
        self.means = rgb_means
        self.std = std
        self.tracking = tracking
        self.max_labels = max_labels
        self.augment = augment
        self.color_augmentor = ColorDistort()

    def __call__(self, image, targets, input_dim):
        assert targets.shape[1] == 6 if self.tracking else 5
        lshape = targets.shape[1]

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if self.tracking:
            tracking_id = targets[:, 5].copy()

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, lshape), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if self.tracking:
            tracking_id_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        # color aug
        if self.augment:
            augment_hsv(image)
            if random.uniform(0,1) > 0.5:
                image = self.color_augmentor(image)
        image_t = image
        # flip
        if self.augment:
            image_t, boxes = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)

        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        if self.tracking:
            tracking_id_t = tracking_id[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if self.tracking:
                tracking_id_t = tracking_id_o

        labels_t = np.expand_dims(labels_t, 1)
        if self.tracking:
            tracking_id_t = np.expand_dims(tracking_id_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, tracking_id_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))

        padded_labels = np.zeros((self.max_labels, lshape))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels


def _clip(image):
    return np.clip(image, 0, 255).astype(np.float32)

def _uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])

def rand_blur(src, flag=0):
    '''
    Apply gassian blur to image.
    Note: function may also return "sharpened" image if flag is not 0
    :param src: input image
    :param flag: determines to return blurred or sharpened image
    :return: blurred or sharpened image
    '''
    #print("src shape",src.shape)
    k_size = random.randint(1, 2) * 2 + 1
    blur = cv2.GaussianBlur(src, (k_size, k_size), 0)
    #blur = np.expand_dims(blur,2)
    sharp = cv2.addWeighted(src, 1.5, blur, -0.5, 0)
    #sharp = np.expand_dims(sharp,2)
    #print("blur shape",blur.shape)
    #print("sharp shape",sharp.shape)

    if flag == 0:
        return blur
    else:
        return sharp

def rand_noise(src, level, flag):
    '''
    Add random noise to input image.
    :param src: input image
    :param level: <float> noise level, always < 1. maximum level * w * h pixels
                  will turn into noise
    :param flag: 0 - add salt-pepper noise
                 1 - add gaussian noise
    :return: image with random noise
    '''
    h, w = src.shape[0:2]
    if flag == 0:           #1 approach - salt&pepper noise
        # add pepper
        noise1 = random.randint(10, int(level/2 * w * h))
        for k in range(noise1):
            ri = random.randint(0, w - 1)
            rj = random.randint(0, h - 1)
            src[rj,ri] = 0
        # add salt
        noise2 = random.randint(10, int(level/2 * w * h))
        for k in range(noise2):
            ri = random.randint(0, w - 1)
            rj = random.randint(0, h - 1)
            src[rj,ri] = 255
        return src
    elif flag != 0:         #2 approach - gaussian noise
        noise3 = random.randint(int(0.02 * w * h), int(level * w * h))
        for k in range(noise3):
            ri = random.randint(0, w - 1)
            rj = random.randint(0, h - 1)
            rv_b = random.randint(0, 255)
            src[rj, ri] = (rv_b)
        return src

def adjust_brightness(image, delta):
    return _clip(image + delta * 255)

def adjust_contrast(image, factor):
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


class ColorDistort():

    def __init__(self, contrast_range=(0.8, 1.2), brightness_range=(-.2, .2), hue_range=(-0.1, 0.1),
                 saturation_range=(0.8, 1.2)):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range

    def __call__(self, image):
        if self.contrast_range is not None:
            contrast_factor = _uniform(self.contrast_range)
            image = adjust_contrast(image, contrast_factor)
        if self.brightness_range is not None:
            brightness_delta = _uniform(self.brightness_range)
            image = adjust_brightness(image, brightness_delta)
        if random.choice([0,1,2]) ==0:
            i=np.random.randint(3)
            if i==0:
                image = rand_blur(image)
            elif i==1:
                image = rand_blur(image, 1)
            elif i==2:
                j = np.random.randint(2)
                image = rand_noise( image, 0.1, j)
            
        if np.random.randint(3) == 1:
            image = image[:,:,::-1]
        
        return image
