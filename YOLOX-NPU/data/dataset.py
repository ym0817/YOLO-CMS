# -*- coding: utf-8 -*-
# @Time    : 21-8-15 :05
# @Author  : MingZhang
# @Email   : zm19921120@126.com


import io
import os
import cv2
import json
import random
import contextlib
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from collections import defaultdict
import copy
import time
import datetime
from data.data_augment import random_perspective, box_candidates, TrainTransform

# for person det, from coco APIs
class Test_COCOeval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))
        imgs = self.cocoGt.loadImgs(p.imgIds)
        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        # for gt in gts:
            # print('gt', gt)
            # gt['width'] = imgs[gt['image_id']-1]['width']
            # gt['height'] = imgs[gt['image_id']-1]['height']
            # gt['file_name'] = imgs[gt['image_id']-1]['file_name']
            # self._gts[gt['image_id'], gt['category_id']].append(gt)
        # for dt in dts:
            # dt['width'] = imgs[dt['image_id']-1]['width']
            # dt['height'] = imgs[dt['image_id']-1]['height']
            # dt['file_name'] = imgs[dt['image_id']-1]['file_name']
            # self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<(aRng[0]*g['width']*g['height']) or g['area']>(aRng[1]*g['width']*g['height'])):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<(aRng[0]*d['width']*d['height']) or d['area']>(aRng[1]*d['width']*d['height']) for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRng = [[0.0077, 1.0],[0.0077, 0.02], [0.02, 0.08], [0.08, 1.0]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None



class COCODataset(torch.utils.data.Dataset):

    def __init__(self, cfg, img_size=(640, 640), name="train2017", json_file="instances_train2017.json", preproc=None,
                 no_aug=True, tracking=False, logger=None):

        super(COCODataset, self).__init__()
        self.opt = cfg
        self.img_size = img_size
        self.name = name
        self.json_file = json_file
        self.preproc = preproc
        self.augment = not no_aug
        self.tracking = tracking
        self.logger = logger
        self.data_dir = self.opt.data_dir
        self.batch_size = self.opt.batch_size

        # data augment params
        self.random_size = self.opt.random_size
        self.degrees = self.opt.degrees
        self.translate = self.opt.translate
        self.scale = self.opt.scale
        self.shear = self.opt.shear
        self.perspective = self.opt.perspective
        self.mixup_scale = (0.5, 1.5)
        self.enable_mosaic = self.augment
        self.enable_mixup = self.opt.enable_mixup
        self.mosaic_prob = self.opt.mosaic_prob
        self.mixup_prob = self.opt.mixup_prob

        #################
        # self.json_file = self.json_file.replace("train", "val")
        # self.name = self.name.replace("train", "val")
        #################
        assert os.path.isfile(self.json_file), 'cannot find {}'.format(self.json_file)
        print("==> Loading {} annotation {}".format(self.name, self.json_file))
        self.coco = COCO(self.json_file)
        self.ids = self.coco.getImgIds()
        self.num_samples = len(self.ids)
        print("images number {}".format(self.num_samples))
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.annotations = self._load_coco_annotations()
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]

        if 'val2017' == self.name:
            print("classes index:", self.class_ids)
            print("class names in dataset:", self.classes)

    def __len__(self):
        return self.num_samples

    def shuffle(self):
        np.random.shuffle(self.annotations)
        print("shuffle images list in {}".format(self.json_file))
        if self.logger:
            self.logger.write("shuffle {} images list...\n".format(self.json_file))

        if self.random_size is not None:
            self.samples_shapes = self.multi_shape()

    def multi_shape(self):
        size_factor = self.img_size[1] * 1. / self.img_size[0]

        multi_shapes = []
        for size in list(range(*self.random_size)):
            random_input_h, random_input_w = (int(32 * size), 32 * int(size * size_factor))
            multi_shapes.append([random_input_h, random_input_w])
        print("multi size training: {}".format(multi_shapes))
        if self.logger:
            self.logger.write("multi size training: {}\n".format(multi_shapes))

        iter_num = int(np.ceil(self.num_samples / self.batch_size))
        samples_shapes = []
        rand_idx = len(multi_shapes) - 1  # initialize with max size, in case of out of memory during training
        for it in range(iter_num):
            if it != 0 and it % 10 == 0:
                rand_idx = np.random.choice(list(range(len(multi_shapes))))
            for _ in range(self.batch_size):
                samples_shapes.append(multi_shapes[rand_idx])
        return samples_shapes

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_img_res = all_bboxes[image_id]
            for res in one_img_res:
                cls, conf, bbox = res[0], res[1], res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self.class_ids[self.classes.index(cls)],
                    'image_id': int(image_id),
                    'score': float(conf)})
        return detections

    def run_coco_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_det = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        str_result = redirect_string.getvalue()
        ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large = coco_eval.stats[:6]
        print(str_result)
        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6 if self.tracking else 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.tracking:
                assert "tracking_id" in obj.keys(), 'cannot find "tracking_id" in your dataset'
                res[ix, 5] = obj['tracking_id']
                # print('errorrrrrrrr: replace tracking_id to cls')
                # res[ix, 5] = cls

        img_info = (height, width)
        file_name = im_ann["file_name"]

        del im_ann, annotations

        return res, img_info, file_name, id_

    def pull_item(self, index):
        res, img_info, file_name, id_ = self.annotations[index]
        # load image and preprocess
        img_file = self.data_dir + "/" + file_name #
        img = cv2.imread(img_file)
        assert img is not None, "error img {}".format(img_file)

        return img, res.copy(), img_info, id_

    def close_random_size(self):
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]
        print("close multi-size training")

    def __getitem__(self, idx):

        if self.enable_mosaic and self.augment and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_h, input_w = self.samples_shapes[idx]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, self.num_samples - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, _ = self.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
            
            # crop_xyxy = [int(random.uniform(0, 0.5*input_w)),int(random.uniform(0, 0.5*input_h)),
            #              int(random.uniform(1.5*input_w, 2.0*input_w)),int(random.uniform(1.5*input_h, 2.0*input_h))]
            # mosaic_img, mosaic_labels = self.merge_mosaic_box(mosaic_img, mosaic_labels, crop_xyxy)
            
            # 仿射变换
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            if self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            return mix_img, padded_labels, img_info, -1
        else:
            img, label, img_info, img_id = self.pull_item(idx)
            # if random.random() < 0.5:
            #     img, label = self.mixup(img, label, self.samples_shapes[idx])
            img, label = self.preproc(img, label, self.samples_shapes[idx])
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        img, cp_labels, _, _ = self.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            if self.tracking:
                tracking_id_labels = cp_labels[keep_list, 5:6].copy()
                labels = np.hstack((box_labels, cls_labels, tracking_id_labels))
            else:
                labels = np.hstack((box_labels, cls_labels))
            labels_mask = np.logical_and((labels[:,2]-labels[:,0])/target_w>0.021,  (labels[:,3]-labels[:,1])/target_h>0.0926)#要求最小检测人体w*h归一化值, 取半值滤除小目标.
            origin_labels = np.vstack((origin_labels, labels[labels_mask]))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
    
    def merge_mosaic_box(self, mosaic_img, mosaic_labels, crop_xyxy):
        h, w, _ = mosaic_img.shape
        x1,y1,x2,y2 = crop_xyxy
        mask = np.zeros(mosaic_labels.shape[0]).astype(np.bool)
        for idx in range(mosaic_labels.shape[0]):
            box = mosaic_labels[idx, :4]
            # 内部box保留
            if box[1] > y1 and box[0] > x1 and box[3] < y2 and box[2] < x2:
                mask[idx] = True
                continue
            # 无交集box, 完整度低的box直接抛弃
            if box[1] > y2 or box[0] > x2 or box[3] < y1 or box[2] < x1 or \
                (box[1] < y1 and (y1-box[1])/(box[3]-box[1]) > 0.1) or \
                    (box[3] > y2 and (box[3]-y2)/(box[3]-box[1]) > 0.2) or \
                        (box[0] < x1 and (x1-box[0])/(box[2]-box[0]) > 0.2) or \
                            (box[2] > x2 and (box[2]-x2)/(box[2]-box[0] > 0.2)): 
                mask[idx] = False
                continue
    
            mosaic_labels[idx, :4] = box
            mask[idx] = True
        resize_mosaic_img = mosaic_img[y1:y2, x1:x2, :]
        merge_labels = mosaic_labels[mask]
        
        np.clip(merge_labels[:, 0] - x1, 0, x2-x1, out=merge_labels[:, 0])
        np.clip(merge_labels[:, 1] - y1, 0, y2-y1, out=merge_labels[:, 1])
        np.clip(merge_labels[:, 2] - x1, 0, x2-x1, out=merge_labels[:, 2])
        np.clip(merge_labels[:, 3] - y1, 0, y2-y1, out=merge_labels[:, 3])
        
        # vis_img = resize_mosaic_img.copy()
        # for box in merge_labels:
        #     cv2.rectangle(vis_img,(int(box[0]),int(box[1])),(int(box[2]), int(box[3])),(0,0,255),3)
        
        return resize_mosaic_img, merge_labels  

def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def get_dataloader(opt, no_aug=False, logger=None, val_loader=True):
    do_tracking = opt.reid_dim > 0
    # train
    datasets = []
    datasets_path = [os.path.join(opt.train_ann, ann) for ann in os.listdir(opt.train_ann)] if opt.is_dataset_dir else [opt.train_ann]
    for dataset in datasets_path:
        train_data = COCODataset(opt,
                                    img_size=opt.input_size,
                                    name='train2017',
                                    json_file=dataset,
                                    preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                        tracking=do_tracking, augment=True),
                                    no_aug=no_aug,
                                    tracking=do_tracking,
                                    logger=logger,
                                    )
        datasets.append(train_data)
    train_dataset = torch.utils.data.ConcatDataset(datasets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,  # cocat dataset, shuffle here
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=True
    )

    if not val_loader:
        return train_loader, None

    # val
    val_dataset = COCODataset(opt,
                              img_size=opt.test_size,
                              name='val2017',
                              json_file=opt.val_ann,
                              preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                     tracking=do_tracking, augment=False),
                              no_aug=True,
                              tracking=do_tracking,
                              logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader


def vis_inputs(inputs, targets, opt):
    from utils.util import label_color

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    for b_i, inp in enumerate(inputs):
        target = targets[b_i]
        img = (((inp.transpose((1, 2, 0)) * opt.std) + opt.rgb_means) * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        gt_n = 0
        for t in target:
            if t.sum() > 0:
                if len(t) == 5:
                    cls, c_x, c_y, w, h = [int(i) for i in t]
                    tracking_id = None
                elif len(t) == 6:
                    cls, c_x, c_y, w, h, tracking_id = [int(i) for i in t]
                else:
                    raise ValueError("target shape != 5 or 6")
                bbox = [c_x - w // 2, c_y - h // 2, c_x + w // 2, c_y + h // 2]
                label = opt.label_name[cls]
                # print(label, bbox)
                color = label_color[cls]
                # show box
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # show label and conf
                txt = '{}-{}'.format(label, tracking_id) if tracking_id is not None else '{}'.format(label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color,
                              -1)
                cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                gt_n += 1

        print("img {}/{} gt number: {}".format(b_i, len(inputs), gt_n))
        cv2.namedWindow("input", 0)
        cv2.imshow("input", img)
        key = cv2.waitKey(0)

        if key == 27:
            exit()


def run_epoch(train_loader, e, opt):
    for batch_i, batch in enumerate(train_loader):
        inps, targets, img_info, ind = batch
        print("------------ epoch {} batch {}/{} ---------------".format(e, batch_i, len(train_loader)))
        print("batch img shape {}, target shape {}".format(inps.shape, targets.shape))
        if opt.show:
            vis_inputs(inps, targets, opt)
        if batch_i >= 21:
            break


def main():
    from config import opt

    # opt.dataset_path = r"D:\work\public_dataset\coco2017"
    opt.dataset_path = "D:\Workspace\CMS\ZG_A\YOLOX2\YOLOX-NPU\coco2017"
    opt.train_ann = opt.dataset_path + "\annotations\instances_train2017.json"
    opt.val_ann = opt.dataset_path + "\annotations\instances_val2017.json"
    opt.data_dir = opt.dataset_path + "\images"

    opt.input_size = (640, 640)
    opt.test_size = (640, 640)
    opt.batch_size = 2
    opt.data_num_workers = 2  # 0
    opt.reid_dim = 0  # 128
    opt.show = True  # False

    train_loader, val_loader = get_dataloader(opt, no_aug=False)

    # train_loader = val_loader
    dataset_label = train_loader.dataset.classes
    assert opt.label_name == dataset_label, "your class_name != dataset's {} {}".format(opt.label_name, dataset_label)
    for e in range(100):
        train_loader.dataset.shuffle()
        if e == 2:
            train_loader.dataset.enable_mosaic = False
        run_epoch(train_loader, e, opt)


if __name__ == "__main__":
    main()
