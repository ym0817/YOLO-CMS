# -*- coding: utf-8 -*-
# @Time    : 21-7-20 20:01
# @Author  : MingZhang
# @Email   : zm19921120@126.com


import numpy as np
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class YOLOXLoss(nn.Module):
    def __init__(self, label_name, reid_dim=0, id_nums=None, strides=[8, 16, 32], in_channels=[256, 512, 1024]):
        super().__init__()

        self.n_anchors = 1
        self.label_name = label_name
        self.num_classes = len(self.label_name)
        self.strides = strides
        self.reid_dim = reid_dim

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.loss_qfl = QualityFocalLoss(reduction="none", loss_weight=10.0)
        # self.loss_dfl = DistributionFocalLoss()
        self.grids = [torch.zeros(1)] * len(in_channels)

        if self.reid_dim > 0:
            assert id_nums is not None, "opt.tracking_id_nums shouldn't be None when reid_dim > 0"
            assert len(id_nums) == self.num_classes, "num_classes={}, which is different from id_nums's length {}" \
                                                     "".format(self.num_classes, len(id_nums))
            # scale_trainable = True
            # self.s_det = nn.Parameter(-1.85 * torch.ones(1), requires_grad=scale_trainable)
            # self.s_id = nn.Parameter(-1.05 * torch.ones(1), requires_grad=scale_trainable)

            self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.classifiers = nn.ModuleList()
            self.emb_scales = []
            for idx, (label, id_num) in enumerate(zip(self.label_name, id_nums)):
                print("{}, tracking label name: '{}', tracking_id number: {}, feat dim: {}".format(idx, label, id_num,
                                                                                                   self.reid_dim))
                self.emb_scales.append(np.math.sqrt(2) * np.math.log(id_num - 1))
                self.classifiers.append(nn.Linear(self.reid_dim, id_num))

    def forward(self, preds, targets, imgs=None):
        outputs, origin_preds, x_shifts, y_shifts, expanded_strides = [], [], [], [], []

        for k, (stride, p) in enumerate(zip(self.strides, preds)):
            pred, grid = self.get_output_and_grid(p, k, stride, p.dtype)

            outputs.append(pred)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.full((1, grid.shape[1]), stride, 
                                    dtype=p.dtype, device=p.device))
            # expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride).type_as(p)) #原

            if self.use_l1:
                reg_output = p[:, :4, :, :]
                batch_size, _, hsize, wsize = reg_output.shape
                reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                reg_output = (reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4))
                origin_preds.append(reg_output.clone())
        
       
        outputs = torch.cat(outputs, 1)
        total_loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, num_fg = self.get_losses(imgs, x_shifts,
                                                                                                y_shifts,
                                                                                                expanded_strides,
                                                                                                targets, outputs,
                                                                                                origin_preds,
                                                                                                dtype=preds[0].dtype)
       
        losses = {"loss": total_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "iou_loss": iou_loss}
        if self.use_l1:
            losses.update({"l1_loss": l1_loss})
        if self.reid_dim > 0:
            losses.update({"reid_loss": reid_loss})
        losses.update({"num_fg": num_fg})
        return losses

    def get_output_and_grid(self, p, k, stride, dtype):
        p = p.clone()
        grid = self.grids[k]
        batch_size, n_ch, hsize, wsize = p.shape

        if grid.shape[2:4] != p.shape[2:4] or grid.device != p.device:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(p.device)
            self.grids[k] = grid

        pred = p.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        pred = (pred.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1))
        grid = grid.view(1, -1, 2)
        pred[..., :2] = (pred[..., :2] + grid) * stride
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * stride
        return pred, grid

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, targets, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, h*w, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, h*w, 1]
        cls_preds = outputs[:, :, 5:self.num_classes + 5]  # [batch, h*w, n_cls]
        if self.reid_dim > 0:
            reid_preds = outputs[:, :, self.num_classes + 5:]  # [batch, h*w, 128]

        assert targets.shape[2] == 6 if self.reid_dim > 0 else 5
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        reid_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                reid_target = outputs.new_zeros((0, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_classes = targets[batch_idx, :num_gt, 0]
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                if self.reid_dim > 0:
                    gt_tracking_id = targets[batch_idx, :num_gt, 5]
                bboxes_preds_per_image = bbox_preds[batch_idx, :, :]

                try:
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs,
                    )
                except RuntimeError:
                    print(traceback.format_exc())
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.reid_dim > 0:
                    reid_target = gt_tracking_id[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            if self.reid_dim > 0:
                reid_targets.append(reid_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        if self.reid_dim > 0:
            reid_targets = torch.cat(reid_targets, 0).type(torch.int64)

        num_fg = max(num_fg, 1) 
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg #(varifocal_loss(obj_preds.view(-1, 1), obj_targets, reduction='none', iou_weighted=False, use_sigmoid=True)).sum() / num_fg   #torch.tensor(0.0,device=loss_iou.device)#
        # loss_cls = torch.tensor(0.0,device=loss_iou.device)#(self.loss_qfl(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg if self.use_l1 else 0.

        reid_loss = 0.
        if self.reid_dim > 0:
            reid_feat = reid_preds.view(-1, self.reid_dim)[fg_masks]
            cls_label_targets = cls_targets.max(1)[1]
            for cls in range(self.num_classes):
                inds = torch.where(cls == cls_label_targets)
                if inds[0].shape[0] == 0:
                    continue
                this_cls_tracking_id = reid_targets[inds]
                this_cls_reid_feat = self.emb_scales[cls] * F.normalize(reid_feat[inds])

                reid_output = self.classifiers[cls](this_cls_reid_feat)
                reid_loss += self.reid_loss(reid_output, this_cls_tracking_id)

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + reid_loss
        fg_r = torch.tensor(num_fg / max(num_gts, 1), device=outputs.device, dtype=dtype)
        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, reid_loss, fg_r

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
            bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
            cls_preds, bbox_preds, obj_preds, targets, imgs, mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # if use_vfl:
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor,
                                                                                                1))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = ( cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)# cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                          * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))
        
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        # del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = ((y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        # n_candidate_k = 10
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            # 原 _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, mode='iou'):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        enclosed_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        enclosed_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
        enclosed_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        enclosed_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    iou = area_i / (area_a[:, None] + area_b - area_i)
    if mode != 'giou':
        return iou
    c_en = (enclosed_tl < enclosed_br).type(enclosed_tl.type()).prod(dim=2)
    area_c = torch.prod(enclosed_br - enclosed_tl, 2) * c_en
    giou = iou - (area_c - area_i) / area_c
    return giou


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = area_i / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "ciou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            # 外接圆wh，对角线距离，中心点距离
            enclose_wh = torch.max(c_br - c_tl, torch.zeros_like(c_tl))
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            center_distance = torch.sum(torch.pow((pred[:, :2] - target[:, :2]), 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(pred[:, 2]/torch.clamp(pred[:, 3],min = 1e-6)) - torch.atan(target[:, 2]/torch.clamp(target[:, 3],min = 1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
            ciou = ciou - alpha * v
            loss = 1.0 - ciou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


def varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   use_sigmoid=False,
                   reduction='mean'):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = pred
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    
    loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction)
    return loss

def focal_loss(pred,
               target,
               weight=None,
               alpha=0.25,
               gamma=2.0,
               reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction)
    return loss


def quality_focal_loss(pred, target, beta=2.0, reduction="none"):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
  
    # label denotes the category id, score denotes the quality score
    score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction=reduction
    ) * scale_factor.pow(beta)

    
    
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score - pred_sigmoid
    loss = F.binary_cross_entropy_with_logits(
        pred, score, reduction=reduction
    ) * scale_factor.abs().pow(beta)
    
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum(dim=1, keepdim=False)
    
    return loss

def distribution_focal_loss(pred, label, reduction="none"):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = (
        F.cross_entropy(pred, dis_left, reduction=reduction) * weight_left
        + F.cross_entropy(pred, dis_right, reduction=reduction) * weight_right
    )
    return loss

class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                beta=self.beta,
                reduction=reduction
            )
        else:
            raise NotImplementedError
        return loss_cls

class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(pred, target,reduction=reduction)
        return loss_cls


if __name__ == "__main__":
    from config import opt

    torch.manual_seed(opt.seed)
    opt.reid_dim = 128  # 0
    opt.batch_size = 2
    dummy_input = [torch.rand([opt.batch_size, 85 + opt.reid_dim, i, i]) for i in [64, 32, 16]]
    dummy_target = torch.rand([opt.batch_size, 3, 6 if opt.reid_dim > 0 else 5]) * 20  # [bs, max_obj_num, 6]
    dummy_target[:, :, 0] = torch.randint(10, (opt.batch_size, 3), dtype=torch.int64)
    if opt.reid_dim > 0:
        dummy_target[:, :, 5] = torch.randint(20, (opt.batch_size, 3), dtype=torch.int64)
        opt.tracking_id_nums = []
        for dummy_id_num in range(50, len(opt.label_name) + 50):
            opt.tracking_id_nums.append(dummy_id_num)

    yolox_loss = YOLOXLoss(label_name=opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums)
    print('input shape:', [i.shape for i in dummy_input])
    print("target shape:", dummy_target, dummy_target.shape)

    loss_status = yolox_loss(dummy_input, dummy_target)
    for l in loss_status:
        print(l, loss_status[l])
