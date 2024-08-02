import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops.poolers import LevelMapper


def encode_boxes(reference_boxes, proposals, weights):
    wx, ww = weights
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_x2 = proposals[:, 1].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 1].unsqueeze(1)

    ex_width = proposals_x2 - proposals_x1
    ex_ctr = proposals_x1 + 0.5 * ex_width

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_ctr = reference_boxes_x1 + 0.5 * gt_widths

    targets_dx = wx * (gt_ctr - ex_ctr) / ex_width
    targets_dw = ww * torch.log(gt_widths / ex_width)

    targets = torch.cat((targets_dx, targets_dw), dim=1)
    return targets


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened, box_regression_flattened = [], []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, W = box_cls_per_level.shape
        Ax2 = box_regression_per_level.shape[1]
        A = Ax2 // 2
        C = AxC // A
        box_cls_flattened.append(permute_and_flatten(box_cls_per_level, N, C, W))
        box_regression_flattened.append(permute_and_flatten(box_regression_per_level, N, 2, W))

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 2)

    return box_cls, box_regression


def permute_and_flatten(layer, N, C, W):
    layer = layer.view(N, -1, C, W)
    layer = layer.permute(0, 3, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def box_similarity(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    l = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    r = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    inter = (r - l).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter)


def _setup_scales(features, seq_shapes, canonical_scale, canonical_level):
    max_x = max(seq_shapes)
    scales = [float(torch.tensor(float(feat.shape[-1]) / float(max_x)).log2().round()) for feat in features]
    map_levels = LevelMapper1d(k_min=-scales[0], k_max=-scales[-1], canonical_scale=canonical_scale, canonical_level=canonical_level)
    return [2**s for s in scales], map_levels


def roi_align1d(input, boxes, output_size, spatial_scale):
    boxes = torch.stack([boxes[:, 0], boxes[:, 1], torch.zeros_like(boxes[:, 1]), boxes[:, 2], torch.full_like(boxes[:, 1], spatial_scale)], dim=1)
    return torch.ops.torchvision.roi_align(input.unsqueeze(-2), boxes, spatial_scale, 1, output_size, -1, True)


def nms1d(boxes, scores, iou_threshold):
    boxes = torch.stack([boxes[:, 0], torch.zeros_like(boxes[:, 1]), boxes[:, 1], torch.ones_like(boxes[:, 1])], dim=1)
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    #max_coordinate = boxes.max()
    #offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    #boxes_for_nms = boxes + offsets[:, None]
    #keep = nms1d(boxes_for_nms, scores, iou_threshold)
    keep = nms1d(boxes, scores, iou_threshold)
    return keep


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 2, 2)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1), nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True))
        self.cls_logits = nn.Conv1d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv1d(in_channels, num_anchors * 2, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits, bbox_reg = [], []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
            #bbox_reg.append(0 * self.bbox_pred(t))
        return logits, bbox_reg


class RegionProposalNetwork1D(RegionProposalNetwork):
    def __init__(
        self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh
    ):
        super().__init__(
            anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh
        )
        self.box_coder = BoxCoder(weights=(1.0, 1.0))
        self.box_similarity = box_similarity

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = boxes.clamp(min=0, max=img_shape)
            keep = torch.where((boxes[:, 1] - boxes[:, 0]) >= 1e-2)[0]  # change how area is calculated
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = _batched_nms_coordinate_trick(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(self, seqs, features, targets):
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(seqs, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 2)
        boxes, scores = self.filter_proposals(proposals, objectness, seqs.seq_length, num_anchors_per_level)
        losses = {}
        cache = {
            "objectness": objectness, 
            "rpn_reg": pred_bbox_deltas, 
            "rpn_boxes": boxes, 
            "rpn_scores": scores, 
        }

        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses, cache


class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size, canonical_scale=128, canonical_level=3):
        super().__init__()
        self.output_size = output_size
        self.canonical_scale = canonical_scale  # which is sampled
        self.canonical_level = canonical_level  # from the typical layer (should be 4?)

        self.scales = None
        self.map_levels = None

    def forward(self, x, boxes, seq_lengths):
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(x, seq_lengths, self.canonical_scale, self.canonical_level)

        num_levels = len(x)
        # convert to roi format
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat([torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)  # MN * 3
        if num_levels == 1:
            return roi_align1d(x[0], rois, output_size=self.output_size, spatial_scale=self.scales[0])

        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, self.output_size), dtype=dtype, device=device)
        for level, (per_level_feature, scale) in enumerate(zip(x, self.scales)):
            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]
            result_idx_in_level = roi_align1d(per_level_feature, rois_per_level, output_size=self.output_size, spatial_scale=scale)
            result[idx_in_level] = result_idx_in_level.squeeze(-2).to(result.dtype)
        return result


class LevelMapper1d(LevelMapper):
    def __init__(self, k_min: int, k_max: int, canonical_scale: int = 224, canonical_level: int = 4, eps: float = 0.000001):
        super().__init__(k_min, k_max, canonical_scale, canonical_level, eps)

    def __call__(self, boxlists):
        s = torch.cat([boxlist[:, 1] - boxlist[:, 0] for boxlist in boxlists])  # we don't do sqrt since 1d data
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


class AnchorGenerator(nn.Module):
    def __init__(self, anchor_sizes) -> None:
        super().__init__()
        self.anchor_sizes = anchor_sizes
        self.num_anchors_per_location = [1 for _ in anchor_sizes]  # TODO: not sure if this is correct

    def forward(self, seq_list, feature_maps):
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        grid_size = [feature_map.shape[-1] for feature_map in feature_maps]
        seq_length = seq_list.tensors.shape[-1]
        strides = [torch.as_tensor(seq_length / g, dtype=dtype, device=device) for g in grid_size]
        cell_anchors = [torch.as_tensor([-s / 2, s / 2], dtype=dtype, device=device).round() for s in self.anchor_sizes]

        anchors_single_seq = []
        for size, stride, base_anchors in zip(grid_size, strides, cell_anchors):
            shift_x = torch.arange(0, size, dtype=torch.int32, device=device) * stride
            shifts = torch.stack((shift_x, shift_x), dim=1)
            anchors_single_seq.append((shifts.view(-1, 1, 2) + base_anchors.view(1, -1, 2)).reshape(-1, 2))

        anchors = [torch.cat([a for a in anchors_single_seq]) for _ in range(len(seq_list.seq_length))]
        return anchors


class SeqList:
    def __init__(self, tensors, seq_length):
        self.tensors = tensors
        self.seq_length = seq_length

    def to(self, device):
        return SeqList(self.tensors.to(device), self.seq_length)


class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000.0 / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        boxes_per_seq = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_seq, 0)

    def encode_single(self, reference_boxes, proposals):
        weights = torch.as_tensor(self.weights, dtype=reference_boxes.dtype, device=reference_boxes.device)
        return encode_boxes(reference_boxes, proposals, weights)

    def decode(self, rel_codes, boxes):
        box_per_seq = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = sum(box_per_seq)
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 2)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)
        width = boxes[:, 1] - boxes[:, 0]
        ctr = boxes[:, 0] + 0.5 * width

        wx, ww = self.weights

        dx = rel_codes[:, 0::2] / wx
        dw = rel_codes[:, 1::2] / ww

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        pred_ctr = dx * width[:, None] + ctr[:, None]
        pred_w = torch.exp(dw) * width[:, None]

        c_to_c_w = pred_ctr.new_tensor(0.5) * pred_w

        pred_boxes1 = pred_ctr - c_to_c_w
        pred_boxes2 = pred_ctr + c_to_c_w
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2), dim=2).flatten(1)
        return pred_boxes


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)
        #return self.cls_score(x), 0 * self.bbox_pred(x)

class RoIHeads1d(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        score_thresh,
        nms_thresh,
        detections_per_img,
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )
        self.box_similarity = box_similarity
        self.box_coder = BoxCoder((10, 10))

    def select_training_samples(self, proposals, targets):
        dtype, device = proposals[0].dtype, proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:  # only change here to 2 for no-object sequence
                gt_boxes_in_image = torch.zeros((1, 2), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_full_scores = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = boxes.clamp(min=0, max=image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            full_scores = scores
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            indices = torch.arange(scores.shape[0])[:, None].expand(scores.shape)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 2)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            indices = indices.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, indices = boxes[inds], scores[inds], labels[inds], indices[inds]

            # remove empty boxes
            keep = torch.where((boxes[:, 1] - boxes[:, 0]) >= 1e-2)[0]  # change how area is calculated
            boxes, scores, labels, indices = boxes[keep], scores[keep], labels[keep], indices[keep]

            # non-maximum suppression, independently done per class
            keep = _batched_nms_coordinate_trick(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, indices = boxes[keep], scores[keep], labels[keep], indices[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_full_scores.append(full_scores[indices])

        return all_boxes, all_scores, all_labels, all_full_scores

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_similarity(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
#         print(len(features))
#         for i in range(len(features)):
#             print(features[i].shape)
#         print(features)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
#         print(box_features.shape)
#         print(box_features)
        box_features = self.box_head(box_features)
#         print(box_features.shape)
#         print(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        result = []
        losses = {}
        cache = {"roi_reg": box_regression, "roi_scores": F.softmax(class_logits, -1)}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels, full_scores = \
                self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i], 
                    "full_scores": full_scores[i]})
        return result, losses, cache


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, size_divisible):
        # we apply minimal transforms here, preprocessing should be added later if needed
        super().__init__()
        self.size_divisible = size_divisible

    def forward(self, seqs, targets):
        if targets is not None:
            # make a deep copy to avoid modifiying it in-place
            targets_copy = []
            for t in targets:
                data = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        seq_lengths = [seq.shape[-1] for seq in seqs]
        seqs = self.batch_seqs(seqs, seq_lengths)
        return SeqList(seqs, seq_lengths), targets

    def batch_seqs(self, seqs, seq_lengths):
        # we assume the sequences is in the same dimension
        max_length = int(math.ceil(float(max(seq_lengths)) / self.size_divisible) * self.size_divisible)
        batch_shape = [len(seqs), 3, 6, max_length]  # exactly the input size
        batched_seqs = seqs[0].new_full(batch_shape, 0)
        for i in range(batched_seqs.shape[0]):
            seq = seqs[i]
            batched_seqs[i, :, :, : seq.shape[-1]].copy_(seq)
        return batched_seqs
