import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import os

from resnet1d import BasicBlock, ResNet
from utils import AnchorGenerator, FastRCNNPredictor, GeneralizedRCNNTransform, MultiScaleRoIAlign, RegionProposalNetwork1D, RoIHeads1d, RPNHead, box_similarity
from stats import StatCalculator, getBatchAcc
from parameters import Parameters
pars = Parameters()

class FasterRCNN(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # RPN parameters
        rpn_pre_nms_top_n_train=20,
        rpn_pre_nms_top_n_test=20,
        rpn_post_nms_top_n_train=20,
        rpn_post_nms_top_n_test=20,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=10,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        rpn_anchor_generator = AnchorGenerator(anchor_sizes=(32, 64, 128, 256, 512))
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork1D(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        box_roi_pool = MultiScaleRoIAlign(output_size=8)
        resolution = box_roi_pool.output_size
        representation_size = 32

        """ create box head blocks and initialize """
#         box_head = TwoMLPHead(out_channels * resolution, representation_size)  # This is not squared since 1d
        box_head_blocks = []
        prev_channels = out_channels
        for channels in [256, 128, 64]:
            box_head_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=prev_channels, out_channels=channels, kernel_size=7, padding=3), 
                    nn.BatchNorm1d(channels), nn.ReLU(inplace=True)
                )
            )
            prev_channels = channels
        box_head_blocks.append(
            nn.Sequential(nn.Flatten(), nn.Linear(prev_channels * resolution, representation_size), nn.ReLU(inplace=True))
        )
        box_head = nn.Sequential(*box_head_blocks)
        for layer in box_head.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        self.roi_heads = RoIHeads1d(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        self.transform = GeneralizedRCNNTransform(size_divisible=32)

    def forward(self, Xs, targets=None):
        losses, cache = {}, {}
        Xs, targets = self.transform(Xs, targets)
        features = self.backbone(Xs.tensors)
        proposals, proposal_losses, rpn_cache = self.rpn(Xs, features, targets)
        losses.update(proposal_losses)
        cache.update(rpn_cache)
        detections = None
        detections, detector_losses, roi_cache = self.roi_heads(features, proposals, Xs.seq_length, targets)
        losses.update(detector_losses)
        cache.update(roi_cache)
        return losses, detections, cache


class FasterRCNNwrap(pl.LightningModule):
    def __init__(self, learning_rate, num_classes):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = FasterRCNN(backbone=ResNet(BasicBlock, [2, 2, 2, 2]), num_classes=num_classes)

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict, _, _ = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dict.update({"total": loss})
        self.log("train loss", loss_dict)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        n_match, n_box = getBatchAcc(self.model, batch)
        return {"val_data": [n_match, n_box]}

    def validation_epoch_end(self, outputs):
        avg_acc = sum([o["val_data"][0] for o in outputs]) / sum([o["val_data"][1] for o in outputs])
        logs = {"val_acc": avg_acc}
        self.log('val_acc', avg_acc)
        return {"avg_val_acc": avg_acc, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.005)
