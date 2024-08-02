import numpy as np
import torch

from utils import box_similarity
from parameters import Parameters

pars = Parameters()

class StatCalculator():
    def __init__(self, detections, targets, iou_thresh=0.75, select_thresh=0.7):
        self.num_data = len(detections)
        self.detections = detections
        self.targets = targets
        self.iou_thresh = iou_thresh
        self.select_thresh = select_thresh
        
        self.match_mats = [None if d["boxes"].shape[0] == 0 else torch.logical_and(
                (box_similarity(d["boxes"], t["boxes"]) >= self.iou_thresh), 
                (d["scores"] >= self.select_thresh).view(-1, 1).expand(-1, t["boxes"].shape[0]))
            for t, d in zip(targets, detections)]
        
        self.nbox = np.array([t["boxes"].shape[0] for t in self.targets])
        self.num_data_nbox = np.array([np.argwhere(self.nbox == nbox + 1).shape[0] for nbox in range(4)])
        self.idx_valid = np.array([idx for idx in range(self.num_data) if self.match_mats[idx] is not None])
        self.idx_ls = [np.array([idx for idx in self.idx_valid if self.nbox[idx] == nbox + 1]) 
            for nbox in range(pars.max_nbox)]
    
    def __call__(self, name):
        if name == "num_gt_match": 
            gt_sel_matches = np.array([0 if mat is None else (mat.sum(axis=0) >= 1).sum().item() 
                for mat in self.match_sel_mats])
            return np.array([((self.nbox == nbox + 1) * gt_sel_matches).sum().item() for nbox in range(pars.max_nbox)])
        if name == "num_pd_match": 
            pd_sel_matches = np.array([0 if mat is None else (mat.sum(axis=1) >= 1).sum().item() 
                for mat in self.match_mats])
            return np.array([((self.nbox == nbox + 1) * pd_sel_matches).sum().item() for nbox in range(pars.max_nbox)])
        
        if name == "labels_arr": 
            labels_all = [np.empty((0, 2)) if match_mat is None or match_mat.sum().item() == 0 else 
                np.array([[int(d["labels"][idx[0]]), int(t["labels"][idx[1]])] for idx in match_mat.nonzero()]) 
                for (d, t, match_mat) in zip(self.detections, self.targets, self.match_mats)]
            return [np.empty((0, 2)) if len(self.idx_ls[nbox]) == 0 else 
                np.concatenate([labels_all[idx] for idx in self.idx_ls[nbox]]) 
                for nbox in range(pars.max_nbox)]
        
        return None


def getBatchAcc(model, batch):
    model.eval()
    images, targets = batch
    
    with torch.no_grad():
        _, detections, _ = model.forward(images)
        stats = StatCalculator(detections, targets, iou_thresh=0.75, select_thresh=0.7)
    
    gt_box_num = stats.num_data_nbox
    gt_box_tt = gt_box_num[0] * 1 + gt_box_num[1] * 2 + gt_box_num[2] * 3 + gt_box_num[3] * 4
    
    num_match = 0
    labels_arr = stats("labels_arr")
    for nbox in range(pars.max_nbox):
        num_match += (labels_arr[nbox][:, 0] == labels_arr[nbox][:, 1]).sum()
    
    return num_match, gt_box_tt