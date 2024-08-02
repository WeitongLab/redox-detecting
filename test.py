import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils import box_similarity
from plot import plotData, plotBoundingBox
from data import retrieveDataset, rescaleData, expandScanRate
from backbone import BackboneNetWrap
from fasterRCNN import FasterRCNNwrap
from parameters import Parameters
from stats import StatCalculator

pars = Parameters()

exp_num_ls = ["230424_00", "230427_00", "230504_03", "230504_04", 
              "230504_05", "230504_06", "230506_00", "230514_00", ]
result_columns = [
    "file", "gt", "lgt", "rgt", "cls", 
    "pd", "lpd", "rpd", "bg", "E", "ECa", "ECb", "ECE", "ECP", "DISP", "SR", "T", 
    "noise_mag", "sr_cache", 
]

class TestPackage(Parameters):
    def __init__(self, exp_num):
        if exp_num not in exp_num_ls:
            print("Experiment not found. ")
            return 0
        else:
            super().__init__()
            self.save_label = exp_num
            self.save_loc = f"{self.cache_dir}/output{self.save_label}"
            
            # configure noise magnitudes and scan rates
            with open(os.path.join(self.cache_dir, "output" + exp_num, "info.txt")) as f:
                lines = f.readlines()
                for line in lines:
                    if "-- noise during training: " in line:
                        noise_infos = line[line.find("(") + 1 : line.find(")")].split(", ")
                        self.min_noise_mag = float(noise_infos[0])
                        self.max_noise_mag = float(noise_infos[1])
                    if "-- minimum number of scan rates: " in line:
                        self.min_sr = int(line[-2])
                    if "-- maximum number of scan rates: " in line:
                        self.max_sr = int(line[-2])
            
            # configure scan rate sampler
            if exp_num == "230504_06":
                self.sr_sampler = [
                    np.array([[mem for mem in comb] for comb in combinations(range(6), num)]) for num in range(1, 7)
                ]
            if exp_num == "230506_00":
                self.sr_sampler = [
                    np.array([[0], [1], [2], [3], [4], [5], ]), 
                    np.array([[0, 5, ], ]), 
                    np.array([[0, 2, 4, ], [1, 3, 5, ], ]), 
                    np.array([[0, 1, 2, 3, ], [1, 2, 3, 4, ], [2, 3, 4, 5, ], ]), 
                    np.array([[0, 1, 2, 3, 4, ], [1, 2, 3, 4, 5, ], ]), 
                    np.array([[0, 1, 2, 3, 4, 5, ], ]), 
                ]
            
            # used to save model outputs on single datas to avoid redundant calculations: 
            self.result_packages = {}


    def loadModel(self):
        ckpt_folder = os.path.join(self.save_loc, "ckpts")
        filename_list = os.listdir(ckpt_folder)
        max_epoch = 0

        for filename in filename_list:
            new_epoch = int(filename[6:].split('-')[0])
            if max_epoch < new_epoch:
                max_epoch = new_epoch
                model_file_name = filename
        
        self.model_file_name = model_file_name
        self.model = FasterRCNNwrap.load_from_checkpoint(os.path.join(ckpt_folder, self.model_file_name), \
            learning_rate=self.lr, num_classes=self.num_cls + 1)
    
    def loadData(self, **kwargs):
        self.test_dataset = retrieveDataset(os.path.join(pars.save_loc, "test_idx.txt"), 
            min_noise_mag = kwargs["min_noise_mag"] if "min_noise_mag" in kwargs.keys() else self.min_noise_mag, 
            max_noise_mag = kwargs["max_noise_mag"] if "max_noise_mag" in kwargs.keys() else self.max_noise_mag, 
            min_sr = kwargs["min_sr"] if "min_sr" in kwargs.keys() else self.min_sr, 
            max_sr = kwargs["max_sr"] if "max_sr" in kwargs.keys() else self.max_sr)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_test, 
            num_workers=16, shuffle=False, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))
    
    
    def getResults(self, file_name=None):
        self.model.eval()
        df = pd.DataFrame(columns=result_columns)
        
        for idx, batch in enumerate(self.test_loader):
            data, targets = batch
            with torch.no_grad():
                _, detections, _ = self.model.model(data, targets)
            
            new_entries = getResultEntries(
                self.test_dataset.file_list[idx*self.batch_size_test : (idx+1)*self.batch_size_test], 
                detections, targets)
            for entry in new_entries:
                df.loc[len(df.index)] = entry
        
        if file_name is not None:
            df.to_csv(file_name, index=False)
        self.results_df = df
    
    
    def testNewData(self, loc):
        df = pd.DataFrame(columns=["file", "lpd", "rpd", "bg", "E", "ECa", "ECb", "ECE", "ECP", "DISP", "SR", "T", ])
        
        for filename in os.listdir(loc): 
            if filename[-4:] == ".txt":
                data = pd.read_csv(os.path.join(loc, filename), sep=',')
                
                Av = np.array(pd.read_csv(os.path.join(loc, filename))
                                [["A", "v"]].values, dtype=np.float32).reshape(-1, 2, data['V'][0] + 1, 2)
                data = np.stack([Av[:, 1, :, 0], np.flip(Av[:, 0, :, 0], axis=1), Av[:, 0, :, 1]])
                data = torch.tensor(rescaleData(data, self.rescale), dtype=torch.float32)
                # noise = torch.zeros_like(data)
                # noise_mag = np.random.uniform(low=self.min_noise_mag, high=self.max_noise_mag)
                # noise[(0, 1), :, :] = torch.normal(noise[(0, 1), :, :], std=noise_mag)
                # data = data + noise
                data = expandScanRate(data)
                
                self.model.eval()
                _, detections, _ = self.model(data.unsqueeze(0))
                boxes = detections[0]["boxes"].detach().tolist()
                full_scores = detections[0]["full_scores"].detach().tolist()
                
                for b, fs in zip(boxes, full_scores):
                    df.loc[len(df.index)] = [filename, b[0], b[1]] + fs
        
        return df
    
    
    def getTrainLoss(self):
        log_paths = [os.path.join(self.save_loc, f"lightning_logs/version_0/train loss/{ln}") 
            for ln in self.loss_names]
        log_files = [os.listdir(lp)[0] for lp in log_paths]
        summaries = [EventAccumulator(os.path.join(lp, lf)).Reload() 
                    for lp, lf in zip(log_paths, log_files)]

        results = {}
        for ln, summary in zip(self.loss_names, summaries):
            steps, values = [], []
            for event in summary.Scalars("train_loss"):
                steps.append(event.step)
                values.append(event.value)
            results[ln] = (steps, values)
        return results
    
    
    def checkPrediction(self, idx):
        data, target = self.test_dataset[idx]
        nbox_data = target['labels'].shape[0]
        
        if idx not in self.result_packages.keys():
            self.model.eval()
            _, detections, cache = self.model.model(data.unsqueeze(0), [target])
            stats = StatCalculator(detections, [target], iou_thresh=0.75, select_thresh=0.7)
            self.result_packages[idx] = {"det": detections, "cache": cache, "stats": stats}
        else:
            stats = self.result_packages[idx]["stats"]
        
        return stats.match_mats[0].shape[0] == nbox_data \
            and stats.match_mats[0].sum().item() == nbox_data \
            and np.all(stats("labels_arr")[nbox_data - 1][:, 0] == stats("labels_arr")[nbox_data - 1][:, 1])
    
    def processImportance(self, tensor):
        tensor = tensor / max(torch.max(tensor).item(), - torch.min(tensor).item())
        return torch.clone(tensor.permute((1, 0, 2))).detach().numpy()
    
    def getImportance(self, idx):
        data, target = self.test_dataset[idx]
        data.requires_grad = True
        
        self.model.eval()
        _, detections, cache = self.model.model(data.unsqueeze(0), [target])
        if idx not in self.result_packages.keys():
            stats = StatCalculator(detections, [target], iou_thresh=0.75, select_thresh=0.7)
            self.result_packages[idx] = {"det": detections, "cache": cache, "stats": stats}
        
        boxes = detections[0]['boxes'].detach().numpy()
        objectness, roi_reg, roi_scores = cache["objectness"], cache["roi_reg"], cache["roi_scores"]
        results = {}
        
        # information on this data in the standard dataframe format
        entries = getResultEntries(self.test_dataset.file_list[idx], detections, [target])
        info_df = pd.DataFrame(columns=result_columns)
        for entry in entries:
            info_df.loc[len(info_df.index)] = entry
        results.update({"info" : info_df})
        
        ## importance w.r.t. classification
        sorted_idx = np.argsort(boxes[:, 0])
        results["imp_cls"] = []
        for s in detections[0]['scores'][sorted_idx]: 
            s.backward(retain_graph=True)
            results["imp_cls"].append(self.processImportance(data.grad))
            data.grad.zero_()
            
        ## importance w.r.t. regression
        _, indices = torch.topk(roi_scores.flatten(), roi_scores.shape[0])
        indices_new = torch.cat((indices * 2, indices * 2 + 1))
        torch.sum(roi_reg.flatten()[indices_new]).backward(retain_graph=True)
        results["imp_reg"] = self.processImportance(data.grad)
        data.grad.zero_()
        
        ## importance w.r.t. objectness
        torch.sum(objectness).backward(retain_graph=True)
        results["imp_obj"] = self.processImportance(data.grad)
        data.grad.zero_()

        return results


## getting prediction results
def getResultEntries(file_list, detections, targets):
    stats = StatCalculator(detections, targets, iou_thresh=0.75, select_thresh=0.7)
    entry_list = []
    
    for idx, (fn_ls, d, t) in enumerate(zip(file_list, detections, targets)):
        fn = os.path.join(fn_ls[0], fn_ls[1])
        boxes, full_scores = d['boxes'].detach().numpy(), d['full_scores'].detach().numpy()
        boxes_gt, labels_gt = t['boxes'].numpy(), t['labels'].numpy()
        matched = np.zeros(boxes.shape[0])
        
        for gt_idx in range(boxes_gt.shape[0]):
            new_data = [fn, 1, round(boxes_gt[gt_idx][0], 2), round(boxes_gt[gt_idx][1], 2), 
                pars.det_labels[int(t["labels"][gt_idx].item()) - 1]]
            
            if stats.match_mats[idx] is None or stats.match_mats[idx][:, gt_idx].sum() == 0:
                new_data += [0, 0, 0]
                for cls in range(pars.num_cls + 1):
                    new_data.append(0.0)
            else:
                pd_idx = torch.nonzero(stats.match_mats[idx][:, gt_idx])[0].item()
                matched[pd_idx] = 1
                new_data += [1, boxes[pd_idx][0].item(), boxes[pd_idx][1].item()]
                for cls in range(pars.num_cls + 1):
                    new_data.append(round(full_scores[pd_idx][cls], 3))
            new_data += [t["cache"]["noise_mag"], t["cache"]["sr_cache"]]
            
            entry_list.append(new_data)
        
        for pd_idx in range(boxes.shape[0]):
            if matched[pd_idx] == 0:
                new_data = [fn, 0, 0, 0, "N/A", 1, 
                            round(boxes[pd_idx][0], 2), round(boxes[pd_idx][1], 2)]
                for cls in range(pars.num_cls + 1):
                    new_data.append(round(full_scores[pd_idx][cls], 3))
                new_data += [t["cache"]["noise_mag"], t["cache"]["sr_cache"]]
                
                entry_list.append(new_data)
    
    return entry_list

def smoothLoss(loss_ls, smth_idx=50):
    max_idx = len(loss_ls)
    return [sum(loss_ls[max(0, idx - smth_idx) : min(max_idx, idx + smth_idx)]) / 
        (min(max_idx, idx + smth_idx) - max(0, idx - smth_idx)) for idx in range(max_idx)]
