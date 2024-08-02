import os
import pickle
import shutil

import pandas as pd
import pytorch_lightning as pl
import numpy as np
from numpy.random import default_rng

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from parameters import Parameters
pars = Parameters()
rng = default_rng()


def expandScanRate(data):
    sr = data.shape[1]
    return data[:, pars.sr_dupe[sr - 1], :]

def maskScanRate(data, sr):
    if sr == 6:
        return data, 123456
    samplers = pars.sr_sampler[sr - 1]
    sr_remain = samplers[np.random.choice(samplers.shape[0])]
    
    sr_cache = 0
    for item in sr_remain:
        sr_cache = 10 * sr_cache + item + 1
    
    return expandScanRate(data[:, sr_remain, :]), sr_cache


def rescaleData(data, scale_new):
    scale_old = data.shape[2]
    data_new = np.empty([data.shape[0], data.shape[1], scale_new])
    for idx in range(scale_new - 1):
        idx_old_raw = (idx / (scale_new - 1)) * (scale_old - 1)
        idx_old = int(idx_old_raw)
        #print(idx, idx_old_raw, idx_old)
        data_new[:, :, idx] = \
            (idx_old + 1 - idx_old_raw) * data[:, :, idx_old] + \
            (idx_old_raw - idx_old) * data[:, :, idx_old + 1]
    data_new[:, :, scale_new - 1] = data[:, :, scale_old - 1]
    
    return data_new


class DetectionDataset(Dataset):
    def __init__(self, file_list, target, idx_list, root=pars.det_data_dir, **kwargs) -> None:
        super().__init__()
        self.target = target
        self.file_list = file_list
        self.idx_list = idx_list
        self.root = root
        
        self.min_sr = kwargs["min_sr"] if "min_sr" in kwargs.keys() else 6
        self.max_sr = kwargs["max_sr"] if "max_sr" in kwargs.keys() else 6
        self.min_noise_mag = kwargs["min_noise_mag"] if "min_noise_mag" in kwargs.keys() else 0.0
        self.max_noise_mag = kwargs["max_noise_mag"] if "max_noise_mag" in kwargs.keys() else 0.01
        self.scale = pars.rescale
        
        if self.scale is not None:
            self.filepath_new = os.path.join(self.root, "data_rescale_" + str(self.scale))
            if not os.path.exists(self.filepath_new):
                os.mkdir(self.filepath_new)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if self.scale is not None:
            filename_old = os.path.join(self.root, "Data", self.file_list[index][0], self.file_list[index][1])
            filename_new = os.path.join(self.filepath_new, self.file_list[index][1])
            if not os.path.exists(filename_new):
                Av = np.array(pd.read_pickle(filename_old)[["A", "v"]].values, dtype=np.float32).reshape(6, 2, -1, 2)
                data = np.stack([Av[:, 0, :, 0], np.flip(Av[:, 1, :, 0], axis=1), Av[:, 0, :, 1]])
                data = rescaleData(data, self.scale)
                pd.DataFrame(data.reshape(18, -1)).to_pickle(filename_new)
                data = torch.tensor(data, dtype=torch.float32)
            else:
                data = torch.tensor(pd.read_pickle(filename_new).values, dtype=torch.float32).view(3, 6, -1)
        else:
            Av = torch.tensor(pd.read_pickle(os.path.join(self.root, "Data", self.file_list[index][0], self.file_list[index][1]))
                              [["A", "v"]].values, dtype=torch.float32).view(6, 2, -1, 2)
            data = torch.stack([Av[:, 0, :, 0], Av[:, 1, :, 0].flip(dims=(1,)), Av[:, 0, :, 1]])
        
        noise = torch.zeros_like(data)
        noise_mag = np.random.uniform(low=self.min_noise_mag, high=self.max_noise_mag)
        # noise_mag = np.random.uniform(high=self.max_noise_mag)
        noise[(0, 1), :, :] = torch.normal(noise[(0, 1), :, :], std=noise_mag)
        data = data + noise
        
        sr_trim = np.random.randint(low=self.min_sr, high=self.max_sr + 1)
        data, sr_cache = maskScanRate(data, sr_trim)
        
        target_ret = {
            "boxes": torch.clone(torch.round(
                (self.target[index]["boxes"] / (self.target[index]["scale"] - 1)) * (self.scale - 1)) \
                    if self.scale is not None else self.target[index]["boxes"]), 
            "labels": torch.clone(self.target[index]["labels"]) + 1, 
            "cache": {
                "noise_mag": noise_mag, 
                "sr_cache": sr_cache, 
            }, 
        }
        # for idx in range(target_ret["labels"].shape[0]):
        #     target_ret["labels"][idx] = int(target_ret["labels"][idx].item()) + 1
        
        return data, target_ret
    
    def log(self, filename):
        f = open(filename, 'w')
        for idx in self.idx_list:
            print(idx, file=f)
        f.close()


def get_dataset(root=pars.det_data_dir, **kwargs):
    try:
        print("retrieving data from {}/save.pkl".format(root))
        with open(os.path.join(root, "save.pkl"), "rb") as f:
            saved_assign = pickle.load(f)
            sanitize_file_list = saved_assign["file_list"]
            target = saved_assign["target"]
    except Exception as e:
        print("pickle files not found, try regenerate")
        tables = pd.concat(map(lambda x: pd.read_csv(os.path.join(root, "Labels", x)), os.listdir(os.path.join(root, "Labels"))))
        dirname_list = os.listdir(os.path.join(root, "Data"))
        filename_list = []
        for idx_list in range(len(dirname_list)):
            filename_list += [[dirname_list[idx_list], file] for file in os.listdir(os.path.join(root, "Data", dirname_list[idx_list]))]
        sanitize_file_list = []
        target = []
        idx = 0
        for f in tqdm(filename_list):
            # we assume the difference in V is a constant
            try:
                Av = torch.tensor(pd.read_pickle(os.path.join(root, "Data", f[0], f[1]))[["A", "v"]].values, dtype=torch.float32).view(6, 2, -1, 2)
                sanitize_file_list.append(f)
                target.append(
                    {
                        "boxes": torch.tensor(tables.loc[tables.File == f[1][:-4]][["Merge_Left", "Merge_Right"]].values, dtype=torch.float32),
                        "labels": torch.tensor([pars.det_key[k] for k in tables.loc[tables.File == f[1][:-4]].Mechanism], dtype=torch.float32),
                        "scale": Av.shape[2],
                    }
                )
                if len(tables.File == f[1][:-4]) == 0:
                    print("no labels found for file ", f)
            except Exception as file_exception:
                print(f, file_exception)
            if (idx + 1) % 5000 == 0:
                print(f"dumping at index {idx}")
                with open(os.path.join(root, "save.pkl"), "wb") as f:
                    pickle.dump({"file_list": sanitize_file_list, "target": target}, f)
            idx += 1
        with open(os.path.join(root, "save.pkl"), "wb") as f:
            pickle.dump({"file_list": sanitize_file_list, "target": target}, f)

    idx_list = [_ for _ in range(len(target))]
    file_train, file_test, target_train, target_test, idx_list_train, idx_list_test = \
        train_test_split(sanitize_file_list, target, idx_list, test_size=pars.test_size)
    file_train, file_val, target_train, target_val, idx_list_train, idx_list_val = \
        train_test_split(file_train, target_train, idx_list_train, test_size=0.1)
    return DetectionDataset(file_train, target_train, idx_list_train, **kwargs), \
           DetectionDataset(file_val, target_val, idx_list_val, **kwargs), \
           DetectionDataset(file_test, target_test, idx_list_test, **kwargs)


def retrieveDataset(filename, root=pars.det_data_dir, **kwargs): 
    try:
        print("retrieving data from {}/save.pkl".format(root))
        with open(os.path.join(root, "save.pkl"), "rb") as f:
            saved_assign = pickle.load(f)
            sanitize_file_list = saved_assign["file_list"]
            target = saved_assign["target"]
    except:
        print("pickle files not found")
        return None

    f = open(filename, 'r')
    idx_list = []
    for line in f:
        idx_list.append(int(line))
    f.close()
    
    file_list, target_list = \
        [sanitize_file_list[idx_list[_]] for _ in range(len(idx_list))], \
        [target[idx_list[_]] for _ in range(len(idx_list))]
    return DetectionDataset(file_list, target_list, idx_list, **kwargs)


class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, root, **kwargs) -> None:
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(root, **kwargs)

    def train_dataloader(self):
        return self._data_loader(self.train_dataset, batch_size=pars.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return self._data_loader(self.val_dataset, batch_size=pars.batch_size_val, shuffle=False)

    def test_dataloader(self):
        return self._data_loader(self.test_dataset, batch_size=pars.batch_size_test, shuffle=False)

    def _data_loader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, num_workers=48, \
            shuffle=shuffle, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))


if __name__ == "__main__":
    A, B, C = get_dataset(root=pars.det_data_dir)
    print(len(A), len(B), len(C))
    print(A[0], B[0], C[0])
