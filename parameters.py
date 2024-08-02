import os
import numpy as np

class Parameters():
    def __init__(self):
        # detection data
        self.det_labels = ['E', 'ECa', 'ECb', 'ECE', 'ECP', 'DISP', 'SR', 'T']
        self.det_key = {"E": 0, "ECa": 1, "ECb": 2, "ECE": 3, "ECP": 4, "DISP": 5, "SR": 6, "T": 7}
        self.det_data_dir = "./data"
        self.num_cls = 8
        self.max_nbox = 4
        
        self.rescale = 1000
        self.min_noise_mag = 0
        self.max_noise_mag = 0.01
        
        self.min_sr = 1
        self.max_sr = 6
        self.sr_dupe = np.array([
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1], 
            [0, 0, 1, 1, 2, 2], 
            [0, 0, 1, 2, 3, 3], 
            [0, 0, 1, 2, 3, 4], 
            [0, 1, 2, 3, 4, 5], 
        ])
        self.sr_sampler = [
            np.array([[0], [1], [2], [3], [4], [5], ]), 
            np.array([[0, 1, ], [0, 2, ], [0, 3, ], [0, 4, ], [0, 5, ], 
                    [1, 2, ], [1, 3, ], [1, 4, ], [1, 5, ], 
                    [2, 3, ], [2, 4, ], [2, 5, ], 
                    [3, 4, ], [3, 5, ], 
                    [4, 5, ], ]), 
            np.array([[0, 1, 2, ], [1, 2, 3, ], [2, 3, 4, ], [3, 4, 5, ], 
                    [0, 2, 4, ], [1, 3, 5, ], ]), 
            np.array([[0, 1, 2, 3, ], [1, 2, 3, 4, ], [2, 3, 4, 5, ], ]), 
            np.array([[0, 1, 2, 3, 4, ], [1, 2, 3, 4, 5, ], ]), 
            np.array([[0, 1, 2, 3, 4, 5, ], ]), 
        ]
        
        # detection training
        self.train_val_size = 0.9
        self.test_size = 1 - self.train_val_size
        self.batch_size_train = 64
        self.batch_size_val = 64
        self.batch_size_test = 128
        self.lr = 1e-3
        self.max_epochs = 100
        self.loss_names = ["total", "loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg", ]
        
        self.cache_dir = "cache/det_train"
        self.save_label = "240802_04"
        self.save_loc = f"{self.cache_dir}/output{self.save_label}"
    
    def print(self, file=None):
        if file is None:
            file = open(os.path.join(self.save_loc, "info.txt"), 'w')
        
        print("- general info", file=file)
        print("-- cache directory: ", self.save_loc, file=file)
        print("-- data directory: ", self.det_data_dir, file=file)
        
        print("\n- data info", file=file)
        print("-- data rescaling: ", self.rescale, file=file)
        print(f"-- noise during training: ({self.min_noise_mag}, {self.max_noise_mag})", file=file)
        print("-- minimum number of scan rates: ", self.min_sr, file=file)
        print("-- maximum number of scan rates: ", self.max_sr, file=file)
        
        print("\n- training info", file=file)
        print("-- epochs: ", self.max_epochs, file=file)
        print("-- learning rate: ", self.lr, file=file)
        print(f"-- train : validation : test = \
            {round(self.train_val_size * 0.9, 3)} : \
            {round(self.train_val_size * 0.1, 3)} : \
            {round(self.test_size, 3)}", file=file)
        print("-- batch size in training: ", self.batch_size_train, file=file)
        print("-- batch size in validation: ", self.batch_size_val, file=file)
        print("-- batch size in testing: ", self.batch_size_test, file=file)


# def inversePermutation(pmt):
#     length = len(pmt)
#     inv = [0 for _ in range(length)]
#     for _ in range(length):
#         inv[pmt[_]] = _
#     return inv
