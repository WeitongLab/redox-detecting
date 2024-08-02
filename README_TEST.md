# Test instructions (V1.0 230529)
## IMPORTANT ##
Please go to parameters.py and change the value of "self.det_data_dir" to your local data directory 
(must contain the same data). Unfortunately you will need to copy "save.pkl" into the data folder, 
otherwise a message "pickle files not found" will show. I will try to fix this in the future. 

- The TestPackage class is a helper for testing trained models 
    against corresponding test datasets or new datasets. (See test.ipynb for examples)
    
  - Initilize with exp_num, which is the identifier of a specific experiment: 
    - "230424_00": 0.0 noise, no scan rate reduction
    - "230427_00": 0.05 dynamic noise, no scan rate reduction
    - "230504_03": 0.1  dynamic noise, no scan rate reduction
    - "230504_04": 0.01 dynamic noise, no scan rate reduction
    - "230504_05": 0.2  dynamic noise, no scan rate reduction
    - "230504_06": 0.01 dynamic noise, reduce scan rate with arbitrary selection
    - "230506_00": 0.01 dynamic noise, reduce scan rate with the following possible selections: 
        np.array([[0, ], [1], [2], [3], [4], [5], ]), 
        np.array([[0, 5, ], ]), 
        np.array([[0, 2, 4, ], [1, 3, 5, ], ]), 
        np.array([[0, 1, 2, 3, ], [1, 2, 3, 4, ], [2, 3, 4, 5, ], ]), 
        np.array([[0, 1, 2, 3, 4, ], [1, 2, 3, 4, 5, ], ]), 
        np.array([[0, 1, 2, 3, 4, 5, ], ]), 
    - "230514_00": 0.01 dynamic noise, reduce scan rate with the following possible selections: 
        np.array([[0, ], [1], [2], [3], [4], [5], ]), 
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
      * Note that this is the default scan rate reduction selector 
          for the experiments without specifications. 

  - loadModel(): load model for TestPackage, call first. 

  - loadData(): load corresponding test data for TestPackage. 
      The user may specify min_noise_mag, max_noise_mag, min_sr, max_sr, 
      which respectively controls the minimum/maximum noise magnitudes added to the data, 
      and the minimum/maximum scan rate reduction applied. 
      If unspecified, these parameters are the same as those used in training. 

  - getResults(file_name=None): returns a DataFrame object with test results
      and prints to file with file_name if specified. Call after loadData(). 
    - Each entry of the DataFrame must contain at least one of the following: 
        a ground truth bounding box or a predicted bounding box. 
        
        An entry contains both a ground truth and a predicted box if and only if
        the iou between the two boxes is above iou_thresh (0.75) 
        and the maximum confidence score of the prediction is above select_thresh (0.7). 
        This scenario will be considered a match between prediction and ground truth. 
        
        The data is rescaled to 1000, so all bounding box boundaries are values within [0, 1000]. 
    - The DataFrame has the following keys: 
      - "file": file name of original data. 
      - "gt": whether this entry includes a ground truth bounding box or not; 
          if 0 then "lgt", "rgt", "cls" are meaningless values. 
      - "lgt", "rgt": left and right boundaries of ground truth bounding box. 
      - "cls": class of ground truth bounding box. 
      - "pd": whether this entry includes a predited bounding box or not;
          if 0 then "lpd", "rpd", "bg", "E", ..., "T" are meaningless vlaues. 
      - "lpd", "rpd": left and right boundaries of predicted bounding box. 
      - "bg", "E", "ECa", "ECb", "ECE", "ECP", "DISP", "SR", "T": full prediction scores; 
          "bg" corresponds to the score for background. 
      - "noise_mag": noise magnitude used for this specific data. 
      - "sr_cache": an integer value indicating which scan rates are preserved for this data; 
          e.g. 135 means the first, third and fifth scan rates are preserved. 

  - testNewData(loc): returns a DataFrame object with test results on data files in loc. 
    - Each entry of the DataFrame simply contains a prediction. 
    - The DataFrame has the following keys, 
      please reference the getResults() entry from above for more information: 
      ["file", "lpd", "rpd", "bg", "E", "ECa", "ECb", "ECE", "ECP", "DISP", "SR", "T", ]
    - Please follow the same format as the test data provided before (see folder test230419)
    
  - getTrainLoss(): get the train loss of this experiment. 
      The user can call this function without calling getModel(). 
    - Returns a dictionary with keys corresponding to different losses: 
      ["total", "loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg", ]
      Here "total" loss is the sum of the other four losses. 
    - Each value of the dictionary is a tuple of two lists (steps, values), 
        where steps contain the steps where the losses are calculated, 
        while values contain the corresponding loss values. 
    - There is also a smoothLoss(loss_ls, smth_idx=50) function available in test.py, 
        which uses average smoothing to generate a better-looking loss plot. 

  - checkPrediction(idx): check if the test_dataset[idx] of the TestPackage 
      has perfect predictions or not, i.e. if for each ground truth event, 
      there exists and only exists one prediction with above 75% iou threshold 
      and 70% confidence score on the correct class, and there are no additional predictions. 
    - Can be utilized to filter the dataset when generating importance data. 

  - getImportance(idx): get the importance data corresponding to test_dataset[idx]. 
    - Returns a dictionary with the following keys and values: 
      - "info": a DataFrame with the standard format (same with getResults()) 
          on the one data point corresponding to test_dataset[idx]. 
      - "imp_cls": a list of importance numpy arrays, each corresponding to 
          the classification of a predicted bounding box. 
          Ordered by the bounding boxes' left boundaries. 
      - "imp_reg": importance numpy array corresponding to roi regression. 
      - "imp_obj": importance numpy array corresponding to objectiveness. 
    
    
