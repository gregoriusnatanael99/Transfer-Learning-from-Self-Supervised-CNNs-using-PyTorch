import os
from easydict import EasyDict as edict
import time
import torch

# init

"""
The train mode can be set to normal or grid_search. Default = normal
normal: use NORMAL_HP_DICT
grid_search: perform grid search with hyperopt using GRID_HP_SEARCH_DICT

The TL_ALGO can be set to swav or supervised. Default = supervised
"""
__C = edict()
cfg = __C

__C.SEED = 1234
__C.MODEL_ARCH = 'resnet50'
__C.TL_ALGO = 'vicreg'      #swav, supervised, vicreg, barlow_twins, SimSiam, MOCO
__C.UNFROZEN_BLOCKS = 0
__C.GPU_ID = [0]    
__C.BATCH_SIZE = 8
__C.NUM_WORKERS = 4
__C.DATASET_DIR = "../preprocessed_data/"
__C.DATASET_MEAN = [0.5094484686851501, 0.5094484686851501, 0.5094484686851501]
__C.DATASET_STD = [0.2523978352546692, 0.2523978352546692, 0.2523978352546692]
__C.TRAIN_MODE = "grid_search"  #normal,grid_search
__C.SAVE_MODEL = True
__C.WEIGHTING = True #use class weighting if True

__C.NORMAL_HP_DICT = {
    'epochs':2,
    'optimizer':'adam',
    'lr':1e-3,
    'weight_decay':0
}

__C.GRID_HP_SEARCH_SPACE_DICT = {
    'epochs':[100],
    'optimizer':['adam'],
    'lr':[1,1e-1,1e-2,1e-3,1e-4,1e-5],
    'weight_decay':[0,1e-1,1e-2,1e-3,1e-4,1e-5]
}

# __C.MODEL_DICT={

# }