import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from misc.utils import *

from evaluators.test_config import cfg

seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

cudnn.benchmark = True

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(cfg.DATASET_MEAN, cfg.DATASET_STD)])
        
test_dataset = datasets.ImageFolder(os.path.join(cfg.DATASET_DIR, 'test'), test_transforms)
test_size = len(test_dataset)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

cfg['class_names'] = test_dataset.classes
cfg['num_class'] = len(cfg['class_names'])

from evaluators.model_evaluator import Model_Evaluator

evaluator = Model_Evaluator(cfg,test_dataloader,test_size)
evaluator.begin_testing()