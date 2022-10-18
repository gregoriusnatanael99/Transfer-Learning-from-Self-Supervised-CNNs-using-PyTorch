import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from misc.utils import *

from config import cfg

seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

cudnn.benchmark = True

data_transforms = {
    'train': transforms.Compose([
#        transforms.RandomResizedCrop(224),
#        transforms.RandomRotation((0,90)),
#        transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET_MEAN, cfg.DATASET_STD)
    ]),
    'val': transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET_MEAN, cfg.DATASET_STD)
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(cfg.DATASET_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.BATCH_SIZE,
                                             shuffle=True, num_workers=cfg.NUM_WORKERS)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

from model_trainer import Model_Trainer

print(class_names)
cfg['num_class'] = len(class_names)

if cfg.WEIGHTING:
    cfg['class_weights'] = calculate_class_weights(image_datasets['train'])
    cfg['class_weights'] = [i for i in cfg['class_weights']]
#    cfg['class_weights'] = [0.01,0.01,0.01,10]
    print(cfg['class_weights'])


trainer = Model_Trainer(cfg,dataloaders,dataset_sizes)
trainer.begin_training()
#print("Training Finished! Have a good day :)")
