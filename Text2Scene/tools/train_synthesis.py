#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt

from composites_utils import *
from composites_config import get_config
from datasets.composites_coco import composites_coco
from datasets.composites_loader import sequence_loader, synthesis_loader
from modules.synthesis_model import SynthesisModel
from modules.synthesis_trainer import SynthesisTrainer

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train_synthesis_model(config):
    traindb = composites_coco(config, 'train', '2017')
    valdb   = composites_coco(config, 'val',   '2017')
    testdb  = composites_coco(config, 'test',  '2017')
    trainer = SynthesisTrainer(config)
    trainer.train(traindb, valdb, testdb)


def overfit_synthesis_model(config):
    config.log_per_steps = 1
    traindb = composites_coco(config, 'train', '2017')
    traindb.scenedb = traindb.scenedb[:config.batch_size*3]
    # print('build pca table')
    # pca_table = AllCategoriesTables(traindb)
    # pca_table.run_PCAs_and_build_nntables_in_feature_space()
    # print('create trainer')
    trainer = SynthesisTrainer(config)
    # print('start training')
    trainer.train(traindb, traindb, traindb)

    

if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # overfit_synthesis_model(config)
    train_synthesis_model(config)
    
