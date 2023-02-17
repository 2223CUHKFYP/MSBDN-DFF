# 17/02/2023 this is for dehazing


from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import re
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', type=str, default='SOTS', help='Path of the validation dataset')
parser.add_argument("--checkpoint", default="models/MSBDN-DFF/1/model.pkl", type=str, help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='MSBDN', help='filename of the training models')
parser.add_argument("--start", type=int, default=2, help="Activated gate module")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

















# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

