from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import functional as F, ToPILImage
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis
from torchvision.io import read_image, ImageReadMode

import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import os, glob
import zipfile
import warnings
warnings.filterwarnings("ignore")

#device 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed()
torch.manual_seed(42)
random.seed()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    random.seed()
    print(f"Using GPU1: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU")

