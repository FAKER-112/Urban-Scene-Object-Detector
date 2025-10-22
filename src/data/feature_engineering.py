import os   
import sys
import json
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection
from torchvision import transforms


class FeatureEngineering:
    def __init__(self):
        pass
    