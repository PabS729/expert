import torch.nn as nn
import numpy as np
import torchvision

class expert_loss():
    def __init__(self, output1, output2):
        