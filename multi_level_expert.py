
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from models.expertcnn import *
import numpy as np
import torchvision.models as models
from moe_component import MoE_mod, Net_sub
import math

class multi_level_expert(nn.Module):
    def __init__(self, in_size, out_size, batch_size, num_experts = 2, k = 2, num_layers = 2):
        super(multi_level_expert, self).__init__()
        self.num_experts = num_experts
        self.layers = num_layers
        self.module = nn.ModuleList()
        hidden_size = in_size
        for i in range(num_layers):
            net = Net_sub(False, 1, 20, i+1)
            out_size = int(math.sqrt(hidden_size)-2)**2 * 2**(i+4)
            exp_layer = MoE_mod(input_size=hidden_size, output_size=out_size, hidden_size=2, num_experts = self.num_experts, model=net, noisy_gating=False, k=k)
            self.module.append(exp_layer)
            hidden_size = out_size
    
        self.fc1 = nn.Linear(hidden_size * batch_size, 128)
        self.fc2 = nn.Linear(128, out_size)


    def forward(self, x):
        i = 0
        x,_ = self.module[0](x)
        x,_ = self.module[1](x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))

        return x