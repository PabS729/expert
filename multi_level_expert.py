
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from models.expertcnn import *
import numpy as np
import torchvision.models as models
from moe_component import MoE_mod, Net_sub
import math
from MOE import MoE

class multi_level_expert(nn.Module):
    def __init__(self, in_size, out_size, batch_size, num_experts=2, k=2, num_layers=2):
        super(multi_level_expert, self).__init__()
        self.num_experts = num_experts
        self.layers = num_layers
        self.module = nn.ModuleList()
        hidden_size = in_size

        self.net_1 = Net_sub(False, 1, 20, 1)
        self.net_2 = Net_sub(False, 1, 20, 2)
        out_size_1 = int(math.sqrt(hidden_size)-2)**2 * 2**(4)
        print(out_size_1)
        out_size_2 = int(math.sqrt(out_size_1)-2)**2 * 2**(5)
        self.exp_layer_1 = MoE_mod(input_size=hidden_size, output_size=out_size_1, hidden_size=2,
                                   num_experts=self.num_experts, model=self.net_1, noisy_gating=False, k=k)
        self.exp_layer_2 = MoE_mod(input_size=out_size_1, output_size=out_size_2, hidden_size=2,
                                   num_experts=self.num_experts, model=self.net_2, noisy_gating=False, k=k)

        # for i in range(num_layers):
        #     net = Net_sub(False, 1, 20, i+1)
        #     out_size = int(math.sqrt(hidden_size)-2)**2 * 2**(i+4)
        #     exp_layer = MoE_mod(input_size=hidden_size, output_size=out_size, hidden_size=2,
        #                         num_experts=self.num_experts, model=net, noisy_gating=False, k=k)
        #     self.module.append(exp_layer)
        #     hidden_size = out_size
        
        self.fc1 = nn.Linear(115200, 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x, ls1, gt_l1 = self.exp_layer_1(x, 3e-4)
        #print("x:", gt_l1.shape)
        x, ls2, gt_l2 = self.exp_layer_2(x, 3e-4)
        
        x = x.view(x.size(0), -1)
        #print("x:", x.shape)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x, gt_l1, gt_l2, ls1 + ls2

class multi_level_mlp_expert(nn.Module):
    def __init__(self, in_size, out_size, batch_size, num_experts=2, k=2, num_layers=2):
        super(multi_level_mlp_expert, self).__init__()
        self.net_1 = Net_sub(False, 1, 20, 1)
        self.net_2 = Net_sub(False, 1, 20, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        hidden_size = 18432
        hidden_size = 10816
        self.fc1 = nn.Linear(hidden_size,128)
        self.fc2 = nn.Linear(128,out_size)
        self.exp_layer_1 = MoE(input_size=hidden_size, output_size=128, hidden_size=2,
                                   num_experts=num_experts, model=self.fc1, noisy_gating=False, k=k)
        self.exp_layer_2 = MoE(input_size=128, output_size=out_size, hidden_size=2,
                                   num_experts=num_experts, model=self.fc2, noisy_gating=False, k=k)

    def forward(self, x):
        x1 = x[:int(x.shape[0]/2)]
        x2 = x[int(x.shape[0]/2):]
        x1 = self.net_1(x1)
        x2 = self.net_1(x2)
        #print(x.shape)
        #x = self.net_2(x)

        #print("x:", x2.shape)

        x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3])
        x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2] * x2.shape[3])
        x = torch.cat((x1, x2))
        x, ls1, gt_l1 = self.exp_layer_1(x, train=True, loss_coef=3e-5)
        x = F.relu(x)
        x, ls2, gt_l2 = self.exp_layer_2(x, train=True, loss_coef=3e-5)
        x = F.log_softmax(x)
        return x, gt_l1, gt_l2, ls1 + ls2

    

