import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

#basic network
class CNN(nn.Module):
    def __init__(self, layers = 5, in_channels = 1, out_features = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.5)

        #self.conv4 = nn.Conv2d(256, 512, 2, 1)
        self.classes = out_features
        self.fc = nn.Linear(73728, 128)
        self.fc2 = nn.Linear(128, self.classes)

    
    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.dropout1(x)
        #x = self.relu(self.conv4(x))
        x = x.view(list(x.size())[0], -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x

class NN(nn.Module):
    def __init__(self, layers = 5, in_channels = 1, out_features = 10):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 2, 1)
        self.classes = out_features
        self.fc = nn.Linear(73728, 100)
        self.out = nn.Linear(100, out_features)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(list(x.size())[0], -1)
        #print(x.shape)
        x = self.fc(x)
        x = self.out(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class expertNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(expertNN, self).__init__()
        self.expert_1 = Net()
        self.expert_2 = Net()
        self.conv = nn.Conv2d(1,3,3,1)
        self.gate = nn.Linear(in_features, 20)
        self.gate_2 = nn.Linear(20, 2)
        self.soft = nn.Softmax(dim=1)
        self.classes = out_features

    def forward(self, x):
        x_1 = self.expert_1(x)
        x_2 = self.expert_2(x)
        x = x.view(list(x.size())[0], -1)
        #print(x_1.shape)
        gated = self.gate(x)
        gated = self.gate_2(gated)
        gated = self.soft(gated)
        #gated = 0.5 * torch.ones((x.shape[0], 2)).cuda()
        #gate = nn.Sigmoid(self.gate(x))
        #print(gated[:,0].shape, x_1.shape)
        #print(gated[:10])
        tens_1 = torch.hstack([gated[:,0] for j in range(self.classes)]).reshape((x.shape[0],self.classes))
        tens_2 = torch.hstack([gated[:,1] for j in range(self.classes)]).reshape((x.shape[0],self.classes))
        inputs = torch.add(tens_1 * x_1, tens_2 * x_2)
        #print(inputs.shape)
        return inputs
        
    # def __call__(self, x):
    #     inputs = Variable(torch.cuda.FloatTensor(x))
    #     prediction = self.forward(inputs)
    #     return np.argmax(prediction.data.cpu().numpy(), 1)

