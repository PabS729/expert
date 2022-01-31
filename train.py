from __future__ import print_function
import argparse
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print("-------------------------------------------------------------------")
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def train_expert(args, model, device, train_loader, ldr2, optimizer, epoch, name):
    saved_gt = []
    model.train()
    ct = 0
    for batch_idx, ((data, target),(data_2, target_2)) in enumerate(zip(train_loader, ldr2)):
        dat = torch.cat((data, data_2))
        targ = torch.cat((target, target_2))
        # plt.imshow(dat[65].reshape(48,48,1))
        # plt.show()
        data, target = dat.to(device), targ.to(device)
        
        #print("----------------------------------------------------------------------")
        #print(targ)
        optimizer.zero_grad()
        #output = model(data)
        if name == "MoE":
            out, aux, gt= model(data)

            loss = F.nll_loss(out, target)
            total_loss = loss + aux

        elif name == "MLMoE":
            out, gt1, gt2, aux = model(data)
            total_loss = F.nll_loss(out, target) + aux
        else:
            out = model(data)
            total_loss = F.nll_loss(out, target)
        ct+=1
        total_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))
            if name == "MoE":
                saved_gt.append(gt)
            if name == "MLMoE":
                saved_gt.append([gt1,gt2])

            if args.dry_run:
                break
        # for im in data[:5]:
        #     img = im.cpu()
        #     plt.imshow(img.reshape((28,28,1)))
        #     plt.show()
        #     print(model.w_gate[:])
    return saved_gt




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_expert(model, device, test_loader, logger, name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if name == "MoE":
                output, aux,g = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                test_loss += aux
            elif name == "MLMoE":
                output, g1, g2, aux = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                test_loss += aux
            else:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))