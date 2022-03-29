from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.expertcnn import * 
from MOE import MoE
from train import * 
from multi_level_expert import multi_level_expert, multi_level_mlp_expert
from faces import datasetfc, datasetfc_test
def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--experiment', default='test', help='the path to store sampled images and models' )
    
    #parser.add_argument('--MOE', type = int, default = 0)
    args = parser.parse_args()
    testingLog = open('{0}/testingLog_{1}.txt'.format(args.experiment, args.epochs), 'w')
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True} 
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize([64,64])
        ])

    transform_s=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize([64,64]),
        transforms.Grayscale()
        ])
    dataset3e = datasets.CIFAR10('../data', train=True, download=True, transform=transform_s)
    dataset3et = datasets.CIFAR10('../data', train=False, download=True, transform=transform_s)
    train_data_path = '../identities_16' 
    face = torchvision.datasets.ImageFolder(train_data_path, transform=transform_s)

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    dataset3 = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
    dataset4 = datasets.FashionMNIST('../data', train=False,
                       transform=transform)
    

    train_ldr_dataset_1 = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    train_ldr_dataset_2 = torch.utils.data.DataLoader(dataset2,**train_kwargs)
    train_ldr_dataset_3 = torch.utils.data.DataLoader(dataset3,**train_kwargs)
    

    dataset_test = ConcatDataset([dataset2, dataset4, datasetfc_test])
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)


    name = "MLMoE"
    num_experts = 3
    num_layers = 2
    model = multi_level_expert(in_size=64*64, out_size=20, batch_size=128, num_experts=num_experts, k=3, num_layers=2).to(device)
    #model = multi_level_mlp_expert(in_size=784, out_size=20, batch_size=128, num_experts=num_experts, k=2, num_layers=2).to(device)
    #model = MoE(input_size=784, output_size=20, num_experts=num_experts, hidden_size=2, model=Net(), k=2).to(device)
    # #name = "d"
    # #model = expertNN(784, 20).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    sfd = []
    for epoch in range(1, args.epochs + 1):
        batch_size = 64
        saved_gt = train_expert(args, model, device, train_ldr_dataset_1, train_ldr_dataset_2, train_ldr_dataset_3, optimizer, epoch, name)
        saved_gt = saved_gt[-1]
        print(saved_gt[0])
        for h in range(num_layers):
            nd = []
            print("layer " + str(h+1) + ":")
            for i in range(num_experts):
                print("expert:", i+1)
                sf1 = saved_gt[h][:,i] > 0.5
                #print(sf1.shape)
                
                for j in range(num_experts):
                    print("dataset:", j+1)
                    sm1d1 = torch.sum(sf1[j * batch_size: batch_size * (j+1)] > 0.5)
                    print(sm1d1)
                    nd.append(sm1d1)
            sfd.append(nd)
                    
        

        test_expert(model, device, test_loader, testingLog, name)
        #testingLog.write()
        scheduler.step()

    if args.save_model:
        print("saving model...")
        torch.save(model.state_dict(), "mnist_cnn_expert_ML.pt")
        torch.save(torch.Tensor([sfd]), "res_MLMOE.pt")
    testingLog.close()


if __name__ == '__main__':
    main()
