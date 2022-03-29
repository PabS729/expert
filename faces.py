from scipy.misc import face
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import cv2
import glob
import numpy
import random
from functools import reduce
from operator import concat
from torchvision import transforms as TR
from augmentations import * 
import torchvision


train_data_path = '../identities_16' 
test_data_path = '../test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values



#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = list(reduce(concat, train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

sf = train_image_paths
#2.
# split train valid from train paths (80,20)
train_image_paths, test_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

# #3.
# # create the test_image_paths
# test_image_paths = []
# for data_path in glob.glob(train_data_path + '/*'):
#     test_image_paths.append(glob.glob(data_path + '/*'))

# test_image_paths = list(flatten(test_image_paths))

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(class_to_idx)

class faces_dataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[1]
        #print(label)
        label = label.split('\\')
        label = label[0]+"\\"+label[1]
        #print(label)
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

transform_s=transforms.Compose([
    TR.ToTensor(),
    # TR.Normalize((0.4810, 0.4055, 0.3830), (0.2742, 0.2440, 0.2387)),
    # TR.Resize([64,64]),
    # TR.Grayscale(),
    ])




dataset_fc_train = faces_dataset(train_image_paths, transform=transform_s)
#print(dataset_fc_train[0])
dataset_fc_test = faces_dataset(test_image_paths, transform=transform_s)

#face = torchvision.datasets.ImageFolder(train_data_path, transform=transform_s)
faces = faces_dataset(sf, transform=transform_s)

# loader = DataLoader(
#     dataset_fc_test,
#     batch_size=64,
#     num_workers=0,
#     shuffle=False
# )
# mean = 0.
# std = 0.
# nb_samples = 0.
# for data in loader:
#     #print(data[0].shape)
#     batch_samples = data[0].size(0)
#     data = data[0].view(batch_samples, data[0].size(1), -1)
#     #print(data.shape)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples
# print(mean, std)




size = [64,64]
rdc = TR.RandomHorizontalFlip()
nd = TR.RandomCrop(size)
kd = TR.RandomRotation(10)
ms = TR.RandomAffine(10)
sk = TR.RandomResizedCrop([64,64], scale = (0.5,1))
pd = TR.ColorJitter(brightness=.5, hue=.3)

arr_tr = [rdc, nd, kd, ms, sk, pd]
for i in range (0,6):
    for tr in arr_tr:
        transform_s=transforms.Compose([
        TR.ToTensor(),
        TR.Normalize((0.4810, 0.4081, 0.3830), (0.2742, 0.2440, 0.2387)),
        TR.Resize([64,64]),
        tr,
        TR.Grayscale(),
        
        ])

        dataset_fc_train_cd = faces_dataset(train_image_paths, transform = transform_s)
        dataset_fc_train = ConcatDataset([dataset_fc_train, dataset_fc_train_cd])
        dataset_fc_test_cd = faces_dataset(test_image_paths, transform = transform_s)
        dataset_fc_test = ConcatDataset([dataset_fc_test, dataset_fc_test_cd])


datasetfc = dataset_fc_train
# datasetfc = ConcatDataset([dataset_fc_train, dataset_fc_train_cd])

datasetfc_test = dataset_fc_test




