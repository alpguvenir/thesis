from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from collections import OrderedDict
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score

from datetime import datetime
import os

import medmnist
from medmnist import INFO, Evaluator
print("MedMNIST", "v", medmnist.__version__, "@", medmnist.HOMEPAGE)

from UNet import UNet
from ViT import ViT

# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "../unet-vit-results-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

BATCH_SIZE = 1

# assign dataset name to data_flag
data_flag = 'nodulemnist3d'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])


# print information about dataset
print("Task on this dataset is", task)
print("Number of channels", n_channels)
print("Number of classes", n_classes)


DataClass = getattr(medmnist, info['python_class'])


# load the data for different splits
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)


# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# turn shuffle=True after debugging


# print split sizes
print("train data size: ", len(train_dataset))
print("val data size: ", len(val_dataset))
print("test data size: ", len(test_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize Unet model
unet = UNet()
unet.to(torch.double)
unet.load_state_dict(torch.load('../model1.pth'))

# use gpu if exists
unet.to(device)
unet.eval()

# initialize ViT model
vit = ViT(
    dim=512,
    num_patches=28,
    patch_dim=512,
    num_classes=2,
    channels=1,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)
vit.to(torch.double)
vit.load_state_dict(torch.load('../model15.pth'))

vit.to(device)
vit.eval()

sigmoid = nn.Sigmoid()

_labels = []
_outputs = []

for inputs, targets in tqdm(test_loader):

    inputs = inputs.permute(2, 0, 1, 3, 4)
    inputs = torch.squeeze(inputs, 1)

    inputs = inputs.to(device)

    features_extracted, _ = unet(inputs)
    features_extracted = features_extracted.permute(2, 3, 0, 1)
    features_extracted = torch.squeeze(features_extracted, 0)

    outputs = vit(features_extracted)
    outputs = sigmoid(outputs)

    labels = targets.to(torch.double)

    _labels.append(labels[0][0])
    _outputs.append(outputs[0][0])

print("number of 1", _labels.count(1))
print("number of 0", _labels.count(0))

arr_labels = [t.cpu().detach().numpy().flat[0] for t in _labels]
arr_outputs = [t.cpu().detach().numpy().flat[0] for t in _outputs]

fpr, tpr, thresholds = roc_curve(arr_labels, arr_outputs)
auc = roc_auc_score(arr_labels, arr_outputs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig(outputs_folder + "/roc.png")

# Selected threshold as 0.5
_outputs = torch.FloatTensor(_outputs)
_labels = torch.FloatTensor(_labels)

_outputs = (_outputs>0.5).float()

metric = BinaryConfusionMatrix()
print(metric(_outputs, _labels))

"""
number of 1 64
number of 0 246
tensor([[228,  18],
        [ 32,  32]])
        
accuracy = 0,8387096774
"""