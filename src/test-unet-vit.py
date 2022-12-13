import yaml
import glob
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nibabel as nib
from PIL import Image, ImageOps
import cv2 
import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score

from tqdm import tqdm

from dataset import dataset
from model.UNet import UNet
from model.ViT import ViT


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "../test-unet-vit-results-realdata-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

def get_file_paths(path):
    return glob.glob(path + "/*")

with open('parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Lists for containing pseudoname of the patient at its class label
ct_scan_list = []
ct_label_list = []

# Read 
ct_file_paths = get_file_paths(params_dict.get("cts.directory"))
ct_labels_path = params_dict.get("cst.label.csv")
ct_labels_exclude_path = params_dict.get("cts.label.problematic")

ct_labels_df = pd.read_csv(ct_labels_path, index_col=0)
ct_labels_exclude_df = pd.read_csv(ct_labels_exclude_path, index_col=False)

for ct_file in ct_file_paths:
    ct_file_name = os.path.basename(ct_file)

    # If the patient name exists only once in pseudonymised_patient_info.csv
    if len(ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()) == 1:
        
        ct_index = ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()[0]

        # If the patient name is not in Problematic_CTs
        if len(ct_labels_exclude_df.index[ct_labels_exclude_df['Patient_name'] == ct_file_name[:-4]].tolist()) == 0:
            ct_scan_list.append(ct_file)
            ct_label_list.append(ct_labels_df.loc[ct_index]['Geschlecht'])
            

transforms = {
                'Clip': {'amin': -150, 'amax': 250},

                # Normalize so the values are between 0 and 1
                'Normalize': {'bounds': [-150, 250]},

                'Resize': {'height': 256, 'width': 256},

                'Crop-Height' : {'begin': 0, 'end': 256},
                'Crop-Width' : {'begin': 0, 'end': 256},

                'Max-Layers' : {'max': 55}
             }


test_dataset = dataset.Dataset(ct_scan_list[700:], ct_label_list[700:], transforms=transforms)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# initialize Unet model
unet = UNet()
unet.load_state_dict(torch.load('model9.pth'))

# use gpu if exists
unet.to(device)
unet.eval()

# initialize ViT model
vit = ViT(
    dim=512,
    num_patches=55,
    patch_dim=512,
    num_classes=2,
    channels=1,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

vit.to(torch.double)
vit.load_state_dict(torch.load('model8.pth'))

vit.to(device)
vit.eval()

sigmoid = nn.Sigmoid()

_labels = []
_outputs = []

with torch.no_grad():
    for inputs, targets in tqdm(test_loader):

        if(str(torch.unsqueeze(targets, 0)[0][0].cpu().detach().numpy().flat[0]) != "nan"):

            inputs = inputs.permute(2, 0, 1, 3, 4)
            inputs = torch.squeeze(inputs, 1)

            inputs = inputs.to(device)

            features_extracted, _ = unet(inputs)
            features_extracted = features_extracted.permute(2, 3, 0, 1)
            features_extracted = torch.squeeze(features_extracted, 0)

            outputs = vit(features_extracted)
            outputs = sigmoid(outputs)

            labels = targets
            labels = labels.to(device)
            labels = torch.unsqueeze(labels, 0).to(torch.float)

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

print(_outputs)
print(_labels)

_outputs = (_outputs>0.5).float()

metric = BinaryConfusionMatrix()
print(metric(_outputs, _labels))
