from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

import torchmetrics
from collections import Counter
import pandas as pd

from matplotlib import pyplot as plt
from collections import OrderedDict

from datetime import datetime
import os
from typing import Optional, Tuple
import yaml
import glob

from dataset import dataset


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "../layer-attention-results-realdata-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

def get_file_paths(path):
    return glob.glob(path + "/*")


NUM_EPOCHS = 10
lr = 0.001


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

                'Max-Layers' : {'max': 700}
             }


train_dataset = dataset.Dataset(ct_scan_list[:700], ct_label_list[:700], transforms=transforms)
val_dataset = dataset.Dataset(ct_scan_list[600:700], ct_label_list[600:700], transforms=transforms)
test_dataset = dataset.Dataset(ct_scan_list[700:], ct_label_list[700:], transforms=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)


get_params = lambda m: sum(p.numel() for p in m.parameters())

# print split sizes
print("train data size: ", len(train_dataset))
print("val data size: ", len(val_dataset))
print("test data size: ", len(test_dataset))


class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size1 = 256
        feature_extractor = models.resnet.resnet18(pretrained=True)
        feature_extractor.fc = (
            nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )
        feature_extractor.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 1), feature_extractor.conv1
        )
        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size1, 8)
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        query = features.mean(0, keepdims=True)
        # print(features.shape)
        # print(query.shape)
        features, att_map = self.att(query, features, features)
        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out


# initialize model
model = AttentionClassifier()
sigmoid = nn.Sigmoid()

print(f"Complete model has {get_params(model)} params")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.01)

"""
data_sample, data_label = train_dataset[0]

print(data_sample.shape)
print(data_label)

data_sample = data_sample.permute(2, 0, 1, 3, 4)
data_sample = torch.squeeze(data_sample, 1)

print(data_sample.shape)
print(data_label)

exit()

preprocess_data = lambda x: x.squeeze().unsqueeze(1).float()
data_sample = preprocess_data(torch.from_numpy(data_sample))
# print(data_sample.shape)
model(data_sample.to(device))
"""


# pbar_epoch = tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS)
for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    total_loss = 0
    model.train()
    if epoch == 0:
        model.feature_extractor.eval()
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        model.feature_extractor.conv1[0].weight.requires_grad = True
        model.feature_extractor.conv1[0].bias.requires_grad = True
    elif epoch == 1:
        model.feature_extractor.eval()
    elif epoch == 2:
        pass
    elif epoch == 3:
        for param in model.feature_extractor.layer4.parameters():
            param.requires_grad = True
    elif epoch == 4:
        for param in model.feature_extractor.layer3.parameters():
            param.requires_grad = True
    elif epoch > 4:
        for param in model.parameters():
            param.requires_grad = True
    pbar_train_loop = tqdm(train_loader, total=len(train_loader), leave=False)
    for input, target in pbar_train_loop:
        optimizer.zero_grad()

        
        input = input.permute(2, 0, 1, 3, 4)
        input = torch.squeeze(input, 1)

        out = model(input.to(device))
        loss = criterion(out, torch.unsqueeze(target, 0).float().to(device))
        loss.backward()
        optimizer.step()
        lv = loss.detach().cpu().item()
        total_loss += lv
        pbar_train_loop.set_description_str(f"Loss: {lv:.2f}")
    print(f"\tMean train loss: {total_loss / ((epoch + 1)*len(train_loader)):.2f}")
    sched.step()
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for input, target in tqdm(val_loader, leave=False):

            input = input.permute(2, 0, 1, 3, 4)
            input = torch.squeeze(input, 1)

            out = model(input.to(device))

            preds.append(sigmoid(out.cpu().flatten()))
            targets.append(target.flatten())
    preds, targets = torch.cat(preds), torch.cat(targets)
    acc = torchmetrics.functional.accuracy(preds, targets, task="binary")
    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(
        preds,
        targets,
    )
    # print(f"\tLabel distribution: {Counter(targets.tolist())}")
    print(f"\tVal accuracy: {acc*100.0:.1f}%")
    print(f"\tVal MCC: {mcc*100.0:.1f}%")
    path_model = outputs_folder + "/model" + str(epoch) + ".pth"
    torch.save(model.state_dict(), path_model)