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

from tqdm import tqdm

from dataset import dataset
from model.UNet import UNet
from model.ViT import ViT


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "../vit-results-realdata-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

def get_file_paths(path):
    return glob.glob(path + "/*")


NUM_EPOCHS = 10
BATCH_SIZE = 1
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


# ROSARIO^CHARLEY^STEWART exists 2 times
# ROSARIO^CHARLEY^STEWART,../data/PDAC_CT/pseudonymised/ROSARIO^CHARLEY^STEWART.nii,PDAC_CT,True,1.0,66,True,True,1.0,False,False,0,False,PDAC,0,,,3,4,2,0,-1,2,1,1,0.0,0.0,1.0,157,1.49,,1,,1,True,SD,False,0,-1,1,0,12.9,16.3
# ROSARIO^CHARLEY^STEWART,../data/PDAC_CT/pseudonymised/ROSARIO^CHARLEY^STEWART.nii,PDAC_CT,True,1.0,57,False,True,-1.0,False,False,2,False,Adeno-CA,1,,,4,c3,c2,1,0,-1,-1,-1,-1.0,-1.0,-1.0,10882,41.16,,,1,1,True,PD,False,1,0,1,1,2.4,8.6

for ct_file in ct_file_paths:
    ct_file_name = os.path.basename(ct_file)

    # If the patient name exists only once in pseudonymised_patient_info.csv
    if len(ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()) == 1:
        
        ct_index = ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()[0]

        # If the patient name is not in Problematic_CTs
        if len(ct_labels_exclude_df.index[ct_labels_exclude_df['Patient_name'] == ct_file_name[:-4]].tolist()) == 0:
            ct_scan_list.append(ct_file)
            ct_label_list.append(ct_labels_df.loc[ct_index]['Geschlecht'])
            
            
#[458,1,128,128]

#[96,1,256,256]
#[92,1,256,256] --> matplotlib
#[55,1,256,256] --> matplotlib + validate

#[136,1,208,256]
#[100+,1,208,256] --> matplotlib
#[68,1,208,256] --> matplotlib + validate

#[23,1,512,512]

transforms = {
                'Clip': {'amin': -150, 'amax': 250},

                # Normalize so the values are between 0 and 1
                'Normalize': {'bounds': [-150, 250]},

                'Resize': {'height': 256, 'width': 256},

                'Crop-Height' : {'begin': 0, 'end': 256},
                'Crop-Width' : {'begin': 0, 'end': 256},

                'Max-Layers' : {'max': 55}
             }


train_dataset = dataset.Dataset(ct_scan_list[:600], ct_label_list[:600], transforms=transforms)
val_dataset = dataset.Dataset(ct_scan_list[600:700], ct_label_list[600:700], transforms=transforms)
test_dataset = dataset.Dataset(ct_scan_list[700:], ct_label_list[700:], transforms=transforms)
#val_dataset = dataset.Dataset(ct_scan_list[700:726] + ct_scan_list[727:789] + ct_scan_list[790:], ct_label_list[700:726] + ct_label_list[727:789] + ct_label_list[790:], transforms=transforms)
#val_dataset = dataset.Dataset(ct_scan_list[700:789] + ct_scan_list[790:], ct_label_list[700:789] + ct_label_list[790:], transforms=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# initialize Unet model
unet = UNet()
unet.load_state_dict(torch.load('model9.pth'))

# use gpu if exists
unet.to(device)
unet.eval()


# initialize ViT model
model = ViT(
    dim=512,
    num_patches=55,
    patch_dim=512,
    num_classes=2,
    channels=1,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# use gpu if exists
model.to(device)

# loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss calculation
train_loss_array = []
val_loss_array = []

for epoch in range(NUM_EPOCHS):
    counter = 0
    total_loss = 0
    
    model.train()
    # TRAIN
    for inputs, targets in tqdm(train_loader):

        inputs = inputs.permute(2, 0, 1, 3, 4)
        inputs = torch.squeeze(inputs, 1)
        #inputs = sigmoid(inputs)

        inputs = inputs.to(device)

        features_extracted, _ = unet(inputs)

        features_extracted = features_extracted.permute(2, 3, 0, 1)
        features_extracted = torch.squeeze(features_extracted, 0)
        
        optimizer.zero_grad()

        outputs = model(features_extracted)

        outputs = sigmoid(outputs)
        labels = targets
        
        labels = labels.to(device)
        labels = torch.unsqueeze(labels, 0).to(torch.float)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if(counter%100 == 0):
            print("outputs", outputs)
            print("labels", labels)
            print("loss", loss)

        counter += 1

    train_loss_array.append((total_loss * 1.0 / len(train_dataset)))

    # VALIDATION
    val_counter = 0
    val_total_loss = 0
    model.eval()
    for val_inputs, val_targets in tqdm(val_loader):

        val_inputs = val_inputs.permute(2, 0, 1, 3, 4)
        val_inputs = torch.squeeze(val_inputs, 1)

        val_inputs = val_inputs.to(device)

        val_features_extracted, _ = unet(val_inputs)
        
        val_features_extracted = val_features_extracted.permute(2, 3, 0, 1)
        val_features_extracted = torch.squeeze(val_features_extracted, 0)
        
        val_outputs = model(val_features_extracted)
        
        
        val_outputs = sigmoid(val_outputs)

        # For sigmoid
        val_labels = val_targets

        val_labels = val_labels.to(device)
        val_labels = torch.unsqueeze(val_labels, 0).to(torch.float)
        
        loss = criterion(val_outputs, val_labels)
        val_total_loss += loss.item()

        val_counter += 1

    model.train()

    val_loss_calculated = val_total_loss * 1.0 / len(val_dataset)

    if len(val_loss_array) == 1 or all(i >= val_loss_calculated for i in val_loss_array):
        path_model = outputs_folder + "/model" + str(epoch) + ".pth"
        torch.save(model.state_dict(), path_model)

    val_loss_array.append((val_total_loss * 1.0 / len(val_dataset)))

    print(train_loss_array)
    print(val_loss_array)


print(train_loss_array)
print(val_loss_array)

indices_list = []
for i in range(0,NUM_EPOCHS):
    indices_list.append(i)


fig = plt.figure(figsize=(20, 20))
plt.plot(indices_list, train_loss_array, label='train loss')
plt.plot(indices_list, val_loss_array, label='val loss')
plt.legend()
plt.savefig(outputs_folder + "/" + "train_val_loss" + str(val_counter) + ".png")
