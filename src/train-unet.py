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


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "../unet-results-realdata-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

def get_file_paths(path):
    return glob.glob(path + "/*")


NUM_EPOCHS = 20
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
            
            
transforms = {
                'Clip': {'amin': -150, 'amax': 250},
                # Normalize so the values are between 0 and 1
                'Normalize': {'bounds': [-150, 250]}
             }


train_dataset = dataset.Dataset(ct_scan_list[:700], ct_label_list[:700], transforms=transforms)
val_dataset = dataset.Dataset(ct_scan_list[700:789] + ct_scan_list[790:], ct_label_list[700:789] + ct_label_list[790:], transforms=transforms)
#val_dataset = dataset.Dataset(ct_scan_list[700:726] + ct_scan_list[727:789] + ct_scan_list[790:], ct_label_list[700:726] + ct_label_list[727:789] + ct_label_list[790:], transforms=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)


# initialize model
model = UNet()
#model.to(torch.double)

# use gpu if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
sigmoid = nn.Sigmoid()

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
        optimizer.zero_grad()

        _, outputs = model(inputs)

        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


        # evaluate model outputs
        # in every 300th instance check the progress
        if(counter%300 == 0):
            model.eval()
            _, outputs = model(inputs)
            
            inputs = inputs.to("cpu")
            outputs = outputs.to("cpu")

            inputs = inputs.detach().numpy()
            outputs = outputs.detach().numpy()

            columns = 2
            rows = 1
            
            fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(inputs[10][0][:,:], cmap='gray')
            
            fig.add_subplot(rows, columns, 2)
            plt.imshow(outputs[10][0][:,:], cmap='gray')

            epoch_outputs_folder = outputs_folder + "/epoch" + str(epoch)
            if not os.path.exists(epoch_outputs_folder):
                os.makedirs(epoch_outputs_folder)
                
            plt.savefig(epoch_outputs_folder + "/" + "counter" + str(counter) + ".png")
            plt.close()

            model.train()
        
        counter += 1

    train_loss_array.append((total_loss * 1.0 / len(train_dataset)))

    # VALIDATION
    val_counter = 0
    val_total_loss = 0
    model.eval()
    for val_inputs, val_targets in tqdm(val_loader):

        val_inputs = val_inputs.permute(2, 0, 1, 3, 4)
        val_inputs = torch.squeeze(val_inputs, 1)
        #val_inputs = sigmoid(val_inputs)

        val_inputs = val_inputs.to(device)
        _, val_outputs = model(val_inputs)
        loss = criterion(val_outputs, val_inputs)

        val_total_loss += loss.item()

        if(val_counter%70 == 0):            
            val_inputs = val_inputs.to("cpu")
            val_outputs = val_outputs.to("cpu")

            val_inputs = val_inputs.detach().numpy()
            val_outputs = val_outputs.detach().numpy()

            columns = 2
            rows = 1
            
            fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(val_inputs[10][0][:,:], cmap='gray')
            
            fig.add_subplot(rows, columns, 2)
            plt.imshow(val_outputs[10][0][:,:], cmap='gray')
                
            plt.savefig(epoch_outputs_folder + "/" + "val_counter" + str(val_counter) + ".png")
            plt.close()

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