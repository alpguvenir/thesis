import yaml
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nibabel as nib
from PIL import Image, ImageOps
import cv2 
import pandas as pd

from tqdm import tqdm

from dataset import dataset
from model import UNet

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
                'Normalize': {'bounds': [-150, 250]}
            }

train_dataset = dataset.Dataset(ct_scan_list, ct_label_list, transforms=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=1)

NUM_EPOCHS = 10
BATCH_SIZE = 1
lr = 0.001

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

total_loss = 0

model.train()
for inputs, targets in tqdm(train_loader):
    print("inputs ", inputs.shape, " targets", targets )


    inputs = inputs.permute(2, 0, 1, 3, 4)
    inputs = torch.squeeze(inputs, 1)
    #inputs = sigmoid(inputs)

    print(inputs.shape)

    inputs = torch.rand(30, 1, 512, 512)

    print(inputs.shape)

    inputs = inputs.to(device)
    optimizer.zero_grad()

    _, outputs = model(inputs)

    loss = criterion(outputs, inputs)
    total_loss += loss.item()

    loss.backward()
    optimizer.step()