from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from collections import OrderedDict

from datetime import datetime
import os

import medmnist
from medmnist import INFO, Evaluator
print("MedMNIST", "v", medmnist.__version__, "@", medmnist.HOMEPAGE)

from UNet import UNet


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = int(datetime.timestamp(dt))

outputs_folder = "unet-results-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)


NUM_EPOCHS = 10
BATCH_SIZE = 1
lr = 0.001


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
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE) # turn shuffle=True after debugging
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# print split sizes
print("train data size: ", len(train_dataset))
print("val data size: ", len(val_dataset))
print("test data size: ", len(test_dataset))


# visualize data type, get a random data sample
x, y = train_dataset[50]
print(x.shape, y.shape)
print(type(x))
print(x[0].shape)


columns = 10
rows = 1

fig = plt.figure(figsize=(20, 20))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(x[0][:,:,2*i], cmap='gray')
plt.savefig(outputs_folder + "/" + "data_slices_from_w.png")
plt.close()


fig = plt.figure(figsize=(20, 20))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(x[0][:,2*i,:], cmap='gray')
plt.savefig(outputs_folder + "/" + "data_slices_from_h.png")
plt.close()


fig = plt.figure(figsize=(20, 20))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(x[0][2*i,:,:], cmap='gray')
plt.savefig(outputs_folder + "/" + "data_slices_from_t.png")
plt.close()


# initialize model
model = UNet()
model.to(torch.double)

# use gpu if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# loss calculation
train_loss_array = []
val_loss_array = []


first_run = True
val_first_run = True

for epoch in range(NUM_EPOCHS):
    counter = 0
    total_loss = 0
    
    model.train()
    # TRAIN
    for inputs, targets in tqdm(train_loader):

        if first_run:
            print("Input shape before permute ", inputs.shape)
            fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(1, 3, 1)
            plt.imshow(inputs[0][0][5,:,:], cmap='gray')
        
        inputs = inputs.permute(2, 0, 1, 3, 4)

        if first_run:
            print("Input shape after permute ", inputs.shape)  
            fig.add_subplot(1, 3, 2)      
            plt.imshow(inputs[5][0][0][:,:], cmap='gray')

        inputs = torch.squeeze(inputs, 1)

        if first_run:
            print("Input shape after squeeze ", inputs.shape)
            fig.add_subplot(1, 3, 3)      
            plt.imshow(inputs[5][0][:,:], cmap='gray')
            plt.savefig(outputs_folder + "/" + "5th_layer_permute_squeeze.png")
            plt.close()

        first_run = False

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

            input_output_folder = "unet-results-" + str(ts) + "/epoch" + str(epoch)
            if not os.path.exists(input_output_folder):
                os.makedirs(input_output_folder)
                
            plt.savefig(input_output_folder + "/" + "counter" + str(counter) + ".png")
            plt.close()

            model.train()
        
        counter += 1

    train_loss_array.append((total_loss * 1.0 / len(train_dataset)))

    # VALIDATION
    val_counter = 0
    val_total_loss = 0
    model.eval()
    for val_inputs, val_targets in tqdm(val_loader):

        if val_first_run:
            print("Input shape before permute ", val_inputs.shape)
            fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(1, 3, 1)
            plt.imshow(val_inputs[0][0][5,:,:], cmap='gray')
        
        val_inputs = val_inputs.permute(2, 0, 1, 3, 4)

        if val_first_run:
            print("Input shape after permute ", val_inputs.shape)  
            fig.add_subplot(1, 3, 2)      
            plt.imshow(val_inputs[5][0][0][:,:], cmap='gray')

        val_inputs = torch.squeeze(val_inputs, 1)

        if val_first_run:
            print("Input shape after squeeze ", val_inputs.shape)
            fig.add_subplot(1, 3, 3)      
            plt.imshow(val_inputs[5][0][:,:], cmap='gray')
            plt.savefig(outputs_folder + "/" + "val_5th_layer_permute_squeeze.png")
            plt.close()

        val_first_run = False

        val_inputs = val_inputs.to(device)
        _, val_outputs = model(val_inputs)
        loss = criterion(val_outputs, val_inputs)

        val_total_loss += loss.item()

        if(val_counter%30 == 0):            
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
                
            plt.savefig(input_output_folder + "/" + "val_counter" + str(val_counter) + ".png")
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
