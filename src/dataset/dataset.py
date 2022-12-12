import json
import os

import torch
import numpy as np
from matplotlib import pyplot as plt

import nibabel as nib
from PIL import Image
from skimage import color
from skimage import io
import cv2 

from typing import Any, Type

class Dataset(torch.utils.data.Dataset):

    def __init__(self, ct_scans: list[str], ct_labels: list[str], transforms: dict = None, scan_type: Type[Any] = np.float32, label_type: Type[Any] = np.int64):

        self.ct_scans = ct_scans
        self.ct_labels = ct_labels

        self.transforms = transforms
        self.scan_type = scan_type
        self.label_type = label_type


    def __len__(self) -> int:
        return len(self.ct_scans)

    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ct_scan, ct_label = (
            self.ct_scans[index],
            self.ct_labels[index]   
        )

        amin = self.transforms['Clip']['amin']
        amax = self.transforms['Clip']['amax']
        lower_bound = self.transforms['Normalize']['bounds'][0]
        upper_bound = self.transforms['Normalize']['bounds'][1]
        

        # Open here the image by nibabel
        ct_instance = nib.load(ct_scan).get_fdata()
        
        # H, W, Layers -> 512 x 512 x L
        ct_instance_shape = ct_instance.shape
        ct_instance_layer_number = ct_instance_shape[2]

        ct_instance_tensor = []
        
        # Executed in the exact order they are specified.
        # Each image would first be clipped and then normalized.
        for ct_instance_layer_index in range(ct_instance_layer_number):
            ct_instance_layer = ct_instance[:,:,ct_instance_layer_index]
            
            """
            plt.imshow(ct_instance_layer, cmap='gray')
            plt.savefig("/home/guevenira/attention_CT/development/src/data_slice.png")
            plt.close()
            """
            
            ct_instance_layer_clipped = np.clip(ct_instance_layer, amin, amax)
            """
            plt.imshow(ct_instance_layer_clipped, cmap='gray')
            plt.savefig("/home/guevenira/attention_CT/development/src/data_slice_clip.png")
            plt.close()
            """

            ct_instance_layer_clippep_normalized = (ct_instance_layer_clipped - (lower_bound)) / ((upper_bound) - (lower_bound))
            #print("XXXXXX")
            #print(ct_instance_layer_clippep_normalized)
            #print(np.max(ct_instance_layer_clippep_normalized))
            #print(np.min(ct_instance_layer_clippep_normalized))
            
            """
            plt.imshow(ct_instance_layer_clippep_normalized, cmap='gray')
            plt.savefig("/home/guevenira/attention_CT/development/src/data_slice_clip_normalize.png")
            plt.close()
            """

            ct_instance_layer_clippep_normalized_rotated = np.rot90(ct_instance_layer_clippep_normalized)

            """
            plt.imshow(ct_instance_layer_clippep_normalized_rotated, cmap='gray')
            plt.savefig("/home/guevenira/attention_CT/development/src/data_slice_clip_normalize_rotate.png")
            plt.close()
            """

            ct_instance_layer_clippep_normalized_rotated = cv2.resize(ct_instance_layer_clippep_normalized_rotated, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

            ct_instance_layer_clippep_normalized_rotated_cropped = ct_instance_layer_clippep_normalized_rotated[16:64,0:64]

            if ct_instance_tensor == []:
                ct_instance_tensor = torch.tensor(ct_instance_layer_clippep_normalized_rotated_cropped.copy(), dtype=torch.float)
                ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)
                #print(ct_instance_tensor.shape)
            else:
                ct_instance_tensor_new_layer = torch.tensor(ct_instance_layer_clippep_normalized_rotated_cropped.copy(), dtype=torch.float)
                ct_instance_tensor_new_layer = torch.unsqueeze(ct_instance_tensor_new_layer, 0)
                #print(ct_instance_tensor_new_layer.shape)
                ct_instance_tensor = torch.cat((ct_instance_tensor, ct_instance_tensor_new_layer), 0)
                #print(ct_instance_tensor.shape)


        ct_instance_tensor = torch.unsqueeze(ct_instance_tensor, 0)

        return ct_instance_tensor, torch.tensor(ct_label)

        # Apply transformation to scan image if transform dict is not empty
        if not self.transform:
            return ct_scan, ct_label
        else:
            if 'clip' in self.transform.keys():
                print("clip")
                return ct_scan, ct_label
            if 'normalize' in self.transform.keys():
                return ct_scan, ct_label
            if 'resize' in self.transform.keys():
                return ct_scan, ct_label
        

