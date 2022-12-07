# thesis

## NoduleMNIST3D UNet
```diff
+ >> python nodulemnist3d_unet.py
```

This will create a folder in syntax unet-results-'timestamp' in the upper directory

- / src
    - nodulemnist3d_unet.py
    - nodulemnist3d_vit.py
    - nodulemnist3d_unet_vit_test.py
- / unet-results-'timestamp'

## NoduleMNIST3D ViT
Move the best unet weights to outside of the unet-results-'timestamp' folder

```diff
+ >> python nodulemnist3d_vit.py
````

This will create a folder in syntax vit-results-'timestamp' in the upper directory

- / src
    - nodulemnist3d_unet.py
    - nodulemnist3d_vit.py
    - nodulemnist3d_unet_vit_test.py
- / unet-results-'timestamp'
- / vit-results-'timestamp'
- unet-weights.pth

## NoduleMNIST3D UNet ViT -> Test
Move the best vit weights to outside of the vit-results-'timestamp' folder

```diff
+ >> python nodulemnist3d_unet_vit_test.py
````

This will create a folder in syntax unet-vit-results-'timestamp' in the upper directory

- / src
    - nodulemnist3d_unet.py
    - nodulemnist3d_vit.py
    - nodulemnist3d_unet_vit_test.py
- / unet-results-'timestamp'
- / vit-results-'timestamp'
- / unet-vit-results-'timestamp'
- unet-weights.pth
- vit-weights.pth