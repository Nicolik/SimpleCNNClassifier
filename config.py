############################################
# Nicola Altini (2020)
#
# This script contains useful global configs
# for this example.
############################################

import torchvision.transforms as transforms
import os

# Root dataset folder
dataset_folder = 'dataset_reduced'
train_folder = os.path.join(dataset_folder, "train")
val_folder = os.path.join(dataset_folder, "val")

# Set to 0 on Windows
num_workers = 0

# For exploiting pretrained models on ImageNet:
sizes = (224,224)

epochs      = 50
steps_loss  = 2
batch_size  = 32
NUM_CLASSES = 2
CHANNELS    = 3

# X
# Z = (X - mu(X)) / std(X)

# Normalization
mean_norm = (0.485, 0.456, 0.406)
std_norm  = (0.229, 0.224, 0.225)

# Train Set Transform
transform_train = transforms.Compose([
    # Always
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ############################
    # Optional
    # transforms.ColorJitter(brightness=(0, 0.5), contrast=(0.2, 0.4), saturation=(0.5, 0.8), hue=(0,0.1)),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomApply( (transforms.RandomRotation((-20,20)),) , p=0.5),
    # Add other augmentations in this section
    ############################
    # Always
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norm, std=std_norm),
])

# Test Set Transform
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norm, std=std_norm),
])
