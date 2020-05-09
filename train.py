############################################
# Nicola Altini (2020)
#
# This is the script which contains the code for training the CNN.
# You have to split your train dataset in two folders before:
#   - cat
#   - dog
# See prepare_dataset.py
############################################


import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from net import Net
from utils import imshow, subsample_dataset
from config import *

cuda = torch.cuda.is_available()

#%% Identification of classes of interest
classes_interest = os.listdir(train_folder)
print(f'Classes = {classes_interest}')

#%% Dataset creation
print("Creating training   dataset from ", train_folder)

train_dataset = torchvision.datasets.ImageFolder(
        root=train_folder,
        transform=transform_train
    )

classes = train_dataset.classes
classes_dict = train_dataset.class_to_idx
class_dict_inverted = {v : k for k, v in classes_dict.items()}
print(f"Classes             = {classes}")
print(f"Classes Dict (k: v) = {classes_dict}")
print(f"Classes Dict (v: k) = {class_dict_inverted}")

#%% Create Train Dataloaders
train_datasampler = subsample_dataset(train_dataset, subsample_portion=.05)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_datasampler, num_workers=num_workers)

#%% Iterable on dataloader
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# Show the batch of input images as grid
imshow(torchvision.utils.make_grid(images), mean=mean_norm, std=std_norm)

image_ = images[0]
label_ = classes[labels[0]]
print(f'Image min = {image_.min()} - max = {image_.max()}')
imshow(image_, mean=mean_norm, std=std_norm)
print(f'Label: {label_}')

#%% Instantiate a network
net = Net(in_channels=CHANNELS, out_features=NUM_CLASSES)

# Move the network on CUDA
if cuda:
    net = net.cuda()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#%% Loop on dataset
for epoch in range(epochs):  # loop over the dataset multiple times

    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # K = 2 classes (cat / dog)
        # inputs : B x C x H x W
        # outputs: B x K

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
            running_loss += loss.item()
            if i % steps_loss == (steps_loss-1):
                # print every steps_loss mini-batches
                np_pred = np.argmax(outputs.cpu().numpy(),-1)
                np_gt = labels.cpu().numpy()
                acc = np.sum(np_pred == np_gt) / batch_size
                print("[Epoch {:2d} - Iter {:3d}] loss: {:.3f} acc: {:.3f}".format(epoch + 1, i + 1, running_loss / steps_loss, acc))
                running_loss = 0.0
    elapsed_time = time.time() - start_time
    print("[Epoch {:2d}] elapsed time: {:.3f}".format(epoch+1, elapsed_time) )

print('Finished Training')

#%% Save the network
logs_dir = './logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
PATH = os.path.join(logs_dir, 'dog_vs_cat.pth')
torch.save(net.state_dict(), PATH)

#%% Save traced model (e.g. use in a C++ project)
net.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(net.cpu(), example)
PATH_TRACED = os.path.join(logs_dir, 'traced_dog_vs_cat.pt')
traced_script_module.save(PATH_TRACED)