############################################
# Nicola Altini (2020)
#
# This script contains utility functions
# for this example
############################################

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
from torch.utils.data import SubsetRandomSampler

from config import *

cuda = torch.cuda.is_available()


def imshow(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    unorm = UnNormalize(mean=mean, std=std)
    uimg = unorm(img)
    npimg = uimg.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def resize_image(image, new_max_size):
    if image.width >= new_max_size and image.height >= new_max_size:
        if image.height < image.width:
            factor = float(new_max_size) / image.height
        else:
            factor = float(new_max_size) / image.width
        # print("Factor = ", factor)
        new_width = int(image.width* factor)
        new_height = int(image.height * factor)
        image = image.resize((new_width, new_height))
    else:
        print("It is not possible to increase image size with this method!")
    return image


def make_pred_from_path(net, image_path):
    # Get PIL Image
    pil_image = Image.open(image_path)

    # Get Tensor from PIL Image
    tensor = transform_test(pil_image)
    # tensor has shape     3 x 224 x 224
    # net needs  shape 1 x 3 x 224 x 224

    # Feed the tensor to the CNN
    out = net(tensor.unsqueeze(0))

    # Take the argmax from out
    _, label = torch.max(out, 1)

    # Convert tensor to string
    class_label = str(label.item())

    return class_label


def make_pred_on_dataloader(net, val_dataloader):
    y_pred = []
    y_true = []

    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            start_time =  time.time()
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Convert tensor from GPU to CPU to Numpy array to List
            y_pred += list(predicted.cpu().numpy())
            y_true += list(labels.cpu().numpy())

            elapsed_time = time.time() - start_time
            print('[Iter {:2d}/{:2d}] - Elapsed time = {:.3f}'.
                  format(idx+1, len(val_dataloader), elapsed_time))

    acc = correct / total
    print('Accuracy of the network on the dataloader images: {:.4f} '.format(acc))

    return y_true, y_pred


def get_classes():
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_folder,
        transform=transform_train
    )
    classes = train_dataset.classes
    classes_dict = train_dataset.class_to_idx
    return classes, classes_dict


def subsample_dataset(train_dataset, subsample_portion=0.05,
                      shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(subsample_portion * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    subset_indices = indices[:split]

    # Creating PT data samplers and loaders:
    train_subset_sampler = SubsetRandomSampler(subset_indices)
    return train_subset_sampler