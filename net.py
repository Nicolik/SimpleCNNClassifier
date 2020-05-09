############################################
# Nicola Altini (2020)
#
# The purpose of this script is to show how to build a simple CNN.
# There is the definition of a class Net and then examples on use.
############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels=3, out_features=2):
        super(Net, self).__init__()

        # input has shape B x in_channels x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        # output has shape B x 32 x 224 x 224

        self.pool1 = nn.MaxPool2d(2, 2)
        # output has shape B x 32 x 112 x 112

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        # output has shape B x 64 x 112 x 112

        self.pool2 = nn.MaxPool2d(2, 2)
        # output has shape B x 64 x 56 x 56

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        # output has shape B x 128 x 56 x 56

        self.pool3 = nn.MaxPool2d(2, 2)
        # output has shape B x 128 x 28 x 28

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        # output has shape B x 256 x 28 x 28

        self.pool4 = nn.MaxPool2d(2, 2)
        # output has shape B x 256 x 14 x 14

        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        # output has shape B x 128

        self.fc2 = nn.Linear(128, out_features=out_features)
        # output has shape B x out_features

        # builtin in CrossEntropyLoss
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # builtin in CrossEntropyLoss
        # x = self.softmax(x)
        return x


#%%
do_test = False
if do_test:
    batch_size = 1
    num_channels = 3
    image_size = 224
    num_classes = 2

    shape_in = (batch_size, num_channels, image_size, image_size)
    shape_out_cnn = (batch_size, num_classes)
    shape_out_gt = (batch_size,)
    net = Net()
    example_in = torch.rand(shape_in)
    example_out = net(example_in)

    print("Out Shape CNN = ", shape_out_cnn)
    print("Out Shape GT  = ", shape_out_gt)

    example_gt = torch.ones(shape_out_gt)
    example_gt = example_gt.long()

    print("CNN Output: ", example_out)
    print("GT  Output: ", example_gt)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(example_out, example_gt)

    print("Cross-Entropy Loss = ", loss.item())
