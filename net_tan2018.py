############################################
# Nicola Altini (2020)
#
# The purpose of this script is to show how to build a CNN as described in
# @article{TAN2018127,
# title = {Age-related Macular Degeneration detection using deep convolutional neural network},
# journal = {Future Generation Computer Systems},
# volume = {87},
# pages = {127-135},
# year = {2018},
# issn = {0167-739X},
# doi = {https://doi.org/10.1016/j.future.2018.05.001},
# url = {https://www.sciencedirect.com/science/article/pii/S0167739X17319167},
# author = {Jen Hong Tan and Sulatha V. Bhandary and Sobha Sivaprasad and Yuki Hagiwara and Akanksha Bagchi and U. Raghavendra and A. {Krishna Rao} and Biju Raju and Nitin Shridhara Shetty and Arkadiusz Gertych and Kuang Chua Chua and U. Rajendra Acharya},
# keywords = {Age-related Macular Degeneration, Aging, Computer-aided diagnosis system, Convolutional neural network, Deep learning, Fundus images},
# abstract = {Age-related Macular Degeneration (AMD) is an eye condition that affects the elderly. Further, the prevalence of AMD is rising because of the aging population in the society. Therefore, early detection is necessary to prevent vision impairment in the elderly. However, organizing a comprehensive eye screening to detect AMD in the elderly is laborious and challenging. To address this need, we have developed a fourteen-layer deep Convolutional Neural Network (CNN) model to automatically and accurately diagnose AMD at an early stage. The performance of the model was evaluated using the blindfold and ten-fold cross-validation strategies, for which the accuracy of 91.17% and 95.45% were respectively achieved. This new model can be utilized in a rapid eye screening for early detection of AMD in the elderly. It is cost-effective and highly portable, hence, it can be utilized anywhere.}
# }
############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTan2018(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super(NetTan2018, self).__init__()
        oc = 16
        self.conv1   = nn.Conv2d(in_channels=in_channels, out_channels=oc, kernel_size=(3,3), padding=0)
        self.max1    = nn.MaxPool2d(2,2)
        self.conv2   = nn.Conv2d(in_channels=oc, out_channels=oc*2, kernel_size=(3,3), padding=0)
        self.max2    = nn.MaxPool2d(2,2)
        self.conv3   = nn.Conv2d(in_channels=oc*2, out_channels=oc*2, kernel_size=(3,3), padding=0)
        self.max3    = nn.MaxPool2d(2,2)
        self.conv4   = nn.Conv2d(in_channels=oc*2, out_channels=oc*4, kernel_size=(3,3), padding=0)
        self.conv5   = nn.Conv2d(in_channels=oc*4, out_channels=oc*4, kernel_size=(3,3), padding=0)
        self.max5    = nn.MaxPool2d(2,2)
        self.conv6   = nn.Conv2d(in_channels=oc*4, out_channels=oc*8, kernel_size=(3,3), padding=0)
        self.conv7   = nn.Conv2d(in_channels=oc*8, out_channels=oc*8, kernel_size=(3,3), padding=0)
        self.hidden1 = nn.Linear(in_features=4*4*128, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=64)
        self.final   = nn.Linear(in_features=64, out_features=out_classes)

    def forward(self, x):
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = self.max3(F.relu(self.conv3(x)))
        x = self.max5(F.relu(self.conv5(F.relu(self.conv4(x)))))
        x = F.relu(self.conv7(F.relu(self.conv6(x))))
        x = x.view(-1, 4*4*128)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.final(x)
        return x


def run():
    batch_size = 1
    num_channels = 3
    image_size = 180
    num_classes = 2

    # Input Shape
    # B x C x H x W
    shape_in = (batch_size, num_channels, image_size, image_size)

    # Output Shape
    # B x K
    shape_out_cnn = (batch_size, num_classes)
    # Ground Truth shape
    shape_out_gt = (batch_size,)

    net = NetTan2018(in_channels=num_channels, out_classes=num_classes)
    # Shapes must be B x C x H x W
    sample_in = torch.rand(shape_in)
    example_out = net(sample_in)

    print("Out Shape CNN = ", shape_out_cnn)
    print("Out Shape GT  = ", shape_out_gt)

    example_gt = torch.ones(shape_out_gt)
    example_gt = example_gt.long()

    print("CNN Output: ", example_out)
    print("GT  Output: ", example_gt)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(example_out, example_gt)

    print("Cross-Entropy Loss = ", loss.item())
