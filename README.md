# Simple CNN Classifier
This is a very simple repo for explaining basic concepts about 
Convolutional Neural Networks (CNNs) to beginners.
The example exploits the PyTorch library (https://pytorch.org/) 
for performing a basic binary classification task on the Kaggle
Dogs vs. Cats dataset (https://www.kaggle.com/c/dogs-vs-cats/data).

## Preparing the dataset
1) Download the dataset from Kaggle 
2) Use the script ``prepare_dataset.py`` in order to split train 
   samples in two folders.
  
## Training
Use the script ``train.py`` in order to train a CNN for this purpose.

## Validation
Use the script ``eval.py`` in order to evaluate the CNN performance on this task.