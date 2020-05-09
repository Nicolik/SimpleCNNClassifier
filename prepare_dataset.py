############################################
# Nicola Altini (2020)
#
# Run this script before training your CNN.
# It will split your train dataset in two folders:
#   - cat
#   - dog
############################################

import os
from config import *

files_train_folder = os.listdir(train_folder)
files_train_folder = [file for file in files_train_folder if file.endswith('.jpg')]

classes = ['cat','dog']

for class_ in classes:
    class_folder = os.path.join(train_folder, class_)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

print("Len Train Folder Files = ", len(files_train_folder))

for file_train in files_train_folder:
    file_class = file_train[:3]
    current_path = os.path.join(train_folder, file_train)
    if file_class == 'cat':
        new_cat_path = os.path.join(train_folder, 'cat', file_train)
        os.rename(current_path, new_cat_path)
    elif file_class == 'dog':
        new_dog_path = os.path.join(train_folder, 'dog', file_train)
        os.rename(current_path, new_dog_path)
