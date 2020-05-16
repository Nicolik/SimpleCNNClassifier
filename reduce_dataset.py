import os
from shutil import copyfile

dataset_folder = os.path.join('dataset','train')

destination_folder = os.path.join('dataset_reduced')
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
train_destination_folder = os.path.join(destination_folder, 'train')
val_destination_folder = os.path.join(destination_folder, 'val')
if not os.path.exists(train_destination_folder):
    os.makedirs(train_destination_folder)
if not os.path.exists(val_destination_folder):
    os.makedirs(val_destination_folder)

NUM_TRAIN_IMAGES_PER_CLASS = 1000
NUM_VAL_IMAGES_PER_CLASS = 500

subfolders = os.listdir(dataset_folder)
print("Subfolders are: ", subfolders)

for subfolder in subfolders:
    print("Subfolder: ", subfolder)
    subfolder_path = os.path.join(dataset_folder, subfolder)
    elements = os.listdir(subfolder_path)
    num_elements = len(elements)
    print("Elements: ", num_elements)

    train_destination_subfolder = os.path.join(train_destination_folder, subfolder)
    val_destination_subfolder = os.path.join(val_destination_folder, subfolder)
    if not os.path.exists(train_destination_subfolder):
        os.makedirs(train_destination_subfolder)
    if not os.path.exists(val_destination_subfolder):
        os.makedirs(val_destination_subfolder)

    train_elem_cnt = 0
    val_elem_cnt = 0

    for idx, element in enumerate(elements):
        element_path = os.path.join(subfolder_path, element)

        if idx < NUM_TRAIN_IMAGES_PER_CLASS:
            train_elem_cnt += 1
            element_destination_path_train = os.path.join(train_destination_subfolder, element)
            copyfile(element_path, element_destination_path_train)

        if idx > num_elements - NUM_VAL_IMAGES_PER_CLASS - 1:
            val_elem_cnt += 1
            element_destination_path_val = os.path.join(val_destination_subfolder, element)
            copyfile(element_path, element_destination_path_val)

    print("Train Elements: ", train_elem_cnt)
    print("Val   Elements: ", val_elem_cnt)


