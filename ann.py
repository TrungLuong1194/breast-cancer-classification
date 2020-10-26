#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:22:10 2020

@author: trungluong
"""
# Import the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'small',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
plt.rcParams.update(params)

from skimage.io import imread
from sklearn.model_selection import train_test_split
import cv2


# ----------------------------------------------------------------------------
# Describe data
# ----------------------------------------------------------------------------

base_path = '../dataset'
folders = os.listdir(base_path)

print('*' * 30)
print('About the data:')
print('*' * 30)

# Number of patients
print('- Number of patients: ' + str(len(folders)))

# Number of images and create dataframe of images
total_images = 0
list_patient_id = []
list_path = []
list_target = []

for i in folders:
    patient_id = i
    for label in [0, 1]:
        patient_path = base_path + '/' + patient_id + '/' + str(label)
        sub_folders = os.listdir(patient_path)
        total_images += len(sub_folders)
        for j in sub_folders:
            image_path = patient_path + '/' + j
            list_patient_id.append(patient_id)
            list_path.append(image_path)
            list_target.append(label)
            
data_dict = {'patient_id': list_patient_id, 
             'path': list_path, 
             'target': list_target}

data = pd.DataFrame(data=data_dict)
            
print('- Number of images: ' + str(total_images))
print('- Data Samples:\n')
print(data.head(10))

# ----------------------------------------------------------------------------
# Data analysis
# ----------------------------------------------------------------------------   

# Number of patches/patient
patch_each_patient = data.groupby('patient_id').target.size()
value_pep = patch_each_patient.values

plt.hist(value_pep, bins=30, facecolor='blue', alpha=0.7)
plt.title('Number of patches/patient')
plt.xlabel("Number of patches")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.savefig('ann_images/Number_of_patches_per_patient.png')
plt.show()

# Percentage of a patient is covered by IDC
cancer_percentage = data.groupby("patient_id").target.value_counts() / data.groupby("patient_id").target.size()
value_cp = cancer_percentage.loc[:, 1]*100

plt.hist(value_cp, bins=30, facecolor='blue', alpha=0.5)
plt.title('Percentage of a patient is covered by IDC')
plt.xlabel("% of patches with IDC")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.savefig('ann_images/Percentage_of_a_patient_is_covered_by_IDC.png')
plt.show()

# Percentage of patchs show IDC
target_percentage = data.target.value_counts()
labels_tp = ['0', '1']
explode_tp = (0.1, 0)

plt.pie(target_percentage, explode=explode_tp, labels=labels_tp, 
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of patchs show IDC')
plt.legend()
plt.savefig('ann_images/Percentage_of_patchs_show_IDC.png')
plt.show()

# Show samples of healthy and cancer patches
data.target = data.target.astype(np.int)

pos_selection = np.random.choice(data[data.target==1].index.values, 
                                 size=50, replace=False)
neg_selection = np.random.choice(data[data.target==0].index.values, 
                                 size=50, replace=False)

# Cancer patches
fig, ax = plt.subplots(5, 10, figsize=(20, 10))

for n in range(5):
    for m in range(10):
        idx = pos_selection[m + 10*n]
        image = imread(data.loc[idx, "path"])
        ax[n,m].imshow(image)
        ax[n,m].grid(False)
        
fig.suptitle("Cancer patches", fontsize=20)
fig.savefig('ann_images/Cancer_patches.png')
plt.show()

# Healthy patches
fig, ax = plt.subplots(5, 10,figsize=(20, 10))

for n in range(5):
    for m in range(10):
        idx = neg_selection[m + 10*n]
        image = imread(data.loc[idx, "path"])
        ax[n,m].imshow(image)
        ax[n,m].grid(False)
        
fig.suptitle("Healthy patches", fontsize=20)
fig.savefig('ann_images/Healthy_patches.png')
plt.show()

# Show cancer patches by coordinates
def get_cancer_dataframe(patient_id, cancer_id):
    path = base_path + '/' + patient_id + '/' + cancer_id
    files = os.listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = path + "/" + dataframe.filename.values
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = np.int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    return dataframe

def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df

example = get_patient_dataframe(data.patient_id.values[0])
print(example.head())

fig, ax = plt.subplots(3, 5, figsize=(50, 30))

patient_ids = data.patient_id.unique()

for n in range(3):
    for m in range(5):
        patient_id = patient_ids[m + 5*n]
        example_df = get_patient_dataframe(patient_id)
        
        ax[n,m].scatter(example_df.x.values, example_df.y.values, 
                        c=example_df.target.values, cmap="coolwarm", s=20);
        ax[n,m].set_title("patient " + patient_id)
        ax[n,m].set_xlabel("y coord")
        ax[n,m].set_ylabel("x coord")

fig.savefig('ann_images/Cancer_patches_by_coordinates.png')
plt.show()

# Show cancer patches on full images
def visualise_breast_tissue(patient_id, pred_df=None):
    example_df = get_patient_dataframe(patient_id)
    max_point = [example_df.y.max()-1, example_df.x.max()-1]
    grid = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    if pred_df is not None:
        patient_df = pred_df[pred_df.patient_id == patient_id].copy()
    mask_proba = np.zeros(shape = (max_point[0] + 50, max_point[1] + 50, 1)).astype(np.float)
    
    broken_patches = []
    for n in range(len(example_df)):
        try:
            image = imread(example_df.path.values[n])
            
            target = example_df.target.values[n]
            
            x_coord = np.int(example_df.x.values[n])
            y_coord = np.int(example_df.y.values[n])
            x_start = x_coord - 1
            y_start = y_coord - 1
            x_end = x_start + 50
            y_end = y_start + 50

            grid[y_start:y_end, x_start:x_end] = image
            if target == 1:
                mask[y_start:y_end, x_start:x_end, 0] = 250
                mask[y_start:y_end, x_start:x_end, 1] = 0
                mask[y_start:y_end, x_start:x_end, 2] = 0
            if pred_df is not None:
                
                proba = patient_df[
                    (patient_df.x==x_coord) & (patient_df.y==y_coord)].proba
                mask_proba[y_start:y_end, x_start:x_end, 0] = np.float(proba)

        except ValueError:
            broken_patches.append(example_df.path.values[n])
    
    
    return grid, mask, broken_patches, mask_proba

patient_id = "14154"
grid, mask, broken_patches,_ = visualise_breast_tissue(patient_id)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(grid, alpha=0.5)
ax[1].imshow(mask, alpha=0.8)
ax[1].imshow(grid, alpha=0.6)
ax[0].grid(False)
ax[1].grid(False)
for m in range(2):
    ax[m].set_xlabel("y-coord")
    ax[m].set_ylabel("y-coord")
ax[0].set_title("Breast tissue slice of patient: " + patient_id)
ax[1].set_title("Cancer tissue colored red \n of patient: " + patient_id)

fig.savefig('ann_images/Cancer_patches_on_full_images.png')
plt.show()

# ----------------------------------------------------------------------------
# Setting the training set, test set
# ----------------------------------------------------------------------------

print(data.head())
data.loc[:, "target"] = data.target.astype(np.str)
print(data.info())

# Split the training patient_ids and test patient_ids
train_ids, test_ids = train_test_split(patient_ids, test_size=0.25, 
                                       random_state=0)

print('train_ids/test_ids: ' + str(len(train_ids)) + '/' + str(len(test_ids)))

# Create the training data and test data
train_df = data.loc[data.patient_id.isin(train_ids),:]
test_df = data.loc[data.patient_id.isin(test_ids),:]

def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

train_df = extract_coords(train_df)
test_df = extract_coords(test_df)

# Target distributions
target_0 = [train_df.target.value_counts()[0], test_df.target.value_counts()[0]]
target_1 = [train_df.target.value_counts()[1], test_df.target.value_counts()[1]]

ind = np.arange(2) 
width = 0.35       
plt.bar(ind, target_0, width, label='0')
plt.bar(ind + width, target_1, width, label='1')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target distributions')
plt.xticks(ind + width / 2, ('Training data', 'Test data'))
plt.legend(loc='best')

plt.savefig('ann_images/Target_distributions.png')
plt.show()


# Shape of images
IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_CHANNELS = 3

# Convert image to array (RGB)
def convert_image_to_array(file_paths):
    images_data = np.empty((file_paths.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
    for index, file in enumerate(file_paths):
        img = cv2.imread(file) 
        res = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        images_data[index] = res
        
    return images_data

X_train = np.array(convert_image_to_array(train_df.path))
X_test = np.array(convert_image_to_array(test_df.path))