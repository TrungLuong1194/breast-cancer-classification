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


# --------------
# Describe data
# --------------

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

# --------------
# Data analysis
# --------------       

# Number of patches/patient
patch_each_patient = data.groupby('patient_id').target.size()
value_pep = patch_each_patient.values

plt.hist(value_pep, bins=30, facecolor='blue', alpha=0.7)
plt.title('Number of patches/patient')
plt.xlabel("Number of patches")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
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
        
plt.show()