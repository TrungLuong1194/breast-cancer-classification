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
