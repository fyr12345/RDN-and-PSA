# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:33:19 2024

@author: fyr
"""

import os
import SimpleITK as sitk
import numpy as np


nii_A = sitk.ReadImage('Resliced_Schaefer2018_300Parcels_7Networks_order_plus_subcort1.nii')
data_A = sitk.GetArrayFromImage(nii_A)


mean_values_array = np.zeros((300, 335))


for i in range(0, 301):

    path = f'xx\Resliced_p{i:03d}_Yeo_300_7_percent_tdi.nii'#White matter disconnection per patient
    if not os.path.exists(path):
        print(f"File {path} does not exist. Skipping...")
        continue
    nii_B=sitk.ReadImage(path)
    data_B = sitk.GetArrayFromImage(nii_B)
    
   
    mean_values = {}


    for value in range(1, 336):  
        
        indices = np.where(data_A == value)       
        corresponding_values = data_B[indices]      
        mean_value = np.sum(corresponding_values)
        mean_values[value] = mean_value

    
    for key, value in mean_values.items():
        mean_values_array[i - 1, key - 1] = value



