
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import glob
import os
# Load atlas.nii
atlas_img = nib.load(r"N:\CCA3\para_Resliced\Resliced_Schaefer2018_300Parcels_7Networks_order_plus_subcort.nii")
atlas_data = atlas_img.get_fdata()
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels != 0]  # Exclude background if labeled as 0
# Get list of patient NIfTI files
patient_files = sorted(glob.glob('N:/*.nii'))

# Load infarct volume data from Excel
infarct_volumes = pd.read_excel(r'D:\xx.xlsx')
# Ensure the order matches the patient_files list
infarct_volumes = infarct_volumes.sort_values('patient_id')
# Initialize a DataFrame to store values
data = pd.DataFrame(columns=['patient_id'] + list(roi_labels))
data['patient_id'] = [os.path.basename(f).split('.')[0] for f in patient_files]
data['infarct_volume'] = infarct_volumes['Volume'].values

for idx, patient_file in enumerate(patient_files):
    # Load patient's NIfTI file
    patient_img = nib.load(patient_file)
    patient_data = patient_img.get_fdata()

    # Extract mean structural connectivity loss per ROI
    for roi in roi_labels:
        roi_mask = atlas_data == roi
        # Since values are constant within ROI, you can take any voxel or compute the mean
        roi_value = np.mean(patient_data[roi_mask])
        data.loc[idx, roi] = roi_value
# Initialize a DataFrame to store residuals
residuals = pd.DataFrame(columns=['patient_id'] + list(roi_labels))
residuals['patient_id'] = data['patient_id']

for roi in roi_labels:
    # Prepare data for regression
    X = data['infarct_volume'].values.reshape(-1, 1)
    y = data[roi].values

    # Fit linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Compute residuals
    resid = y - y_pred
    residuals[roi] = resid
for idx, patient_file in enumerate(patient_files):
    # Create an empty array for residuals
    residual_img_data = np.zeros(atlas_data.shape)

    for roi in roi_labels:
        roi_mask = atlas_data == roi
        # Assign the residual value to all voxels in the ROI
        residual_value = residuals.loc[idx, roi]
        residual_img_data[roi_mask] = residual_value

    # Save the new NIfTI file
    residual_img = nib.Nifti1Image(residual_img_data, affine=atlas_img.affine, header=atlas_img.header)
    patient_id = residuals.loc[idx, 'patient_id']
    nib.save(residual_img, f'N:/CCA3/clsm_cancha/{patient_id}_residual.nii')
