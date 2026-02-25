#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 03:51:17 2026

@author: owenjohnson
"""


import pandas as pd
import numpy as np

import mne
from mne.datasets.brainstorm import bst_phantom_elekta

from mne.beamformer import make_lcmv

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import Spatial_resolution_or_SSS_functions as sr_SSS
print(__doc__)

# %%
source_pos = np.array([[0.0, 0.0, 0.05]])

# 
sources = 2
a = 5          # in cm
d = 20         # in mm

word = "single_shell_dual_source"
reg_param = .05
SNR = 1.7


# source space
res = 10       # in dmm
sub = 3         # in cm  

#plotting
zoom_radius_z = 30
zoom_radius_y = 30

# %%

# reading forward

full_path = f"/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_sub_{sub}cm_rad_loc_{a}_{res}dmm-fwd.fif"
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_sub_3cm_loc_5cm_10dmm_single_shell-fwd.fif"
# full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_2mm_single_shell-fwd.fif"



fwd = mne.read_forward_solution(full_path)
src = fwd["src"]

# %%

# creating data path, directory, subject, sfreq

data_path = bst_phantom_elekta.data_path(verbose=True)
subjects_dir = data_path
subject = "phantom_otaniemi"
sfreq = 150

# %%

# --- Getting Brain Noise Sources ---

filename = f'/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_{a}cm_SNR_1_150Hz_meg.fif'
# filename = f'/Users/owenjohnson/MEG/Two_source_dif_ori_d_{d}_a_{a}_SNR_{SNR}_10cm_head_gain_error_meg.fif'
# filename = f'/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_{SNR}_150Hz_meg_w_filtered_gaussian_noise_raw.fif'
filename = '/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_3cm_SNR_1_150Hz_single_shell_meg.fif'
filename = '/Users/owenjohnson/MEG/Two_source_dif_ori_d_20_a_5_SNR_1.7_10cm_head_gain_error_single_shell_meg.fif'

# 2. Load files
raw = mne.io.read_raw_fif(filename, preload=True)
raw = raw.copy()

d = d / 1000
a = a / 100
spatial_resolution_mm = res / 10

# %%


# --- 1. Parameters (Matching the Signal Generation) ---
duration = 100.0          # Total duration (seconds)
n_events = 156            # Required number of events
fs = sfreq                # Sampling rate (e.g., 150)

t_pre_trigger = 0.2
t_post_decay = 0.2        # Adjusted to 0.2s to fit 156 events without overlap

# Calculate sample counts
pre_trigger_samples = int(t_pre_trigger * fs)
n_total_samples = int(duration * fs)

# --- 2. Spacing Calculation ---
# We force the spacing to accommodate exactly n_events
spacing_samples = int(n_total_samples / n_events)

# Generate trigger indices 
# (Indices where the "event" occurs in the time series)
trigger_indices = pre_trigger_samples + np.arange(n_events) * spacing_samples

# --- 3. MNE Events Array Construction ---
# MNE format: [sample_index, 0, event_id]
events = np.zeros((n_events, 3), dtype=int)

# Column 0: Sample indices
events[:, 0] = trigger_indices

# Column 1: Value before the trigger (Standard MNE practice is 0)
events[:, 1] = 0

# Column 2: Event ID
event_id = 1
events[:, 2] = event_id

print(f"Created event array for {len(events)} events.")
print(f"First 5 event indices: {events[:5, 0]}")



# %%

# Create Epochs

# Creating projectors
# projs = mne.compute_proj_raw(raw_noise, n_grad=1, n_mag=4, n_eeg=0)
# raw_proj_applied = raw.add_proj(projs)


tmin = -.2
tmax = .2
proj = False

# raw_proj_applied.del_proj()

epochs = mne.Epochs(
    raw=raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(-.2,-0.05),
    preload=True,
    proj=proj
)

# epochs.plot()

# %%
    
# --- Create Evoked ---

# Compute the Evoked data (average across all epochs)
evoked = epochs.average()

# evoked.plot()
    

# %%
    
# get noise and data cov

noise_cov = mne.compute_covariance(
    epochs,
    tmin=-0.2,
    tmax=-0.05, 
    method='empirical',
    # method='diagonal_fixed', 
    # method_params={'diagonal_fixed': {'grad': reg_param, 'mag': reg_param}},
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
)


data_cov = mne.compute_covariance(
    epochs,
    tmin=0.0, 
    tmax=0.15, 
    # method='empirical',
    method='diagonal_fixed', 
    method_params={'diagonal_fixed': {'grad': reg_param, 'mag': reg_param}},
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

# data_cov.plot(epochs.info)
# noise_cov.plot(epochs.info)

# diag_noise_cov = make_covariance_diagonal(noise_cov)
# diag_noise_cov.plot(epochs.info)

    
# %%

# make filter

rank={"meg":200}

filters = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=.00,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm=None,
    rank=None,
    reduce_rank=True
)


# apply filter and get stc

# stc = mne.beamformer.apply_lcmv(evoked, filters)
# src = fwd["src"]
# stc.plot(src, subject=subject, subjects_dir=subjects_dir)


# %%

# Getting data
cov_data = data_cov.data
cov_noise_data = noise_cov.data
# diag_noise_data = diag_noise_cov.data
fwd_data = fwd['sol']['data']

# Getting data from filters
n_sources = filters['n_sources']
W = filters['weights'] # Shape (n_sources, n_channels)
whitener = filters['whitener']
max_ori = filters['max_power_ori'] # Listed using hyphens not underscore in mne python, error in MNE python?
vertices = filters['vertices']

# Applying whitening to data and noise cov and then taking pseudoinverse
white_cov = whitener @ cov_data @ whitener.T
white_noise_cov = whitener @ cov_noise_data @ whitener.T
pi_white_cov = np.linalg.pinv(white_cov)
pi_white_noise_cov = np.linalg.pinv(white_noise_cov)


pi_cov = np.linalg.pinv(cov_data)
pi_noise_cov = np.linalg.pinv(cov_noise_data)


# white_diag_noise_cov = whitener @ diag_noise_data @ whitener.T
# pi_white_diag_noise_cov = np.linalg.pinv(white_diag_noise_cov)

# Setting up list of pseudo-t values
pseudo_t_list = []

# %%

for n in range(n_sources):
    w_n = W[n, :]
    
    # getting fwd at n
    fwd_at_n = fwd_data[:, 3 * n: 3*(n + 1)]
    fixed_fwd = fwd_at_n @ max_ori[n]
    
    # applying whitening to fixed fwd
    white_fwd = whitener @ fixed_fwd
    
    # Power in active window
    power_active = 1 / (white_fwd.T @ pi_white_cov @ white_fwd)
    
    # Power in control window (i.e. noise baseline)
    power_control = 1 / (white_fwd.T @ pi_white_noise_cov @ white_fwd)
    
    # Power of noise from filter
    power_noise = w_n.T @ white_noise_cov @ w_n
    
    # Pseudo-t calculation
    # (power_active - power_noise) / power_noise * 2
    if power_noise > 0:
        t_val = (power_active - power_control) / (2 * power_noise)
    else:
        t_val = 0
        print("zero power noise")
    pseudo_t_list.append(t_val)
    
    

# %%


pseudo_t_values = np.array(pseudo_t_list)

stc = mne.VolSourceEstimate(
    data=pseudo_t_values,          
    vertices=vertices, 
    tmin=0.0, 
    tstep=0.05, 
    subject=subject,
)


src = fwd["src"]

# apply filter and get stc

stc.plot(src, subject=subject, subjects_dir=subjects_dir)


# %%

peak_value = np.max(np.abs(stc.data))
threshold = peak_value / 2.0
stc_thresholded = stc.copy()
mask = np.abs(stc_thresholded.data) < threshold
stc_thresholded.data[mask] = 0.0

# --- Plotting the Thresholded STC ---

stc_thresholded.plot(
    src=src,
    subject=subject,
    subjects_dir=subjects_dir,
    clim={'kind': 'value', 'pos_lims': [0.0, threshold, peak_value]} 
)


# %%

# --- 1. Define Spatial Parameters ---

spatial_resolution_m = spatial_resolution_mm / 1000.0 # Convert to meters for consistency
voxel_volume_m3 = spatial_resolution_m ** 3

# --- 2. Thresholding and Counting Voxels  ---

peak_value = np.max(np.abs(stc.data))
threshold = peak_value / 2.0

# Find the index of the absolute peak in the data
peak_idx = np.unravel_index(np.argmax(np.abs(stc.data)), stc.shape)

# Extract the beamformer output at the time of the peak: a vector of (n_voxels,)
peak_spatial_map = np.abs(stc.data[:, peak_idx[1]])

# Count the voxels in this spatial map that are above the half-maximum threshold
n_voxels_at_peak_half_max = np.sum(peak_spatial_map >= threshold)

# --- 3. Calculate Peak Volume and Dimension ---

# Peak Volume = (Number of suprathreshold voxels) * (Voxel Volume)
peak_volume_m3 = n_voxels_at_peak_half_max * voxel_volume_m3

# Peak Dimension is approximated as the cube root of the peak volume
peak_dimension_m = peak_volume_m3 ** (1/3)
peak_dimension_mm = peak_dimension_m * 1000.0
v_id, t = stc.get_peak()
est_pos = fwd['src'][0]['rr'][v_id]
          
print(est_pos)
# RMS error in location
error = sr_SSS.compute_location_error(source_pos, est_pos)
error *= 1000 # error in mm
rounded_pos = np.round(est_pos, decimals=4)

# --- 4. Report Results ---

print("--- Conventional Beamformer Spatial Resolution Metrics ---")
print(f"1. Voxel Resolution: {spatial_resolution_mm} mm")
print(f"2. Absolute Peak Value: {peak_value:.3e}")
print(f"3. Count of threshold Voxels: {n_voxels_at_peak_half_max}")
print(f"4. Approximated Peak Dimension (Cube Root of Volume): **{peak_dimension_mm:.2f} mm**")
print(f"5. Source Location Error: {error:.2f} mm")


# %%

# --- Plotting 2d contour and color maps ---
# Plotting 2d contour and color maps
v_id, t = stc.get_peak()
est_pos = fwd['src'][0]['rr'][v_id]
v_id, t = stc.get_peak()
est_pos_cm = est_pos * 100
_, peak_time_idx = stc.get_peak(time_as_index=True)

img = stc.as_volume(src, mri_resolution=False) 
data_3d = np.array(img.dataobj[:, :, :, peak_time_idx])
peak_voxel_idx = np.unravel_index(np.argmax(np.abs(data_3d)), data_3d.shape)
x_peak, y_peak, z_peak = peak_voxel_idx

print(f"Peak voxel found at indices: X={x_peak}, Y={y_peak}, Z={z_peak}")

# These functions take a voxel index and return the physical cm value
def voxel_to_cm_y(idx, a):
    offset_voxels = idx - y_peak
    val = est_pos_cm[1] + (offset_voxels * (spatial_resolution_mm / 10))
    return f"{val:.2f}"

def voxel_to_cm_z(idx, a):
    offset_voxels = idx - z_peak
    val = est_pos_cm[2] + (offset_voxels * (spatial_resolution_mm / 10))
    return f"{val:.2f}"

# 1. Define Zoom Parameters (Adjust radius as needed)
y_start = max(0, y_peak - zoom_radius_y)
y_end = min(data_3d.shape[1], y_peak + zoom_radius_y)
z_start = max(0, z_peak - zoom_radius_z)
z_end = min(data_3d.shape[2], z_peak + zoom_radius_z)

# Define parameters in meters
z_val = np.sqrt(a**2 - (d/2)**2)

# True locations in mm
dipole1_mm = np.array([0, d/2, z_val]) * 1000
dipole2_mm = np.array([0, -d/2, z_val]) * 1000
center_mm  = np.array([0, 0, z_val]) * 1000  # The midpoint between them

# Map real-world mm to voxel indices using the inverse affine
inv_affine = np.linalg.inv(img.affine)
idx1 = inv_affine.dot(np.append(dipole1_mm, 1))[:3]
idx2 = inv_affine.dot(np.append(dipole2_mm, 1))[:3]

rel_y1, rel_z1 = idx1[1] - y_start, idx1[2] - z_start
rel_y2, rel_z2 = idx2[1] - y_start, idx2[2] - z_start

# Slice the data for the zoom
# Note: slice_yz is [Y, Z], so we slice those dimensions
plot_data_zoomed = data_3d[x_peak, y_start:y_end, z_start:z_end].T

fig, ax = plt.subplots(figsize=(8, 8))

# --- A. Plot the Color Map ---
# 'extent' maps the pixel indices to the original coordinate space
extent = [y_start, y_end, z_start, z_end]
img_plot = ax.imshow(plot_data_zoomed, origin='lower', cmap='magma', 
                     aspect='equal', extent=extent)

# --- B. Plot Enhanced Contour Map ---
levels = np.linspace(np.min(plot_data_zoomed), np.max(plot_data_zoomed), 10)
contours = ax.contour(plot_data_zoomed, levels=levels, origin='lower', 
                      colors='white', linewidths=0.8, alpha=0.6,
                      extent=extent)

y_dis = zoom_radius_y * spatial_resolution_mm * 2 / 10
z_dis = zoom_radius_z * spatial_resolution_mm * 2 / 10
x_value = est_pos[0] * 1000

ax.xaxis.set_major_formatter(FuncFormatter(voxel_to_cm_y))
ax.yaxis.set_major_formatter(FuncFormatter(voxel_to_cm_z))

#  Mark Dipoles
ax.scatter(idx1[1], idx1[2], marker='x', color='cyan', s=120, 
           linewidths=2, label='Dipole 1')
ax.scatter(idx2[1], idx2[2], marker='x', color='lime', s=120, 
           linewidths=2, label='Dipole 2')
 
ax.set_title(f"Conventional Beamformer X = {x_value:.1f} mm")
ax.set_xlabel(f"Y-axis {y_dis:.2f} (cm)")
ax.set_ylabel(f"Z-axis {z_dis:.2f} (cm)")

plt.tight_layout()
plt.show()

# %%


# Define parameters in meters
z_val = np.sqrt(a**2 - (d/2)**2)

# True locations in mm
dipole1_mm = np.array([0, d/2, z_val]) * 1000
dipole2_mm = np.array([0, -d/2, z_val]) * 1000
center_mm  = np.array([0, 0, z_val]) * 1000  # The midpoint between them

_, peak_time_idx = stc.get_peak(time_as_index=True)
# Convert the 1D STC data into a 3D/4D NIfTI-like volumetric image
img = stc.as_volume(src, mri_resolution=False) 

# Map real-world mm to voxel indices using the inverse affine
inv_affine = np.linalg.inv(img.affine)
idx1 = inv_affine.dot(np.append(dipole1_mm, 1))[:3]
idx2 = inv_affine.dot(np.append(dipole2_mm, 1))[:3]
center_idx = inv_affine.dot(np.append(center_mm, 1))[:3]
for x in range(3):
    center_idx[x] = int(round(center_idx[x]))

# Extract the 3D data array for the specific peak time point
data_3d = np.array(img.dataobj[:, :, :, peak_time_idx])

# Find the 3D coordinates (X, Y, Z) of the maximum voxel activity
peak_voxel_idx = np.unravel_index(np.argmax(np.abs(data_3d)), data_3d.shape)

x_peak, y_peak, z_peak = (peak_voxel_idx)

x_peak = round(int(x_peak))
y_peak = round(int(y_peak))
z_peak = round(int(z_peak))

print(f"Peak voxel found at indices: X={x_peak}, Y={y_peak}, Z={z_peak}")

# Extract the Y-Z plane (Sagittal view) at the peak X coordinate
# By indexing the X dimension and keeping Y and Z, we get a 2D array of shape (Y, Z)
slice_yz = data_3d[x_peak, :, :]

# Extract the Y-Z plane at X = 0 (which is center_idx[0])
x_slice_idx = int(round(center_idx[0]))
slice_yz = data_3d[x_slice_idx, :, :]

# Setup the crop window:
window_mm = zoom_radius_z
vox_size = res / 10
half_win_vox = int((window_mm / 2) / vox_size)  # 10 voxels

y_c = int(round(center_idx[1]))
z_c = int(round(center_idx[2]))

# Bounding box coordinates
y_start, y_end = y_c - half_win_vox, y_c + half_win_vox
z_start, z_end = z_c - half_win_vox, z_c + half_win_vox

# Slice and transpose for the correct Matplotlib orientation
cropped_data = slice_yz[y_start:y_end, z_start:z_end].T

# Calculate relative positions for the 'X' markers inside the cropped window
# We use the exact float indices (idx1, idx2) so the markers are perfectly accurate
rel_y1, rel_z1 = idx1[1] - y_start, idx1[2] - z_start
rel_y2, rel_z2 = idx2[1] - y_start, idx2[2] - z_start

fig, ax = plt.subplots(figsize=(7, 6))

# Contrast adjustments (lift the floor to 5-10% of the max so edges aren't pitch black)
vmax_val = np.max(cropped_data)
vmin_val = vmax_val * 0.05

# Plot the colormap
img_plot = ax.imshow(
    cropped_data, 
    origin='lower', 
    cmap='magma', 
    aspect='equal',
    vmin=vmin_val, 
    vmax=vmax_val
)

# Plot the contours
contours = ax.contour(
    cropped_data, 
    levels=4, 
    origin='lower', 
    colors='white', 
    linewidths=0.8, 
    alpha=0.7
)

# Mark Dipole 1
ax.scatter(rel_y1, rel_z1, marker='x', color='cyan', s=120, linewidths=2, label='Dipole 1')
# Mark Dipole 2
ax.scatter(rel_y2, rel_z2, marker='x', color='lime', s=120, linewidths=2, label='Dipole 2')


window_cm = window_mm / 10
ax.set_xlabel(f"Y-axis {window_cm:.2f} (cm)")
ax.set_ylabel(f"Z-axis {window_cm:.2f} (cm)")

# Clean up
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Conventional Beamformer Output")
ax.legend(loc='upper right', fontsize=9)

plt.show()



# %%
if sources == 1:
    results_list = []
    current_run = {
        "Voxel_Res_mm": spatial_resolution_mm,
        "Peak_Value": peak_value,
        "Threshold_Voxel_Count": n_voxels_at_peak_half_max,
        "Peak_Dimension_mm": peak_dimension_mm,
        "Source_Error_mm": error,
        "True_Dipole_Location": source_pos,
        "Estimated_Dipole_Location": rounded_pos,
        "Regularization": reg_param
        
    }
    
    # Add to the master list
    results_list.append(current_run)
    
    # Create the DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Display the results
    print(df_results.to_string())
    
    import os
    
    file_path = f"/Users/owenjohnson/MEG/single_source_data/conv_beamformer_{a}cm{word}_SNR_{SNR}.csv"
    file_exists = os.path.isfile(file_path)
    
    # Save using mode='a' (append)
    df_results.to_csv(file_path, mode='a', index=False, header=not file_exists)

elif sources == 2:
    results_list = []
    current_run = {
        "Voxel_Res_mm": spatial_resolution_mm,
        "Peak_Value": peak_value,
        "Threshold_Voxel_Count": n_voxels_at_peak_half_max,
        "Peak_Dimension_mm": peak_dimension_mm,
        "Source_Error_mm": error,
        "True_Dipole_Location": source_pos,
        "Estimated_Dipole_Location": rounded_pos,
        "Regularization": reg_param
        
    }
    
    # Add to the master list
    results_list.append(current_run)
    
    # Create the DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Display the results
    print(df_results.to_string())
    
    import os
    
    file_path = f"/Users/owenjohnson/MEG/dual_source_data/conv_beamformer_{a}cm_SNR{SNR}_{word}.csv"
    file_exists = os.path.isfile(file_path)
    
    # Save using mode='a' (append)
    df_results.to_csv(file_path, mode='a', index=False, header=not file_exists)

else:
    print("source number not one or two")
# %%


try:
    # 1. Load the existing data
    existing_df = pd.read_csv(file_path)
    
    if not existing_df.empty:
        # 2. Remove the last row
        # .iloc[:-1] selects everything EXCEPT the last row
        updated_df = existing_df.iloc[-1:]
        
        # 3. Save it back to the same filename
        updated_df.to_csv(file_path, index=False)
        print(f"Successfully removed other entries. New row count: {len(updated_df)}")
    else:
        print("The file is already empty.")

except FileNotFoundError:
    print("The file does not exist yet.")










