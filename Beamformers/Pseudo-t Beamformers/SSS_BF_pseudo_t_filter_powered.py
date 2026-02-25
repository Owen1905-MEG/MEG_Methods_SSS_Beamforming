#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 13:00:38 2026

@author: owenjohnson
"""


import numpy as np
import pandas as pd
import mne
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.beamformer import make_lcmv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append('/Users/owenjohnson/MEG')
import Spatial_resolution_or_SSS_functions as sr_SSS

# %%

# beamformer details
diagonal_status = True
L_in = 8
L_ext = 3
word = "filter_powered"

source_pos = np.array([[0.0, 0.0, 0.05]])
plot = True

# source details
num_sources = 1
a = 5       # in cm
d = 0       # in mm
SNR = 1
reg_param = .05

#  forward details
res = .5     # in dmm
sub = 2     # in cm

# plot details
zoom_radius_z = 20
zoom_radius_y = 20

# %%

# reading forward

full_path = f"/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_sub_{sub}cm_rad_loc_{a}_{res}dmm-fwd.fif"
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_res_2mm-fwd.fif"
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

# filename = f'/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_{a}cm_SNR_1_150Hz_meg.fif'
filename = f'/Users/owenjohnson/MEG/Two_source_dif_ori_d_{d}_a_{a}_SNR_{SNR}_10cm_head_gain_error_meg.fif'
filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_.5_150Hz_meg_w_filtered_gaussian_noise_raw.fif'
filename = f'/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_{a}cm_SNR_1_150Hz_meg.fif'

# 2. Load files
raw = mne.io.read_raw_fif(filename, preload=True)
raw = raw.copy()


# --- Parameters (Matching the Signal Generation) ---
duration = 100.0          # Total duration (seconds)
n_events = 156            # Required number of events
fs = sfreq                # Sampling rate (e.g., 150)
t_pre_trigger = 0.2
t_post_decay = 0.2        # Adjusted to 0.2s to fit 156 events without overlap
pre_trigger_samples = int(t_pre_trigger * fs)
n_total_samples = int(duration * fs)
spacing_samples = int(n_total_samples / n_events)
trigger_indices = pre_trigger_samples + np.arange(n_events) * spacing_samples
events = np.zeros((n_events, 3), dtype=int)
events[:, 0] = trigger_indices
events[:, 1] = 0
event_id = 1
events[:, 2] = event_id

print(f"Created event array for {len(events)} events.")
print(f"First 5 event indices: {events[:5, 0]}")


  
  # %%
  
  # Create Epochs
  
tmin = -.2
tmax = .2
proj = False

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

# Making scaled evoked
raw_scaled = raw.copy()
raw_scaled.rescale({"mag":100})  # make sure mag is scaled by 100

raw_scaled.del_proj()

epochs_scaled = mne.Epochs(
    raw=raw_scaled,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(-.2,-.05),
    preload=True,
    proj=proj,
)

evoked_scaled = epochs_scaled.average()

# picking good channels for forward

new_raw_ch_names = raw.ch_names

raw.pick(['meg'])
mag_picks = mne.pick_types(raw.info, meg='mag', exclude='bads')

forward = fwd # since no bad channels

# %%

d = d / 1000  # converting from mm to m
a = a / 100   # converting from cm to m

# %%

# get SSS basis and pS

# Load the metadata from file
df = pd.read_csv("sss_results_metadata.csv")
matrices = np.load("sss_pS_matrices.npz")
row = df[df['L_in'] == L_in]
matrix_key = row['matrix_key'].values[0]

pS_final = matrices[matrix_key]

new_n_use_in = row['n_use_in'].values[0]

sss_channels = []
for n in range(new_n_use_in):
    ch_number = n + 1
    sss_channels.append("MEG" + f"{ch_number}".zfill(4))

# Not needed unless computation 
# pS_final, new_n_use_in, reduced_S_in, sss_channels = sr_SSS.get_pS_and_SSS_basis(L_in, L_ext, raw_scaled.info)

# update info and raw to sss basis

evoked_w_sss_info = sr_SSS.change_of_evoked_info_to_sss_efficient(raw, evoked_scaled, sss_channels, new_n_use_in) 

sss_raw = sr_SSS.change_of_raw_basis(
        raw=raw, raw_sss_info=evoked_w_sss_info.info, pS_final=pS_final
        )


# %%

sss_epochs = mne.Epochs(
    raw=sss_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(-.2,-.05),
    preload=True,
    proj=proj
)


sss_noise_cov = mne.compute_covariance(
    sss_epochs,
    tmin=-0.20,
    tmax=-.05, 
    method='empirical',
    # method='diagonal_fixed', 
    # method_params={'diagonal_fixed': {'grad': reg_param, 'mag': reg_param}},
    scalings=dict(mag=1, grad=1, eeg=1)
)

sss_data_cov = mne.compute_covariance(
    sss_epochs,
    tmin=0.000, 
    tmax=0.10, 
    # method='empirical'
    method='diagonal_fixed', 
    method_params={'diagonal_fixed': {'grad': reg_param, 'mag': reg_param}},
    scalings=dict(mag=1, grad=1, eeg=1)
    )

sss_data_cov.plot(evoked_w_sss_info.info)
sss_noise_cov.plot(evoked_w_sss_info.info)

# del sss_epochs

# %%

# change data and noise cov to sss basis

sss_noise_cov_diag = sr_SSS.change_of_cov_basis_into_diag(raw, sss_noise_cov, pS_final)

# forward lead fields transformed to SSS basis
sss_forward = sr_SSS.change_of_lead_fields_basis(forward, pS_final, sss_channels, new_n_use_in)

# del forward

    # %%

# make filter

#  sss_data_cov_no_reg
#  sss_data_cov
    
filters = make_lcmv(
    evoked_w_sss_info.info,
    sss_forward,
    sss_data_cov,
    reg=0.00,
    noise_cov=sss_noise_cov_diag,    # sets whitening
    pick_ori="max-power",
    weight_norm=None,
    rank=None,
    reduce_rank=True
)

# filters_noise = make_lcmv(
#     evoked_w_sss_info.info,
#     sss_forward,
#     sss_noise_cov,
#     reg=0.00,
#     # noise_cov=sss_noise_cov,    # sets whitening
#     pick_ori="max-power",
#     weight_norm=None,
#     rank=None,
#     reduce_rank=True
# )


# %%

# Getting cov and fwd data
cov_data = sss_data_cov.data
cov_noise_data = sss_noise_cov.data
noise_cov_diag_data = sss_noise_cov_diag.data
fwd_data = sss_forward['sol']['data']

# Getting info from filters
n_sources = filters['n_sources']
W = filters['weights'] # Shape (n_sources, n_channels)
# W_noise = filters_noise['weights']
whitener = filters['whitener']
max_ori = filters['max_power_ori'] # Listed using hyphens not underscore in mne python, error in MNE python?

# pseudoinverse of cov
# pi_cov = np.linalg.pinv(cov_data)
# pi_noise_cov = np.linalg.pinv(cov_noise_data)


# Applying whitening to cov
white_cov = whitener @ cov_data @ whitener.T
white_noise_cov = whitener @ cov_noise_data @ whitener.T
pi_white_cov = np.linalg.pinv(white_cov)
pi_white_noise_cov = np.linalg.pinv(white_noise_cov)

white_diag_noise_cov = whitener @ noise_cov_diag_data @ whitener.T
pi_white_diag_noise_cov = np.linalg.pinv(white_diag_noise_cov)


# Diag noise cov
# pi_diag_noise_cov = np.linalg.pinv(noise_cov_diag_data)

# %%


if diagonal_status == True:
    noise_cov = white_diag_noise_cov
    pi_noise_cov = pi_white_diag_noise_cov
    diagonalized = "Diagonal Noise Covariance"
else:
    noise_cov = white_noise_cov
    pi_noise_cov = pi_white_noise_cov
    diagonalized = "Non-Diagonal Noise Covariance"
    

pseudo_t_list = []

# If whitening is applied: pi_noise_cov, cov_noise_data 

for n in range(n_sources):
    w_n = W[n, :]
    
    # getting fwd at n
    fwd_at_n = fwd_data[:, 3 * n: 3*(n + 1)]
    fixed_fwd = fwd_at_n @ max_ori[n]
    
    # applying whitening to fixed fwd
    white_fwd = whitener @ fixed_fwd
    
    # Power in active window
    power_active = w_n.T @ cov_data @ w_n
    
    power_control = w_n.T @ noise_cov_diag_data @ w_n

    
    # Power in noise window assuming identity matrix as noise due to whitening
    power_noise = w_n.T @ noise_cov_diag_data @ w_n
    
    # Pseudo-t calculation
    # We use (power_active - power_noise) / power_noise *2
    # or simply power_active / power_noise
    if power_noise > 0:
        t_val = (power_active - power_control) / (2 * power_noise)
    else:
        t_val = 0
        print("zero power noise")
    pseudo_t_list.append(t_val)


# %%


pseudo_t_values = np.array(pseudo_t_list)

vertices = filters['vertices']

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
    # Optional: Set a consistent color scale limit based on the original peak
    # Using vmin/vmax ensures the colorbar reflects the magnitude of the sources.
    clim={'kind': 'value', 'pos_lims': [0.0, threshold, peak_value]} 
)

# %%

# --- Spatial Resolution Statistic ---

peak_dimension_mm, spatial_resolution_mm, n_voxels = sr_SSS.get_peak_dimensions(stc, src)

# --- Calculate Localization Error ---

v_id, t = stc.get_peak()
est_pos = fwd['src'][0]['rr'][v_id]
rounded_pos = [round(x, 4) for x in est_pos]
error = sr_SSS.compute_location_error(source_pos, est_pos)
error *= 1000

# --- Report Results ---

print(f"--- {diagonalized} SSS Beamformer Spatial Resolution Metrics ---")
print(f"0. Internal Order of {L_in} and External Order of {L_ext} for SSS Basis")
print(f"1. Voxel Resolution: {spatial_resolution_mm} mm")
print(f"2. Absolute Peak Value: {peak_value:.3e}")
print(f"3. Count of threshold Voxels: {n_voxels}")
print(f"4. Peak Dimension: **{peak_dimension_mm:.2f} mm**")
print(f"5. Source Location Error: {error:.2f} mm")


# Initialize the storage list
if num_sources == 2:
    results_list = []
    current_run = {
        "Method": "Diagonalized Noise Covariance" if diagonal_status else "Non-Diagonalized Noise Covariance",
        "L_in": L_in,
        "L_ext": L_ext,
        "Voxel_Res_mm": spatial_resolution_mm,
        "Peak_Value": peak_value,
        "Threshold_Voxel_Count": n_voxels,
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
    source_num = int(round(source_pos[0][2] * 100))
    
    file_path = f"/Users/owenjohnson/MEG/single_source_data/sss_beamformer_{num_sources}_sources_depth_{source_num}cm{word}.csv"
    file_exists = os.path.isfile(file_path)
    
    # Save using mode='a' (append)
    df_results.to_csv(file_path, mode='a', index=False, header=not file_exists)
else:
    print("two sources not selected")

# %%
if plot == True:
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
    source_pos_cm = source_pos[0] * 100
    x_value = est_pos[0] * 1000
    
    ax.xaxis.set_major_formatter(FuncFormatter(voxel_to_cm_y))
    ax.yaxis.set_major_formatter(FuncFormatter(voxel_to_cm_z))
    
    #  Mark Dipoles
    ax.scatter(idx1[1], idx1[2], marker='x', color='cyan', s=120, 
               linewidths=2, label='Dipole 1')
    ax.scatter(idx2[1], idx2[2], marker='x', color='lime', s=120, 
               linewidths=2, label='Dipole 2')
     
    ax.set_title(f"SSS Beamformer X = {x_value:.1f} mm ({diagonalized}) int. order {L_in} ")
    ax.set_xlabel(f"Y-axis {y_dis:.2f} (cm)")
    ax.set_ylabel(f"Z-axis {z_dis:.2f} (cm)")
    
    plt.tight_layout()
    plt.show()
else:
    print("not plotting")
    
    
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
window_mm = 20
vox_size = .5
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
ax.set_title(f"SSS Beamformer Output: ({diagonalized} with int. order {L_in})")
ax.legend(loc='upper right', fontsize=9)

plt.show()
