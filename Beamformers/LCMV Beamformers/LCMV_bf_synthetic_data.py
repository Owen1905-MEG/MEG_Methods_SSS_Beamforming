#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:18:48 2025

@author: owenjohnson
"""

import numpy as np

import mne
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

from mne.beamformer import make_lcmv

import matplotlib.pyplot as plt

print(__doc__)


# %%

# all created functions used below
def compute_location_error(pos: np.ndarray, est_pos: np.ndarray) -> float:
    """
    Computes the rms location error between the true 
    dipole location and the estimated location.

    Parameters
    ----------
    pos : np.ndarray
        The true 3D coordinates of the dipole (e.g., [x, y, z] in meters).
    est_pos : np.ndarray
        The estimated 3D coordinates from the MNE source time course 
        (e.g., [x_est, y_est, z_est] in meters).

    Returns
    -------
    location error : float
        The location error in the unit of cm for input in m.
    """
    # The location error is the magnitude (norm) of the difference vector.
    location_error = np.linalg.norm(est_pos - pos) * 10**2
    return location_error

# %%

# reading forward

# creating volume source space
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_res_2mm-fwd.fif"
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_sub_2cm_rad_loc_5_1mm-fwd.fif"
fwd = mne.read_forward_solution(full_path)

src = fwd["src"]


# %%

# creating data path, directory, subject
# using phantom
data_path = bst_phantom_elekta.data_path(verbose=True)
subjects_dir = data_path
subject = "phantom_otaniemi"

raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"

info = mne.io.read_info(raw_fname)
tstep = 1.0 / info["sfreq"]
raw = read_raw_fif(raw_fname, preload=False)
raw.del_proj()
raw.crop(tmin=0, tmax=1)
raw.pick("meg")
info = raw.info

# If sfreq is not wanted to be equal to 1000 Hz change here
sfreq = 150
tstep = 1.0 /sfreq
raw = read_raw_fif(raw_fname, preload=False)
raw.del_proj()
raw.crop(tmin=0, tmax=1)
new_raw = raw.resample(sfreq=sfreq)
new_raw.del_proj()
info = new_raw.info

# %%

# --- Getting Brain Noise Sources ---

# filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_meg.fif'      # data with just brain noise
# filename = '/Users/owenjohnson/MEG/synthetic_data_w_gain_error_5cm_SNR_1_150Hz_meg.fif' # Data with just gain error and brain noise
# filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_200s_400event_meg.fif'
# filename = '/Users/owenjohnson/MEG/Two_source_d_05_a_5_SNR_2_150Hz_meg.fif'
# filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_10cm_rad_meg.fif'
filename = '/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_5cm_SNR_1_150Hz_meg.fif'

# 2. Load files
raw = mne.io.read_raw_fif(filename, preload=True)
raw = raw.copy()


# %%

fs = sfreq

# --- 1. Parameters ---
duration = 100.0  
n_events = 156    

n_total_samples = int(duration * fs)
spacing_samples = int(n_total_samples / n_events)

# Requirements: at least 0.2s pre and 0.2s post
t_pre_trigger = 0.20
pre_trigger_samples = int(t_pre_trigger * fs)
trigger_indices = pre_trigger_samples + np.arange(n_events) * spacing_samples


events = np.zeros((n_events, 3), dtype=int)
events[:, 0] = trigger_indices
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


epochs.plot()


# %%
    
# --- Create Evoked ---

# Compute the Evoked data (average across all epochs)
evoked = epochs.average()

evoked.plot()
    

# %%

    
# get noise and data cov

noise_cov = mne.compute_covariance(
    epochs,
    tmin=-0.2,
    tmax=-0.1, 
    method='ledoit_wolf', 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
)


data_cov = mne.compute_covariance(
    epochs,
    tmin=0.0, 
    tmax=0.1, 
    method='ledoit_wolf', 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

# data_cov.plot(epochs.info)
# noise_cov.plot(epochs.info)
# %%

# make filter

filters = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
    reduce_rank=True
)

src = fwd["src"]

# apply filter and get stc

stc = mne.beamformer.apply_lcmv(evoked, filters)

stc.plot(src, subject=subject, subjects_dir=subjects_dir)


# %%


# Find the absolute peak value across all time points and voxels
peak_value = np.max(np.abs(stc.data))

# Calculate the threshold: half of the peak value
threshold = peak_value / 2.0

# Create a copy of the STC to modify the data
stc_thresholded = stc.copy()

# Apply the threshold: set all values below the threshold to zero
# We check against the absolute value to ensure both positive and negative
# sources below the threshold are masked out.
mask = np.abs(stc_thresholded.data) < threshold
stc_thresholded.data[mask] = 0.0

# --- 3. Plotting the Thresholded STC ---

# The plot method takes the Source space (src) object and subject information
stc_thresholded.plot(
    src=src,
    subject=subject,
    subjects_dir=subjects_dir,
    # Optional: Set a consistent color scale limit based on the original peak
    # Using vmin/vmax ensures the colorbar reflects the magnitude of the sources.
    clim={'kind': 'value', 'pos_lims': [0.0, threshold, peak_value]} 
)

# %%

# --- 1. Define Spatial Parameters ---

spatial_resolution_mm = 1.0
spatial_resolution_m = spatial_resolution_mm / 1000.0 # Convert to meters
voxel_volume_m3 = spatial_resolution_m ** 3

# --- 2. Thresholding and Counting Voxels ---
# Getting threshold value
peak_value = np.max(np.abs(stc.data))
threshold = peak_value / 2.0

# Getting number of voxels greater than half maximum at time of peak
peak_idx = np.unravel_index(np.argmax(np.abs(stc.data)), stc.data.shape) # gives tuple of indices for peak voxel and peak time
peak_spatial_map = np.abs(stc.data[:, peak_idx[1]])
n_voxels_at_peak_half_max = np.sum(peak_spatial_map >= threshold)

# --- 3. Calculate Peak Volume and Dimension ---

peak_volume_m3 = n_voxels_at_peak_half_max * voxel_volume_m3
peak_dimension_m = peak_volume_m3 ** (1/3)

peak_dimension_mm = peak_dimension_m * 1000.0

# --- Results ---

print("--- Beamformer Spatial Resolution Metrics ---")
print(f"1. Voxel Resolution: {spatial_resolution_mm} mm")
print(f"2. Absolute Peak Value: {peak_value:.3e}")
print(f"3. Half-Maximum Threshold: {threshold:.3e}")
print(f"4. Count of threshold Voxels: {n_voxels_at_peak_half_max}")
print(f"5. Calculated Peak Volume: {peak_volume_m3:.3e} m^3")
print(f"6. Approximated Peak Dimension (Cube Root of Volume): **{peak_dimension_mm:.2f} mm**")




# %%

v_id, t = stc.get_peak()
est_pos = fwd['src'][0]['rr'][v_id]

          
print(est_pos)
source_pos = [[0.0, 0, .05]]




# rms displacement error in 

error = compute_location_error(source_pos, est_pos)

error *= 1000

print(error)

print(f"rms error of {error} mm")



# %%
# (0,0,50) mm # conv BF radius 10 cm sphere SNR .1


# --- Beamformer Spatial Resolution Metrics ---
# 1. Voxel Resolution: 2.0 mm
# 2. Absolute Peak Value: 1.432e+00
# 3. Half-Maximum Threshold: 7.161e-01
# 4. Count of threshold Voxels: 6816
# 5. Calculated Peak Volume: 5.453e-05 m^3
# 6. Approximated Peak Dimension (Cube Root of Volume): **37.92 mm**

















