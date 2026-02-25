#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:50:25 2025

@author: owenjohnson
"""


import numpy as np

import mne
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif


from scipy.spatial import KDTree
import matplotlib.pyplot as plt

print(__doc__)


# %%

# all created functions used below

def find_closest_vertices(src, pos):
    """
    Finds the indices of the closest source space vertices to a list of Cartesian coordinates.

    Args:
        src: The MNE SourceSpaces object (expected to be a volume source space).
        pos: A numpy array of shape (N_coords, 3) containing the Cartesian 
             coordinates (x, y, z) for which to find the closest vertex.

    Returns:
        A numpy array of shape (N_coords,) containing the index of the 
        closest vertex in src for each coordinate in pos.
    """
    if len(src) == 0:
        raise ValueError("Source space object is empty.")
    
    # The 'rr' field contains the coordinates of all source points (vertices).
    all_coords = src[0]['rr']
    
    # 1. Create a KDTree from the source space vertices
    kdtree = KDTree(all_coords)
    
    # 2. Query the KDTree with the list of positions
    distances, indices = kdtree.query(pos, k=1)

    return distances, indices



def mix_linear_weighted_snr(fwd, raw_signal, raw_noise, indices, q_moment, ori, target_snr, source_pos):
    """
    Scales the raw_signal by calculating type-specific Power SNRs, converting 
    them to Linear SNRs, combining them using the Weighted Average formula, and 
    applying LINEAR scaling factor (S = Target/Current).
    """
    # --- 1. Prepare Parameters and Channel Info ---
    
    L_free = fwd['sol']['data']
    fwd_vertno = fwd['src'][0]['vertno']

    # Search for your source_idx in the forward's vertex list
    try:
        # This gives you the spatial location index in the forward matrix
        fwd_idx = np.where(fwd_vertno == indices)[0][0]
        print(f"The index in the forward data is: {fwd_idx}")
    except IndexError:
        print("The chosen vertex is not included in the forward solution (likely outside the mask).")
    q = q_moment
    
    ori_vec = np.array(ori)
    ori_vec = ori_vec / np.linalg.norm(ori_vec)
    
    picks_mag = mne.pick_types(raw_noise.info, meg='mag', eeg=False, misc=False)
    picks_grad = mne.pick_types(raw_noise.info, meg='grad', eeg=False, misc=False)
    
    Mm = len(picks_mag)
    Mp = len(picks_grad)
    M_total = Mm + Mp
    if M_total == 0:
        raise ValueError("No MEG channels found in data.")

    # --- 2. Calculate Noise Power Components (v_rms^2 * M) ---
    v_rms_mag = np.std(raw_noise.get_data(picks=picks_mag)) if Mm > 0 else 0.0
    v_rms_grad = np.std(raw_noise.get_data(picks=picks_grad)) if Mp > 0 else 0.0
    
    noise_power_mag_den = (v_rms_mag**2 * Mm) if Mm > 0 else 0.0
    noise_power_grad_den = (v_rms_grad**2 * Mp) if Mp > 0 else 0.0

    # --- 3. Calculate Signal Power Components (q^2 * |L|^2) ---

    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])
    start_col = fwd_idx * 3
    end_col = fwd_idx * 3 + 3
    L_source = L_free[:, start_col:end_col]
    print(L_source)
    print(ori_vec)
    L_eff = L_source @ ori_vec.T
    signal_magnitude_qL += np.abs(q * np.squeeze(L_eff))

    sig_power_mag_num = np.sum(signal_magnitude_qL[picks_mag]**2)
    sig_power_grad_num = np.sum(signal_magnitude_qL[picks_grad]**2)

    # --- 4. Calculate Linear SNR_m and SNR_p for Weighted Average ---

    # Step A: Calculate Power SNR (q^2|L|^2 / v_rms^2 M) for each type
    snr_m_power = sig_power_mag_num / noise_power_mag_den if noise_power_mag_den > 0 else 0.0
    snr_p_power = sig_power_grad_num / noise_power_grad_den if noise_power_grad_den > 0 else 0.0

    snr_m_linear = np.sqrt(snr_m_power)
    snr_p_linear = np.sqrt(snr_p_power)

    # Weighted SNR (Linear Average): (Mm * SNRm + Mp * SNRp) / (Mm + Mp)
    current_weighted_snr = ((Mm *  snr_m_linear) + (Mp * snr_p_linear)) / M_total

    # --- Print Summary ---
    print("--- SNR Calculation Summary (Hybrid Approach) ---")
    print(f"Linear SNR_m: {snr_m_linear:.4f}")
    print(f"Linear SNR_p: {snr_p_linear:.4f}")
    print(f"Current Weighted SNR: {current_weighted_snr:.4f}")
    print("-------------------------------------------------")

    # --- 5. Determine Scaling Factor (Linear Ratio) ---

    if current_weighted_snr <= 0:
        print("Warning: Current Weighted SNR is 0 or negative. Cannot scale.")
        return raw_noise.copy()

    # CORRECTED SCALING: S = Target_SNR / Current_SNR
    scaler = target_snr / current_weighted_snr 
    print(f"Scaling Factor required for Target {target_snr} (linear ratio): {scaler}")
    
    # --- 6. Scale and Mix ---
        
    raw_combined = raw_noise.copy()
    scaled_signal_data = raw_signal.get_data() * scaler
    raw_combined._data += scaled_signal_data
    
    return raw_combined

def mix_power_weighted_snr(fwd, raw_signal, raw_noise, indices, q_moment, ori, target_snr, source_pos):
    """
    Scales the raw_signal by calculating type-specific Power SNRs, converting 
    them to Linear SNRs, combining them using the Weighted Average formula, and 
    applying LINEAR scaling factor (S = Target/Current).
    """
    # --- 1. Prepare Parameters and Channel Info ---
    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]
    
    L_free = fwd['sol']['data']
    fwd_vertno = fwd['src'][0]['vertno']

    # Search for your source_idx in the forward's vertex list
    try:
        # This gives you the spatial location index in the forward matrix
        fwd_idx = np.where(fwd_vertno == indices)[0][0]
        print(f"The index in the forward data is: {fwd_idx}")
    except IndexError:
        print("The chosen vertex is not included in the forward solution (likely outside the mask).")
  
    q = q_moment
    
    ori_vec = np.array(ori)
    ori_vec = ori_vec / np.linalg.norm(ori_vec)
    
    picks_mag = mne.pick_types(raw_noise.info, meg='mag', eeg=False, misc=False)
    picks_grad = mne.pick_types(raw_noise.info, meg='grad', eeg=False, misc=False)
    
    Mm = len(picks_mag)
    Mp = len(picks_grad)
    M_total = Mm + Mp
    if M_total == 0:
        raise ValueError("No MEG channels found in data.")

    # --- 2. Calculate Noise Power Components (v_rms^2 * M) ---
    v_rms_mag = np.std(raw_noise.get_data(picks=picks_mag)) if Mm > 0 else 0.0
    v_rms_grad = np.std(raw_noise.get_data(picks=picks_grad)) if Mp > 0 else 0.0
    
    noise_power_mag_den = (v_rms_mag**2 * Mm) if Mm > 0 else 0.0
    noise_power_grad_den = (v_rms_grad**2 * Mp) if Mp > 0 else 0.0

    # --- 3. Calculate Signal Power Components (q^2 * |L|^2) ---

    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])

    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])
    start_col = fwd_idx * 3
    end_col = fwd_idx * 3 + 3
    L_source = L_free[:, start_col:end_col]
    print(L_source)
    print(ori_vec)
    L_eff = L_source @ ori_vec.T
    signal_magnitude_qL += np.abs(q * np.squeeze(L_eff))

    sig_power_mag_num = np.sum(signal_magnitude_qL[picks_mag]**2)
    sig_power_grad_num = np.sum(signal_magnitude_qL[picks_grad]**2)

    # --- 4. Calculate Linear SNR_m and SNR_p for Weighted Average ---

    # Step A: Calculate Power SNR (q^2|L|^2 / v_rms^2 M) for each type
    snr_m_power = sig_power_mag_num / noise_power_mag_den if noise_power_mag_den > 0 else 0.0
    snr_p_power = sig_power_grad_num / noise_power_grad_den if noise_power_grad_den > 0 else 0.0

    # Weighted SNR (Linear Average): (Mm * SNRm + Mp * SNRp) / (Mm + Mp)
    current_weighted_snr = ((Mm *  snr_m_power) + (Mp * snr_p_power)) / M_total

    # --- Print Summary ---
    print("--- SNR Calculation Summary (Hybrid Approach) ---")
    print(f"Linear SNR_m: {snr_m_power:.4f}")
    print(f"Linear SNR_p: {snr_p_power:.4f}")
    print(f"Current Weighted SNR: {current_weighted_snr:.4f}")
    print("-------------------------------------------------")

    # --- 5. Determine Scaling Factor (Linear Ratio) ---

    if current_weighted_snr <= 0:
        print("Warning: Current Weighted SNR is 0 or negative. Cannot scale.")
        return raw_noise.copy()

    # CORRECTED SCALING: S = Target_SNR / Current_SNR
    power_ratio = target_snr / current_weighted_snr
    scaler = np.sqrt(power_ratio)
    print(f"Scaling Factor required for Target {target_snr} (power ratio): {scaler}")
    
    # --- 6. Scale and Mix ---
        
    raw_combined = raw_noise.copy()
    scaled_signal_data = raw_signal.get_data() * scaler
    raw_combined._data += scaled_signal_data
    
    return raw_combined

def mix_noise_to_snr(fwd, raw_signal, raw_noise, indices, q_moment, ori, target_snr, source_pos):
    """
    Scales the raw_signal by calculating type-specific Power SNRs, converting 
    them to Linear SNRs, combining them using the Weighted Average formula, and 
    applying LINEAR scaling factor (S = Target/Current).
    """
    # --- 1. Prepare Parameters and Channel Info ---
    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]
    
    L_free = fwd['sol']['data']
    fwd_vertno = fwd['src'][0]['vertno']

    # Search for your source_idx in the forward's vertex list
    try:
        # This gives you the spatial location index in the forward matrix
        fwd_idx = np.where(fwd_vertno == indices)[0][0]
        print(f"The index in the forward data is: {fwd_idx}")
    except IndexError:
        print("The chosen vertex is not included in the forward solution (likely outside the mask).")
  
    q = q_moment
    
    ori_vec = np.array(ori)
    ori_vec = ori_vec / np.linalg.norm(ori_vec)
    
    picks_mag = mne.pick_types(raw_noise.info, meg='mag', eeg=False, misc=False)
    picks_grad = mne.pick_types(raw_noise.info, meg='grad', eeg=False, misc=False)
    
    Mm = len(picks_mag)
    Mp = len(picks_grad)
    M_total = Mm + Mp
    if M_total == 0:
        raise ValueError("No MEG channels found in data.")

    # --- 2. Calculate Noise Power Components (v_rms^2 * M) ---
    v_rms_mag = np.std(raw_noise.get_data(picks=picks_mag)) if Mm > 0 else 0.0
    v_rms_grad = np.std(raw_noise.get_data(picks=picks_grad)) if Mp > 0 else 0.0
    
    noise_power_mag_den = (v_rms_mag**2 * Mm) if Mm > 0 else 0.0
    noise_power_grad_den = (v_rms_grad**2 * Mp) if Mp > 0 else 0.0

    # --- 3. Calculate Signal Power Components (q^2 * |L|^2) ---

    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])

    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])
    start_col = fwd_idx * 3
    end_col = fwd_idx * 3 + 3
    L_source = L_free[:, start_col:end_col]
    print(L_source)
    print(ori_vec)
    L_eff = L_source @ ori_vec.T
    signal_magnitude_qL += np.abs(q * np.squeeze(L_eff))

    sig_power_mag_num = np.sum(signal_magnitude_qL[picks_mag]**2)
    sig_power_grad_num = np.sum(signal_magnitude_qL[picks_grad]**2)

    # --- 4. Calculate Linear SNR_m and SNR_p for Weighted Average ---

    # Step A: Calculate Power SNR (q^2|L|^2 / v_rms^2 M) for each type
    snr_m_power = sig_power_mag_num / noise_power_mag_den if noise_power_mag_den > 0 else 0.0
    snr_p_power = sig_power_grad_num / noise_power_grad_den if noise_power_grad_den > 0 else 0.0

    # Weighted SNR (Linear Average): (Mm * SNRm + Mp * SNRp) / (Mm + Mp)
    current_weighted_snr = ((Mm *  snr_m_power) + (Mp * snr_p_power)) / M_total

    # --- Print Summary ---
    print("--- SNR Calculation Summary (Hybrid Approach) ---")
    print(f"Linear SNR_m: {snr_m_power:.4f}")
    print(f"Linear SNR_p: {snr_p_power:.4f}")
    print(f"Current Weighted SNR: {current_weighted_snr:.4f}")
    print("-------------------------------------------------")

    # --- 5. Determine Scaling Factor (Linear Ratio) ---

    if current_weighted_snr <= 0:
        print("Warning: Current Weighted SNR is 0 or negative. Cannot scale.")
        return raw_noise.copy()

    # CORRECTED SCALING: S = Target_SNR / Current_SNR
    power_ratio = target_snr / current_weighted_snr
    scaler = np.sqrt(power_ratio)
    print(f"Scaling Factor required for Target {target_snr} (power ratio): {scaler}")
    
    # --- 6. Scale and Mix ---
        
    raw_combined = raw_noise.copy()
    scaled_signal_data = raw_signal.get_data() * scaler
    raw_combined._data += scaled_signal_data
    
    return raw_combined

# %%

# creating forward

# creating volume source space
# full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_res_2mm-fwd.fif"
# full_path = "spherical_head_model_rad_93cm_25mm-fwd.fif"
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_2mm_single_shell-fwd.fif"

fwd = mne.read_forward_solution(full_path)

src = fwd["src"]

# %%

# create source space and forward solution
# using phantom

data_path = bst_phantom_elekta.data_path(verbose=True)
subjects_dir = data_path
subject = "phantom_otaniemi"

raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"

info = mne.io.read_info(raw_fname)
tstep = 1.0 / info["sfreq"]
raw = read_raw_fif(raw_fname, preload=False)
raw.crop(tmin=0, tmax=1)
raw.pick("meg")
info = raw.info


# If sfreq does not equal 1000 Hz
sfreq = 150
tstep = 1.0 / sfreq
raw = read_raw_fif(raw_fname, preload=False)
raw.del_proj()
raw.crop(tmin=0, tmax=1)
new_raw = raw.resample(sfreq=sfreq)
new_raw.del_proj()
info = new_raw.info

# %%

# --- Getting Brain Noise Sources ---

filename = '/Users/owenjohnson/MEG/brain_noise_5000_dipoles_150Hz_100s_rad_10cm_meg.fif'
# filename = 'normalized_synthetic_brain_noise_5000_dipoles_150Hz_meg.fif'
filename = 'brain_noise_5000_dipoles_150Hz_100s_single_shell_rad_10cm_meg.fif'

#   Load files
raw_noise = mne.io.read_raw_fif(filename, preload=True)

# %%

# Creating dipole signals of interest.

n_active_dipoles = 1

# getting locations and orientations
source_pos = [0, 0, 0.03]

# # converting pos to vertex
# Find the closest vertex for each random point
dist, source_idx = find_closest_vertices(src, source_pos)

# Position in head coordinates (MNE uses meters)

# Fixed orientation (e.g., pointing purely in the +x direction)
ori = np.array([[1.0, 0.0, 0.0]])   # shape (n_dipoles, 3), unit vector

# creating samples for all time, activation, and activity individually

# 1. Prepare the orientation vector
vector = ori
orient_vec = np.array(vector, dtype=float)
orient_vec_3d = orient_vec[:n_active_dipoles].reshape(n_active_dipoles, 3, 1)


# %%

# For Generalized Lorentzian Activity

# Assume fs is defined (e.g., 150 Hz or 1000 Hz)
fs = sfreq  # Adjust this to your actual sampling rate

# --- 1. Parameters ---
duration = 100.0  # Total duration (seconds)
n_events = 156    # Fixed number of triggers
q_moment = 20 * 1e-9

# Requirements: at least 0.2s pre and 0.2s post
t_pre_trigger = 0.20
t_post_decay = 0.20  # Adjusted to meet your "at least .2s" requirement

# Waveform morphology
width_ms = 25.0 
latency_ms = 50.0 
amp_jitter = 0.3 

# --- 2. Spacing Calculation (Forced 156 Events) ---

# Total samples available
n_total_samples = int(duration * fs)
spacing_samples = int(n_total_samples / n_events)
spacing_sec = spacing_samples / fs

# Define the window size based on your .2s/.2s requirement
pre_trigger_samples = int(t_pre_trigger * fs)
post_decay_samples = int(t_post_decay * fs)
window_samples = pre_trigger_samples + post_decay_samples

print(f"Required Window: {t_pre_trigger + t_post_decay}s")
print(f"Available Spacing: {spacing_sec:.4f}s")

if spacing_samples < window_samples:
    print("WARNING: Spacing is smaller than window. Overlap will occur.")
else:
    print("Success: No overlap. Each peak has its required buffer.")

# Generate trigger indices
# We start the first trigger at pre_trigger_samples so the first peak's 
# pre-buffer starts at index 0.
trigger_indices = pre_trigger_samples + np.arange(n_events) * spacing_samples
trigger_times = trigger_indices / fs

# --- 3. Waveform Template ---
t_template = np.linspace(-t_pre_trigger, t_post_decay, window_samples, endpoint=False)
tau = (width_ms / 1000.0) / 2.0
latency_s = latency_ms / 1000.0

# Lorentzian: 1 / (1 + ((t - latency)/tau)^2)
template_wave = 1.0 / (1.0 + ((t_template - latency_s) / tau)**2)

# --- 4. Sequence Generation ---
time_series = np.zeros(n_total_samples)
rng = np.random.default_rng(42)
amplitude = 1.0
amplitudes = 1.0 + amp_jitter * (2.0 * rng.random(n_events) - 1.0)

for trig_idx, amp in zip(trigger_indices, amplitudes):
    start_idx = trig_idx - pre_trigger_samples
    end_idx = start_idx + window_samples
    
    if end_idx <= n_total_samples:
        time_series[start_idx:end_idx] += template_wave * amp

# --- 5. Plotting ---
# view_limit_sec = 1
# view_limit_samples = int(view_limit_sec * fs)
# t_view = np.arange(view_limit_samples) / fs
# data_view = time_series[:view_limit_samples]
# triggers_view = trigger_times[trigger_times < view_limit_sec]

# plt.figure(figsize=(12, 5))
# plt.plot(t_view, data_view, color='#1f77b4', label='Lorentzian Peaks')
# plt.vlines(triggers_view, ymin=-0.1, ymax=0, color='red', label='Triggers')
# plt.title(f"Simulation: {n_events} Events over {duration}s (Spacing: {spacing_sec:.3f}s)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


# %%


# Calculate vector data for dipoles (N_dipoles, 3, N_times)
dipole_data_3d = orient_vec_3d * time_series 
_, vertex_idx_scalar = find_closest_vertices(src, source_pos)
vertices = [np.atleast_1d(vertex_idx_scalar).astype(int)]

# 3. Create the Vector Source Estimate
stc_vec = mne.VolVectorSourceEstimate(
    data=dipole_data_3d,
    vertices=vertices,
    tmin=0,
    tstep=tstep, 
    subject=subject
)



stc_vec.data *= q_moment

# %%

# 4. Simulate Raw Data

raw_source = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)



# raw_source.plot()

# %%
# --- MNE Event Creation using Fixed Indices ---

events = np.zeros((n_events, 3), dtype=int)
events[:, 0] = trigger_indices
event_id = 1
events[:, 2] = event_id

print(f"Generated {events.shape[0]} MNE events.")
print(f"First 5 events:\n{events[:5]}")

raw_source.plot(events=events, duration=2.0, start=0, title="Simulated Source with Events")


# %%


# output_filename = '/Users/owenjohnson/MEG/synthetic_data_5cm_no_noise_meg_150Hz_raw.fif'
# raw_source.save(output_filename, overwrite=True)

# %%



# 1. Define the window around the peak 
# (e.g., 0.0 to 0.1 captures the 50ms latency peak)
tmin_peak = 0.05
tmax_peak = 0.05

# 2. Run the mixer

# final_raw = mix_linear_weighted_snr(
#             fwd=fwd, 
#             raw_signal=raw_source,
#             raw_noise=raw_noise,
#             indices=source_idx,
#             q_moment=amplitude,
#             ori=ori,
#             target_snr=1,
#             source_pos=source_pos
#             )

final_raw = mix_power_weighted_snr(
            fwd=fwd, 
            raw_signal=raw_source,
            raw_noise=raw_noise,
            indices=source_idx,
            q_moment=q_moment,
            ori=ori,
            target_snr=1,
            source_pos=source_pos
            )


# --- 3. Visualize Verification ---

#Average the epochs to see the signal clearly out of the noise
# evoked = mne.Epochs(final_raw, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# # # Plot
# fig = evoked.plot(show=False)


evoked_power = mne.Epochs(final_raw, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# Plot
fig = evoked_power.plot(show=False)

# %%

# filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_linear_SNR_1_150Hz_10cm_rad_meg.fif'
# final_raw.save(filename, overwrite=True)

# %%

filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_10cm_rad_meg.fif'
final_raw.save(filename, overwrite=True)

# %%

# Adding Noise


# %%

normal_device_noise_cov = mne.make_ad_hoc_cov(info = raw_source.info, std = None)

raw = raw_source.copy() 

info = raw.info

# 1. Define Parameters
n_channels = 306
sfreq = 150.0  # New Sampling frequency (Hz)
duration = 100 # Seconds
n_samples = int(sfreq * duration) # Total: 15,000
data = np.zeros((n_channels, n_samples))

raw_noise = mne.io.RawArray(data, info)


raw_noise = mne.simulation.add_noise(inst = raw_noise, cov = normal_device_noise_cov, iir_filter=[0.2, -0.2, 0.04], random_state = 42)

# %%


raw_mix = mix_noise_to_snr(fwd=fwd, 
                             raw_signal=raw_source,
                             raw_noise=raw_noise,
                             indices=source_idx,
                             q_moment=q_moment,
                             ori=ori,
                             target_snr=1,
                             source_pos=source_pos
                             )


evoked = mne.Epochs(raw_mix, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()
epochs = mne.Epochs(raw_mix, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05))
epochs.plot()

# Plot
fig = evoked.plot()
# %%


def check_mag_density(raw):
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False)
    spectrum_mag = raw.compute_psd(picks=mag_picks, fmin=1.0, fmax=75.0, n_fft=2048)
    
    # Mags are already in T, just convert to fT
    asd_mag_fT = np.mean(np.sqrt(spectrum_mag.get_data())) * 1e15
    print(f"Resulting Magnetometer Density: {asd_mag_fT:.2f} fT/sqrt(Hz)")


check_mag_density(raw_mix)

check_mag_density(raw_noise)

# %%


output_filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_meg_w_filtered_gaussian_noise_raw.fif'
raw_mix.save(output_filename, overwrite=True)


# %%

# Gain Error Implemented

n_channels = 306
gain_error_std = .001

# Generate N_channels random gain factors (G_i) following a Gaussian distribution
# with a mean of 1 and a standard deviation.
gain_factors = 1 + np.random.normal(loc=0.0, scale=gain_error_std, size=n_channels)

# --- Adding Channel Gain Error ---
scaling_matrix = np.diag(gain_factors)

print(scaling_matrix)

final_raw_copy = final_raw.copy()

data_transformed = final_raw_copy.get_data()

data_transformed = scaling_matrix @ data_transformed


info = final_raw.info
raw_w_gain_noise = mne.io.RawArray(data_transformed, info, copy = "info")

# raw_w_gain_noise.plot()

evoked = mne.Epochs(raw_w_gain_noise, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# Plot
fig = evoked_power.plot(show=False)

# %%

# Save data for comparison with other methods like trad. LCMV beamformer
output_filename = '/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_3cm_SNR_1_150Hz_single_shell_meg.fif'
raw_w_gain_noise.save(output_filename, overwrite=True)

# %%

print(np.std(raw_w_gain_noise.get_data(),axis=1))

print(np.std(final_raw.get_data(),axis=1))
