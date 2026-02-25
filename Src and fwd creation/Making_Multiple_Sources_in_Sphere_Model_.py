#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:19:47 2026

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


SNR = 1.7

a_num = 5     # distance from origin to dipole in cm
d_num = 20   # distance between sources in mm


# %%

# all created functions used below

def generate_spherical_shell_points(num_points, r_in, r_out):
    """
    Generates points uniformly distributed inside a spherical shell.
    The resulting coordinates are in the same unit as r_in and r_out.
    """
    if r_in >= r_out or r_in < 0 or r_out <= 0:
        raise ValueError("r_in must be less than r_out and both must be positive.")

    # 1. Generate radial coordinate (r)
    # Samples r^3 uniformly, then takes the cube root.
    r_in_cubed = r_in**3
    r_out_cubed = r_out**3
    r_cubed = np.random.uniform(r_in_cubed, r_out_cubed, num_points)
    r = np.cbrt(r_cubed)

    # 2. Generate polar angle (phi)
    # Samples cos(phi) uniformly, then takes arccos.
    u = np.random.uniform(-1.0, 1.0, num_points)
    phi = np.arccos(u)

    # 3. Generate azimuthal angle (theta)
    # Samples theta uniformly from [0, 2*pi]
    theta = np.random.uniform(0.0, 2 * np.pi, num_points)

    # 4. Convert Spherical (r, phi, theta) to Cartesian (x, y, z)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Combine coordinates into a single array
    points = np.stack((x, y, z), axis=-1)
    return points

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

def generate_random_unit_vectors(N: int):
    """
    Generates N randomly and uniformly distributed unit vectors (orientations)
    in 3D Cartesian space.

    The method used is to sample from a standard 3D Gaussian distribution and
    then normalize the resulting vectors, which guarantees uniformity on the
    surface of the unit sphere.

    Args:
        N: The number of random orientations to generate.

    Returns:
        A NumPy array of shape (N, 3) where each row is a unit vector.
    """
    # 1. Generate N random vectors (x, y, z) where each component is
    # sampled from a standard normal (Gaussian) distribution.
    # The shape is (5000, 3).
    random_vectors = np.random.randn(N, 3)

    # 2. Calculate the L2 norm (magnitude) of each vector.
    # axis=1 sums up the squares of components for each row (vector).
    # keepdims=True ensures the resulting shape is (5000, 1), which allows
    # for easy element-wise division in the next step (broadcasting).
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)

    # 3. Normalize the vectors by dividing each vector by its magnitude.
    # This results in vectors lying on the unit sphere (magnitude = 1).
    unit_vectors = random_vectors / norms

    # Sanity check (optional): Print the norm of the first vector to confirm it's 1
    # print(f"Norm of the first vector: {np.linalg.norm(unit_vectors[0])}")

    return unit_vectors    


def generate_gaussian_random_time_course(duration_s, sampling_rate_hz, mean, std_dev, n_dipoles=1):
    """
    Generates N independent Gaussian random time courses.
    """
    n_times = int(duration_s * sampling_rate_hz)
    
    # 1. Generate the time axis
    time = np.linspace(0, duration_s, n_times, endpoint=False)
    
    # 2. Generate N independent random time series (Vectorized!)
    # Shape will be (n_dipoles, n_times)
    source_time_series_matrix = np.random.normal(
        loc=mean, 
        scale=std_dev, 
        size=(n_dipoles, n_times)
    )
    
    return time, source_time_series_matrix


def mix_linear_weighted_snr(fwd, raw_signal, raw_noise, indices, q_moment, ori, target_snr, source_pos):
    """
    Scales the raw_signal by calculating type-specific linear SNRs, converting 
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
        fwd_idx_1 = np.where(fwd_vertno == indices[0][1])[0][0]
        fwd_idx_2 = np.where(fwd_vertno == indices[1][1])[0][0]
        
        print(f"The indices in the forward data is: {fwd_idx_1} and {fwd_idx_2}")
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
   
    # Dipole 1 contribution to signal
    signal_magnitude_qL = np.zeros(raw_noise.info['nchan'])
    start_col = fwd_idx_1 * 3
    end_col = fwd_idx_1 * 3 + 3
    L_source = L_free[:, start_col:end_col]
    print(L_source)
    print(ori_vec)
    L_eff = L_source @ ori_vec.T
    signal_magnitude_qL += np.abs(q * np.squeeze(L_eff))
    
    # Dipole 2 contribution to signal
    start_col = fwd_idx_2 * 3
    end_col = fwd_idx_2 * 3 + 3
    L_source = L_free[:, start_col:end_col]
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

def mix_power_weighted_snr(fwd, raw_signal, raw_noise, indices, q_moment, oris, target_snr, source_pos):
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
        fwd_idx_1 = np.where(fwd_vertno == indices[0])[0][0]
        fwd_idx_2 = np.where(fwd_vertno == indices[1])[0][0]
        
        print(f"The indices in the forward data is: {fwd_idx_1} and {fwd_idx_2}")
    except IndexError:
        print("The chosen vertex is not included in the forward solution (likely outside the mask).")
  
    q = q_moment
    
    ori_vec_1 = np.array(oris[0])
    ori_vec_1 = ori_vec_1 / np.linalg.norm(ori_vec_1)
    
    ori_vec_2 = np.array(oris[1])
    ori_vec_2 = ori_vec_2 / np.linalg.norm(ori_vec_2)
    
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

    # Dipole 1 power
    L_source_1 = L_free[:, fwd_idx_1 * 3 : fwd_idx_1 * 3 + 3]
    L_eff_1 = L_source_1 @ ori_vec_1.T
    mag_1 = np.abs(q * np.squeeze(L_eff_1))
    
    # Dipole 2 power
    L_source_2 = L_free[:, fwd_idx_2 * 3 : fwd_idx_2 * 3 + 3]
    L_eff_2 = L_source_2 @ ori_vec_2.T
    mag_2 = np.abs(q * np.squeeze(L_eff_2))
    
    # Sum of powers (since sources are temporally uncorrelated)
    sig_power_mag_num = np.sum(mag_1[picks_mag]**2) + np.sum(mag_2[picks_mag]**2)
    sig_power_grad_num = np.sum(mag_1[picks_grad]**2) + np.sum(mag_2[picks_grad]**2)

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
    
    cos_sim = np.dot(L_eff_2, L_eff_1) / (np.linalg.norm(L_eff_1) * np.linalg.norm(L_eff_2))
    
    return raw_combined, cos_sim


# %%

# creating forward

# creating volume source space
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_res_2mm-fwd.fif"
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

filename = '/Users/owenjohnson/MEG/brain_noise_5000_dipoles_150Hz_100s_rad_10cm_src_meg.fif'
filename = 'brain_noise_5000_dipoles_150Hz_100s_single_shell_rad_10cm_meg.fif'

#   Load files
raw_noise = mne.io.read_raw_fif(filename, preload=True)

# %%

# Normalizing rms brain noise density over all samples and all channels

# normalization to 14 fT/sqrt(Hz) for planar grad, resulting in ~ 33.8fT/sqrt(Hz)

# raw_noise.compute_psd().plot()


# After PSD of raw it appears the rms mean channel noise is already close to good
# so I don't change it for now


# %%

# Creating dipole signals of interest.

n_dipoles = 2 

d = d_num / 1000 # conv to m from mm
a = a_num / 100  # conv to m from cm

fs = 150 # sampling frequency

source_pos1 = [0, d/2, np.sqrt(a**2 - ((d**2) / 4))]
source_pos2 = [0, -d/2, np.sqrt(a**2 - ((d**2) / 4))]

source_pos = [source_pos1, source_pos2]

ori_1 = [1,0,0]
ori_2 = [0,1,0]
oris = [ori_1,ori_2]

# Lorentzian waveform features
width_1 = 25.0 
latency_1 = 50.0 
amp_1 = 0.3 

width_2 = 40.0 
latency_2 = 90.0 
amp_2 = 0.4 

source_pos1

waveforms = {'width': [width_1, width_2], 'latency': [latency_1, latency_2], 'amp' : [amp_1,amp_2]}

def create_raw_from_dipoles(n_dipoles, source_pos, oris, src, fs, waveforms, plot="plot"):
    duration = 100.0
    n_events = 156
    tstep = 1.0 / fs
    n_total_samples = int(duration * fs)
    
    t_pre_trigger = 0.2
    t_post_decay = 0.2
    pre_trigger_samples = int(t_pre_trigger * fs)
    window_samples = pre_trigger_samples + int(t_post_decay * fs)
    t_template = np.linspace(-t_pre_trigger, t_post_decay, window_samples, endpoint=False)
    
    trigger_indices = pre_trigger_samples + np.arange(n_events) * int(n_total_samples / n_events)
    rng = np.random.default_rng(42)
    
    data_list = []
    vertices_list = []
    
    for n in range(n_dipoles):
        # New time series for each dipole
        this_dipole_time_series = np.zeros(n_total_samples)
        
        # Waveform generation
        tau = (waveforms["width"][n] / 1000.0) / 2.0
        latency_s = waveforms['latency'][n] / 1000.0
        template_wave = 1.0 / (1.0 + ((t_template - latency_s) / tau)**2)
        amplitudes = 1.0 + waveforms['amp'][n] * (2.0 * rng.random(n_events) - 1.0)
        # amplitudes = 1.0 + np.random.normal(loc=0.0, scale=waveforms['amp'][n], size=n_events)
        
        for trig_idx, amp in zip(trigger_indices, amplitudes):
            start_idx = trig_idx - pre_trigger_samples
            end_idx = start_idx + window_samples
            if end_idx <= n_total_samples:
                this_dipole_time_series[start_idx:end_idx] += template_wave * amp

        # Find vertex
        _, v_idx = find_closest_vertices(src, [source_pos[n]])
        v_idx = int(v_idx[0])
        vertices_list.append(v_idx)

        # Correct shape for VolVectorSourceEstimate (n_vertices, 3, n_times)
        orient_vec = np.array(oris[n], dtype=float).reshape(1, 3, 1) # (1 vertex, 3 axes, 1 time)
        dipole_data = orient_vec * this_dipole_time_series       # (1, 3, n_times)
        data_list.append(dipole_data)

    # Combine data
    # final shape: (n_dipoles, 3, n_times)
    dipoles_data = np.concatenate(data_list, axis=0)
    vertices = np.array(vertices_list)

    # Sort to satisfy MNE requirements
    sort_idx = np.argsort(vertices)
    vertices = vertices[sort_idx]
    dipoles_data = dipoles_data[sort_idx]

    stc_vec = mne.VolVectorSourceEstimate(
        data=dipoles_data,
        vertices=[vertices], # Volume expects list of arrays
        tmin=0,
        tstep=tstep,
        subject='sample'
    )

    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = trigger_indices
    events[:, 2] = 1

    return stc_vec, events, vertices_list

# %%

stc_vec, events, vertices_list = create_raw_from_dipoles(n_dipoles, source_pos, oris, src, fs, waveforms, plot="plot")

# %%


# Simulate Raw Data
raw_source = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)
    

#raw_source.plot(events=events, duration=2.0, start=0, title="Simulated Source with Events")


# %%

# output_filename = '/Users/owenjohnson/MEG/No_noise_meg_150Hz_two_dip_raw.fif'
# raw_source.save(output_filename, overwrite=True)

# %%

# Mixing brain noise into raw

print(vertices_list)

final_raw, cos_sim = mix_power_weighted_snr(
            fwd=fwd, 
            raw_signal=raw_source,
            raw_noise=raw_noise,
            indices=vertices_list,
            oris=oris,
            q_moment=1,
            target_snr=SNR, 
            source_pos=source_pos
            )


# Average the epochs to see the signal clearly out of the noise
evoked = mne.Epochs(final_raw, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# Plot
fig = evoked.plot(show=False)

# final_raw = mix_linear_weighted_snr(
#             fwd=fwd, 
#             raw_signal=raw_source,
#             raw_noise=raw_noise,
#             indices=vertices_list,
#             ori=ori,
#             q_moment=1,
#             target_snr=1.7, 
#             source_pos=source_pos
#             )


# # Plotting evoked

# # Average the epochs to see the signal clearly out of the noise
# evoked = mne.Epochs(final_raw, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# # Plot
# fig = evoked.plot(show=False)

# %%

# output_filename = '/Users/owenjohnson/MEG/Two_source_dif_ori_d_02_a_5_SNR_2_93cm_head_meg.fif'
# final_raw.save(output_filename, overwrite=True)

# %%

# # Adding Noise


# # %%

# normal_device_noise_cov = mne.make_ad_hoc_cov(info = raw_source.info, std = None)

# raw = final_raw.copy()

# final_raw_w_device_noise = mne.simulation.add_noise(inst = raw, cov = normal_device_noise_cov, random_state = 42)

# evoked = mne.Epochs(final_raw_w_device_noise, events, tmin=-0.2, tmax=0.5, baseline=(-.2,-.05)).average()

# # Plot
# fig = evoked.plot(show=False)

# def check_mag_density(raw):
#     mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False)
#     spectrum_mag = raw.compute_psd(picks=mag_picks, fmin=1.0, fmax=75.0, n_fft=2048)
    
#     # Mags are already in T, just convert to fT
#     asd_mag_fT = np.mean(np.sqrt(spectrum_mag.get_data())) * 1e15
#     print(f"Resulting Magnetometer Density: {asd_mag_fT:.2f} fT/sqrt(Hz)")


# raw_noise = final_raw.copy()

# check_mag_density(raw_noise)

# raw_noise = final_raw_w_device_noise.copy()

# check_mag_density(raw_noise)

# final_raw_w_device_noise = mne.simulation.add_noise(inst = raw_noise, cov = normal_device_noise_cov, random_state = 42)


# raw_noise = final_raw_w_device_noise.copy()

# check_mag_density(raw_noise)
# # %%


# output_filename = '/Users/owenjohnson/MEG/synthetic_lorentzian_5cm_SNR_1_150Hz_meg_w_5fT_device_noise_raw.fif'
# raw_noise.save(output_filename, overwrite=True)

# %%

# Gain Error Implemented

n_channels = 306
gain_error_std = .001

# Generate N_channels random gain factors (G_i) following a Gaussian distribution
# with a mean of 1 and a standard deviation.
gain_factors = 1 + np.random.normal(loc=0.0, scale=gain_error_std, size=n_channels)

# --- Adding Channel Gain Error ---
scaling_matrix = np.diag(gain_factors)

final_raw_copy = final_raw.copy()

data_transformed = final_raw_copy.get_data()

data_transformed = scaling_matrix @ data_transformed


info = final_raw.info
raw_w_gain_noise = mne.io.RawArray(data_transformed, info, copy = "info")

# raw_w_gain_noise.plot()

# %%

# Save data for comparison with other methods like trad. LCMV beamformer
output_filename = f'/Users/owenjohnson/MEG/Two_source_dif_ori_d_{d_num}_a_{a_num}_SNR_{SNR}_10cm_head_gain_error_single_shell_meg.fif'
raw_w_gain_noise.save(output_filename, overwrite=True)

# %%

# print(np.std(raw_w_gain_noise.get_data(),axis=1))

# print(np.std(final_raw.get_data(),axis=1))