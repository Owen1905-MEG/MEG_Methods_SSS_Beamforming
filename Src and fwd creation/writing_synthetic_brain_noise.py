#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 20:12:49 2025

@author: owenjohnson
"""



import numpy as np

import mne
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif


from scipy.spatial import KDTree

print(__doc__)


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
    
    return indices

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


# %%

# creating forward

# creating volume source space
full_path = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_res_2mm-fwd.fif"
full_path = "spherical_head_model_rad_10cm_res_2mm-fwd.fif"
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
raw.del_proj()
raw.crop(tmin=0, tmax=1)
raw.pick("meg")
info = raw.info


# If sfreq does not equal 1000 Hz
sampling_rate = 150
tstep = 1.0 / sampling_rate
raw = read_raw_fif(raw_fname, preload=False)
raw.crop(tmin=0, tmax=1)
raw.del_proj()
new_raw = raw.resample(sfreq=sampling_rate)
new_raw.del_proj()
info = new_raw.info

# %%

# --- Creating Brain Noise Sources ---

n_active_dipoles = 5100


# getting locations and orientations
pos = generate_spherical_shell_points(num_points=n_active_dipoles, r_in=.05, r_out=.080)


# # converting pos to vertex
# Find the closest vertex for each random point
indices = find_closest_vertices(src, pos)
indices = np.unique(find_closest_vertices(src, pos))
ordered_indices = sorted(indices)

n_dipoles_reduced = indices.shape[0]
print(n_dipoles_reduced)

# source time course
time, source_time_matrix = generate_gaussian_random_time_course(
                            duration_s=25, 
                            sampling_rate_hz=sampling_rate,
                            mean=0.0,
                            std_dev=1.0,
                            n_dipoles=n_dipoles_reduced
                            )

source_time_matrix *= source_time_matrix * 1e-9

# Prepare the orientation vector
vector = generate_random_unit_vectors(N=n_dipoles_reduced)
orient_vec = np.array(vector, dtype=float)
orient_vec_3d = orient_vec[:n_dipoles_reduced].reshape(n_dipoles_reduced, 3, 1)


# Reshape time series for broadcasting: (N_dipoles, 1, N_times)
time_series_3d = source_time_matrix[:, np.newaxis, :]

# Calculate vector data for ALL dipoles (N_dipoles, 3, N_times)
# The result is calculated using optimized NumPy C code.
dipole_data_3d = orient_vec_3d * time_series_3d 


# %%

# print(vertno_array_mne_ordered.shape)

# Create the Vector Source Estimate
stc_vec = mne.VolVectorSourceEstimate(
    data=dipole_data_3d,
    vertices=[indices], # Array of all unique vertex indices
    tmin=0,
    tstep=tstep, 
    subject=subject
)

# %%

# Simulate Raw Data
raw_noise = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)

raw_noise.plot()

# %%

output_filename = 'synthetic_brain_noise_1_meg.fif'
raw_noise.save(output_filename, overwrite=True)



# --- Creating Brain Noise Sources ---

# source time course
time, source_time_matrix = generate_gaussian_random_time_course(
                            duration_s=25, 
                            sampling_rate_hz=sampling_rate,
                            mean=0.0,
                            std_dev=1.0,
                            n_dipoles=n_dipoles_reduced
                            )

source_time_matrix *= source_time_matrix * 1e-9

# Reshape time series for broadcasting: (N_dipoles, 1, N_times)
time_series_3d = source_time_matrix[:, np.newaxis, :]

# Calculate vector data for ALL dipoles (N_dipoles, 3, N_times)
# The result is calculated using optimized NumPy C code.
dipole_data_3d = orient_vec_3d * time_series_3d 


# %%

# print(vertno_array_mne_ordered.shape)

# Create the Vector Source Estimate
stc_vec = mne.VolVectorSourceEstimate(
    data=dipole_data_3d,
    vertices=[indices], # Array of all unique vertex indices
    tmin=25,
    tstep=tstep, 
    subject=subject
)

# %%

# Simulate Raw Data
raw_noise = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)

# %%

output_filename = 'synthetic_brain_noise_2_meg.fif'
raw_noise.save(output_filename, overwrite=True)



# --- Creating Brain Noise Sources ---

# source time course
time, source_time_matrix = generate_gaussian_random_time_course(
                            duration_s=25, 
                            sampling_rate_hz=sampling_rate,
                            mean=0.0,
                            std_dev=1.0,
                            n_dipoles=n_dipoles_reduced
                            )

source_time_matrix *= source_time_matrix * 1e-9

# Reshape time series for broadcasting: (N_dipoles, 1, N_times)
time_series_3d = source_time_matrix[:, np.newaxis, :]

# Calculate vector data for ALL dipoles (N_dipoles, 3, N_times)
# The result is calculated using optimized NumPy C code.
dipole_data_3d = orient_vec_3d * time_series_3d 


# %%

# print(vertno_array_mne_ordered.shape)

# Create the Vector Source Estimate
stc_vec = mne.VolVectorSourceEstimate(
    data=dipole_data_3d,
    vertices=[indices], # Array of all unique vertex indices
    tmin=50,
    tstep=tstep, 
    subject=subject
)

# %%

# Simulate Raw Data
raw_noise = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)

# %%

output_filename = 'synthetic_brain_noise_3_meg.fif'
raw_noise.save(output_filename, overwrite=True)


# --- Creating Brain Noise Sources ---

# source time course
time, source_time_matrix = generate_gaussian_random_time_course(
                            duration_s=25, 
                            sampling_rate_hz=sampling_rate,
                            mean=0.0,
                            std_dev=1.0,
                            n_dipoles=n_dipoles_reduced
                            )

source_time_matrix *= source_time_matrix * 1e-9

# Reshape time series for broadcasting: (N_dipoles, 1, N_times)
time_series_3d = source_time_matrix[:, np.newaxis, :]

# Calculate vector data for ALL dipoles (N_dipoles, 3, N_times)
# The result is calculated using optimized NumPy C code.
dipole_data_3d = orient_vec_3d * time_series_3d 


# %%

# print(vertno_array_mne_ordered.shape)

# Create the Vector Source Estimate
stc_vec = mne.VolVectorSourceEstimate(
    data=dipole_data_3d,
    vertices=[indices], # Array of all unique vertex indices
    tmin=75, # change to 25s??????
    tstep=tstep, 
    subject=subject
)

# %%

# Simulate Raw Data
raw_noise = mne.apply_forward_raw(fwd, stc_vec, info, on_missing='raise', use_cps=False)

# %%

output_filename = 'synthetic_brain_noise_4_meg.fif'
raw_noise.save(output_filename, overwrite=True)
# %%

# Combing raw

# 1. Define the exact filenames in the order you want them concatenated
filenames = [
    "synthetic_brain_noise_1_meg.fif",
    "synthetic_brain_noise_2_meg.fif",
    "synthetic_brain_noise_3_meg.fif",
    "synthetic_brain_noise_4_meg.fif"
]

# 2. Load files
raw_list = [mne.io.read_raw_fif(f, preload=True) for f in filenames]

# 3. Concatenate
raw_noise = mne.concatenate_raws(raw_list)

# %%



raw_noise.plot()
# output_filename = 'synthetic_brain_noise_5000_dipoles_150Hz_meg.fif'
# raw_noise.save(output_filename, overwrite=True)

raw_noise.compute_psd().plot()

# %%

filename = 'synthetic_brain_noise_5000_dipoles_150Hz_meg.fif'

#   Load files
raw_noise = mne.io.read_raw_fif(filename, preload=True)




# %%

def apply_simulation_normalization(raw, target_grad_fT=14.0):
    """
    Normalizes the entire Raw instance based on Gradiometer noise density.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data containing simulated brain noise.
    target_grad_fT : float
        Target density for gradiometers in fT/sqrt(Hz) (default 14.0).
    """
    
    # 1. Compute PSD for Gradiometers only
    #    We use fmin=1, fmax=40 to capture the "bulk" of the noise density
    grad_picks = mne.pick_types(raw.info, meg='grad', eeg=False)
    spectrum_grad = raw.compute_psd(picks=grad_picks, fmin=1.0, fmax=40.0, n_fft=2048)
    
    # 2. Get Data in MNE Units (T/m)
    #    Shape: (n_grad_channels, n_freqs)
    psd_grad_tm = spectrum_grad.get_data()
    asd_grad_tm = np.sqrt(psd_grad_tm)
    
    # 3. Convert MNE Gradiometer Data (T/m) to Field Difference (T) for comparison
    #    Neuromag/VectorView baseline is typically 1.68 cm (0.0168 m)
    #    We use 1.7 cm (0.017 m) as per your description.
    baseline_m = 0.017
    asd_grad_T = asd_grad_tm * baseline_m
    
    # 4. Calculate current mean density in fT
    #    (Convert T -> fT by multiplying by 1e15)
    current_density_fT = np.mean(asd_grad_T) * 1e15
    
    print(f"Current Gradiometer Density: {current_density_fT:.2f} fT/sqrt(Hz)")

    # 5. Determine Global Scaling Factor
    #    ratio = Target / Current
    scaling_factor = target_grad_fT / current_density_fT
    
    print(f"Applying global scaling factor: {scaling_factor:.4f}")

    # 6. Apply Scaling to ALL CHANNELS (Mags + Grads)
    #    This preserves the physical relationship (Maxwell's equations)
    #    so the Magnetometers scale 'naturally' alongside the Gradiometers.
    raw.apply_function(lambda x: x * scaling_factor, picks='all')
    
    return raw

# --- Validation Step ---
# To confirm the Magnetometers landed at ~33.8 fT/sqrt(Hz)
def check_mag_density(raw):
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False)
    spectrum_mag = raw.compute_psd(picks=mag_picks, fmin=1.0, fmax=40.0, n_fft=2048)
    
    # Mags are already in T, just convert to fT
    asd_mag_fT = np.mean(np.sqrt(spectrum_mag.get_data())) * 1e15
    print(f"Resulting Magnetometer Density: {asd_mag_fT:.2f} fT/sqrt(Hz)")


raw_noise_copy = raw_noise.copy()
raw_normalized = apply_simulation_normalization(raw_noise_copy)



check_mag_density(raw_normalized)
# 36.77 fT/sqrt(Hz) for 1000 Hz signal after procedure.


raw_normalized.compute_psd().plot()

# %%


output_filename = 'brain_noise_5000_dipoles_150Hz_100s_single_shell_rad_10cm_meg.fif'
raw_normalized.save(output_filename, overwrite=True)






