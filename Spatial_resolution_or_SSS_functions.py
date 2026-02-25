#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:05:34 2026

@author: owenjohnson
"""

import numpy as np
import mne


def get_peak_dimensions(stc, src):
    """
    Computes peak dimension, i.e. cuberoot of volume of all voxels with half or more max activity.

    Parameters
    ----------
    stc : mne.VolSourceEstimate
        Volume source estimate
    spatial_resolution_mm : 
        spatial resolution of source space used

    Returns
    -------
    float :
    
    """

    res_m = np.diff(src[0]['rr'][:, 0])[np.diff(src[0]['rr'][:, 0]) > 0].min()
    spatial_resolution_mm = res_m * 1000
    spatial_resolution_mm = round(spatial_resolution_mm, 2)
    spatial_resolution_m = spatial_resolution_mm / 1000.0 # Convert to meters for consistency
    voxel_volume_m3 = spatial_resolution_m ** 3
    
    # Find voxels with activity greater than or equal to half peak value
    
    peak_value = np.max(np.abs(stc.data))
    threshold = peak_value / 2.0
    peak_idx = np.unravel_index(np.argmax(np.abs(stc.data)), stc.shape)
    peak_spatial_map = np.abs(stc.data[:, peak_idx[1]])
    n_voxels_at_peak_half_max = np.sum(peak_spatial_map >= threshold)
    
    # --- Calculate Peak Volume and Dimension ---
    
    peak_volume_m3 = n_voxels_at_peak_half_max * voxel_volume_m3
    peak_dimension_m = peak_volume_m3 ** (1/3)
    peak_dimension_mm = peak_dimension_m * 1000.0
    n_voxels = n_voxels_at_peak_half_max
    return peak_dimension_mm, spatial_resolution_mm, n_voxels


def compute_location_error(pos: np.ndarray, est_pos: np.ndarray) -> float:
    """
    Computes the location error (Euclidean distance) between the true 
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
    float
        The location error in the unit of the input coordinates (typically meters).
    """
    # The location error is the magnitude (norm) of the difference vector.
    location_error = np.linalg.norm(est_pos - pos)
    return location_error



#  --- SSS Basis and Functions ---

def get_pS_and_SSS_basis(L_in, L_ext, info, mag_scale = 100, coord_frame = "head", origin = (0.,0.,0.)):
    
    S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(
        info,
        origin=(0., 0., 0.),
        int_order=L_in,
        ext_order=L_ext,
        coord_frame='head',
        regularize=None,
        bad_condition='warning',
        mag_scale=100
        )
    
    # normalize S and reduce condition number before computing pS
    
    ST_S = S.transpose() @ S
    print(np.linalg.cond(ST_S,None))
    
    # print(np.linalg.cond(S,None))
    col_n_S = np.linalg.norm(S, axis=0)
    
    n_S = S / col_n_S
    
    ST_S_norm = n_S.transpose() @ n_S
    
    print(np.linalg.cond(ST_S_norm,None))
    print(np.linalg.cond(n_S,None))
    
    # This tracks the indices of the *original* basis vectors (0 to S.shape[1]-1)
    # that are *still present* in the reduced_S matrix.
    current_original_indices = np.arange(S.shape[1])
    dropped_vector_indices = []
    dropped_vector_type = []
    dropped_vector_cond_step = []
    step_counter = 0
    
    # Removing column vectors to reduce condition number of S transpose @ S 
    # according to Vrba, J., Taulu, S., Nenonen, J. et al. Signal Space Separation Beamformer. Brain Topogr 23, 128–133 (2010). https://doi.org/10.1007/s10548-009-0120-7
    
    reduced_S = n_S.copy()
    
    print(n_use_in)
    
    ST_S = reduced_S.transpose() @ reduced_S
    
    new_n_use_in = n_use_in
    
    while np.linalg.cond(ST_S, None) >= 10**5:
        step_counter += 1
        cond_compare_list =[]
        for n in range(reduced_S.shape[1]):
            reduced_S_copy = reduced_S.copy()
            reduced_S_n = np.delete(reduced_S_copy, n, axis=1)
            ST_S = reduced_S_n.transpose() @ reduced_S_n
            cond_n = np.linalg.cond(ST_S, None)
            cond_compare_list.append(cond_n)
        
        if not cond_compare_list :
            print("cond_compare_list is empty. Breaking loop.")
            break 
    
        min_value = min(cond_compare_list)
        min_index = cond_compare_list.index(min_value)
        
        original_index_dropped = current_original_indices[min_index]
        
        if min_index < new_n_use_in:
            new_n_use_in -= 1
            v_type = 'Internal'
        else:
            print("index is for external basis vector")
            v_type = 'External'
            
        dropped_vector_indices.append(original_index_dropped)
        dropped_vector_type.append(v_type)
        dropped_vector_cond_step.append(step_counter)
        
        print(new_n_use_in)
        print(min_index)
        current_original_indices = np.delete(current_original_indices, min_index)
        reduced_S = np.delete(reduced_S, min_index, axis=1)
        ST_S = reduced_S.transpose() @ reduced_S
        print(f"Current condition number: {np.linalg.cond(ST_S, None):.2e}")
    
    
    reduced_S_in = reduced_S[:, :new_n_use_in]
    
    # calculating our own pseudo inverse from S
    pS_from_S = np.linalg.pinv(reduced_S)
    
    
    print(np.linalg.cond(pS_from_S, None))
    
    # getting final pS internal
    pS_final = pS_from_S[:new_n_use_in, :]
    
    sss_channels = []
    for n in range(new_n_use_in):
        ch_number = n + 1
        sss_channels.append("MEG" + f"{ch_number}".zfill(4))
        
    return pS_final, new_n_use_in, reduced_S_in, sss_channels
    


def mag_scaling_lead_field_update(fwd, mag_picks, bad_indices):
    # getting attributes of forward
    forward_copy = fwd.copy()
    sol = forward_copy['sol']
    data = sol['data']
    
    # updating lead field "mag" scale
    data[mag_picks] *= 100
    
    # dropping bad row
    bad_indices.sort(reverse=True)
    for bad_index in bad_indices:
        np.delete(data, bad_index, 1)
            
    
    forward_copy['sol']['data'] = data
    
    return forward_copy

def change_of_raw_info_to_sss(
        raw, old_channel_names, sss_channels, new_n_use_in
        ):
    
    new_raw = raw.copy()
    
    # modifying  raw attribute ch_names
    # dropping channels down to n_use_in
    raw_channels_to_drop = raw.info['ch_names'][new_n_use_in:]
    new_raw.drop_channels(raw_channels_to_drop)
    
    
    # rename channels
    ch_dict_mapping = {}
    for n in range(new_n_use_in):
        ch_dict_mapping[raw.info['ch_names'][n]] = sss_channels[n]
    new_raw.rename_channels(ch_dict_mapping)

    # Create a mapping to set all channels to 'mag'
    channel_type_mapping = {ch_name: 'mag' for ch_name in new_raw.info['ch_names']}

    # Update the channel types in the info object
    new_raw.info.set_channel_types(channel_type_mapping)
    
    return new_raw

def change_of_evoked_info_to_sss_efficient(
    raw, evoked, sss_channels, new_n_use_in):
    
    # 1. Use deepcopy to ensure no side effects on the original evoked object
    evoked_copy = evoked.copy()
    
    # --- Channel Selection (Dropping) ---
    # 2. Get the list of channels to KEEP, based on the desired new_n_use_in
    # This is more direct than getting channels to drop.
    channels_to_keep = raw.info['ch_names'][:new_n_use_in]
    
    # 3. Pick the channels in the evoked object directly.
    # This efficiently handles both the data and the info structure.
    evoked_copy.pick(channels_to_keep)
    
    # --- Channel Renaming ---
    # 4. Create the mapping dictionary using slicing, avoiding an explicit loop
    # Maps channels picked from RAW (and now in evoked_copy) to the new SSS names
    old_names = evoked_copy.info['ch_names']
    new_names = sss_channels[:new_n_use_in]
    
    # Ensure the lengths match before zipping (a good check, though they should)
    if len(old_names) != len(new_names):
        raise ValueError("Channel counts for renaming do not match.")

    ch_dict_mapping = dict(zip(old_names, new_names))
    evoked_copy.rename_channels(ch_dict_mapping)
    
    # --- Channel Type Setting ---
    # 5. MNE provides the set_channel_types method which is efficient.
    # This can be simplified further by setting the type for ALL channels
    # in the object, as they should all be 'mag' after SSS processing.
    evoked_copy.set_channel_types({ch_name: 'mag' 
                                   for ch_name in evoked_copy.info['ch_names']})
    
    return evoked_copy


def change_of_cov_basis_into_sss(
        raw,covariance, pS_final, new_ch_names: list = None
    ):

    #preparing for transformation
    
    cov_data = covariance.data
    pS_T = pS_final.transpose()
    
    
    #getting new attributes
    transformed_cov_data1 = cov_data @ pS_T
    transformed_cov_data2 = pS_final @ transformed_cov_data1
    
    
    new_n_channels = transformed_cov_data2.shape[0]
    new_ch_names = [f'MEG{i+1:04d}' for i in range(new_n_channels)]
    
    transformed_covariance = mne.Covariance(
        data = transformed_cov_data2,
        names = new_ch_names,
        bads = raw.info["bads"],
        nfree = covariance.nfree,
        projs = raw.info['projs']
        )

    return transformed_covariance

def change_of_raw_basis(
        raw, raw_sss_info, pS_final
        ):
    
    # preparing for transformation
    data = raw.get_data(['meg'])
    
    transformed_data = pS_final @ data
    
    transformed_raw = mne.io.RawArray(
        data = transformed_data, 
        info = raw_sss_info, 
        copy = "info"
        )
    
    return transformed_raw
    
    
def change_of_lead_fields_basis(fwd, pS, sss_ch, new_n_use_in):
    # getting attributes of forward
    forward_copy = fwd.copy()
    data = forward_copy['sol']['data']
    

    # transforming data
    transformed_data = pS @ data
        
    
   # creating attributes for new forward instance
    new_channels = []
    new_n_chan = new_n_use_in
    
    for n in range(new_n_use_in):
        ch_number = n + 1
        new_channels.append("MEG" + f"{ch_number}".zfill(4))
    
    # modifying forward with transformed lead fields and new attributes
    forward_copy['sol']['data'] = transformed_data
    forward_copy['sol']['row_names'] = new_channels
    forward_copy['nchan'] = new_n_chan
    
    # updating info
    # getting rid of extra channels
    mapping = {}
    for n in range(new_n_use_in):
        ch_number = n + 1
        mapping[forward_copy.ch_names[n]] = "MEG" + f"{ch_number}".zfill(4)
    mne.channels.rename_channels(forward_copy['info'], mapping)
  
    #Picking channels
    final_sss_forward = mne.pick_channels_forward(forward_copy, sss_ch)
    
    return final_sss_forward


def change_of_cov_basis_into_diag(
        raw,covariance, pS_final, new_ch_names: list = None
    ):

    #preparing for transformation
    
    cov_data = covariance.data
    
    # Get the diagonal elements
    diagonal_elements = np.diag(cov_data)

    # Create a new matrix with only the diagonal elements
    C_sss_diag = np.diag(diagonal_elements)
    
    new_n_channels = cov_data.shape[0]
    new_ch_names = [f'MEG{i+1:04d}' for i in range(new_n_channels)]
    
    transformed_covariance = mne.Covariance(
        data = C_sss_diag,
        names = new_ch_names,
        bads = raw.info["bads"],
        nfree = covariance.nfree,
        projs = raw.info['projs']
        )

    return transformed_covariance

if __name__ == "__main__":
    # This only runs if you run my_functions.py directly
    print("Testing function...")
