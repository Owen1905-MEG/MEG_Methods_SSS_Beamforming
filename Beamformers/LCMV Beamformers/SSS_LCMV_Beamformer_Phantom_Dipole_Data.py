#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:11:27 2025

@author: owenjohnson
"""


import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

from mne.beamformer import apply_lcmv, make_lcmv
from mne.datasets import fetch_fsaverage


print(__doc__)

# %%


# all created functions used below

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

def change_of_evoked_info_to_sss(evoked, old_channel_names, sss_channels, new_n_use_in):
    evoked_copy = evoked.copy()
    
    # modifying  evoked attribute ch_names
    # dropping channels down to n_use_in
    evoked_channels_to_drop = evoked.info['ch_names'][new_n_use_in:]
    evoked_copy.drop_channels(evoked_channels_to_drop)
    
    
    # rename channels
    ch_dict_mapping = {}
    for n in range(new_n_use_in):
        ch_dict_mapping[raw.info['ch_names'][n]] = sss_channels[n]
    evoked_copy.rename_channels(ch_dict_mapping)

    # Create a mapping to set all channels to 'mag'
    channel_type_mapping = {ch_name: 'mag' for ch_name in evoked_copy.info['ch_names']}

    # Update the channel types in the info object
    evoked_copy.info.set_channel_types(channel_type_mapping)
    
    return evoked_copy
    
def change_of_cov_basis_into_sss(
    covariance, pS_final, new_ch_names: list = None
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
    data = raw.get_data()
    
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
    transformed_data = pS_final @ data
        
    
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

# %%

# getting raw data from phantom

data_path = bst_phantom_elekta.data_path(verbose=True)

raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname, preload=True)
# raw.compute_psd().plot(average=False, amplitude=False, picks="data", exclude="bads")

print(raw.info)

# %%

# subject information (otaniemi has same external configuration as vecterview)

subjects_dir = data_path
subject = "phantom_otaniemi"

# %%
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
print(actual_pos)

# %%

# reading forward solution

fname = "brainstorm_elekta_phantom_dipoles-fwd.fif"
full_path = "/Users/owenjohnson/MEG/brainstorm_elekta_phantom_dipoles_2mm-fwd.fif"
fwd = mne.read_forward_solution(full_path)


# %%

# Creating SSS Beamformer
# getting events

events = find_events(raw, "STI201")

# raw.plot(events=events)

raw.info["bads"] = ["MEG1933", "MEG2421"]  # known bad channels
raw.del_proj()


# computing epochs
tmin, tmax = -0.1, 0.1
events = find_events(raw, "STI201")
event_id = list(range(1, 33))
bmax = -0.05
proj = False

epochs = mne.Epochs(
    raw, events, event_id, tmin=tmin, tmax=tmax, baseline=(tmin, bmax), preload=True, proj=proj,
)


evoked = epochs["4"][1:-1].average()

evoked.plot()

# %%


# for dipole 1

# get noise and data cov

# use [1:-1] epochs to avoid dipole switching artifacts

noise_cov = mne.compute_covariance(
    epochs[1:-1],
    tmin=-0.1,
    tmax=bmax, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
)

# evoked.plot(noise_cov=noise_cov, time_unit="s")

data_cov = mne.compute_covariance(
    epochs["4"][1:-1],
    tmin=.00, 
    tmax=.1, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

# data_cov.plot(epochs.info)
# noise_cov.plot(epochs.info)


# Making scaled evoked
raw_scaled = raw.copy()
raw_scaled.rescale({"mag":100})
raw_scaled.del_proj()
                     
proj = False
epochs_scaled = mne.Epochs(
    raw_scaled,
    events,
    event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    proj=proj,
)

evoked_scaled = epochs_scaled["4"][1:-1].average()
evoked_scaled.drop_channels("MEG1933")
evoked_scaled.drop_channels("MEG2421")

meg_channel_names = raw.ch_names
bad_index1 = meg_channel_names.index("MEG1933")
bad_index2 = meg_channel_names.index("MEG2421")

bad_indices = [bad_index1, bad_index2]


# picking good channels for forward

new_raw_ch_names = [ch for ch in raw.ch_names if ch not in raw.info['bads']]

raw.pick(['meg'])
mag_picks = mne.pick_types(raw.info, meg='mag', exclude='bads')

forward = mag_scaling_lead_field_update(fwd, mag_picks, bad_indices)

forward.pick_channels(new_raw_ch_names)

# %%


# get S

S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(
    raw.info,
    origin=(0., 0., 0.),
    int_order=12,
    ext_order=3,
    coord_frame='head',
    regularize=None,
    bad_condition='warning'
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

# Removing column vectors to reduce condition number of S transpose @ S 
# according to Vrba, J., Taulu, S., Nenonen, J. et al. Signal Space Separation Beamformer. Brain Topogr 23, 128–133 (2010). https://doi.org/10.1007/s10548-009-0120-7

reduced_S = n_S.copy()

print(n_use_in)

ST_S = reduced_S.transpose() @ reduced_S

new_n_use_in = n_use_in

while np.linalg.cond(ST_S, None) >= 10**2:
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
    if min_index < new_n_use_in:
        new_n_use_in -= 1
    else:
        print("index is for external basis vector")
    print(new_n_use_in)
    print(min_index)
    reduced_S = np.delete(reduced_S, min_index, axis=1)
    ST_S = reduced_S.transpose() @ reduced_S
    print(f"Current condition number: {np.linalg.cond(ST_S, None):.2e}")

# calculating our own pseudo inverse from S
pS_from_S = np.linalg.pinv(reduced_S)

#finding column to cut in pS as 306 -> 304 to fit covariance
meg_channel_names = raw.ch_names

bad_index1 = meg_channel_names.index("MEG1933")
bad_index2 = meg_channel_names.index("MEG2421")


# cutting bad column from pS
pS_from_S_no_bads = np.delete(pS_from_S, (bad_index1, bad_index2), 1)

# getting final pS internal
pS_final = pS_from_S_no_bads[:new_n_use_in, :]

sss_channels = []
for n in range(new_n_use_in):
    ch_number = n + 1
    sss_channels.append("MEG" + f"{ch_number}".zfill(4))
    
# %%



# update  evoked info to sss basis

evoked_w_sss_info = change_of_evoked_info_to_sss(evoked_scaled, new_raw_ch_names, sss_channels, new_n_use_in) 


# change evoked scaled to sss basis
evoked_scaled_data = evoked_scaled.get_data()
sss_data_evoked_scaled = pS_final @ evoked_scaled_data
sss_evoked_scaled = mne.EvokedArray(sss_data_evoked_scaled, evoked_w_sss_info.info, tmin=tmin)

# plotting evoked in sss basis

# sss_evoked_scaled.plot()

# change data and noise cov to sss basis

sss_data_cov = change_of_cov_basis_into_sss(data_cov, pS_final)
sss_noise_cov = change_of_cov_basis_into_sss(noise_cov, pS_final)

# plotting 

# sss_data_cov.plot(sss_evoked_scaled.info)
# sss_noise_cov.plot(sss_evoked_scaled.info)


# change lead field to sss basis

# forward lead fields transformed to SSS basis
sss_forward = change_of_lead_fields_basis(forward, pS_final, sss_channels, new_n_use_in)

# %%



# make filter

filters_sss = make_lcmv(
    sss_evoked_scaled.info,
    sss_forward,
    sss_data_cov,
    reg=.05,
    noise_cov=sss_noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank="info",
    reduce_rank=True
)

src = fwd["src"]

# apply filter and get stc

stc_sss = mne.beamformer.apply_lcmv(sss_evoked_scaled, filters_sss)

stc_sss.plot(src, subject=subject, subjects_dir=subjects_dir)


print(actual_pos[3])

v_id, t = stc_sss.get_peak()
est_pos = fwd['src'][0]['rr'][v_id]
print(est_pos)
diffs = 1000 * np.sqrt(np.sum((est_pos - actual_pos[3]) ** 2, axis=-1))
print(f"{diffs} mm localization error")
