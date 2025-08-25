#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:17:50 2025

@author: owenjohnson
"""

import os


import mne
from mne.beamformer import make_lcmv
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt

# %%


data_path = sample.data_path()

# the raw file containing the channel location + types
sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
# The paths to Freesurfer reconstructions
subjects_dir = data_path / "subjects"
subject = "sample"


raw = mne.io.read_raw_fif(raw_fname, preload = True)
raw.info["bads"] = ["MEG 2443"]
raw.info["bads"] += ["MEG 2313"]

# deleting projectors
raw.del_proj()

event_id = 1  # those are the trials with left-ear auditory stimuli
tmin, tmax = -0.2, 0.5
events = mne.find_events(raw)

# picking channels
raw.pick(["meg", "eog"])

# Create epochs
proj = False  
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    proj=proj,
    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6)
)

evoked = epochs.average().crop(0.00,0.15)
evoked.pick(picks="meg")
evoked.drop_channels("MEG 2443")
evoked.drop_channels("MEG 2313")

# %%

# Getting data covariances over different periods
# and noise covariances from pre stimulus baseline period

noise_cov_epochs = mne.compute_covariance(
    epochs, 
    keep_sample_mean = False,
    tmin=-0.20, 
    tmax=0.0, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
)

noise_cov_epochs_kept_mean = mne.compute_covariance(
    epochs, 
    keep_sample_mean=True,
    tmin=-0.20, 
    tmax=0.0, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
)

# data cov from part of epochs from .01 to .15 s

data_cov_epochs_15 = mne.compute_covariance(
    epochs, 
    keep_sample_mean = False,
    tmin=0.01, 
    tmax=.15, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

data_cov_epochs_kept_mean_15 = mne.compute_covariance(
    epochs, 
    keep_sample_mean = True,
    tmin=.01, 
    tmax=.15, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )


# data cov from part of epochs from .01 to .20 s

data_cov_epochs_20 = mne.compute_covariance(
    epochs, 
    keep_sample_mean = False,
    tmin=0.01, 
    tmax=.20, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

data_cov_epochs_kept_mean_20 = mne.compute_covariance(
    epochs, 
    keep_sample_mean = True,
    tmin=.01, 
    tmax=.20, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )


# data cov from part of epochs from .01 to .25 s

data_cov_epochs = mne.compute_covariance(
    epochs, 
    keep_sample_mean = False,
    tmin=0.01, 
    tmax=.25, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

data_cov_epochs_kept_mean = mne.compute_covariance(
    epochs, 
    keep_sample_mean = True,
    tmin=.01, 
    tmax=.25, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )


# data cov using entire epochs

data_cov_entire_epochs = mne.compute_covariance(
    epochs, 
    keep_sample_mean = False,
    tmin=None, 
    tmax=None, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )

data_cov_entire_epochs_kept_mean = mne.compute_covariance(
    epochs, 
    keep_sample_mean = True,
    tmin=None, 
    tmax=None, 
    method="empirical", 
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )


# %%

# Making scaled evoked
raw_scaled = raw.copy()

raw_scaled.rescale({"mag":100})
       
                     
proj = False
epochs_scaled = mne.Epochs(
    raw_scaled,
    events,
    event_id,
    tmin=-0.2,
    tmax=0.5,
    baseline=None,
    preload=True,
    proj=proj,
    reject=dict(grad=4000e-13, mag=4e-10, eog=150e-6)
)

evoked_scaled = epochs.average().crop(.00,.15)
evoked_scaled.pick(picks="meg")
evoked_scaled.drop_channels("MEG 2443")
evoked_scaled.drop_channels("MEG 2313")


# %%

# getting forward solution
outputfilefwd = 'my_forward_solution-fwd.fif'

fwd = mne.read_forward_solution(outputfilefwd)


# %%


# What data should we use?
data_path = sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"


# for calibration etc.
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)


# for using SSS with data from Elekta Neuromag® systems
fine_cal_file = os.path.join(sample_data_folder, "SSS", "sss_cal_mgh.dat")
crosstalk_file = os.path.join(sample_data_folder, "SSS", "ct_sparse_mgh.fif")


# find bad channels then input here using find_bad_channels_maxwell()
raw.info["bads"] = ["MEG 2443"]
raw.info["bads"] += ["MEG 2313"]

raw.pick(['meg'])


mag_picks = mne.pick_types(raw.info, meg='mag', exclude='bads')

#raw_sss = mne.preprocessing.maxwell_filter(
#    raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True
#)

# getting good channels and index of bads
good_channels_from_raw = [ch for ch in raw.ch_names if ch not in raw.info['bads']]
new_raw_ch_names = good_channels_from_raw


meg_channel_names = raw.ch_names

bad_index1 = meg_channel_names.index("MEG 2443")
bad_index2 = meg_channel_names.index("MEG 2313")


# %%

# get SSS basis and Covariance in SSS basis, multipole moments

S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(
    raw.info,
    regularize=None
    )

S_in = S[:, :n_use_in]

pS_in = pS[:n_use_in]

pS_in_T = pS_in.transpose()



# %%

# Lead fields rescaling gradiometers by 100

def mag_scaling_lead_field_update(fwd, mag_picks):
    # getting attributes of forward
    forward_copy = fwd.copy()
    sol = forward_copy['sol']
    data = sol['data']
    
    # updating lead field "mag" scale
    data[mag_picks] *= 100
    
    # dropping bad row
    #np.delete(data, bad_index, 1)
    
    forward_copy['sol']['data'] = data
    
    return forward_copy


forward = mag_scaling_lead_field_update(fwd, mag_picks)

# picking good channels from forward
forward.pick_channels(new_raw_ch_names)


# %%

# Computing data covariance from raw. MNE python automatically scales grad and mag to same unit
# Note to self: raw is not time centered, function below will subtract mean

data_cov = mne.compute_raw_covariance(
    raw,
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6)
    )


# %%

# Column normalization of S

ST_S = S.transpose() @ S
print(np.linalg.cond(ST_S,None))

# print(np.linalg.cond(S,None))
col_n_S = np.linalg.norm(S, axis=0)

n_S = S / col_n_S

ST_S_norm = n_S.transpose() @ n_S

print(np.linalg.cond(ST_S_norm,None))



# Removing column vectors to reduce condition number of S transpose @ S 
# according to Vrba, J., Taulu, S., Nenonen, J. et al. Signal Space Separation Beamformer. Brain Topogr 23, 128–133 (2010). https://doi.org/10.1007/s10548-009-0120-7

reduced_S = n_S.copy()

ST_S = reduced_S.transpose() @ reduced_S

new_n_use_in = n_use_in

while np.linalg.cond(ST_S, None) >= 10**5:
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
    new_n_use_in = n_use_in - 1
    print(new_n_use_in)
    print(min_index)
    reduced_S = np.delete(reduced_S, min_index, axis=1)
    ST_S = reduced_S.transpose() @ reduced_S
    print(f"Current condition number: {np.linalg.cond(ST_S, None):.2e}")


# calculating our own pseudo inverse from S
pS_from_S = np.linalg.pinv(reduced_S)


#finding column to cut in pS as 306 -> 304 to fit covariance
meg_channel_names = raw.ch_names

bad_index1 = meg_channel_names.index("MEG 2443")
bad_index2 = meg_channel_names.index("MEG 2313")


# cutting bad column from pS
pS_from_S_no_bads = np.delete(pS_from_S, (bad_index1, bad_index2), 1)

# getting final pS internal
pS_final = pS_from_S_no_bads[:new_n_use_in, :]

sss_channels = []
for n in range(new_n_use_in):
    ch_number = n + 1
    sss_channels.append("MEG " + f"{ch_number}".zfill(4))
    

    
# %%

# Updating raw info to sss basis

def change_of_raw_info_to_sss(raw, old_channel_names, sss_channels):
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

sss_info_raw = change_of_raw_info_to_sss(raw, new_raw_ch_names, sss_channels)    


# %%
# Creating raw with sss basis data and info


# # transforming raw data to sss basis

# sss_raw_info = sss_info_raw.info

# data, times = raw.get_data(return_times=True)

# # 168800 time points for reference


# # deleting bad channels from data and transforming to sss basis

# data_no_bads = np.delete(data, (bad_index1, bad_index2), 0)

# sss_data = pS_final @ data_no_bads
    

# # creating raw in sss basis
# sss_raw = mne.io.RawArray(sss_data, sss_raw_info, copy = 'info')

# %%

# # Creating Evoked in SSS basis using sss_raw
# # Note: not time centered
# # For some reason below evoked when applied using filter provides
# # weaker and seemingly incorrect stc, applying pS to evoked data resolves this


# sss_epochs = mne.Epochs(
#     sss_raw,
#     events,
#     event_id,
#     tmin=-.2,
#     tmax=.5,
#     baseline=None,
#     preload=True,
#     proj=False,
# )

# sss_evoked_from_sss_raw = sss_epochs.average().crop(0.00,0.15)

# %%


# Alternative Evoked in SSS basis for when applying filter
# Note: in application, scaling evoked seems to have no impact on stc after beamforming

# transforming evoked to SSS basis
evoked_data = evoked.get_data()
sss_data_evoked = pS_final @ evoked_data
sss_evoked = mne.EvokedArray(sss_data_evoked, sss_info_raw.info)

# evoked scaled to SSS basis
evoked_scaled_data = evoked_scaled.get_data()
sss_data_evoked_scaled = pS_final @ evoked_scaled_data
sss_evoked_scaled = mne.EvokedArray(sss_data_evoked_scaled, sss_info_raw.info)

# %%

#changing covariance
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
    new_ch_names = [f'MEG {i+1:04d}' for i in range(new_n_channels)]
    
    transformed_covariance = mne.Covariance(
        data = transformed_cov_data2,
        names = new_ch_names,
        bads = raw.info["bads"],
        nfree = covariance.nfree,
        projs = raw.info['projs']
        )

    return transformed_covariance
    


# for setting off diagonal noise covariance terms to zero
def change_of_cov_basis_into_sss_and_drop_off_diag_terms(
    covariance, pS_final, new_ch_names: list = None
    ):

    #preparing for transformation
    
    cov_data = covariance.data
    pS_T = pS_final.transpose()
    
    
    #getting new attributes
    transformed_cov_data1 = cov_data @ pS_T
    transformed_cov_data2 = pS_final @ transformed_cov_data1
    
    diag_terms = np.diag(transformed_cov_data2)
    transformed_cov_data3 = np.diag(diag_terms)
    
    new_n_channels = transformed_cov_data2.shape[0]
    new_ch_names = [f'MEG {i+1:04d}' for i in range(new_n_channels)]
    
    transformed_covariance = mne.Covariance(
        data = transformed_cov_data3,
        names = new_ch_names,
        bads = raw.info["bads"],
        nfree = covariance.nfree,
        projs = raw.info['projs']
        )

    return transformed_covariance

# covariances calculated using different periods and with/without mean subtracted


# data cov from raw

sss_data_cov = change_of_cov_basis_into_sss(data_cov, pS_final)

# noise cov from baseline

sss_noise_cov_epochs = change_of_cov_basis_into_sss(noise_cov_epochs, pS_final)

sss_noise_cov_epochs_diag = change_of_cov_basis_into_sss_and_drop_off_diag_terms(noise_cov_epochs, pS_final)

sss_noise_cov_epochs_kept_mean = change_of_cov_basis_into_sss(noise_cov_epochs_kept_mean, pS_final)


# data cov .01 to .15


sss_data_cov_epochs_15 = change_of_cov_basis_into_sss(data_cov_epochs_15, pS_final)

sss_data_cov_epochs_kept_mean_15 = change_of_cov_basis_into_sss(data_cov_epochs_kept_mean_15, pS_final)


# data cov .01 to .20

sss_data_cov_epochs_20 = change_of_cov_basis_into_sss(data_cov_epochs_20, pS_final)

sss_data_cov_epochs_kept_mean_20 = change_of_cov_basis_into_sss(data_cov_epochs_kept_mean_20, pS_final)


# data cov .01 to .25

sss_data_cov_epochs = change_of_cov_basis_into_sss(data_cov_epochs, pS_final)

sss_data_cov_epochs_kept_mean = change_of_cov_basis_into_sss(data_cov_epochs_kept_mean, pS_final)


# data cov over all epochs

sss_data_cov_entire_epochs = change_of_cov_basis_into_sss(data_cov_entire_epochs, pS_final)

sss_data_cov_entire_epochs_kept_mean = change_of_cov_basis_into_sss(data_cov_entire_epochs_kept_mean, pS_final)

# %%

# Transforming lead fields and updating info


def change_of_lead_fields_basis(fwd, pS, sss_ch):
    # getting attributes of forward
    forward_copy = fwd.copy()
    sol = forward_copy['sol']
    data = sol['data']


    # transforming data
    lead_field_list = []
    for n in range(data.shape[1]):
        new_lead_field = pS @ data[0:, n]
        lead_field_list.append(new_lead_field)
    lead_field_tuple = tuple(lead_field_list)
    transformed_data = np.column_stack(lead_field_tuple)
        
    
   # creating attributes for new forward instance
    new_channels = []
    new_n_chan = new_n_use_in
    
    for n in range(new_n_use_in):
        ch_number = n + 1
        new_channels.append("MEG " + f"{ch_number}".zfill(4))
    
    # modifying forward with transformed lead fields and new attributes
    forward_copy['sol']['data'] = transformed_data
    forward_copy['sol']['row_names'] = new_channels
    forward_copy['nchan'] = new_n_chan
    
    # updating info
    # getting rid of extra channels
    mapping = {}
    for n in range(new_n_use_in):
        ch_number = n + 1
        mapping[forward_copy.ch_names[n]] = "MEG " + f"{ch_number}".zfill(4)
    mne.channels.rename_channels(forward_copy['info'], mapping)
  
    #Picking channels
    final_sss_forward = mne.pick_channels_forward(forward_copy, sss_channels)
    
    return final_sss_forward

# forward from sample for comparison
fwd_fname = meg_path / "sample_audvis-meg-vol-7-fwd.fif"
forward_sample = mne.read_forward_solution(fwd_fname)

# scaling lead fields
forward_sample_scaled = mag_scaling_lead_field_update(forward_sample, mag_picks)
# picking channels
forward_sample_scaled.pick_channels(new_raw_ch_names)
#transforming with S basis
sss_forward_sample = change_of_lead_fields_basis(forward_sample_scaled, pS_final, sss_channels)

# forward computed previously

sss_forward = change_of_lead_fields_basis(forward, pS_final, sss_channels)

#%%

# Filters 
# Note: rank of noise is 79
# Note: reg parameter at .0 produces very similar stc


# data cov over .01 to .15 s
filters = make_lcmv(
    sss_info_raw.info,
    sss_forward,
    sss_data_cov_epochs_kept_mean_15,
    reg=0.05,
    noise_cov=sss_noise_cov_epochs_kept_mean,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)

# data cov over .01 to .20 s
filters1 = make_lcmv(
    sss_info_raw.info,
    sss_forward,
    sss_data_cov_epochs_kept_mean_20,
    reg=0.05,
    noise_cov=sss_noise_cov_epochs_kept_mean,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)

# data cov over .01 to .15 s
filters2 = make_lcmv(
    sss_info_raw.info,
    sss_forward,
    sss_data_cov_epochs_kept_mean,
    reg=0.05,
    noise_cov=sss_noise_cov_epochs_kept_mean,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)


# data cov over entire epoch
filters3 = make_lcmv(
    sss_info_raw.info,
    sss_forward,
    sss_data_cov_entire_epochs_kept_mean,
    reg=0.05,
    noise_cov=sss_noise_cov_epochs_kept_mean,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)



src = fwd["src"]


# %%

# Using filter on Evoked in SSS basis

# using kept mean for all and data cov over just .01 to .15
stc = mne.beamformer.apply_lcmv(sss_evoked_scaled, filters)

# using kept mean for all and data cov over just .01 to .20
stc1 = mne.beamformer.apply_lcmv(sss_evoked_scaled, filters1)

# using kept mean for all and data cov over just .01 to .25
stc2 = mne.beamformer.apply_lcmv(sss_evoked_scaled, filters2)

# kept mean for all but data cov over all epochs
stc3 = mne.beamformer.apply_lcmv(sss_evoked_scaled, filters3)

# %%

# visualizing 


lims = [0.30, 0.45, 0.90]
kwargs = dict(
    src=src,
    subject="sample",
    subjects_dir=subjects_dir,
    initial_time=0.083,
    verbose=True,
)

stc.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)

stc1.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)

stc2.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)

stc3.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)


