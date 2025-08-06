#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:17:50 2025

@author: owenjohnson
"""

import os


import mne
from mne.beamformer import apply_lcmv_cov, make_lcmv
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import subspace_angles





# %%


data_path = sample.data_path()

# the raw file containing the channel location + types
sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
# The paths to Freesurfer reconstructions
subjects_dir = data_path / "subjects"
subject = "sample"


raw = mne.io.read_raw_fif(raw_fname, preload = True)
raw.info["bads"] += ["MEG 2443"]

# deleting projectors and rescaling magnetometers 
raw.del_proj()

event_id = 1  # those are the trials with left-ear auditory stimuli
tmin, tmax = -0.2, 0.5
events = mne.find_events(raw)

# pick relevant channels
raw.pick(["meg", "eog"])  # pick channels of interest

# Create epochs
proj = False  # already applied
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None, 0),
    preload=True,
    proj=proj,
    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6),
)

evoked = epochs.average().crop(0.05, 0.15)

# Visualize averaged sensor space data
evoked.plot_joint()

# epochs.plot(['meg'])

noise_cov_epochs = mne.compute_covariance(
    epochs, tmin=tmin, tmax=0, method="empirical", keep_sample_mean = False
                                          )
noise_cov_epochs.plot(epochs.info)


# %%

# The transformation file obtained by coregistration
trans = sample_dir / "sample_audvis_raw-trans.fif"

info = mne.io.read_info(raw_fname)


# %%
surface = subjects_dir / subject / "bem" / "inner_skull.surf"
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)

# %%
conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject="sample", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)


# %%

fwd = mne.make_forward_solution(
    raw.info,
    trans=trans,
    src=vol_src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)


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

raw = mne.io.read_raw_fif(raw_fname, preload = True)

# get rid of unnecessary projectors

raw.del_proj()


# for using SSS with data from Elekta Neuromag® systems
fine_cal_file = os.path.join(sample_data_folder, "SSS", "sss_cal_mgh.dat")
crosstalk_file = os.path.join(sample_data_folder, "SSS", "ct_sparse_mgh.fif")


# find bad channels then input here using find_bad_channels_maxwell()
raw.info["bads"] += ["MEG 2443"]

raw.pick(['meg'])


mag_picks = mne.pick_types(raw.info, meg='mag', exclude='bads')

#raw_sss = mne.preprocessing.maxwell_filter(
#    raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True
#)

# getting good channels and index of bads
good_channels_from_raw = [ch for ch in raw.ch_names if ch not in raw.info['bads']]
new_raw_ch_names = good_channels_from_raw


meg_channel_names = raw.ch_names

bad_index = meg_channel_names.index("MEG 2443")


# %%



# get SSS basis and Covariance in SSS basis, multipole moments

S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(raw.info)

S_in = S[:, :n_use_in]

pS_in = pS[:n_use_in]

pS_in_T = pS_in.transpose()



# %%

# finding angle between S basis and data

# data, times = raw.get_data(return_times=True)
# print(times)

# print(data.shape)
# print(times.shape)


# angle_list = []

# for i in range(np.shape(data)[1]):
#     data_col = data[:, i, None]
#     angle = subspace_angles(S, data_col)
#     print(angle)
#     angle_list.append(angle)
    
# min_angle = min(angle_list)
# max_angle = max(angle_list)
# print(min_angle)
# print(max_angle)

# %%

# rescale gradiometers by 100

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



forward.pick_channels(new_raw_ch_names)


#raw.rescale({"mag":100})


# %%

# computing covariances. MNE python automatically scales grad and mag to same unit
data_cov = mne.compute_raw_covariance(raw)
noise_cov = mne.compute_raw_covariance(raw, tmin = 0, tmax = .9)

data_cov.plot(raw.info)

# %%




# column normalization of S

ST_S = S.transpose() @ S
print(np.linalg.cond(ST_S,None))

# print(np.linalg.cond(S,None))
col_n_S = np.linalg.norm(S, axis=0)

n_S = S / col_n_S

ST_S_norm = n_S.transpose() @ n_S

print(np.linalg.cond(ST_S_norm,None))



# removing column vectors to reduce condition number of S transpose @ S

reduced_S = n_S.copy()

ST_S = reduced_S.transpose() @ reduced_S

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
    reduced_S = np.delete(reduced_S, min_index, axis=1)
    ST_S = reduced_S.transpose() @ reduced_S
    print(f"Current condition number: {np.linalg.cond(ST_S, None):.2e}")



# doing above to S_in

# reduced_S_in = S_in.copy()
# ST_S = reduced_S_in.transpose() @ reduced_S_in

# while np.linalg.cond(ST_S, None) >= 10**5:
#     cond_compare_list =[]
#     for n in range(reduced_S_in.shape[1]):
#         reduced_S_in_copy = reduced_S_in.copy()
#         reduced_S_in_n = np.delete(reduced_S_in_copy, n, axis=1)
#         ST_S = reduced_S_in_n.transpose() @ reduced_S_in_n
#         cond_n = np.linalg.cond(ST_S, None)
#         cond_compare_list.append(cond_n)
    
#     if not cond_compare_list :
#         print("cond_compare_list is empty. Breaking loop.")
#         break 

#     min_value = min(cond_compare_list)
#     min_index = cond_compare_list.index(min_value)
#     reduced_S_in = np.delete(reduced_S_in, min_index, axis=1)
#     ST_S = reduced_S_in.transpose() @ reduced_S_in
#     print(f"Current condition number: {np.linalg.cond(ST_S, None):.2e}")


# calculating our own pseudo inverse from S_in
pS_from_S = np.linalg.pinv(n_S)


#finding column to cut in n_pS as 306 -> 305 to fit covariance

meg_channel_names = raw.ch_names

bad_index = meg_channel_names.index("MEG 2443")



# cutting bad column from pS

pS_from_S_no_bads = np.delete(pS_from_S, bad_index, 1)

# getting final pS internal

pS_final = pS_from_S_no_bads[:n_use_in, :]

sss_channels = []
for n in range(n_use_in):
    ch_number = n + 1
    sss_channels.append("MEG " + f"{ch_number}".zfill(4))
    

    
# %%

# updating raw

def change_of_raw_info_to_sss(raw, old_channel_names, sss_channels):
    new_raw = raw.copy()
    
    # modifying  raw attribute ch_names
    # dropping channels down to n_use_in
    raw_channels_to_drop = raw.info['ch_names'][n_use_in:]
    new_raw.drop_channels(raw_channels_to_drop)
    
    
    # rename channels
    ch_dict_mapping = {}
    for n in range(n_use_in):
        ch_dict_mapping[raw.info['ch_names'][n]] = sss_channels[n]
    new_raw.rename_channels(ch_dict_mapping)

    # Create a mapping to set all channels to 'mag'
    channel_type_mapping = {ch_name: 'mag' for ch_name in new_raw.info['ch_names']}

    # Update the channel types in the info object
    new_raw.info.set_channel_types(channel_type_mapping)
    
    return new_raw

sss_info_raw = change_of_raw_info_to_sss(raw, new_raw_ch_names, sss_channels)    


# getting attributes for transformation of raw data

sss_raw_info = sss_info_raw.info
data, times = raw.get_data(return_times=True)

data_no_bads = np.delete(data, bad_index, 0)


list_data = []
for t in range(166800):
    sss_data_col = pS_final @ data_no_bads[:, t]
    reshaped_sss_col = np.reshape(sss_data_col, (sss_data_col.shape[0],1))
    list_data.append(reshaped_sss_col)

sss_data_in_tuple = tuple(list_data)
sss_data = np.concatenate(sss_data_in_tuple, axis = 1)
print(sss_data.shape)
    
sss_raw = mne.io.RawArray(sss_data, sss_info_raw.info, copy = 'info')

# pass through maxwellfilter to then use info for lcmv 

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

# covariances calculated using varing methods to use/compare

sss_data_cov = change_of_cov_basis_into_sss(data_cov, pS_final)

sss_noise_cov_epochs = change_of_cov_basis_into_sss(noise_cov_epochs, pS_final)

sss_noise_cov_epochs_diag = change_of_cov_basis_into_sss_and_drop_off_diag_terms(noise_cov_epochs, pS_final)

sss_noise_cov  = change_of_cov_basis_into_sss(noise_cov, pS_final)

sss_noise_cov_diag = change_of_cov_basis_into_sss_and_drop_off_diag_terms(noise_cov, pS_final)




# plotting cov
sss_data_cov.plot(sss_raw.info, proj=False)
sss_noise_cov.plot(sss_raw.info, proj=False)
sss_noise_cov_epochs.plot(sss_raw.info, proj=False)

# sss_noise_cov_diag.plot(sss_raw.info, proj=False)

# %%

#changing lead fields


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
    new_n_chan = n_use_in
    
    for n in range(n_use_in):
        ch_number = n + 1
        new_channels.append("MEG " + f"{ch_number}".zfill(4))
    
    # modifying forward with transformed lead fields and new attributes
    forward_copy['sol']['data'] = transformed_data
    forward_copy['sol']['row_names'] = new_channels
    forward_copy['nchan'] = new_n_chan
    
    # updating info
    # getting rid of extra channels
    mapping = {}
    for n in range(n_use_in):
        ch_number = n + 1
        mapping[forward_copy.ch_names[n]] = "MEG " + f"{ch_number}".zfill(4)
    mne.channels.rename_channels(forward_copy['info'], mapping)
  
    #Picking channels
    final_sss_forward = mne.pick_channels_forward(forward_copy, sss_channels)
    
    
    # setting names
  
    # does forward.info need to be updated/changed? yes, but what?
    
    # updating forward.info
    #forward_copy['info']['ch_names'] = sss_channels
    #forward_copy['info']['nchan'] = new_n_chan
    
    
    return final_sss_forward



sss_forward = change_of_lead_fields_basis(forward, pS_final, sss_channels)



# %%
# making lcmv. (what should reg be?)



#%%
# modifications before filter




# making filter and getting source space

filters = make_lcmv(
    sss_raw.info,
    sss_forward,
    sss_data_cov,
    reg=0.05,
    noise_cov=sss_noise_cov_epochs_diag,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)

src = fwd["src"]


# vector lcmv
   # filters_vec = make_lcmv(
       # sss_raw.info,
      #  sss_forward,
       # sss_data_cov,
       # reg=0.05,
      #  noise_cov=sss_noise_cov,
       # pick_ori="vector",
       # weight_norm="unit-noise-gain-invariant",
       # rank='full',
   # )





del fwd

# %%

#using filter

sss_raw.crop(7.55, tmax = 7.660)

stc = mne.beamformer.apply_lcmv_raw(sss_raw, filters)

# epoched covariance filter



#stc_vector = apply_lcmv_cov(sss_data_cov, filters)

del filters




# %%

# visualizing 

lims = [0.3, 0.45, 1.6]
kwargs = dict(
    src=src,
    subject="sample",
    subjects_dir=subjects_dir,
    initial_time=7.587,
    verbose=True,
)



# %%

#mri 2d slices

stc.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)











