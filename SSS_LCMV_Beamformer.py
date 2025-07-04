#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:17:50 2025

@author: owenjohnson
"""

import os

import matplotlib.pyplot as plt

import mne
from mne.beamformer import apply_lcmv, make_lcmv
from mne.datasets import fetch_fsaverage, sample
import numpy as np
import pandas as pd
import seaborn as sns



# %%


# What data should we use?
data_path = sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_filt-0-40_raw.fif"


# for calibration etc.
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)

raw = mne.io.read_raw_fif(raw_fname, verbose=False)
#raw.crop(tmax=60)

raw.pick(["meg"])  # pick channels of interest

# get rid of unnecessary projectors

raw.del_proj()


# for using SSS with data from Elekta Neuromag® systems
fine_cal_file = os.path.join(sample_data_folder, "SSS", "sss_cal_mgh.dat")
crosstalk_file = os.path.join(sample_data_folder, "SSS", "ct_sparse_mgh.fif")


# find bad channels then input here using find_bad_channels_maxwell()
raw.info["bads"] += ["MEG 2443"]



#raw_sss = mne.preprocessing.maxwell_filter(
#    raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True
#)

# getting good channels
good_channels_from_raw = [ch for ch in raw.ch_names if ch not in raw.info['bads']]
new_raw_ch_names = good_channels_from_raw


# %%

# What time ranges should be used?
data_cov = mne.compute_raw_covariance(raw)
noise_cov = mne.compute_raw_covariance(raw, tmin = 0, tmax = .9)

# %%
# Read forward model
fwd_fname = meg_path / "sample_audvis-meg-vol-7-fwd.fif"
forward = mne.read_forward_solution(fwd_fname)

forward.pick_channels(new_raw_ch_names,True)

# %%

# get SSS basis and Covariance in SSS basis, multipole moments

S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(raw.info)

S_in = S[:, :n_use_in]

pS_in = pS[:n_use_in]

pS_in_T = pS_in.transpose()


# normalize columns of S

# condition number of S
# get SVD, ratio of smallest and largest singular value 


# to mitigate: regularization
# column normalize
# can reverse normalization

# print(np.linalg.cond(S_in,None))
column_norms = np.linalg.norm(S_in, axis=0)

# Normalize each column by dividing by its norm
# Use broadcasting for efficient division
n_S = S_in / column_norms

# print(np.linalg.cond(n_S,None))


# calculating our own pseudo inverse from S_in
pS_from_S = np.linalg.pinv(n_S)


#finding column to cut in n_pS as 306 -> 305 to fit covariance

meg_channel_names = raw.ch_names

bad_index = meg_channel_names.index("MEG 2443")

print(bad_index)

# cutting bad column from pS

pS_from_S_no_bads = np.delete(pS_from_S, bad_index, 1)

print(pS_from_S_no_bads.shape)



# Getting pS from S

column_norms_pS_from_S = np.linalg.norm(pS_from_S_no_bads, axis=0, keepdims=True)

n_pS_from_S = pS_from_S_no_bads / column_norms_pS_from_S



print(np.linalg.cond(pS_from_S,None))
print(np.linalg.cond(pS_from_S_no_bads,None))
print(np.linalg.cond(n_pS_from_S,None))



# normalizing pS from compute maxwell basis (large condition number is issue)
column_norms_pS = np.linalg.norm(pS_in, axis=0, keepdims=True)

n_pS = pS_in / column_norms_pS

print(np.linalg.cond(pS_in,None))
print(np.linalg.cond(n_pS,None))




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
    return new_raw

sss_raw = change_of_raw_info_to_sss(raw, new_raw_ch_names, sss_channels)    



# pass through maxwellfilter to then use info for lcmv 

# %%

#changing covariance
def change_of_cov_basis_into_sss(
    covariance, pS_in, new_ch_names: list = None
    ):

    #preparing for transformation
    
    cov_data = covariance.data
    pS_T = pS_in.transpose()
    
    
    #getting new attributes
    transformed_cov_data1 = cov_data @ pS_T
    transformed_cov_data2 = pS_in @ transformed_cov_data1
    
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
    


sss_data_cov = change_of_cov_basis_into_sss(data_cov, pS_from_S_no_bads)
sss_noise_cov = change_of_cov_basis_into_sss(noise_cov, pS_from_S_no_bads)


# regularize data cov

reg_sss_data_cov = mne.cov.regularize(sss_data_cov, sss_raw.info)


# condition number
print(np.linalg.cond(sss_data_cov["data"],None))
print(np.linalg.cond(reg_sss_data_cov["data"],None))
print(np.linalg.cond(sss_noise_cov["data"],None))


# computing rank
    #mne.compute_rank(sss_noise_cov, info= sss_raw.info)
    #mne.compute_rank(sss_data_cov, info= sss_raw.info)


# plotting cov
sss_data_cov.plot(sss_raw.info, proj=False)
reg_sss_data_cov.plot(sss_raw.info, proj=False)
sss_noise_cov.plot(sss_raw.info, proj=False)


# %%

#changing lead fields

# nchan needs to change. 
# data of solutions needs to change. 
# does info need to change?

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
    # getting rid of extre channels
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

sss_forward = change_of_lead_fields_basis(forward, n_pS_from_S, sss_channels)



# %%
# making lcmv. (what should reg be?)


print(sss_raw.info['bads'])
print(sss_noise_cov['bads'])

#%%
# modifications before filter
# sss_raw.info.normalize_proj()




# making filter and getting source space

filters = make_lcmv(
    sss_raw.info,
    sss_forward,
    reg_sss_data_cov,
    reg=0.1,
    noise_cov=sss_noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank='full',
)

src = forward["src"]


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


del forward

# %%

#using filter

stc = mne.beamformer.apply_lcmv_cov(reg_sss_data_cov, filters)


#stc_vector = apply_lcmv_cov(sss_data_cov, filters)

del filters
# del filters_vec


# %%

# visualizing 

lims = [0.3, 0.45, 0.6]
kwargs = dict(
    src=src,
    subject="sample",
    subjects_dir=subjects_dir,
    initial_time=0.087,
    verbose=True,
)

# %%

#mri 2d slices

stc.plot(mode="stat_map", clim=dict(kind="value", pos_lims=lims), **kwargs)
















