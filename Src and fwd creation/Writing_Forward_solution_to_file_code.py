#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:17:00 2025

@author: owenjohnson
"""

import numpy as np

import mne
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

print(__doc__)


# getting raw data from phantom

data_path = bst_phantom_elekta.data_path(verbose=True)

raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname, preload=True)
# raw.compute_psd().plot(average=False, amplitude=False, picks="data", exclude="bads")
raw.info



subjects_dir = data_path

# use "otaniemi" as geometry is the same though dipole positions are different
fetch_phantom("otaniemi", subjects_dir=subjects_dir)


subject = "phantom_otaniemi"

# transformation from head frame to mri frame
trans = mne.transforms.Transform("head", "mri", np.eye(4))



subjects_dir = data_path
fetch_phantom("otaniemi", subjects_dir=subjects_dir)
# %%
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.05), head_radius=0.03)

src = mne.setup_volume_source_space(
    subject=subject, pos=1, sphere=sphere, mindist=0.0, subjects_dir=subjects_dir
    )

# %%


# creating fwd
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.1,relative_radii=(0.999, 0.9999, 0.99999, 1.0), sigmas=(1.0, 1.0, 1.0, 1.0))

fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=sphere,
    meg=True,
    eeg=False,
    n_jobs=None,
    verbose=True,
)

# %%


output_file_name = "/Users/owenjohnson/MEG/spherical_head_model_rad_10cm_sub_3cm_loc_5cm_10dmm_single_shell-fwd.fif"
# output_file_name = "/Users/owenjohnson/MEG/brainstorm_elekta_phantom_dipoles_1mm-fwd.fif"


mne.write_forward_solution(output_file_name, fwd, overwrite=True, verbose=None)

