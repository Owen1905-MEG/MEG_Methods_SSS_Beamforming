#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:34:52 2026

@author: owenjohnson
"""

import numpy as np
import pandas as pd
import mne
import sys
sys.path.append('/Users/owenjohnson/MEG')
import Spatial_resolution_or_SSS_functions as sr_SSS

# --- CONFIGURATION ---
filename = '/Users/owenjohnson/MEG/synthetic_data_w_gain_error001_7cm_SNR_1_150Hz_meg.fif'
L_in_values = range(7, 15)
L_ext = 3
csv_output = "sss_results_metadata.csv"
matrix_output = "sss_pS_matrices.npz"

# --- 1. PREP DATA (Outside loop) ---
raw = mne.io.read_raw_fif(filename, preload=True)
raw_scaled = raw.copy().rescale({"mag": 100})
raw_scaled.del_proj()

# Setup events (simplified for brevity)
fs, duration, n_events, t_pre_trigger = 150, 100.0, 156, 0.2
trigger_indices = int(t_pre_trigger * fs) + np.arange(n_events) * int((duration * fs) / n_events)
events = np.zeros((n_events, 3), dtype=int)
events[:, 0], events[:, 2] = trigger_indices, 1

# --- 2. LOOP AND COLLECT ---
results_metadata = []
matrix_dict = {}

for l_val in L_in_values:
    print(f"Computing for L_in = {l_val}...")
    
    # Run your function
    pS_final, new_n_use_in, reduced_S_in, sss_channels = sr_SSS.get_pS_and_SSS_basis(
        l_val, L_ext, raw_scaled.info
    )
    
    # Store metadata for the CSV
    results_metadata.append({
        "L_in": l_val,
        "L_ext": L_ext,
        "n_use_in": new_n_use_in,
        "matrix_key": f"pS_Lin_{l_val}"  # This links CSV to the NPZ file
    })
    
    # Store the actual matrix in our dictionary
    matrix_dict[f"pS_Lin_{l_val}"] = pS_final

# --- 3. SAVE EVERYTHING ---
# Save the CSV
pd.DataFrame(results_metadata).to_csv(csv_output, index=False)

# Save all matrices into one compressed file
np.savez_compressed(matrix_output, **matrix_dict)

print(f"Saved metadata to {csv_output} and matrices to {matrix_output}")


# %%


import numpy as np
import pandas as pd

# 1. Load the metadata
df = pd.read_csv("sss_results_metadata.csv")

# 2. Load the matrices
matrices = np.load("sss_pS_matrices.npz")

# 3. Pull up information for L_in = 9
row = df[df['L_in'] == 8]
matrix_key = row['matrix_key'].values[0]
my_pS_matrix = matrices[matrix_key]

print(f"Loaded matrix for L_in 9 with shape: {my_pS_matrix.shape}")



















