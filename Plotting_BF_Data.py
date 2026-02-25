#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:03:51 2026

@author: owenjohnson
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


depth = 0.03  # Change this to 7 if testing with the 7cm file

# --- 1. SET THE CORRECT FILE PATH ---
file_path = '/Users/owenjohnson/MEG/single_source_data/sss_beamformer_1_sources_depth_3cm_single_shell_1.csv'
file_path_conv = '/Users/owenjohnson/MEG/single_source_data/conv_beamformer_0.03cmsingle_shell_SNR_1.csv'


if not os.path.exists(file_path):
    print(f"ERROR: File not found at: {file_path}")
else:
    # --- 2. LOAD AND PREPARE DATA ---
    df = pd.read_csv(file_path)
    
    # Sort by L_in
    df = df.sort_values(by='L_in')

    # Split the data - using .str.strip() to remove any accidental hidden spaces
    non_diag = df[df['Method'].str.strip() == 'Non-Diagonalized Noise Covariance']
    diag = df[df['Method'].str.strip() == 'Diagonalized Noise Covariance']
    
    print(f"Rows found for Non-Diagonalized: {len(non_diag)}")
    print(f"Rows found for Diagonalized: {len(diag)}")
    # ------------------------

    if len(non_diag) == 0:
        print("WARNING: Non-Diagonalized data is empty! Check the spelling of the method name in your CSV.")

    # Conv BF data
    df_conv = pd.read_csv(file_path_conv)
    peak_dim_conv = df_conv['Peak_Dimension_mm'].iloc[-1]
    source_error_conv = df_conv['Source_Error_mm'].iloc[-1]

    # --- 3. CREATE THE PLOTS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'SSS Beamformer Performance ({depth}cm depth)', fontsize=18, fontweight='bold')

    # --- PLOT 1: Peak Dimension ---
    # We add zorder=5 to ensure the lines are drawn on top of the grid
    ax1.plot(non_diag['L_in'], non_diag['Peak_Dimension_mm'], 'o-', 
             label='Non-Diagonalized', linewidth=2.5, markersize=8, zorder=5)
    ax1.plot(diag['L_in'], diag['Peak_Dimension_mm'], 's-', 
             label='Diagonalized', linewidth=2.5, markersize=8, zorder=4)
    
    ax1.axhline(y=peak_dim_conv, color='crimson', linestyle='--', linewidth=2, 
                label=f'Conventional ({peak_dim_conv:.2f} mm)', zorder=3)

    ax1.set_title('Spatial Resolution', fontsize=14)
    ax1.set_xlabel('Internal Order ($L_{in}$)')
    ax1.set_ylabel('Peak Dimension (mm)')
    ax1.set_xticks([8, 10, 12, 14])
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- PLOT 2: Localization Error ---
    ax2.plot(non_diag['L_in'], non_diag['Source_Error_mm'], 'o-', 
             label='Non-Diagonalized', linewidth=2.5, markersize=8, zorder=5)
    ax2.plot(diag['L_in'], diag['Source_Error_mm'], 's-', 
             label='Diagonalized', linewidth=2.5, markersize=8, zorder=4)
    
    ax2.axhline(y=source_error_conv, color='crimson', linestyle='--', linewidth=2, 
                label=f'Conventional ({source_error_conv:.2f} mm)', zorder=3)

    ax2.set_title('Localization Error', fontsize=14)
    ax2.set_xlabel('Internal Order ($L_{in}$)')
    ax2.set_ylabel('RMS Location Error (mm)')
    ax2.set_xticks([8, 10, 12, 14])
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()