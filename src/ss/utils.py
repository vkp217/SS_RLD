# import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_ratio(t0=0, step=0.1, T=12.5):
    w = np.arange(1, 20 + step, step)
    tau = np.arange(300, 3001, 100) * 1e-3  # tau values in picoseconds
    ratio = np.zeros((len(w), len(tau)))
    
    for i in range(len(w)):
        cal1 = np.exp(-t0 / tau)
        cal2 = np.exp(-(t0 + w[i]) / tau)
        cal3 = 1 - np.exp(-T / tau)
        cal4 = cal1 - cal2
        ratio[i, :] = cal4 / cal3
    
    return w, tau, ratio

def ss3_read_hdf5_file(fname):
    """
    Reads Bottom G2 and INT gate images from a .hdf5 file. 
    (top and bottom are the upper and lower part of the sensors size 500x250 pixels each )
    
    Parameters:
    - fname (str): Full path to the .hdf5 file.
    
    Returns:
    - tpsfs1 (ndarray): 3D array of Bottom G2 Gate images.
    - inten1 (ndarray): 2D intensity image summed across G2 gates.
    - tpsfs2 (ndarray): 3D array of Bottom INT Gate images.
    - inten2 (ndarray): 2D intensity image summed across INT gates.
    """
    with h5py.File(fname, 'r') as f:
        level2 = list(f['/Gate Images'].keys())
        
        # Filter and sort G2 and INT gate datasets
        bottomG2_keys = sorted([key for key in level2 if key.startswith('Bottom G2 Gate')],
                               key=lambda x: int(x.split()[-1]))
        bottomINT_keys = sorted([key for key in level2 if key.startswith('Bottom INT Gate')],
                                key=lambda x: int(x.split()[-1]))

        # Read one dataset to get shape
        sample_data = f['/Gate Images'][bottomG2_keys[0]][()]
        x, y = sample_data.shape

        # Allocate memory
        tpsfs1 = np.zeros((x, y, len(bottomG2_keys)), dtype=np.float64)
        tpsfs2 = np.zeros((x, y, len(bottomINT_keys)), dtype=np.float64)

        # Read data into arrays
        for i, key in enumerate(bottomG2_keys):
            tpsfs1[:, :, i] = f['/Gate Images'][key][()].astype(np.float64)

        for j, key in enumerate(bottomINT_keys):
            tpsfs2[:, :, j] = f['/Gate Images'][key][()].astype(np.float64)

    # Intensity images by summing across gates
    inten1 = np.sum(tpsfs1, axis=2)
    inten2 = np.sum(tpsfs2, axis=2)

    return tpsfs1, inten1, tpsfs2, inten2

def generate_ratio_table(t0=0, step=0.1, T=12.5):
    """
    Generates a tau vs. ratio lookup table for lifetime estimation.

    Returns:
    - tau (ndarray): Array of decay constants in seconds.
    - ratio (ndarray): Ratio values for interpolation.
    """
    w = np.arange(1, 20 + step, step)
    tau = np.arange(300, 3001, 100) * 1e-3  # tau in seconds
    ratio = np.zeros((len(w), len(tau)))

    for i in range(len(w)):
        cal1 = np.exp(-t0 / tau)
        cal2 = np.exp(-(t0 + w[i]) / tau)
        cal3 = 1 - np.exp(-T / tau)
        cal4 = cal1 - cal2
        ratio[i, :] = cal4 / cal3

    return tau, ratio

def process_single_shot_lifetime(fname, gate=61, gate_width=3, step=0.1):
    """
    Processes SS3 HDF5 file for single-shot fluorescence lifetime estimation.

    Parameters:
    - fname (str): Path to the HDF5 file.
    - gate (int): Gate number to use (default: 61).
    - gate_width (float): Gate width in ns (default: 3).
    - step (float): Step size in ns for the ratio table (default: 0.1).

    Returns:
    - output (ndarray): Computed lifetime image.
    """
    # Read data
    tpsfs1, inten1, tpsfs2, inten2 = ss3_read_hdf5_file(fname)

    # Extract gate slice (MATLAB gate=61 → Python index=60)
    ratio_vec1_temp = tpsfs1[:, :, gate - 1]
    ratio_vec2_temp = tpsfs2[:, :, gate - 1]

    ratio_vec1 = ratio_vec1_temp.flatten()
    ratio_vec2 = ratio_vec2_temp.flatten()

    # Initial ratio image (clipped at 2)
    rratio = np.divide(ratio_vec1_temp, ratio_vec2_temp, out=np.zeros_like(ratio_vec1_temp), where=ratio_vec2_temp!=0)
    rratio = np.clip(rratio, None, 2)

    # Filter for valid ratios
    rratio_vec = np.full_like(ratio_vec1, np.nan)
    valid_mask = (ratio_vec1 > 3) & (ratio_vec2 > 5)
    rratio_vec[valid_mask] = ratio_vec1[valid_mask] / ratio_vec2[valid_mask]

    # Generate lookup table
    tau, ratio_table = generate_ratio_table(step=step)

    # Find appropriate row (k) for gate width
    k = int((gate_width - 1) / step)

    # Interpolate lifetime from ratio table
    interp_func = interp1d(ratio_table[k, :], tau, bounds_error=False, fill_value=np.nan)
    vq_vec = interp_func(rratio_vec)
    vq = vq_vec.reshape(rratio.shape) - (500e-3)

    # Replace NaNs with zeros
    output = np.nan_to_num(vq, nan=0.0)

    # Visualization
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(ratio_vec1_temp, cmap='jet')
    plt.title('G2 channel')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(ratio_vec2_temp, cmap='jet')
    plt.title('INT channel')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(output, cmap='jet')
    plt.title('Computed Lifetime')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return output