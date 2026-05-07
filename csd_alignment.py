import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ch_info = pd.read_csv('NKI_data/ch_info.csv')
erp_dir = 'NKI_data/csd_erps/'

dat = {}
for idx, file in enumerate(ch_info['BBN files']):
    fid = file[1:-8]
    dat[fid] = {'supra_ch': ch_info['Supra Ch'].iloc[idx],
                'gran_ch': ch_info['Gran Ch'].iloc[idx],
                'infra_ch': ch_info['Infra Ch'].iloc[idx]}

    print(f"{file}"
          f"\nSupra Chan: {ch_info['Supra Ch'].iloc[idx]}"
          f"\nGran Chan: {ch_info['Gran Ch'].iloc[idx]}"
          f"\nInfra Chan: {ch_info['Infra Ch'].iloc[idx]}\n\n")

for file in os.listdir(erp_dir):
    if file == 'pc1_erp.npy':
        continue
    if file.endswith('.npy'):
        fn = file[:-12]
        erp = np.load(erp_dir + file)
        dat[fn]['erp'] = erp


def align_laminar(data, supra_ch, gran_ch, infra_ch, n_out=21):
    """
    data: (channels, time)
    supra_ch, gran_ch, infra_ch: anchor channel indices
    n_out: number of output (aligned) channels
    """

    n_ch, n_t = data.shape

    # Original depth axis (channel indices)
    old_depth = np.arange(n_ch)

    # Define target anchor positions (canonical space)
    target_supra = int(0.25 * n_out)
    target_gran = int(0.50 * n_out)
    target_infra = int(0.75 * n_out)

    # Define control points
    src = np.array([0, supra_ch, gran_ch, infra_ch, n_ch - 1])
    tgt = np.array([0, target_supra, target_gran, target_infra, n_out - 1])

    # Interpolate mapping function
    new_depth = np.interp(old_depth, src, tgt)

    # Create uniform target grid
    target_grid = np.arange(n_out)

    # Interpolate data onto target grid
    aligned = np.zeros((n_out, n_t))

    for t in range(n_t):
        aligned[:, t] = np.interp(target_grid, new_depth, data[:, t])

    return aligned


aligned_dir = 'NKI_data/aligned_data_21/'
os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(aligned_dir+'plots/', exist_ok=True)

for animal in dat:
    aligned = align_laminar(dat[animal]['erp'], dat[animal]['supra_ch'], dat[animal]['gran_ch'], dat[animal]['infra_ch'])
    np.save(f"{aligned_dir}{animal}_aligned_erp.npy", aligned)

    v = np.percentile(np.abs(aligned), 99)
    levels = np.linspace(-v, v, 41)
    fig, ax = plt.subplots(figsize=(6, 12))
    time_ms = np.arange(aligned.shape[1])
    channels = np.arange(aligned.shape[0])

    cf = ax.contourf(
        time_ms/10,
        channels,
        aligned,
        levels=levels,
        cmap='RdBu',
        extend='both'
    )

    ax.invert_yaxis()
    plt.xlim([0, 50])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel")
    ax.set_title("Average CSD ERP (Aligned)")
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(f"{aligned_dir}plots/{animal}_aligned_erp.jpg")
    plt.close()
