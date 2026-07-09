from pyprep.prep_pipeline import PrepPipeline
import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt
from utils import get_csd, getbandpass, get_trigger_times
import os
from mpi4py import MPI

data_dir = 'NKI_data/aligned_data_21/'
save_dir = f'{data_dir}/plots/'
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

for file in all_files:
    fn = data_dir + file
    file_id = file[:-16]
    erp_csd = np.load(data_dir + file)
    fs = 1000
    time_ms = np.arange(erp_csd.shape[1]) / fs * 1000
    channels = np.arange(erp_csd.shape[0])

    v = np.percentile(np.abs(erp_csd), 99)
    levels = np.linspace(-v, v, 41)

    plot_tmin = 0
    plot_tmax = 150

    mask = (time_ms >= plot_tmin) & (time_ms <= plot_tmax)

    print("ERP shape:", erp_csd.shape)
    print("Available time range:", time_ms[0], "to", time_ms[-1], "ms")
    print("Samples selected for plot:", mask.sum())

    fig, ax = plt.subplots(figsize=(6, 12))

    cf = ax.contourf(
        time_ms[mask],
        channels,
        erp_csd[:, mask],
        levels=levels,
        cmap="RdBu",
        extend="both"
    )

    ax.set_xlim(plot_tmin, plot_tmax)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel")
    ax.set_title("Average CSD ERP")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)

    # fig.colorbar(cf, ax=ax, label="CSD")
    plt.tight_layout()
    os.makedirs(f"{save_dir}", exist_ok=True)
    plt.savefig(f"{save_dir}{file_id}_aligned21_erp.jpg")
    plt.close()
