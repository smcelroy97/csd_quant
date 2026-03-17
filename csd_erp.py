'''
Eventual use is to load a directory of LFP or CSD files to
then be downsampled, flattened, and run PCA to create an
"ideal" case CSD patten for an auditory ERP as seen in
Rimehaug et al. (2023) https://doi.org/10.7554/eLife.87169
'''
from pyprep.prep_pipeline import PrepPipeline
import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt
from utils import get_csd, getbandpass, get_trigger_times
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_dir = 'NKI_data/'
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mat")])
files_per_rank = np.array_split(all_files, size)
for file in files_per_rank[rank]:
    if '.mat' in file:
        fn = data_dir + file
        file_id = file[:-7]
        samprds = 10000

        fp = h5py.File(fn, 'r')

        sampr = float(fp['craw']['adrate'][0][0])   # sampling rate
        orig_sampr = sampr
        dat = fp['craw']['cnt']  # cnt record stores the electrophys data
        print('fn:', fn, 'sampr:', sampr, 'samprds:', samprds)

        dt = 1.0 / sampr  # time-step in seconds
        npdat = np.zeros(dat.shape)
        tmax = (len(npdat) - 1.0) * dt  # note: len(npdat) is first dimension only
        dat.read_direct(npdat)  # read it into memory; note that this LFP data usually stored in microVolt
        npdat *= 0.001  # convert microVolt to milliVolt here

        print("raw npdat shape:", npdat.shape)

        bpdat = getbandpass(npdat, sampr)
        print("bpdat shape after getbandpass:", bpdat.shape)

        # downsample continuous LFP to match their pipeline order
        if sampr != samprds:
            bpdat = scipy.signal.resample(bpdat, int(bpdat.shape[1] * samprds / sampr), axis=1)
            sampr = float(samprds)

        print("bpdat shape after resample:", bpdat.shape, "sampr:", sampr)

        # compute continuous CSD before epoching, matching the MATLAB order
        csd_dat = get_csd(bpdat, sampr)
        print("csd_dat shape:", csd_dat.shape)
        fp.close()


        trigs = get_trigger_times(fn)

        trigs = np.asarray(trigs, dtype=float)
        trigs_samp = np.rint(trigs * (sampr / orig_sampr)).astype(np.int64)

        fs = sampr
        tmin = 0.0  # 0 ms pre
        tmax = 0.100   # 100 ms post

        pre = int(round(abs(tmin) * fs))       # samples before trigger
        post = int(round(tmax * fs))            # samples after trigger
        n_times = pre + post

        good_trig_mask = (trigs_samp - pre >= 0) & (trigs_samp + post <= csd_dat.shape[1])
        trigs_samp = trigs_samp[good_trig_mask]

        # Build epochs: (n_epochs, n_channels, n_times)
        E = np.stack([csd_dat[:, (t - pre):(t + post)] for t in trigs_samp], axis=0)

        print("Epoch tensor:", E.shape)  # (n_epochs, n_channels, n_times)

        # # peak-to-peak per epoch per channel
        # ptp = E.max(axis=2) - E.min(axis=2)          # (n_epochs, n_channels)
        #
        # # reduce to a single score per epoch (max across channels is conservative)
        # ptp_epoch = ptp.max(axis=1)                  # (n_epochs,)
        #
        # # robust threshold: median + k * MAD
        # med = np.median(ptp_epoch)
        # mad = np.median(np.abs(ptp_epoch - med)) + 1e-12
        # k = 8.0  # typical: 6–12; lower = stricter
        # thr = med + k * 1.4826 * mad
        #
        # good_epochs = ptp_epoch < thr
        # E_good = E[good_epochs]
        #
        # print(f"Kept {E_good.shape[0]}/{E.shape[0]} epochs (ptp thr={thr:.4g})")

        erp_csd = E.mean(axis=0)
        os.makedirs(f'{data_dir}csd_erps', exist_ok=True)
        np.save(f'{data_dir}csd_erps/{file_id}_csd_erp', erp_csd)

        #  plotting
        erp_csd_plot = erp_csd

        time_ms = np.arange(erp_csd_plot.shape[1]) / sampr * 1000 + tmin * 1000
        channels = np.arange(erp_csd_plot.shape[0])

        v = np.percentile(np.abs(erp_csd_plot), 99)
        levels = np.linspace(-v, v, 41)

        plot_tmin = -5
        plot_tmax = 50

        mask = (time_ms >= plot_tmin) & (time_ms <= plot_tmax)

        fig, ax = plt.subplots(figsize=(6, 12))

        cf = ax.contourf(
            time_ms[mask],
            channels,
            erp_csd_plot[:, mask],
            levels=levels,
            cmap='RdBu',
            extend='both'
        )

        ax.invert_yaxis()
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Channel")
        ax.set_title("Average CSD ERP")
        ax.axvline(0, color='k', linestyle='--', linewidth=1)

        plt.tight_layout()
        os.makedirs(f"{data_dir}csd_erps/plots", exist_ok=True)
        plt.savefig(f"{data_dir}csd_erps/plots/{file_id}_csd_erp.jpg")
