'''
Performs a wasserstein distance calculation between the sinks and sources
of two CSD files, and sums them to provide a total wasserstein distance
Additionally there a pairwise calculation can be done to compare more
than two files at a time

CSD 1 - should be an ideal CSD from PC 1 of a large set of ERPs and animals
CSD 2 - In our case, simulated CSD from a model, but can be any CSD
'''

import numpy as np
from utils import wasserstein_csd, pairwise_wd_csd


csd_template = np.load("../csd_quant/aligned_30_erp.npy", allow_pickle=True)  # shape (depth,time) but can vary across animals
# csd_template = csd_template[:, 0:500]


def preprocess_csd(csd, threshold_frac=0.15):
    csd = csd - np.mean(csd)
    csd = csd / (np.max(np.abs(csd)) + 1e-12)

    thr = threshold_frac * np.max(np.abs(csd))
    csd[np.abs(csd) < thr] = 0.0
    return csd


def wd_from_template(sim_csd):
    print('Calculating Wasserstein Distance...')
    temp_pp = preprocess_csd(csd_template)
    sim_csd_pp = preprocess_csd(sim_csd)
    d = wasserstein_csd(csd_template, sim_csd, interpolate=True, sp_len=30, t_len=100)
    d_pp = wasserstein_csd(temp_pp, sim_csd_pp, interpolate = True, sp_len = 30, t_len = 100)
    print(f'Wasserstein Distance from simulated CSD to NHP template = {d}')
    print(f'Wasserstein Distance from simulated CSD to NHP template = {d_pp} after PREPROCESSING')
    return(d)

if __name__ == '__main__':
    csd_template = np.load("../csd_quant/pc1_erp.npy", allow_pickle=True)
    csd2 = np.zeros(csd_template.shape)



    d = wasserstein_csd(csd_template, csd2, interpolate=False, sp_len=len, t_len=100)
    print("WD =", d)

    # Pairwise
    # csds = [csd1, csd2, np.load("erpdata/19aug23_50dB_bbn_avgERP.pkl", allow_pickle=True)]
    # D = pairwise_wd_csd(csds, interpolate=True, sp_len=30, t_len=100)
    # print(D)
