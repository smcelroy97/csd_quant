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

csd1 = np.load("erpdata/2-rb023024011@os.mat_20kHz_avgERP.pkl", allow_pickle=True)  # shape (depth,time) but can vary across animals
csd2 = np.load("erpdata/2-rb023024052@os.mat_20kHz_avgERP.pkl", allow_pickle=True)

csdA = csd1['avgCSD']
csdB = csd2['avgCSD']

d = wasserstein_csd(csd1, csd2, interpolate=True, sp_len=30, t_len=100)
print("WD =", d)

# Pairwise
# csds = [csd1, csd2, np.load("erpdata/19aug23_50dB_bbn_avgERP.pkl", allow_pickle=True)]
# D = pairwise_wd_csd(csds, interpolate=True, sp_len=30, t_len=100)
# print(D)
