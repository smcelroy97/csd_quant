import numpy as np
import os
import scipy
from mpi4py import MPI
from sklearn.decomposition import PCA

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rank = 0

data_dir = ('NKI_data/csd_erps/')
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
files_per_rank = np.array_split(all_files, size)

n_channels = 21
n_time = 1000

flat_erps = []

for file in files_per_rank[rank]:
    fname = file[:-12]
    data = np.load(file)
    flat_erp = data.flatten()
    flat_erps.append(flat_erp)

X = np.stack(flat_erps)

pca = PCA(n_components=5).fit(X)
pca_erps = pca.fit_transform(flat_erps)
template = pca.components_[0].reshape(n_channels, n_time)
