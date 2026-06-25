import numpy as np
import os
import scipy
from mpi4py import MPI
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_dir = ('NKI_data/aligned_data/')
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

first = np.load(os.path.join(data_dir, all_files[0]))
n_channels, n_time = first.shape

flat_erps = []

for file in all_files:
    fname = file[:-12]
    data = np.load(data_dir + file)
    flat_erp = data.flatten()
    flat_erps.append(flat_erp)

X = np.stack(flat_erps)

pca = PCA(n_components=5)
pca_erps = pca.fit_transform(X)
template = pca.components_[0].reshape(n_channels, n_time)

# np.save(f'{data_dir}/pc1_erp_redo.npy', template)

fs = 1000  # Hz
time_ms = np.arange(template.shape[1]) / fs * 1000
channels = np.arange(template.shape[0])

v = np.percentile(np.abs(template), 99)
levels = np.linspace(-v, v, 41)

plot_tmin = 0
plot_tmax = 150

mask = (time_ms >= plot_tmin) & (time_ms <= plot_tmax)

fig, ax = plt.subplots(figsize=(6, 12))

cf = ax.contourf(
    time_ms[mask],
    channels,
    template[:, mask],
    levels=levels,
    cmap='RdBu',
    extend='both'
)

ax.set_xlim(plot_tmin, plot_tmax)
ax.invert_yaxis()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Channel")
ax.set_title("30 channel aligned CSD ERP")
ax.axvline(0, color='k', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(f"{data_dir}/plots/aligned30_erp_150ms.jpg")
