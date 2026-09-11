'''
Performs a wasserstein distance calculation between the sinks and sources
of two CSD files, and sums them to provide a total wasserstein distance
Additionally there a pairwise calculation can be done to compare more
than two files at a time

CSD 1 - should be an ideal CSD from PC 1 of a large set of ERPs and animals
CSD 2 - In our case, simulated CSD from a model, but can be any CSD
'''

from pathlib import Path
import numpy as np

if __package__:
    from .utils import wasserstein_csd, pairwise_wd_csd
else:
    # Preserve direct script execution from the csd_quant directory.
    from utils import wasserstein_csd, pairwise_wd_csd

MODULE_DIR = Path(__file__).resolve().parent


# Keep prestimulus data for visualization; select the scoring window below.
with np.load(MODULE_DIR / "aligned_30_erp_prestim10.npz", allow_pickle=False) as target:
    csd_template = target['csd'].copy()
    template_times_ms = target['times_ms'].copy()


def poststimulus_csd(csd, times_ms):
    """Select [0, 200) ms and resample to 0..199 ms without time stretching.

    Prestimulus samples are excluded before interpolation and preprocessing.
    Require complete, uniformly sampled coverage of the requested window.
    """
    csd = np.asarray(csd, dtype=float)
    times = np.asarray(times_ms, dtype=float)
    if csd.ndim != 2 or times.ndim != 1 or csd.shape[1] != len(times):
        raise ValueError('Expected CSD (depth, time) and matching times_ms')
    if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(csd).all():
        raise ValueError('CSD and times must be finite, with at least two samples')
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0]):
        raise ValueError('times_ms must be increasing and uniformly sampled')
    keep = (times >= 0.0) & (times < 200.0)
    t = times[keep]
    if len(t) < 2 or t[0] > 1e-8 or t[-1] < 199.0 - 1e-8 or t[-1] + dt[0] < 200.0 - 1e-8:
        raise ValueError('CSD must cover the full [0, 200) ms scoring window')
    return np.stack([np.interp(np.arange(200), t, row[keep]) for row in csd])


def preprocess_csd(csd, threshold_frac=0.15):
    csd = csd - np.mean(csd)
    csd = csd / (np.max(np.abs(csd)) + 1e-12)

    thr = threshold_frac * np.max(np.abs(csd))
    csd[np.abs(csd) < thr] = 0.0
    return csd


def wd_from_template(sim_csd, sim_times_ms=None):
    """Return raw and legacy preprocessed WD, both restricted to [0, 200) ms.

    Omitted times are supported only for 200 samples at 1 kHz starting at 0.
    Pass explicit times for simulated data or any epoch containing prestimulus.
    """
    sim_csd = np.asarray(sim_csd, dtype=float)
    if sim_times_ms is None:
        if sim_csd.ndim != 2 or sim_csd.shape[1] != 200:
            raise ValueError('Pass sim_times_ms unless input is 200 samples at 1 kHz, starting at 0 ms')
        sim_times_ms = np.arange(200)
    temp = poststimulus_csd(csd_template, template_times_ms)
    simulated = poststimulus_csd(sim_csd, sim_times_ms)
    d = wasserstein_csd(temp, simulated, interpolate=True, sp_len=30, t_len=200)
    d_pp = wasserstein_csd(preprocess_csd(temp), preprocess_csd(simulated),
                           interpolate=True, sp_len=30, t_len=200)
    print(f'Poststimulus WD [0, 200) ms = {d}; legacy pp_wd = {d_pp}')
    return d, d_pp


if __name__ == '__main__':
    wd_from_template(csd_template, template_times_ms)
