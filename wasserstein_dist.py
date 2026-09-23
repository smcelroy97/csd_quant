'''
Performs a wasserstein distance calculation between the sinks and sources
of two CSD files, and sums them to provide a total wasserstein distance
Additionally there a pairwise calculation can be done to compare more
than two files at a time

CSD 1 - QC-derived, anatomically aligned short-SOA mean
CSD 2 - In our case, simulated CSD from a model, but can be any CSD
'''

from pathlib import Path
import numpy as np

if __package__:
    from .utils import wasserstein_csd, align_model_csd
else:
    # Preserve direct script execution from the csd_quant directory.
    from utils import wasserstein_csd, align_model_csd

MODULE_DIR = Path(__file__).resolve().parent


# Keep prestimulus data for visualization; select the scoring window below.
with np.load(MODULE_DIR / "qc_workflow/csd_channel_interpretation/qc_results/run_20260923_csd_one/short/mean_short.npz", allow_pickle=False) as target:
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


def wd_from_template(sim_csd, sim_times_ms=None, *, sim_depths_um=None,
                     anchor_depths_um=(475.0, 1100.0, 1625.0)):
    """Return (total, sink, source) unthresholded WD on the 1-ms [0,200) grid.

    With sim_depths_um, align physical model CSD using its landmarks.
    Without depths, input must already use the experimental 30-row alignment.

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
    if sim_depths_um is not None:
        simulated = align_model_csd(simulated, sim_depths_um, anchor_depths_um)
    if temp.shape != (30, 200) or simulated.shape != (30, 200):
        raise ValueError("Scoring requires aligned 30-depth x 200 one-ms samples")
    d, d_sink, d_src = wasserstein_csd(temp, simulated)

    print(f'Poststimulus WD [0, 200) ms = {d}\n'
          f'WD of Sinks = {d_sink}\n'
          f'WD of Sources = {d_src}')
    return d, d_sink, d_src


if __name__ == '__main__':
    wd_from_template(csd_template, template_times_ms)
