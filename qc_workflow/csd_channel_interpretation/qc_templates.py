"""Auditable LFP QC, anatomical alignment, and condition-specific CSD templates.

Edit the configuration block and run this file in your Python editor.
Raw data and historical templates are never modified.
Dependencies: numpy, scipy, h5py, matplotlib; optional POT for WD validation.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

VERSION = "1.0"

# ======================= EDITABLE CONFIGURATION =======================
# During development outputs stay in the workspace. Set these paths as needed.
DATA_REPO = Path("/Users/scoot/dev/csd_quant")
OUTPUT_ROOT = Path(__file__).resolve().parent / "qc_results"
RUN_NAME = "run_20260923_csd_one"
CACHE_DIRECTORY = Path("/Users/scoot/dev/csd_quant/qc_workflow/qc_results/continuous_cache")
OVERRIDES_FILE = None  # Optional Path to a JSON file; see QC_WORKFLOW.md.
RECORDING_LIMIT = None  # Set to 2 for a quick ingestion/QC smoke test.


@dataclass
class Settings:
    fs: int = 1000
    low_hz: float = 0.5
    high_hz: float = 100.0
    filter_order: int = 4
    spacing_um: float = 100.0
    pre_ms: int = 100
    post_ms: int = 200
    qc_pre_ms: int = 200
    qc_post_ms: int = 300
    edge_ms: int = 1000
    baseline: bool = True
    anchor_convention: str = "raw-one"
    channel_z: float = 6.0
    trial_z: float = 6.0
    max_bad_channels: int = 2
    max_bad_run: int = 2
    min_trials: int = 30
    min_retained_fraction: float = 0.5
    bootstrap: int = 100
    jitter: int = 100
    seed: int = 314159


# CSV landmarks are interpreted as one-based CSD rows for this comparison.
# All signal processing and QC settings are preserved from the raw-one run.
SETTINGS = Settings(anchor_convention="csd-one", low_hz=0.5, high_hz=100.0,
                    baseline=True, bootstrap=100, jitter=100)
# ======================================================================


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corr(a, b):
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    x, y = x-x.mean(), y-y.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den > 1e-15 else 0.0


def robust_log_z(values, axis=0):
    """Positive metric outliers; log-space MAD floor avoids tiny-MAD explosions."""
    x = np.log(np.maximum(values, 1e-12))
    med = np.median(x, axis=axis, keepdims=True)
    scale = np.maximum(1.4826*np.median(abs(x-med), axis=axis, keepdims=True), 0.1)
    return (x-med)/scale


def anchors_to_csd(anchors, convention):
    # A centered second difference's row 0 corresponds to raw contact index 1.
    offsets = {"raw-one": -2, "raw-zero": -1, "csd-one": -1, "csd-zero": 0}
    return np.asarray(anchors, dtype=float) + offsets[convention]


def align_laminar(data, anchors, n_out=30):
    """Piecewise-linear depth warp; last two axes are depth and time.

    Anchors are CSD-row coordinates, NOT anatomical layer boundaries.
    CSD is computed in physical space before this feature-space warp.
    """
    a = np.asarray(data, dtype=float)
    n = a.shape[-2]
    src = np.array([0, *anchors, n-1], dtype=float)
    if not np.all(np.diff(src) > 0):
        raise ValueError(f"Anchors must be strictly interior and ordered: {src}")
    target = np.array([0, int(.25*n_out), int(.5*n_out), int(.75*n_out), n_out-1])
    positions = np.interp(np.arange(n_out), target, src)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo+1, n-1)
    frac = positions-lo
    return a[..., lo, :]*(1-frac[:, None])+a[..., hi, :]*frac[:, None]


def csd_from_lfp(lfp, spacing_um=100):
    """LFP last axes are contacts,time; result is mV/mm^2 (no conductivity factor)."""
    return -np.diff(lfp, n=2, axis=-2)/(spacing_um/1000)**2


def longest_true_run(mask):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    return int(np.max(np.flatnonzero(np.diff(padded) == -1)-
                      np.flatnonzero(np.diff(padded) == 1), initial=0))


def repair_channels(epochs, bad, max_bad=2, max_run=2):
    """Only interpolate interior LFP contacts. Never extrapolate an edge contact."""
    bad = np.asarray(bad, bool)
    if bad[0] or bad[-1]:
        raise ValueError("Bad edge contact: cannot safely interpolate")
    if bad.sum() > max_bad or longest_true_run(bad) > max_run:
        raise ValueError("Too many or adjacent bad contacts for interpolation")
    out = epochs.copy()
    good = np.flatnonzero(~bad)
    for ch in np.flatnonzero(bad):
        left, right = good[good < ch][-1], good[good > ch][0]
        w = (ch-left)/(right-left)
        out[:, ch] = (1-w)*epochs[:, left]+w*epochs[:, right]
    return out


def load_continuous(path, cache_dir, fs_out=1000):
    """Read HDF5 in bounded blocks, anti-alias, then downsample on a fixed clock.

    Overlap exceeds the polyphase FIR support; block lengths preserve phase.
    The cache includes the raw file stat and ingest version; no QC decisions cached.
    """
    stat = path.stat()
    signature = {"path": str(path.resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "fs_out": fs_out, "ingest": VERSION}
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir/(path.stem+"_continuous.npz")
    if dest.exists():
        with np.load(dest, allow_pickle=False) as z:
            if json.loads(str(z["signature"])) == signature:
                return {k: z[k].copy() for k in z.files if k != "signature"}
    with h5py.File(path, "r") as f:
        fs = float(f["craw/adrate"][0, 0])
        source = f["craw/cnt"]
        n, n_ch = source.shape
        if n_ch < 5 or n < n_ch:
            raise ValueError(f"Expected samples x contacts, found {source.shape}")
        key = next((k for k in ("trig/anatrig", "anatrig") if k in f), None)
        if key is None:
            raise ValueError("No trigger array")
        refs = f[key][()]
        trigs = np.asarray(f[refs[0, 0]][()]).ravel().astype(float)
        if not np.isfinite(trigs).all() or np.any(np.diff(trigs) <= 0):
            raise ValueError("Trigger samples must be finite and strictly increasing")
        ratio = Fraction(fs_out/fs).limit_denominator(100000)
        up, down = ratio.numerator, ratio.denominator
        if not np.isclose(fs*up/down, fs_out, atol=1e-8):
            raise ValueError("Sampling-rate ratio is not representable")
        block = max(down, int(fs*10)//down*down)
        overlap = 32*max(up, down)
        overlap = int(np.ceil(overlap/down))*down
        total = int(np.ceil(n*up/down))
        result = np.empty((total, n_ch), dtype=np.float32)
        invalid = np.zeros((total, n_ch), dtype=bool)
        equal_count = np.zeros(n_ch, dtype=np.int64)
        invalid_count = np.zeros(n_ch, dtype=np.int64)
        digest = hashlib.sha256()
        for start in range(0, n, block):
            end = min(start+block, n)
            left, right = max(0, start-overlap), min(n, end+overlap)
            x = np.asarray(source[left:right], dtype=float)
            core = x[start-left:end-left]
            digest.update(core.tobytes())
            equal_count += np.sum(np.diff(core, axis=0) == 0, axis=0)
            invalid_count += np.sum(~np.isfinite(core), axis=0)
            bad = ~np.isfinite(x)
            if bad.any():
                x[bad] = 0  # these neighborhoods are marked invalid below
            y = resample_poly(x*.001, up, down, axis=0)
            a, b = int(start*up/down), int(np.ceil(end*up/down))
            local = int((start-left)*up/down)
            result[a:b] = y[local:local+b-a]
            if bad.any():
                inds, chans = np.nonzero(bad)
                points = np.clip(np.rint((inds+left)*up/down).astype(int), 0, total-1)
                invalid[points, chans] = True
        output = dict(lfp=result, invalid=invalid, raw_fs=np.array(fs),
                      triggers_ms=trigs/fs*1000, raw_flat_fraction=equal_count/max(n-1, 1),
                      raw_invalid_fraction=invalid_count/n, signal_sha256=np.array(digest.hexdigest()))
    np.savez_compressed(dest, signature=np.array(json.dumps(signature)), **output)
    return output


def channel_qc(unfiltered, filtered, times, raw_flat_fraction, raw_invalid_fraction, cfg):
    """Conservative persistent-noise screening; high evoked amplitude alone is not bad."""
    base = times < 0
    # First differences remove DC/slow offsets from the fast-residual metric.
    hf = np.diff(unfiltered-filtered, axis=-1)
    rms = lambda x: np.sqrt(np.mean(x*x, axis=-1))
    base_centered = filtered[..., base]-np.median(filtered[..., base], axis=-1, keepdims=True)
    noise = np.median(rms(hf[..., base[1:]]), axis=0)
    amp = np.median(rms(base_centered), axis=0)
    residual = np.zeros_like(filtered[..., base])
    for c in range(filtered.shape[1]):
        neighbors = [j for j in (c-1, c+1) if 0 <= j < filtered.shape[1]]
        residual[:, c] = base_centered[:, c]-np.mean(base_centered[:, neighbors], axis=1)
    residual_epoch = rms(residual)
    residual_med = np.median(residual_epoch, axis=0)
    z_noise, z_amp, z_res = (robust_log_z(v) for v in (noise, amp, residual_med))
    # Narrow 60-Hz/harmonic contamination is additional corroborating evidence,
    # not a reason to discard all channels with physiological rhythmic power.
    baseline_raw = unfiltered[..., base]
    baseline_raw = baseline_raw-baseline_raw.mean(-1, keepdims=True)
    spectrum = abs(np.fft.rfft(baseline_raw*np.hanning(baseline_raw.shape[-1]), axis=-1))**2
    freq = np.fft.rfftfreq(baseline_raw.shape[-1], 1/cfg.fs)
    broad = (freq >= 20) & (freq <= 400)
    line_bins = np.logical_or.reduce([abs(freq-f) <= 5 for f in (60, 120, 180)])
    line_fraction = np.median(spectrum[..., line_bins].sum(-1)/
                             np.maximum(spectrum[..., broad].sum(-1), 1e-20), axis=0)
    z_line = robust_log_z(line_fraction)
    persistent = np.mean(robust_log_z(residual_epoch, axis=1) > cfg.channel_z, axis=0)
    flat = (amp < max(np.median(amp)*.001, 1e-10)) | (raw_flat_fraction > .99)
    invalid = raw_invalid_fraction > .001
    noisy = (z_noise > cfg.channel_z) & (z_res > cfg.channel_z) & (persistent > .25)
    line_bad = (line_fraction > .5) & (z_line > cfg.channel_z) & (z_res > cfg.channel_z) & (persistent > .25)
    bad = flat | invalid | noisy | line_bad
    rows = []
    for c in range(len(bad)):
        reasons = [name for name, flag in (("flat", flat[c]), ("nonfinite", invalid[c]),
                   ("persistent_noise_and_neighbor_residual", noisy[c]),
                   ("line_noise_and_neighbor_residual", line_bad[c])) if flag]
        rows.append(dict(contact_zero=c, contact_one=c+1, bad=bool(bad[c]),
                         reasons=";".join(reasons), baseline_rms_mV=float(amp[c]),
                         fast_residual_step_rms_mV=float(noise[c]), noise_z=float(z_noise[c]),
                         amplitude_z=float(z_amp[c]), residual_z=float(z_res[c]),
                         line_power_fraction=float(line_fraction[c]), line_fraction_z=float(z_line[c]),
                         residual_outlier_fraction=float(persistent[c]),
                         raw_flat_fraction=float(raw_flat_fraction[c]),
                         raw_invalid_fraction=float(raw_invalid_fraction[c])))
    return bad, rows


def trial_qc(unfiltered, filtered, times, good_channels, invalid_trials, cfg):
    x, raw = filtered[:, good_channels], unfiltered[:, good_channels]
    base = x[..., times < 0]
    metrics = dict(baseline_rms=np.std(base, axis=-1),
                   peak_to_peak=np.ptp(x, axis=-1),
                   max_step=np.max(abs(np.diff(x, axis=-1)), axis=-1),
                   hf_rms=np.sqrt(np.mean(np.diff(raw-x, axis=-1)**2, axis=-1)))
    z = {k: robust_log_z(v, axis=0) for k, v in metrics.items()}
    # A large physiological response alone is insufficient for rejection.
    baseline_bad = np.any(z["baseline_rms"] > cfg.trial_z+2, axis=1)
    joint = (z["peak_to_peak"] > cfg.trial_z) & (
             (z["max_step"] > cfg.trial_z) | (z["hf_rms"] > cfg.trial_z))
    transient_bad = np.any(joint, axis=1)
    flat_bad = np.any(np.ptp(raw, axis=-1) < 1e-10, axis=1)
    reject = baseline_bad | transient_bad | flat_bad | invalid_trials
    rows = []
    for i in range(len(x)):
        reasons = [name for name, flag in (("baseline_outlier", baseline_bad[i]),
                   ("amplitude_and_transient_outlier", transient_bad[i]),
                   ("flat_epoch", flat_bad[i]), ("nonfinite_near_epoch", invalid_trials[i])) if flag]
        rows.append(dict(epoch_index=i, rejected=bool(reject[i]), reasons=";".join(reasons),
                         **{k+"_max_z": float(v[i].max()) for k, v in z.items()}))
    return reject, rows


def process_recording(path, row, cache, out, cfg, override):
    data = load_continuous(path, cache, cfg.fs)
    lfp = data["lfp"].astype(float)
    n = len(lfp)
    sos = butter(cfg.filter_order, [cfg.low_hz, cfg.high_hz], btype="bandpass", fs=cfg.fs, output="sos")
    filtered = sosfiltfilt(sos, lfp, axis=0)
    times = np.arange(-cfg.qc_pre_ms, cfg.qc_post_ms, 1000/cfg.fs)
    relative = np.rint(times*cfg.fs/1000).astype(int)
    t = np.rint(data["triggers_ms"]*cfg.fs/1000).astype(int)
    margin = int(cfg.edge_ms*cfg.fs/1000)
    complete = (t+relative[0] >= margin) & (t+relative[-1] < n-margin)
    ids = np.flatnonzero(complete)
    if not len(ids):
        raise ValueError("No complete epochs outside filter-edge margin")
    indices = t[ids, None]+relative[None, :]
    raw_ep = lfp[indices].transpose(0, 2, 1)
    ep = filtered[indices].transpose(0, 2, 1)
    bad, channels = channel_qc(raw_ep, ep, times, data["raw_flat_fraction"], data["raw_invalid_fraction"], cfg)
    for contact in override.get("bad_contacts_zero", []):
        if not 0 <= contact < len(bad):
            raise ValueError(f"Invalid override contact {contact}")
        bad[contact] = True
        channels[contact]["bad"] = True
        channels[contact]["reasons"] += ";manual"
    for contact in override.get("keep_contacts_zero", []):
        if not 0 <= contact < len(bad) or data["raw_invalid_fraction"][contact] > .001:
            raise ValueError("Cannot retain invalid/nonfinite contact")
        bad[contact] = False
        channels[contact]["bad"] = False
        channels[contact]["reasons"] += ";manual_keep"
    invalid_trials = np.array([np.any(data["invalid"][max(0, c+relative[0]-500):
                               min(n, c+relative[-1]+501), ~bad]) for c in t[ids]])
    if np.all(bad):
        raise ValueError("All channels failed QC")
    reject, trials = trial_qc(raw_ep, ep, times, ~bad, invalid_trials, cfg)
    manual_trials = set(override.get("bad_trials_zero", []))
    if any(i < 0 or i >= len(t) for i in manual_trials):
        raise ValueError("Manual trial index outside trigger array")
    for i, original_id in enumerate(ids):
        if int(original_id) in manual_trials:
            reject[i] = True
            trials[i]["reasons"] += ";manual"
            trials[i]["rejected"] = True
        trials[i].update(trigger_index_zero=int(original_id), onset_ms=float(data["triggers_ms"][original_id]))
    excluded = "manual_recording_exclusion" if override.get("exclude_recording", False) else ""
    try:
        repaired = repair_channels(ep, bad, cfg.max_bad_channels, cfg.max_bad_run)
    except ValueError as e:
        excluded = str(e)
        repaired = ep.copy()
    kept = ~reject
    if kept.sum() < cfg.min_trials or kept.mean() < cfg.min_retained_fraction:
        excluded = excluded or "Insufficient retained trials"
    anchors_raw = override.get("anchors", [float(row[k]) for k in ("Supra Ch", "Gran Ch", "Infra Ch")])
    anchors = anchors_to_csd(anchors_raw, cfg.anchor_convention)
    selected = (times >= -cfg.pre_ms) & (times < cfg.post_ms)
    base = (times >= -cfg.pre_ms) & (times < 0)
    def transform(x):
        if cfg.baseline:
            x = x-x[..., base].mean(axis=-1, keepdims=True)
        return csd_from_lfp(x[..., selected], cfg.spacing_um)
    before = transform(ep).mean(axis=0)
    physical = transform(repaired[kept])
    after = physical.mean(axis=0) if len(physical) else np.zeros_like(before)
    try:
        aligned_before = align_laminar(before, anchors)
        aligned = align_laminar(after, anchors)
        aligned_epochs = align_laminar(physical, anchors)
    except ValueError as e:
        excluded = excluded or str(e)
        aligned = aligned_before = np.zeros((30, selected.sum()))
        aligned_epochs = np.empty((0, 30, selected.sum()))
    # Track which physical CSD stencils include reconstructed contacts.
    stencil = np.convolve(bad.astype(int), np.ones(3, dtype=int), mode="valid") > 0
    rng = np.random.default_rng(cfg.seed)
    if len(aligned_epochs) >= 4:
        perm = rng.permutation(len(aligned_epochs))
        halves = np.stack([aligned_epochs[perm[::2]].mean(0), aligned_epochs[perm[1::2]].mean(0)])
        reliability = corr(halves[0], halves[1])
    else:
        halves = np.zeros((2, *aligned.shape)); reliability = 0.0
    gaps = np.diff(data["triggers_ms"])
    info = dict(file=path.name, soa_code=int(row["SOA Code"]),
                group="short" if int(row["SOA Code"]) == 1 else "long",
                n_triggers=len(t), incomplete_or_edge=int((~complete).sum()),
                n_eligible=len(ids), n_rejected=int(reject.sum()), n_kept=int(kept.sum()),
                bad_contacts_zero=np.flatnonzero(bad).tolist(),
                reconstructed_csd_rows=np.flatnonzero(stencil).tolist(),
                anchors_csv=list(map(float, anchors_raw)), anchors_csd=anchors.tolist(),
                excluded=excluded, median_soa_ms=float(np.median(gaps)),
                min_soa_ms=float(gaps.min()), max_soa_ms=float(gaps.max()),
                split_half_correlation=reliability, signal_sha256=str(data["signal_sha256"]),
                source_fs=float(data["raw_fs"]))
    dest = out/"recordings"/path.stem
    dest.mkdir(parents=True, exist_ok=True)
    save_json(dest/"qc.json", info)
    write_csv(dest/"channels.csv", channels)
    write_csv(dest/"trials.csv", trials)
    write_csv(dest/"excluded_triggers.csv", [dict(trigger_index_zero=int(i), onset_ms=float(data["triggers_ms"][i]),
                reason="incomplete_or_filter_edge") for i in np.flatnonzero(~complete)])
    np.savez_compressed(dest/"erp.npz", before=aligned_before, after=aligned,
                        physical_after=after, aligned_epochs=aligned_epochs, halves=halves,
                        times_ms=times[selected], anchors_csd=anchors, stencil_repaired=stencil,
                        retained_trigger_indices=ids[kept])
    # A small review sample, not a second copy of every trial.
    example_ids = np.flatnonzero(reject)[:5]
    np.savez_compressed(dest/"qc_examples.npz", times_ms=times,
                        rejected_lfp=ep[example_ids], rejected_trigger_indices=ids[example_ids],
                        median_retained_lfp=np.median(repaired[kept], axis=0) if kept.any() else np.zeros(ep.shape[1:]))
    return info, aligned_before, aligned, after, halves


def pca_fit(X, normalize=False):
    """Fast sample-space eigensystem, exact centered PCA up to roundoff."""
    shape = X.shape[1:]
    x = X.reshape(len(X), -1).astype(float)
    norms = np.linalg.norm(x, axis=1)
    if normalize:
        if np.any(norms <= 1e-12):
            raise ValueError("Cannot shape-normalize a zero recording")
        x = x/norms[:, None]
    mean = x.mean(0)
    centered = x-mean
    val, u = np.linalg.eigh(centered@centered.T)
    order = np.argsort(val)[::-1]
    val, u = np.maximum(val[order], 0), u[:, order]
    valid = val > max(val[0]*1e-12, 1e-20)
    if not np.any(valid):
        raise ValueError("No between-recording variance for PCA")
    val, u = val[valid], u[:, valid]
    components = (u.T@centered)/np.sqrt(val[:, None])
    scores = centered@components.T
    for i in range(len(components)):
        # Physical mean supplies a consistent sign; PC1 remains a variation mode.
        if components[i]@mean < 0:
            components[i] *= -1; scores[:, i] *= -1
    return dict(components=components.reshape((-1, *shape)), mean=mean.reshape(shape),
                scores=scores, evr=val/val.sum(), norms=norms)


def stability(X, normalize, cfg):
    fit = pca_fit(X, normalize)
    pc = fit["components"][0]
    influence = fit["scores"][:, 0]**2/np.sum(fit["scores"][:, 0]**2)
    loo = []
    for i in range(len(X)):
        other = pca_fit(np.delete(X, i, axis=0), normalize)
        loo.append(abs(corr(pc, other["components"][0])))
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap):
        ids = rng.integers(0, len(X), len(X))
        if len(np.unique(ids)) < 2:
            continue
        other = pca_fit(X[ids], normalize)
        boot.append(abs(corr(pc, other["components"][0])))
    return dict(pc1_explained_variance=float(fit["evr"][0]),
                cumulative_variance=fit["evr"].cumsum().tolist(),
                pc1_mean_correlation=corr(pc, fit["mean"]),
                pc1_influence=influence.tolist(), loo_pc1_abs_correlation=loo,
                bootstrap_pc1_abs_correlation_quantiles=np.quantile(boot, [.05, .5, .95]).tolist() if boot else [],
                normalization="per-recording L2" if normalize else "none"), fit


def landmark_sensitivity(physical, anchors, reference, cfg, normalize=False):
    rng = np.random.default_rng(cfg.seed+1)
    correlations, mean_correlations = [], []
    ref_mean = np.mean([align_laminar(p, a) for p, a in zip(physical, anchors)], axis=0)
    for _ in range(cfg.jitter):
        arrays = []
        for p, a in zip(physical, anchors):
            for attempt in range(100):
                shifted = a+rng.integers(-1, 2, 3)
                if np.all(np.diff([0, *shifted, p.shape[0]-1]) > 0):
                    break
            else:
                raise ValueError("Cannot generate valid landmark perturbation")
            arrays.append(align_laminar(p, shifted))
        fit = pca_fit(np.stack(arrays), normalize)
        correlations.append(abs(corr(reference, fit["components"][0])))
        mean_correlations.append(corr(ref_mean, np.mean(arrays, axis=0)))
    return dict(jitter_contacts=1, draws=cfg.jitter,
                pc1_abs_correlation_quantiles=np.quantile(correlations, [.05, .5, .95]).tolist() if correlations else [],
                mean_correlation_quantiles=np.quantile(mean_correlations, [.05, .5, .95]).tolist() if correlations else [])


def build_groups(records, cfg, out):
    groups = {}
    times = np.arange(-cfg.pre_ms, cfg.post_ms)
    score = times >= 0
    for group in ("short", "long"):
        selected = [r for r in records if r[0]["group"] == group and not r[0]["excluded"]]
        if len(selected) < 4:
            raise ValueError(f"Need >=4 retained recordings for {group} validation")
        X = np.stack([r[2][:, score] for r in selected])
        before = np.stack([r[1][:, score] for r in selected])
        physical = [r[3][:, score] for r in selected]
        anchors = [np.asarray(r[0]["anchors_csd"]) for r in selected]
        dest = out/group
        dest.mkdir(exist_ok=True)
        stats, fit = stability(X, False, cfg)
        shape_stats, shape_fit = stability(X, True, cfg)
        before_stats, _ = stability(before, False, cfg)
        landmark = landmark_sensitivity(physical, anchors, fit["components"][0], cfg)
        shape_landmark = landmark_sensitivity(physical, anchors, shape_fit["components"][0], cfg, True)
        indexing = {}
        # Raw-zero is one CSD row deeper than provisional raw-one.
        for delta in (-1, 1, 2):
            try:
                shifted = np.stack([align_laminar(p, a+delta) for p, a in zip(physical, anchors)])
                alt = pca_fit(shifted)
                indexing[str(delta)] = dict(pc1_abs_correlation=abs(corr(fit["components"][0], alt["components"][0])),
                                           mean_correlation=corr(X.mean(0), shifted.mean(0)))
            except ValueError as e:
                indexing[str(delta)] = {"unavailable": str(e)}
        for name, arr, units in (("pc1", fit["components"][0], "arbitrary PC loading"),
                                 ("pc1_shape", shape_fit["components"][0], "arbitrary PC loading"),
                                 ("mean", X.mean(0), "mV/mm^2"),
                                 ("median", np.median(X, axis=0), "mV/mm^2")):
            metadata = dict(group=group, kind=name, units=units, settings=asdict(cfg),
                            filenames=[r[0]["file"] for r in selected],
                            median_soa_ms=float(np.median([r[0]["median_soa_ms"] for r in selected])),
                            alignment="piecewise-linear: supra=7,gran=15,infra=22; anchors are not boundaries")
            np.save(dest/f"{name}_{group}.npy", arr)
            np.savez(dest/f"{name}_{group}.npz", csd=arr, times_ms=times[score], depth_bins=np.arange(30),
                     metadata_json=np.array(json.dumps(metadata)))
        np.savez_compressed(dest/"pca_details.npz", erps=X, before_erps=before,
                            components=fit["components"], mean=fit["mean"], scores=fit["scores"],
                            explained_variance_ratio=fit["evr"],
                            shape_components=shape_fit["components"], shape_scores=shape_fit["scores"],
                            shape_explained_variance_ratio=shape_fit["evr"],
                            filenames=np.array([r[0]["file"] for r in selected]),
                            halves=np.stack([r[4][..., score] for r in selected]))
        summary = dict(n_recordings=len(selected), before_qc=before_stats, after_qc=stats,
                       shape_normalized=shape_stats, landmarks=landmark,
                       shape_landmarks=shape_landmark, uniform_index_shift=indexing,
                       median_soa_ms=float(np.median([r[0]["median_soa_ms"] for r in selected])))
        uniform = np.stack([align_laminar(p, np.array([7, 15, 22])*(p.shape[0]-1)/29) for p in physical])
        uniform_fit = pca_fit(uniform)
        summary["uniform_depth_comparison"] = dict(pc1_explained_variance=float(uniform_fit["evr"][0]),
             pc1_mean_correlation=corr(uniform_fit["components"][0], uniform_fit["mean"]),
             note="Whole-depth linear rescaling without anatomical anchors; diagnostic only.")
        save_json(dest/"diagnostics.json", summary)
        write_csv(dest/"recording_influence.csv", [dict(file=r[0]["file"],
                  pc1_score=float(fit["scores"][i, 0]), pc1_influence=stats["pc1_influence"][i],
                  loo_pc1_abs_correlation=stats["loo_pc1_abs_correlation"][i],
                  shape_pc1_influence=shape_stats["pc1_influence"][i],
                  erp_l2=float(fit["norms"][i]), split_half_correlation=r[0]["split_half_correlation"])
                  for i, r in enumerate(selected)])
        groups[group] = summary
    return groups


def plot_outputs(out, records, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for row, group in enumerate(("short", "long")):
        for col, name in enumerate(("mean", "median", "pc1", "pc1_shape")):
            a = np.load(out/group/f"{name}_{group}.npy")
            scale = float(np.max(abs(a))) or 1
            im = axes[row, col].imshow(a/scale, aspect="auto", origin="upper", cmap="RdBu",
                                      vmin=-1, vmax=1, extent=[0, 200, 29.5, -.5])
            for y in (7, 15, 22):
                axes[row, col].axhline(y, color="k", ls=":", lw=.6)
            axes[row, col].set(title=f"{group} SOA: {name}\nmax |value|={scale:.3g}", xlabel="Time (ms)")
        axes[row, 0].set_ylabel("Aligned depth bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Display normalized to each panel's peak; negative = sink")
    fig.savefig(out/"templates.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, group in zip(axes, ("short", "long")):
        for name, label in (("before_qc", "Before QC"), ("after_qc", "After QC"),
                             ("shape_normalized", "After QC, shape normalized")):
            v = groups[group][name]["cumulative_variance"]
            ax.plot(np.arange(1, len(v)+1), v, marker="o", label=label)
        ax.set(title=f"{group} SOA", xlabel="Number of components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.savefig(out/"variance.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    x = np.arange(len(records))
    axes[0].bar(x, [r[0]["n_rejected"]/r[0]["n_eligible"] for r in records])
    axes[0].set_ylabel("Rejected trial fraction")
    axes[1].bar(x, [len(r[0]["bad_contacts_zero"]) for r in records])
    axes[1].set_ylabel("Bad contacts")
    axes[2].bar(x, [r[0]["split_half_correlation"] for r in records])
    axes[2].set_ylabel("Split-half correlation")
    axes[2].set_xticks(x, [r[0]["file"].replace("@os.mat", "") for r in records], rotation=90, fontsize=8)
    fig.savefig(out/"qc_overview.png", dpi=160)
    plt.close(fig)
    for info, before, after, _, _ in records:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        limit = max(float(np.max(abs(before))), float(np.max(abs(after))), 1e-12)
        for ax, arr, label in zip(axes, (before, after), ("Before QC", "After QC")):
            im = ax.imshow(arr, origin="upper", aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit,
                           extent=[-100, 200, 29.5, -.5])
            ax.axvline(0, color="k", ls=":")
            ax.set(title=label, xlabel="Time (ms)", ylabel="Aligned depth bin")
        fig.suptitle(f"{info['file']}: {info['n_rejected']} trials rejected, {len(info['bad_contacts_zero'])} bad contacts")
        fig.colorbar(im, ax=axes, label="CSD (mV/mm²), common scale")
        fig.savefig(out/"recordings"/Path(info["file"]).stem/"qc_comparison.png", dpi=120)
        plt.close(fig)
        example_path = out/"recordings"/Path(info["file"]).stem/"qc_examples.npz"
        if example_path.exists():
            with np.load(example_path, allow_pickle=False) as z:
                examples, median, example_times, example_ids = (z[k] for k in
                    ("rejected_lfp", "median_retained_lfp", "times_ms", "rejected_trigger_indices"))
            if len(examples):
                fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.2*len(examples)),
                                         squeeze=False, constrained_layout=True)
                for ax, example, trial_id in zip(axes[:, 0], examples, example_ids):
                    pre = example_times < 0
                    ex = example-example[:, pre].mean(-1, keepdims=True)
                    med = median-median[:, pre].mean(-1, keepdims=True)
                    contact = np.argmax(np.max(abs(ex-med), axis=-1))
                    ax.plot(example_times, ex[contact], label=f"Rejected trigger {trial_id}, contact {contact} (zero-based)")
                    ax.plot(example_times, med[contact], color="k", lw=1, label="Median retained LFP")
                    ax.axvline(0, color="gray", ls=":")
                    ax.set(xlabel="Time (ms)", ylabel="LFP (mV)")
                    ax.legend(fontsize=8)
                fig.savefig(example_path.with_name("rejected_examples.png"), dpi=120)
                plt.close(fig)


def paired_variance(records):
    """Descriptive balanced two-way SS; interaction and noise cannot be separated."""
    pairs = {}
    for info, _, after, _, _ in records:
        if info["excluded"]:
            continue
        key = info["file"].split("@")[0][:-3]
        if info["group"] in pairs.setdefault(key, {}):
            raise ValueError(f"Ambiguous recording pair: {key}")
        pairs[key][info["group"]] = after[:, -200:]
    complete = {k: v for k, v in pairs.items() if set(v) == {"short", "long"}}
    if len(complete) < 2:
        return {"unavailable": "Fewer than two complete recording pairs"}
    y = np.array([[v["short"], v["long"]] for v in complete.values()])
    grand = y.mean(axis=(0, 1))
    pair_mean, soa_mean = y.mean(axis=1), y.mean(axis=0)
    total = np.sum((y-grand)**2)
    pair_ss = 2*np.sum((pair_mean-grand)**2)
    soa_ss = len(y)*np.sum((soa_mean-grand)**2)
    residual_ss = np.sum((y-pair_mean[:, None]-soa_mean[None]+grand)**2)
    return dict(pair_ids=list(complete), n_complete_pairs=len(complete),
                fractions=dict(recording_pair=float(pair_ss/total), soa=float(soa_ss/total),
                               interaction_and_noise=float(residual_ss/total)),
                note="Descriptive sums of squares, not causal effects or an animal-level statistical test. Pair IDs drop the final three sequence digits from the filename; verify against acquisition metadata.")


def main():
    cfg = SETTINGS
    if not 0 < cfg.low_hz < cfg.high_hz < cfg.fs/2 or min(cfg.bootstrap, cfg.jitter) < 0:
        raise ValueError("Invalid filter or resampling settings")
    if cfg.fs != 1000:
        raise ValueError("This workflow uses a 1 kHz analysis grid")
    if cfg.anchor_convention not in ("raw-one", "raw-zero", "csd-one", "csd-zero"):
        raise ValueError("Unknown anchor convention")
    out = (OUTPUT_ROOT / RUN_NAME).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        raise ValueError("Output already contains a run; change RUN_NAME (cache may be reused)")
    cache = CACHE_DIRECTORY
    overrides = json.loads(OVERRIDES_FILE.read_text()) if OVERRIDES_FILE else {}
    rows = list(csv.DictReader((DATA_REPO/"NKI_data/ch_info.csv").open()))
    if set(int(r["SOA Code"]) for r in rows) != {1, 2}:
        raise ValueError("Expected SOA codes 1 (short) and 2 (long)")
    if RECORDING_LIMIT:
        rows = rows[:RECORDING_LIMIT]
    records = []
    import scipy
    save_json(out/"manifest.json", dict(version=VERSION, settings=asdict(cfg), repo=str(DATA_REPO.resolve()),
              metadata_sha256=hashlib.sha256((DATA_REPO/"NKI_data/ch_info.csv").read_bytes()).hexdigest(),
              overrides=overrides, status="running", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              dependency_versions=dict(numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__),
              processing_order="continuous polyphase anti-alias resampling; continuous zero-phase bandpass; QC; LFP repair; per-trial baseline; physical CSD; piecewise-linear alignment",
              input_units_assumption="microvolts", anchor_numbering_provisional=cfg.anchor_convention == "raw-one"))
    for i, row in enumerate(rows):
        name = row["BBN files"].strip("'")
        print(f"[{i+1}/{len(rows)}] {name}", flush=True)
        record = process_recording(DATA_REPO/"NKI_data/raw_files"/name, row, cache, out, cfg, overrides.get(name, {}))
        records.append(record)
        print(json.dumps(record[0]), flush=True)
    hashes = [r[0]["signal_sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate raw signal content detected; review before constructing templates")
    rows_out = [dict(file=r[0]["file"], group=r[0]["group"], n_triggers=r[0]["n_triggers"],
               n_eligible=r[0]["n_eligible"], n_kept=r[0]["n_kept"], n_rejected=r[0]["n_rejected"],
               bad_contacts_zero=";".join(map(str, r[0]["bad_contacts_zero"])), excluded=r[0]["excluded"],
               split_half_correlation=r[0]["split_half_correlation"], median_soa_ms=r[0]["median_soa_ms"]) for r in records]
    write_csv(out/"recording_qc.csv", rows_out)
    counts = [sum(r[0]["group"] == g and not r[0]["excluded"] for r in records) for g in ("short", "long")]
    if min(counts) >= 4:
        groups = build_groups(records, cfg, out)
        save_json(out/"summary.json", groups)
        save_json(out/"paired_variance.json", paired_variance(records))
        plot_outputs(out, records, groups)
    elif not RECORDING_LIMIT:
        raise ValueError(f"Insufficient retained recordings for templates: {counts}")
    manifest = json.loads((out/"manifest.json").read_text())
    manifest["status"] = "smoke_test_complete" if RECORDING_LIMIT else "complete"
    save_json(out/"manifest.json", manifest)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
