"""Run in your editor after qc_templates.py. No command-line arguments.

Calibrates shape-only WD against held-out recordings, split halves, ten-trial
resampling, and landmark uncertainty. Does not change the production scorer.
"""
from pathlib import Path
import json
import numpy as np
import ot
from scipy.spatial.distance import cdist

from qc_templates import align_laminar, corr, pca_fit, save_json, write_csv

# ======================= EDITABLE CONFIGURATION =======================
RESULTS_ROOT = Path(__file__).resolve().parent / "qc_results"
RUN_DIRECTORY = None  # Set Path("...") to select a run, or use newest complete run.
TIME_BIN_MS = 1       # Exact EMD on 5-ms means, not a regularized approximation.
TEN_TRIAL_DRAWS = 20
LANDMARK_DRAWS = 20
RANDOM_SEED = 314159
# Optional simulation NPZ: csd (depth,time), times_ms, depths_um, plus preprocessing_json.
SIMULATION_FILE = None
SIMULATION_GROUP = "short"
SIMULATION_ANCHOR_DEPTHS_UM = None  # Required anatomical homologs of CSV anchors.
SIMULATION_SOA_MS = 500.0
# ======================================================================


class ShapeWD:
    """Sink/source normalized transport on a common fixed 0..200-ms grid."""
    def __init__(self, time_bin_ms=5, n_depth=30):
        if time_bin_ms < 1 or 200 % time_bin_ms:
            raise ValueError("Time bin must divide 200 ms")
        self.bin_ms = time_bin_ms
        self.n_depth = n_depth
        d, t = np.meshgrid(np.linspace(0, 1, n_depth),
                           np.linspace(0, 1, 200//time_bin_ms), indexing="ij")
        coords = np.column_stack([d.ravel(), t.ravel()])
        self.cost = cdist(coords, coords)
        self.cost /= self.cost.max()

    def grid(self, a):
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_depth, 200) or not np.isfinite(a).all():
            raise ValueError("Expected finite aligned CSD with shape (30,200)")
        return a.reshape(self.n_depth, -1, self.bin_ms).mean(-1)

    def __call__(self, a, b):
        # Split signs BEFORE temporal averaging: opposite currents must not cancel
        # just because a diagnostic grid uses coarser temporal bins.
        distances = []
        for sign in (-1, 1):
            x = self.grid(np.maximum(sign*a, 0)).ravel()
            y = self.grid(np.maximum(sign*b, 0)).ravel()
            sx, sy = x.sum(), y.sum()
            if sx <= 1e-12 and sy <= 1e-12:
                distances.append(0.0)
            elif min(sx, sy) <= 1e-12:
                distances.append(1.0)
            else:
                value, log = ot.emd2(x/sx, y/sy, self.cost, numItermax=1000000, log=True)
                if log.get("warning"):
                    raise RuntimeError(log["warning"])
                distances.append(float(value))
        return dict(wd=sum(distances), wd_sink=distances[0], wd_source=distances[1])


def pick_run():
    if RUN_DIRECTORY is not None:
        path = Path(RUN_DIRECTORY)
    else:
        completed = [p.parent for p in RESULTS_ROOT.glob("*/manifest.json")
                     if json.loads(p.read_text()).get("status") == "complete"]
        if not completed:
            raise ValueError("Run qc_templates.py first, or set RUN_DIRECTORY")
        path = max(completed, key=lambda p: (p/"manifest.json").stat().st_mtime_ns)
    if json.loads((path/"manifest.json").read_text()).get("status") != "complete":
        raise ValueError("Selected QC run is incomplete")
    return path


def quantiles(values):
    return dict(zip(("p05", "median", "p95"), map(float, np.quantile(values, [.05, .5, .95]))))


def validate_group(run, group, metric):
    with np.load(run/group/"pca_details.npz", allow_pickle=False) as z:
        X, names, halves = z["erps"], z["filenames"].tolist(), z["halves"]
    rows, sampling, landmarks = [], [], []
    rng = np.random.default_rng(RANDOM_SEED)
    for i, (a, name) in enumerate(zip(X, names)):
        train = np.delete(X, i, axis=0)
        pc1 = pca_fit(train)["components"][0]
        pc1_shape = pca_fit(train, True)["components"][0]
        targets = dict(mean=train.mean(0), median=np.median(train, axis=0),
                       pc1=pc1, pc1_shape=pc1_shape)
        for kind, target in targets.items():
            rows.append(dict(file=name, target=kind, **metric(a, target), correlation=corr(a, target)))
        rows.append(dict(file=name, target="split_half_same_recording", **metric(*halves[i]),
                         correlation=corr(*halves[i])))
        with np.load(run/"recordings"/Path(name).stem/"erp.npz", allow_pickle=False) as z:
            epochs = z["aligned_epochs"][..., z["times_ms"] >= 0]
            physical = z["physical_after"][:, z["times_ms"] >= 0]
            anchors = z["anchors_csd"]
        # Use disjoint held-out trials, avoiding optimistic subset-vs-full scoring.
        if len(epochs) >= 20:
            for draw in range(TEN_TRIAL_DRAWS):
                perm = rng.permutation(len(epochs))
                sample, rest = epochs[perm[:10]].mean(0), epochs[perm[10:]].mean(0)
                sampling.append(dict(file=name, draw=draw, **metric(sample, rest)))
        for draw in range(LANDMARK_DRAWS):
            for attempt in range(100):
                shifted = anchors+rng.integers(-1, 2, 3)
                if np.all(np.diff([0, *shifted, physical.shape[0]-1]) > 0):
                    break
            else:
                raise ValueError("Cannot generate valid landmarks")
            warped = align_laminar(physical, shifted)
            landmarks.append(dict(file=name, draw=draw, **metric(a, warped)))
        print(f"WD {group}: {i+1}/{len(X)} {name}", flush=True)
    dest = run/group
    write_csv(dest/"wd_leave_one_recording_out.csv", rows)
    write_csv(dest/"wd_ten_trials.csv", sampling)
    write_csv(dest/"wd_landmark_jitter.csv", landmarks)
    kinds = sorted(set(r["target"] for r in rows))
    summary = {kind: quantiles([r["wd"] for r in rows if r["target"] == kind]) for kind in kinds}
    if sampling:
        summary["ten_trials_vs_disjoint_remainder"] = quantiles([r["wd"] for r in sampling])
    if landmarks:
        summary["same_recording_landmark_jitter"] = quantiles([r["wd"] for r in landmarks])
    summary["grid"] = dict(depth_bins=30, time_bin_ms=TIME_BIN_MS, window_ms=[0, 200],
                           solver="exact balanced EMD", polarity_split_before_binning=True)
    summary["interpretation"] = "Descriptive quantiles, not confidence intervals; repeated draws are not independent recordings."
    save_json(dest/"wd_validation.json", summary)
    return summary


def score_simulation(run, metric):
    """Require explicit preprocessing and anatomical mapping; never guess from shape."""
    if SIMULATION_ANCHOR_DEPTHS_UM is None:
        raise ValueError("Set the three simulation anatomical anchor depths before scoring")
    with np.load(SIMULATION_FILE, allow_pickle=False) as z:
        a, times, depths = z["csd"], z["times_ms"], z["depths_um"]
        pre = json.loads(str(z["preprocessing_json"]))
    cfg = json.loads((run/"manifest.json").read_text())["settings"]
    expected = {"low_hz": cfg["low_hz"], "high_hz": cfg["high_hz"],
                "filter_order": cfg["filter_order"], "baseline": cfg["baseline"],
                "baseline_ms": [-cfg["pre_ms"], 0] if cfg["baseline"] else None}
    if any(pre.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Simulation preprocessing must match {expected}")
    if a.shape != (len(depths), len(times)) or np.any(np.diff(depths) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid depth/time coordinates")
    if len(depths) < 5 or not np.allclose(np.diff(depths), np.diff(depths)[0]):
        raise ValueError("Simulation CSD must use uniformly spaced physical contacts")
    if times[0] > 0 or times[-1] < 199:
        raise ValueError("Simulation must cover 0..199 ms")
    anchor_depths = np.asarray(SIMULATION_ANCHOR_DEPTHS_UM, dtype=float)
    if len(anchor_depths) != 3 or np.any(np.diff([depths[0], *anchor_depths, depths[-1]]) <= 0):
        raise ValueError("Simulation anchors must be interior and ordered")
    rows = np.interp(anchor_depths, depths, np.arange(len(depths)))
    a = np.stack([np.interp(np.arange(200), times, channel) for channel in a])
    aligned = align_laminar(a, rows)
    result = dict(simulation_file=str(SIMULATION_FILE), condition=SIMULATION_GROUP,
                  simulation_soa_ms=SIMULATION_SOA_MS, metrics={})
    for kind in ("mean", "median", "pc1", "pc1_shape"):
        with np.load(run/SIMULATION_GROUP/f"{kind}_{SIMULATION_GROUP}.npz", allow_pickle=False) as z:
            target = z["csd"]
            target_soa = json.loads(str(z["metadata_json"]))["median_soa_ms"]
        result["metrics"][kind] = metric(aligned, target)
    result.update(target_soa_ms=target_soa, soa_mismatch=abs(target_soa-SIMULATION_SOA_MS) > 1,
                  rms_csd=float(np.sqrt(np.mean(aligned**2))),
                  near_zero_response=bool(np.max(abs(aligned)) < 1e-10),
                  note="Shape distance is not response strength or a pass/fail criterion.")
    save_json(run/"simulation_wd.json", result)


def plot_validation(run, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    keys = ["mean", "median", "pc1", "pc1_shape", "split_half_same_recording",
            "ten_trials_vs_disjoint_remainder", "same_recording_landmark_jitter"]
    labels = ["Mean", "Median", "PC1", "Shape PC1", "Split half", "10 trials", "Anchor ±1"]
    for ax, group in zip(axes, ("short", "long")):
        vals = [results[group][k] for k in keys]
        center = np.array([v["median"] for v in vals])
        error = np.array([[v["median"]-v["p05"] for v in vals], [v["p95"]-v["median"] for v in vals]])
        ax.errorbar(np.arange(len(vals)), center, yerr=error, fmt="o", capsize=4)
        ax.set(xticks=np.arange(len(vals)), xticklabels=labels, ylabel="WD (median and 5–95% range)", title=group+" SOA")
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(run/"wd_validation.png", dpi=160)
    plt.close(fig)


def main():
    run = pick_run()
    metric = ShapeWD(TIME_BIN_MS)
    results = {group: validate_group(run, group, metric) for group in ("short", "long")}
    save_json(run/"wd_validation.json", results)
    plot_validation(run, results)
    if SIMULATION_FILE is not None:
        score_simulation(run, metric)
    print(f"Saved WD validation to {run}", flush=True)


if __name__ == "__main__":
    main()
