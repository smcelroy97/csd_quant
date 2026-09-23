# CSD quality control and SOA-specific templates

Open `qc_templates.py` in your Python editor, edit its configuration block, and run the file. Then open and run `validate_wd.py`. There are no command-line arguments. Use the existing `csd_quant` Python environment, which supplies NumPy, SciPy, h5py, Matplotlib, and POT (`ot`). Run `test_qc_templates.py` directly to check the implementation.

The scripts are additive. They do not replace `wasserstein_dist.py`, edit raw recordings, or replace the historical template. Every template run has a timestamped output folder. Do not switch model optimization to a new template until its diagnostics have been reviewed.

## What to edit

In `qc_templates.py`:

- `DATA_REPO`: folder containing `NKI_data/ch_info.csv` and `NKI_data/raw_files`.
- `OUTPUT_ROOT`, `RUN_NAME`, `CACHE_DIRECTORY`: outputs and reusable downsampled continuous recordings. Processing/QC decisions are recalculated every run. Reuse the cache for filter, baseline, rejection, or landmark sensitivity runs. Source size, timestamp, and ingestion version invalidate stale caches.
- `SETTINGS`: filter band, baseline policy, landmark convention, rejection thresholds, interpolation limits, minimum retained trials, and bootstrap/jitter repetitions. Defaults use 0.5–100 Hz, fourth-order zero-phase filtering, and per-trial −100–0 ms baseline subtraction to match the reviewed simulation settings.
- `RECORDING_LIMIT`: leave as `None` for all recordings. Set to 2 for a smoke test; smoke tests are not selected automatically by the WD script.
- `OVERRIDES_FILE`: optional JSON path for manual decisions. All manual contact and trial IDs in this file are **zero-based**. Raw CSV landmarks use the separate `anchor_convention` setting.

In `validate_wd.py`:

- `RESULTS_ROOT` and `RUN_DIRECTORY`: select a completed template run. `None` selects the newest completed run by manifest modification time.
- `TIME_BIN_MS`: default 5-ms temporal means for feasible repeated exact EMD. Set to 1 for the original temporal resolution. Distances from different grids should not be compared directly.
- `TEN_TRIAL_DRAWS`, `LANDMARK_DRAWS`: number of repeated descriptive validation draws, not independent observations.
- Optional simulation fields at the top enable scoring an explicitly aligned and preprocessed simulation. They are disabled by default.

## `qc_templates.py`, block by block

### 1. Configuration and settings

Defines explicit, recorded processing choices. The 1-kHz analysis grid is fixed. The CSV is interpreted as raw electrode contacts, based on the supplied clarification. `raw-one` is the provisional default because the source is MATLAB-style data; the zero/one-based convention still needs confirmation.

The centered second difference at CSD row 0 belongs to raw electrode index 1. Therefore:

| CSV convention | Conversion to zero-based CSD row |
| --- | --- |
| Raw contacts, one-based | subtract 2 |
| Raw contacts, zero-based | subtract 1 |
| CSD rows, one-based | subtract 1 |
| CSD rows, zero-based | unchanged |

### 2. Continuous ingestion and cache

Reads compressed HDF5 in bounded, overlapping blocks. Converts µV to mV and uses polyphase anti-alias filtering to resample from the native rate to 1 kHz. Blocks preserve sampling phase and overlap beyond the FIR support. A regression test compares blocked and whole-recording resampling.

This deliberately differs from the historical FFT resampling: polyphase downsampling occurs before the analysis bandpass and avoids assuming a periodic full recording. The continuous fourth-order analysis bandpass follows downsampling. Both data and simulations must still use the same intended analysis band and baseline.

The cache records raw-signal SHA-256, source sampling rate, trigger times, nonfinite samples, and flat-sample fractions. Nonfinite values are temporarily replaced for numerical filtering, but their affected contacts/epochs are not silently accepted. Identical raw signal hashes stop template construction for duplicate review.

### 3. Epoch extraction and channel QC

Extracts −200–300 ms for QC and −100–200 ms for saved ERPs. A one-second recording-edge margin protects the filter boundaries; it is a practical default, not a guarantee for every high-pass cutoff. For a legacy 0.05-Hz analysis, inspect boundary transients and consider a longer margin.

Channel screening uses prestimulus measurements to avoid rejecting channels simply because their evoked responses are large:

- Near-flat baseline or more than 99% repeated raw samples.
- More than 0.1% nonfinite raw samples.
- Persistent noise supported by **both** an unusually large fast-residual step metric and an unusually large residual relative to neighboring contacts. Both must exceed six robust log-space standard deviations, with spatial residual outliers in more than 25% of eligible epochs.
- A narrowband 60-Hz/harmonic contamination flag requires more than 50% of 20–400 Hz baseline power in the tested line bands, a robust line-fraction outlier, and the same persistent spatial-residual evidence. It is not triggered merely by a 60-Hz peak. Change the explicit line frequencies for a 50-Hz recording environment.

The fast-residual measure is the RMS of first differences of the downsampled signal minus the bandpassed signal. It is a broadband quality indicator, not a calibrated spectral-band power estimate. All channel measurements are saved even when they do not trigger rejection. Log-MAD scales have a floor to avoid absurd z-scores when variability is tiny.

This is conservative automatic screening, not proof that every retained contact is good. Saturation at unknown hardware rails, persistent physiological gradients, and correlated/common-mode artifacts require visual review. High amplitude alone does not reject a contact.

### 4. Trial QC and manual overrides

Computes per-contact distributions across trials for prestimulus RMS, full-epoch peak-to-peak amplitude, largest filtered time step, and fast-residual step RMS. A trial is rejected for:

- An extreme prestimulus RMS outlier (>8 robust log z by default).
- Large peak-to-peak amplitude **and** a large step or fast residual (>6 robust log z in the same contact).
- A flat epoch or nonfinite data within an expanded neighborhood.
- A supplied manual trial exclusion.

The code does not reject trials for failing to resemble the target, having poor WD, or reducing PC1 variance. Strong unusual physiology can still trigger an artifact rule, so review `trials.csv` and saved examples before interpreting rejection as ground truth. Transient flat segments that occupy only part of an epoch are not exhaustively detected by the whole-epoch flat check.

Example override file:

```json
{
  "EXACT_RECORDING_FILENAME.mat": {
    "bad_contacts_zero": [8],
    "bad_trials_zero": [12, 40],
    "keep_contacts_zero": [],
    "exclude_recording": false
  }
}
```

Overrides may also provide `anchors: [supra, granular, infra]` in the selected CSV convention. Use that only to correct documented anatomical metadata, never to minimize model WD. Keeping a contact overrides the automatic bad flag but cannot retain a contact with substantial nonfinite data. A new run logs the full overrides.

### 5. LFP repair and physical CSD

Repairs at most two bad interior contacts, with at most two consecutive bad contacts, by linear interpolation between good LFP contacts. Bad edge contacts or more extensive damage exclude the recording; the code does not extrapolate or bridge arbitrary gaps.

Repair occurs before computing the spatial second derivative. The affected three-contact CSD stencils are explicitly saved. An interpolated contact can flatten local curvature, so a repaired CSD is not independent evidence of a missing or weak sink. Repeat important analyses with repaired recordings excluded.

At least 30 trials and 50% of eligible trials must remain. No anatomical anchor is silently moved or clamped to an edge. Invalid anchors exclude the recording.

Baseline subtraction, when enabled, is performed per trial and contact. The physical CSD is `−diff(LFP, n=2)/(spacing_mm**2)`, yielding mV/mm² without a conductivity multiplier. The outer LFP contacts are lost.

### 6. Piecewise-linear anatomical warping

Preserves the original idea: supra, granular, and infra landmarks map to output bins 7, 15, and 22 on a 30-bin depth axis. Linear interpolation between anchors allows individual segments to stretch/compress. These are alignment landmarks, not layer boundaries.

CSD is calculated in physical coordinates first, then warped for pattern comparison. The warped map is a feature representation, not a literal uniformly spaced physical column. No Jacobian amplitude adjustment is applied; identical choices must be used for experimental and simulation alignment.

Uncertainty is measured with independently jittered anchors by ±1 contact, constrained to valid order. This is a sensitivity experiment, not a posterior distribution for anatomy. It does not fit landmarks to improve WD. A separate uniform-depth comparison shows PCA without landmark-based warping, and whole-anchor shifts test numbering uncertainty (including +2 rows relative to `raw-one`, the historical direct-index convention).

### 7. Separate SOA templates and PCA

Code 1 and code 2 are kept separate as short and long SOA. Actual intervals are measured from triggers, not inferred from their labels. In the supplied data they are about 624.5 and 1524.5 ms. Neither is exactly the simulation's 500-ms train.

Each retained recording contributes one ERP, independent of how many trials it contains. PCA fits the 0–199 ms poststimulus window only. Outputs for each SOA:

- `pc1_short/long.npy` and `.npz`: ordinary centered PCA, retaining amplitude variation.
- `pc1_shape_short/long.npy` and `.npz`: centered PCA after L2-normalizing each recording. This reduces amplitude dominance but can promote noisy low-amplitude recordings; use the reliability diagnostics.
- `mean_short/long` and `median_short/long`: alternative targets in physical CSD units before warping interpretation.
- `pca_details.npz`: complete components, explained variance, scores, input ERPs, and split halves.

PC signs are oriented toward their dataset mean for repeatability. PC1 remains a variation mode, not automatically an average physiological CSD. A component loading's sign alone is not evidence of a shared physical sink/source. No trial/channel threshold is optimized to increase PC1 variance.

### 8. Variance, influence, and landmark diagnostics

Reports before/after-QC variance on the **same retained recording set**, cumulative explained variance, PC1–mean correlation, each recording's squared-score share, leave-one-recording-out PC1 stability, and recording-bootstrap stability.

The bootstrap samples recordings, not known independent animals. Where recordings share subjects/sites, this can underestimate uncertainty; use acquisition metadata to define true independent units before inferential claims. Individual trial split-half correlations measure repeatability but can also be high for a repeatable artifact.

`paired_variance.json` describes variation attributable to recording-pair differences, SOA differences, and the unresolved interaction/noise residual. Pair keys are formed from filenames by removing the final three sequence digits and must be verified. This is descriptive variance decomposition, not a causal effect or a formal repeated-measures test.

### 9. Saved audit trail and plots

`manifest.json` records settings, code/metadata hashes, overrides, and completion status. Each recording has `qc.json`, channel/trial CSVs, excluded triggers, saved ERPs/retained epochs, up to five rejected LFP examples, and a common-scale before/after CSD plot.

`templates.png`, `variance.png`, and `qc_overview.png` summarize the run. Template panels are normalized individually for visualization and labeled with original maxima; the saved numerical arrays are not altered by plot scaling.

## `validate_wd.py`, block by block

### 1. Run selection and metric

Selects a completed run. Builds a fixed depth–time ground-cost matrix on normalized coordinates. Splits positive and negative currents before temporal binning so opposite polarities do not cancel within a coarser diagnostic bin. Computes exact balanced optimal transport separately for sources and sinks, then sums them.

The default 5-ms grid reduces repeated validation cost. It is **not** numerically interchangeable with historical 1-ms WD. There is no global demeaning, arbitrary 15% threshold, or entropy regularization. Empty polarity distributions are handled explicitly. Amplitude invariance is tested and intentional.

### 2. Held-out template validation

For each recording, builds mean, median, ordinary PC1, and shape-normalized PC1 from all other recordings of that SOA. Scores the held-out recording against these targets. This avoids validating a recording against a template partly made from itself.

Compare the **same held-out recordings** across target types. Use the distributions, not one pooled minimum, to decide whether PC1 is defensible. The script does not automatically select whichever template flatters a simulation.

### 3. Trial-count and landmark uncertainty

Scores disjoint split halves. Repeatedly compares ten randomly selected trials with the remaining trials of the same recording. These subsets are disjoint to avoid optimistic subset-versus-full comparisons. This approximates finite-trial uncertainty; random ten-trial subsets are not equivalent to a ten-pulse train with adaptation history.

Also measures how much same-recording WD changes when anatomical landmarks move by ±1 contact. Percentile ranges are descriptive, not confidence intervals. A model improvement smaller than these variations should not be treated as decisive without more repetitions.

### 4. Optional simulation scoring

Requires an NPZ with `csd` (depth,time), `times_ms`, `depths_um`, and `preprocessing_json`. It requires explicit anatomical simulation anchor depths and checks filter/baseline metadata against the target run. Both depth and time coordinates are validated; shape equality is insufficient.

The script reports separate source/sink distances, total WD, response RMS, and the SOA mismatch. It does not manufacture a composite physiological score or a pass/fail threshold. Absolute amplitude comparisons are appropriate against physical mean/median targets, not unit PC loading maps, and need electrode/model calibration.

Required `preprocessing_json` content for the defaults:

```json
{"low_hz": 0.5, "high_hz": 100.0, "filter_order": 4,
 "baseline": true, "baseline_ms": [-100, 0]}
```

## `test_qc_templates.py`, block by block

Synthetic checks cover bounded-block resampling and cache consistency; centered CSD polarity/indexing; exact landmark placement; interpolation in LFP space and unsafe edge rejection; noisy-channel and transient-trial rejection without rejecting the ordinary common response; PCA equivalence to direct SVD; and WD's expected scale, polarity, and timing behavior. These validate implementation, not the biological correctness of a template.

## How to judge whether WD is useful

1. Confirm raw-contact numbering and inspect influential recordings and rejected examples.
2. Check whether mean/median or PC1 best represents held-out data. Higher PC1 variance alone is not an improvement.
3. Compare changes in model WD with finite-trial, landmark, and between-recording variation on the same grid.
4. Match stimulus timing, baseline, filters, and anatomical mapping before interpreting model distances.
5. Keep a separate uptake criterion: evoked spike excess, latency, pulse reliability, recovery, and laminar recruitment. Shape-only WD must not select a nearly unresponsive model by itself.

The matched-preprocessing settings differ from the historical template. Before/after QC comparisons within a run isolate rejection; comparisons with the old template also include filtering, baseline, and anchor-convention changes. Attribute those effects separately.
