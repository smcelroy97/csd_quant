# CSD-channel interpretation rerun

CSV landmarks were interpreted as **one-based CSD rows** (`csd-one`), rather than one-based raw LFP contacts (`raw-one`). Thus the internal zero-based anchor is CSV minus one, versus CSV minus two previously. This moves each landmark one row deeper in the physical CSD before the same piecewise-linear interpolation. Zero-based CSD numbering was not assumed. Data, filters, baseline, QC thresholds, PCA, random seeds, and WD settings were unchanged.

All 26 recordings (13 per SOA) completed. QC decisions were verified identical: 63/4,163 trials rejected, no channels rejected, no recordings excluded. Separate short/long PC1, mean, median, and normalized-PC1 files are saved under `qc_results/run_20260923_csd_one/`.

| Measurement | Short: LFP → CSD interpretation | Long: LFP → CSD interpretation |
|---|---:|---:|
| PC1 explained variance | 33.06% → 33.07% | 32.43% → 31.53% |
| Largest squared-PC1-score contribution | 80.0% → 85.7% | 65.7% → 77.9% |
| Minimum leave-one-recording-out PC1 absolute correlation | 0.396 → 0.225 | 0.502 → 0.355 |
| Median PC1 absolute correlation with ±1-contact landmark jitter | 0.392 → 0.224 | 0.420 → 0.300 |
| Median recording-bootstrap PC1 absolute correlation | 0.914 → 0.966 | 0.757 → 0.800 |
| PC1 correlation with mean | 0.252 → 0.260 | 0.002 → 0.262 |

The new and previous PC1s have absolute correlations of 0.247 short and 0.297 long (absolute values account for arbitrary PC sign). Mean-template correlations are 0.507 and 0.462. Thus landmark interpretation materially affects spatial patterns even though explained variance barely changes. Median bootstrap correlation improves, but leave-one-out influence and independent landmark-jitter stability worsen; stability is not uniformly improved.

## Held-out Wasserstein comparison

Median WD, lower is closer. Same exact balanced EMD, separate source/sink normalization, 30 depth bins, 5-ms time bins, 0–200 ms window. These are descriptive comparisons, not significance tests or confirmation of anatomical correctness.

| Target | Short: LFP → CSD | Long: LFP → CSD |
|---|---:|---:|
| Mean | 0.185 → 0.184 | 0.179 → 0.175 |
| Median | 0.203 → 0.182 | 0.188 → 0.182 |
| PC1 | 0.217 → 0.196 | 0.207 → 0.189 |
| Recording-normalized PC1 | 0.189 → 0.218 | 0.166 → 0.175 |
| Split halves of same recording | 0.032 → 0.037 | 0.022 → 0.022 |
| Ten trials versus disjoint remainder | 0.068 → 0.068 | 0.054 → 0.053 |
| Same recording, landmark jitter | 0.060 → 0.057 | 0.058 → 0.056 |

Ordinary PC1's held-out median WD improves about 9.4% short and 9.0% long. It remains worse than the mean/median targets. Normalized PC1 worsens for both conditions. These mixed results do not justify selecting a channel convention by whichever gives lower WD.

The descriptive paired variance decomposition remains dominated by recording-pair differences: 91.1% recording pair, 1.3% SOA, 7.6% interaction/residual. Pair IDs are inferred from filenames; this is not an animal-level inferential test.

**Conclusion:** Interpreting the landmarks as CSD channels changes the templates substantially and modestly improves ordinary-PC1 held-out WD, but does not resolve dominant-recording influence or landmark sensitivity. It does not establish a canonical response or a robust single PC1 for model tuning. No model reranking was performed.

## Reusable scripts, block by block

- `qc_templates.py`: configuration (`csd-one`); continuous loading/resampling/cache; channel QC; trial QC/manual overrides; limited LFP repair and physical CSD; CSD-coordinate landmark interpolation; separate SOA PCA/templates; influence/bootstrap/landmark diagnostics; plots and provenance. Run directly in an editor. Change `RUN_NAME` before repeating to preserve existing results.
- `validate_wd.py`: select completed run; configure source/sink WD; hold out each recording for template comparison; split-half/ten-trial/landmark uncertainty; optional metadata-checked simulation comparison; save tables and figures.
- `compare_interpretations.py`: load previous and new diagnostics; tabulate matched measurements; calculate old/new pattern correlations; verify identical QC decisions; save `interpretation_comparison.csv`.

No argparse or CLI configuration is used. The continuous cache is reused from the parent workflow. Detailed processing explanations remain in the parent `QC_WORKFLOW.md`. The previous run is preserved.
