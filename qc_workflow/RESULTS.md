# Results: quality control, anatomical alignment, and Wasserstein validation

Completed run: `qc_results/run_20260922_161348`. All 26 recordings were processed, with 13 recordings per SOA. Seven synthetic/numerical tests passed. Scripts have editable configuration blocks and run directly in an editor; no argparse or CLI configuration is used. See `QC_WORKFLOW.md` for the complete block-by-block explanation.

## Quality control

The conservative automatic rules rejected 63 of 4,163 trials (1.51%), leaving 4,100 trials. No channels were flagged and no recordings were excluded. This does not certify every channel as good: the rules deliberately avoid treating a strong, repeatable response as an artifact. Review the per-recording channel/trial CSVs and rejection plots before adding manual overrides. Channel repair, when needed, interpolates limited interior LFP contacts before calculating CSD, and records affected derivative stencils.

QC alone did not materially change ordinary PC1 variance: short SOA changed from 32.5% to 33.1%, and long SOA from 32.8% to 32.4%. Bad trials are therefore not the main explanation for the unstable template in this analysis.

## Separate SOA templates and variance

| Measurement | Short SOA | Long SOA |
|---|---:|---:|
| Ordinary PC1 explained variance | 33.1% | 32.4% |
| Recording-normalized PC1 explained variance | 15.7% | 18.5% |
| Ordinary PC1 correlation with mean | 0.252 | 0.002 |
| Largest recording share of squared PC1 scores | 80.0% | 65.7% |
| Absolute PC1 correlation after removing the most influential recording | 0.396 | 0.502 |
| Median absolute PC1 correlation under independent ±1-contact landmark perturbations | 0.392 | 0.420 |

The influential rb043044 response is highly repeatable in the short-SOA split-half comparison (correlation 0.992), so influence alone is not a defensible reason to discard it. Centered PC1 describes variation around the mean, and especially for long SOA it is not representative of the mean evoked response. Recording normalization reduces amplitude dominance but can increase the contribution of weak/noisy recordings.

A descriptive paired sum-of-squares decomposition attributes 90.8% to recording-pair differences, 1.2% to the SOA main effect, and 7.9% to interaction/residual variation. Pairs are inferred from filenames; this is not a causal decomposition, animal-level significance test, or proof that SOA has no physiological effect.

## Anatomical alignment

The scripts interpret CSV landmarks as raw electrode contacts. One-based indexing remains provisional. With an interior second derivative, raw one-based contact numbers map to CSD rows by subtracting two. The alternative raw-zero convention and explicit index-shift sensitivity are documented.

Physical CSD is computed before piecewise-linear depth warping. Supra/gran/infra landmarks map to common bins 7/15/22 on a 30-bin grid. This accommodates unequal anatomical intervals without recomputing physical derivatives on warped coordinates. It is an anatomical comparison coordinate, not a uniform physical depth axis.

Anatomical warping does not remove uncertainty in the supplied landmarks. Under independent ±1-contact perturbations, mean templates were more stable than PC1 (median correlations 0.787 short and 0.756 long). A global +1-contact shift reduced ordinary PC1 correlation to 0.247 and 0.297 respectively. Confirming channel indexing and landmark definitions remains consequential. The script reports invalid shifts rather than silently clipping them.

## Wasserstein validation

These are median diagnostic distances using exact balanced EMD, separate source/sink probability distributions, a 30-depth-bin by 5-ms-time-bin grid, and a 0–200 ms window. They must not be compared numerically with the old 1-ms scores. Template comparisons hold the tested recording out of template construction. Quantiles in the accompanying plot are descriptive ranges, not confidence intervals.

| Comparison | Short SOA | Long SOA |
|---|---:|---:|
| Held-out recording versus mean | 0.185 | 0.179 |
| Held-out recording versus median | 0.203 | 0.188 |
| Held-out recording versus ordinary PC1 | 0.217 | 0.207 |
| Held-out recording versus recording-normalized PC1 | 0.189 | 0.166 |
| Split halves of the same recording | 0.032 | 0.022 |
| Ten trials versus disjoint remaining trials | 0.068 | 0.054 |
| Same recording with ±1-contact landmark perturbations | 0.060 | 0.058 |

Ordinary PC1 is not the preferred target on these descriptive held-out results. Use the mean as a primary evoked-response reference, retain normalized PC1 as a complementary comparison (particularly long SOA), and save ordinary PC1 for continuity and variability analysis. Do not choose or adjust landmarks to minimize a simulation's WD. Compare candidate models against held-out biological distances, trial-count uncertainty, and anatomical uncertainty rather than interpreting a small absolute WD in isolation.

WD remains a shape metric: separate polarity normalization discards overall amplitude and source/sink mass balance. Evaluate evoked response amplitude, latency, laminar sink/source structure, PSTH, and adaptation alongside it. These results do not establish that excessive model L4 activity is the cause of the previous ranking, nor that adding laminar noise would improve physiological accuracy.

Raw trigger intervals were approximately 624.5 ms for short SOA and 1524.477 ms for long SOA. The described simulation uses 500 ms. Match stimulus timing, pulse duration, trial/pulse selection, baseline, filtering, and depth/time conventions before interpreting model-versus-experiment scores. Optional simulation scoring is implemented with explicit metadata checks, but this run did not rerank the model batch.

## Saved outputs

- `short/pc1_short.npz` and `long/pc1_long.npz`: requested separate PC1 templates, including time coordinates and metadata; `.npy` copies are also provided.
- Each SOA directory also contains mean, median, recording-normalized PC1, full PCA details, influence diagnostics, and WD tables.
- `templates.png`, `variance.png`, `qc_overview.png`, and `wd_validation.png`: summary figures.
- `recordings/`: per-recording audits, retained epochs, processed CSD, QC comparison plots, and rejected-trial examples where applicable.
- `manifest.json`: configuration, provenance hashes, package versions, and completion status.

The original processing scripts and outputs are preserved. The new workflow and its reusable continuous-data cache are contained in `qc_workflow/`; generated results and cache are ignored by its `.gitignore`.
