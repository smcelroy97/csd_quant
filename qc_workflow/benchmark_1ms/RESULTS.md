# One-millisecond short-SOA benchmark

“Keep 1s” was interpreted as keeping the 1-ms grid discussed immediately beforehand, not one-second bins. Production scoring files were not changed.

Source: the QC-derived short-SOA ordinary-mean target, one-based CSD landmark interpretation, 13 retained recordings. Fixed 30-depth anatomical grid; 0–200 ms window. Exact balanced EMD separately normalizes sink/source mass, then sums distances. Depth and time span normalized 0–1 with equal weights, matching the existing implementation. No thresholding or whole-map demeaning.

| Comparison | 1-ms median WD | 5-ms median WD | 1-ms descriptive 5th–95th percentiles |
|---|---:|---:|---:|
| Held-out recording versus mean of other 12 | 0.182295 | 0.183897 | 0.126565–0.231534 |
| Split halves within recording | 0.036585 | 0.036887 | 0.024933–0.155129 |

Held-out median is approximately 0.87% lower at 1 ms. Spearman distance-rank correlation between grids is 0.9945 for held-out comparisons and 1.0 for split halves. The 5-ms benchmark conclusions for this target are little changed. These are descriptive experimental results, not a statistical equivalence test and not a check of simulation ranking. Percentiles are not acceptance thresholds or confidence intervals.

Use the **1-ms results** as the reference for simulation scoring at 1 ms with the same target, anatomical mapping, time window, and cost weights. A simulation score near 0.182 is near the observed median held-out biological distance, not proof of biological correctness. The split-half benchmark measures within-recording repeatability, not an achievable universal model target.

Only the selected short-SOA mean and split-half benchmark were recomputed here. Earlier PC1, long-SOA, pooled, ten-trial, and landmark-jitter WD results remain 5-ms diagnostics and must not be relabeled as 1-ms results. No model reranking was performed.

## Executable script, block by block

`benchmark_1ms.py` runs in an editor with editable paths and no argparse/CLI configuration.
1. Configuration: source CSD-channel QC run, short SOA, output directory, shared scoring helpers.
2. Load the 13 maps, filenames and split halves.
3. For 5-ms and 1-ms grids, leave each recording out when building its mean reference; independently compare its two split halves.
4. Save every total/sink/source distance and execution time as each comparison finishes.
5. Calculate medians, descriptive quantiles, paired differences and rank correlation; save summary and completion manifest.

Saved arrays already have matching 30 × 200 depth/time dimensions. At 1 ms there is no temporal averaging; at 5 ms signs are split before averaging. This compares existing grid conventions, which also place temporal bin coordinates uniformly across 0–1 at each resolution.

Files: `results/distances.csv`, `results/summary.json`, `results/manifest.json`. Dependencies: existing qc_workflow helpers and their Python environment. Re-running replaces this benchmark's output files only.
