# Normalized mean and correction to the PCA comparison

## Important correction

The original `erp_pca.py` and previous QC reruns fit PCA to X shaped recordings × flattened depth-time points, centering each depth-time feature across recordings. Their PC1 is a direction of between-recording variation. This does not prevent that direction from resembling a shared physiological response: if X_i = a_i S, centered PCA recovers S when the amplitudes vary. The earlier implication that a variation component cannot be canonical was incorrect.

Rimehaug et al. use a different orientation. Their Figure_2_and_3 notebook fits `PCA` to `X.T`, so each recording is a feature and each depth-time point is an observation. Centering removes each recording's scalar mean over its map, not the across-recording mean CSD map. The component contains recording weights; their plotted map is `(weights @ X).reshape(depth,time) * explained_variance_ratio`. Thus this PCA directly extracts a weighted spatiotemporal pattern shared across recordings. It does not simply locate the largest instantaneous CSD value or time derivative.

Verified in https://github.com/atleer/CINPLA_Allen_V1_analysis/blob/7e2390763e69f182e09a87999194a85e4e087b3d/make_figures/Figure_2_and_3.ipynb . Paper: https://elifesciences.org/articles/87169 . The paper reports 50.4% and checks alternative alignment and plain-mean targets. Earlier comparisons of our ~33% with that figure did not use the same PCA orientation and should not be treated as equivalent. No claim that the paper's PCA is inappropriate is warranted.

We reproduced their PCA orientation/reconstruction on our processed A1 maps, resolving arbitrary component sign by correlation with the training mean. This is not a full replication of their preprocessing or V1 experiment. Eigensystem results were checked numerically against sklearn PCA(X.T), including explained variance and reconstructed map.

## Normalized-average test

Source: completed `csd-one` run, 13 recordings per SOA, same QC, depth warp, 0–199 ms maps and exact 5-ms-grid sign-separated WD. Each recording is divided by its full-map L2 norm, then averaged with equal recording weights. This is an arithmetic mean of normalized maps, not a Wasserstein barycenter. Normalizing the final ordinary mean would not produce this result.

Every held-out template is built using only the other 12 recordings. Lower median WD is closer; descriptive results do not establish significant superiority.

| Target | Short median WD | Long median WD |
|---|---:|---:|
| Ordinary mean | 0.184 | 0.175 |
| Normalized mean | 0.189 | 0.181 |
| Previous recording-as-observation PC1 | 0.196 | 0.189 |
| Previous shape PC1 | 0.218 | 0.175 |
| Rimehaug-orientation PC1 | 0.205 | 0.175 |

Normalized mean is not a clear improvement over ordinary mean: it improves individual held-out distances in 7/13 short and 4/13 long recordings. Its pattern correlates 0.975 with the ordinary mean for both SOAs. Differences of medians are not the median of paired differences; both are saved in diagnostics.

| Stability measurement | Short normalized mean | Long normalized mean | Short paper PCA | Long paper PCA |
|---|---:|---:|---:|---:|
| Minimum leave-one-out pattern correlation | 0.984 | 0.986 | 0.242 | 0.631 |
| Median bootstrap pattern correlation | 0.868 | 0.884 | 0.841 | 0.781 |
| Median independent ±1-anchor jitter correlation | 0.796 | 0.783 | 0.317 | 0.510 |

PC correlations use absolute values to account for arbitrary sign. Mean correlations retain the physical sign. Leave-one-out averages share 12/13 inputs, so their high stability is not independent proof of biological validity. Bootstrap/jitter draws are descriptive, not animal-level confidence intervals.

The Rimehaug-orientation PC1 explains 30.3% short and 30.1% long of its appropriately centered spatial-temporal variance. Its correlations with ordinary mean are 0.598 and 0.735, respectively. Its long-SOA WD is competitive with ordinary mean, but short-SOA influence and landmark sensitivity remain concerns. These results narrow the previous conclusion: the data have not demonstrated a robust universal PC1 target under these settings; PCA itself is not disqualified, and this is not proof that the data cannot support a useful reference.

## Saved scripts and outputs, block by block

`test_normalized_mean.py` runs directly in an editor, with no argparse:
1. Configure the existing source run, output directory, random seed and 100 draws.
2. Check independent positive amplitude-scaling invariance and identical-map recovery.
3. Load retained maps, normalize and average, save separate short/long NPY/NPZ templates with times/depth/metadata.
4. Calculate held-out WD and paired comparisons against all previous targets.
5. Measure leave-one-out, bootstrap and anatomical jitter stability.
6. Save comparison plots, summaries and provenance.

`test_paper_pca.py` also runs directly:
1. Import the same configuration and source maps.
2. Implement the transpose-PCA covariance calculation and weighted-map reconstruction; choose sign using the training mean only.
3. Verify against sklearn's PCA(X.T).
4. Save separate paper-orientation PC1 maps, weights and explained variance.
5. Refit within every held-out, bootstrap and landmark-jitter sample; save metrics.

Run the normalized-mean script first, then the paper-PCA script. Helper copies `qc_templates.py` and `validate_wd.py` provide the exact existing alignment and distance functions. Dependencies: numpy, scipy, h5py, matplotlib, POT, scikit-learn. Output directory is `normalized_mean_results/`; repeated runs replace this experiment's outputs but do not modify source QC runs.

Separate normalized means: `normalized_mean_results/short/mean_shape_short.npz` and `normalized_mean_results/long/mean_shape_long.npz`. Summary figures compare ordinary mean, normalized mean, and the previous PC1 definitions; paper-orientation PC1 is saved separately. No model batch was reranked.
