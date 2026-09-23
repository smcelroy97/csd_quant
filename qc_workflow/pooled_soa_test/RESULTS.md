# Pooled-SOA analysis

Combined 13 short and 13 long retained CSD maps using the latest one-based CSD landmark interpretation and unchanged preprocessing. Each trial-averaged recording is one input; trials are not pooled with unequal trial-count weights. PCA uses the verified Rimehaug orientation (depth-time observations, recording variables), weighted-map reconstruction, and training-mean sign convention. Ordinary and per-recording-L2-normalized means are also saved.

## Main result

PC1 explained variance is **28.57% pooled**, versus **30.29% short** and **30.08% long**. Pooling does not increase the fraction explained. Pooled PC1 has absolute pattern correlations of **0.935 with short** and **0.977 with long**. This largely preserves the separate templates rather than producing a distinctly different common component.

Twenty-six recordings are not 26 verified independent animals. Filename-derived pairing identifies 13 pairs, each containing one recording of each SOA. Pair identities are provisional acquisition groupings, not confirmed animal IDs. Therefore both ordinary leave-one-recording-out and leave-one-pair-out validation are included.

## Held-out WD

Same 5-ms-grid exact balanced sign-separated WD as earlier. Lower is closer. For pooled pair-held-out scores, both members are removed before template fitting; each is then scored independently and the resulting 26 values are summarized. These are descriptive medians, not significance tests.

| Target | Separate short | Pooled target, short tested, pair held out | Separate long | Pooled target, long tested, pair held out |
|---|---:|---:|---:|---:|
| Ordinary mean | 0.184 | 0.187 | 0.175 | 0.176 |
| Normalized mean | 0.189 | 0.183 | 0.181 | 0.187 |
| Rimehaug PC1 | 0.205 | 0.195 | 0.175 | 0.185 |

Pooled PC1 modestly improves short-SOA median WD and worsens long-SOA median WD. There is no consistent improvement across SOAs. Overall pair-held-out medians are 0.181 for ordinary mean, 0.185 for normalized mean, and 0.190 for PC1.

## Stability

- Pooled PC1 minimum leave-one-recording-out correlation: **0.582**; minimum leave-one-pair-out correlation: **0.481**. Separate short/long minima were 0.242/0.631.
- Pooled PC1 median bootstrap correlation: **0.876** when independently resampling 26 recordings, versus **0.754** when resampling the 13 pairs together. Separate-SOA medians were 0.841/0.781. Bootstrap draw ranges are descriptive, not confidence intervals.
- Independent per-recording ±1-contact landmark jitter gives median pooled PC1 correlation **0.521**, versus separate 0.317/0.510. This jitter diagnostic does not model correlated anatomical error shared by both recordings in a pair.
- Pooled normalized mean is stable to leaving pairs out (minimum correlation **0.985**) and has median pair-bootstrap correlation **0.886**. High leave-out stability partly reflects overlapping training sets and does not prove biological validity.

Pooling adds repeated-condition measurements; it does not demonstrate the benefit of recruiting more independent animals. It has not resolved the PC1 uncertainty. The direction of effects is mixed, so a pooled reference should be considered an exploratory condition mixture rather than a validated replacement for SOA-specific targets.

## Script, block by block

`analyze_pooled_soa.py` is executable in an editor, with editable configuration and no argparse/CLI configuration:
1. Set source run, helper directory, output folder, seed, and 100 resampling draws.
2. Load both SOA groups, verify one member per SOA per inferred pair, construct and save three pooled templates with time/depth metadata.
3. Refit templates excluding one recording or a complete pair; calculate held-out WD and template stability.
4. Bootstrap individual recordings and complete pairs separately.
5. Perturb anatomical anchors independently by ±1 contact and rebuild templates.
6. Save summaries by test SOA, leave-out CSVs, PCA weights, explained variance, and correlations with separate templates.
7. Plot separate and pooled targets and save a completion/provenance manifest.

The script imports verified helpers from the existing `normalized_mean_test` directory. Dependencies remain numpy, scipy, h5py, matplotlib, POT, scikit-learn. Outputs are in `results/`; rerunning replaces this experiment's outputs but leaves the source QC runs unchanged. All folds were checked for the expected 25-recording or 24-recording training size. Output plots were inspected. No simulation ranking was performed.
