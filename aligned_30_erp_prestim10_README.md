# Prestimulus CSD template

`aligned_30_erp_prestim10.npy` has shape (30, 210): aligned depth by time. Time samples are -10 through 199 ms at 1 kHz. Column 10 is stimulus onset. The equivalent NPZ includes `csd`, `times_ms`, and `depth_bins`.

The template is PC1 of centered flattened, anatomically aligned recording-average CSDs from the 26 recordings in `NKI_data/ch_info.csv`. It preserves the historical template construction, continuous fourth-order 0.05–300 Hz zero-phase filtering, 100 µm contact spacing, and direct use of metadata anchor indices. No baseline subtraction, peak normalization, or hard threshold is applied. PCA sign is oriented to the original template's poststimulus pattern.

The plotted horizontal lines mark supra/granular/infra alignment anchors at zero-based bins 7, 15, and 22; they are not inferred layer boundaries. The vertical line marks 0 ms. PC1 loadings have arbitrary units.

The old active template is not replaced. Before using this file with the existing shape-only Wasserstein interpolation, supply simulated CSD covering the same -10 to 200 ms half-open window. A 0 to 200 ms simulation epoch alone would be misaligned. Prefer resampling both using their actual time coordinates.

Rebuild with the csd_quant environment:

    python rebuild_template.py --repo /path/to/csd_quant --output /path/to/new_output

The script caches per-recording arrays in its output directory. Use a fresh output directory when changing processing settings or source recordings. `metadata.json` records the processing and contributing epochs.
