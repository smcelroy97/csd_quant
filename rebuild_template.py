"""Rebuild aligned PC1 from continuous recordings, retaining -10 <= t < 200 ms.
Preserves historical filtering, anchor-index convention and PCA construction.
Run with --repo and --output; never overwrites the active scoring template.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.signal import butter, sosfiltfilt, resample
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def align(data, anchors):
    src = np.array([0, * anchors, data.shape[0]-1], dtype=float)
    if not np.all(np.diff(src) > 0):
        raise ValueError(f'Invalid depth anchors: {src}')
    mapped = np.interp(np.arange(data.shape[0]), src, [0, 7, 15, 22, 29])
    return np.stack([np.interp(np.arange(30), mapped, col) for col in data.T], axis=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.repo, args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(root/'NKI_data/ch_info.csv')
    records = []
    arrays = []
    for _, row in rows.iterrows():
        name = row['BBN files'].strip("'")
        path = root/'NKI_data/raw_files'/name
        cache = out/(path.stem+'_aligned_pre10.npy')
        info_path = cache.with_suffix('.json')
        if cache.exists() and info_path.exists():
            arrays.append(np.load(cache))
            records.append(json.loads(info_path.read_text()))
            continue
        with h5py.File(path) as f:
            fs = float(f['craw/adrate'][0, 0])
            raw = f['craw/cnt'][()]
            n, ch = raw.shape
            key = next(k for k in ['trig/anatrig', 'anatrig'] if k in f)
            refs = f[key][()]
            trigs = np.asarray(f[refs[0, 0]][()]).reshape(-1)
            trigger = np.rint(trigs*1000/fs).astype(int)
            nout = int(n*1000/fs)
            good = (trigger >= 10) & (trigger+200 <= nout)
            trigger = trigger[good]
            if not len(trigger):
                raise ValueError(f'No complete epochs: {name}')
            sos = butter(4, [.05, 300], btype='bandpass', fs=fs, output='sos')
            erp = np.empty((ch, 210))
            # Channel-wise processing bounds memory; all filtering remains continuous.
            for j in range(ch):
                x = sosfiltfilt(sos,np.asarray(raw[:,j],dtype=float)*.001)
                x = resample(x, nout) if fs != 1000 else x
                erp[j] = np.mean([x[t-10:t+200] for t in trigger], axis=0)
        csd = -np.diff(erp, n=2, axis=0)/.1**2
        aligned = align(csd, [row['Supra Ch'], row['Gran Ch'], row['Infra Ch']])
        assert aligned.shape == (30, 210) and np.isfinite(aligned).all()

        info = {'file': name,
                'source_fs_hz': fs,
                'n_epochs': int(len(trigger)),
                'excluded_epochs': int((~good).sum()),
                'anchors_as_in_existing_code': [int(row[k]) for k in ['Supra Ch', 'Gran Ch', 'Infra Ch']]
                }
        np.save(cache, aligned)
        info_path.write_text(json.dumps(info, indent=2))
        arrays.append(aligned)
        records.append(info)
        print(f'{len(arrays)}/{len(rows)} {name}: {len(trigger)} epochs', flush=True)

    X = np.stack(arrays)
    pca = PCA(n_components=5, svd_solver='full').fit(X.reshape(len(X), -1))
    template = pca.components_[0].reshape(30,210)
    old = np.load(root/'aligned_30_erp.npy')
    corr = np.corrcoef(template[:, 10:].ravel(), old.ravel())[0, 1]
    sign = 1 if corr >= 0 else -1
    template *= sign
    np.save(out/'aligned_30_erp_prestim10.npy', template)
    np.save(out/'times_ms.npy', np.arange(-10, 200, dtype=float))
    np.savez(out/'aligned_30_erp_prestim10.npz', csd=template, times_ms=np.arange(-10, 200), depth_bins=np.arange(30))
    meta = {'time_window_ms': [-10, 200],
            'endpoint': 'exclusive',
            'sampling_rate_hz': 1000,
            'shape': list(template.shape),
            'method': 'PC1 of centered flattened aligned recording ERPs; not the mean',
            'filter_hz': [.05, 100],
            'filter_order': 4,
            'filter': 'continuous zero-phase Butterworth SOS',
            'baseline_subtraction': False,
            'spacing_um': 100,
            'anchor_convention': 'unchanged from csd_alignment.py; metadata indices used directly',
            'pc1_explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
            'sign_reference': 'aligned_30_erp.npy poststimulus correlation',
            'poststimulus_correlation_old': float(abs(corr)),
            'old_template_sha256': hashlib.sha256((root/'aligned_30_erp.npy').read_bytes()).hexdigest(),
            'recordings': records}

    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    v = np.max(abs(template))
    im = ax.imshow(template, extent=[-10.5, 199.5, 29.5, -.5], aspect='auto', interpolation='nearest', cmap='RdBu', vmin=-v, vmax=v)
    ax.axvline(0, color='k', ls='--', lw=1)
    for y, label in [(7, 'Supra anchor'), (15, 'Granular anchor'), (22, 'Infra anchor')]:
        ax.axhline(y, color='gray', ls=':', lw=.8)
        ax.text(202, y, label, va='center', fontsize=9)
    ax.set(xlabel='Time relative to stimulus (ms)', ylabel='Aligned depth bin', title='30-channel CSD PC1 template: 10 ms before, 200 ms after onset')
    fig.colorbar(im, ax=ax, pad=.19, label='PC1 loading (arbitrary units; negative = sink)')
    fig.savefig(out/'aligned_30_erp_prestim10.png', dpi=180)
    print(json.dumps({k: v for k, v in meta.items() if k != 'recordings'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
