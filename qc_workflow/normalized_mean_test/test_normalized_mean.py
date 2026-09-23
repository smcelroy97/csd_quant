"""Editor-runnable normalized-average evaluation. Edit configuration below; no CLI."""
from pathlib import Path
import json, hashlib
import numpy as np
from qc_templates import align_laminar, corr, save_json, write_csv
from validate_wd import ShapeWD, quantiles
# 1. Configuration: reuse completed QC and CSD-channel anatomical interpretation.
RUN=Path('/Users/scoot/dev/csd_quant/qc_workflow/csd_channel_interpretation/qc_results/run_20260923_csd_one')
OUTPUT=Path(__file__).resolve().parent/'normalized_mean_results'
DRAWS=100
SEED=314159

def normalize(X):
    norms=np.linalg.norm(X.reshape(len(X),-1),axis=1)
    if np.any(norms<=1e-12) or not np.isfinite(X).all():raise ValueError('Invalid/zero CSD')
    return X/norms[:,None,None]

def mean_shape(X):return normalize(X).mean(0)

def main():
    # 2. Verify normalization: scaling any recording cannot alter the result.
    rng=np.random.default_rng(1); example=rng.normal(size=(4,30,200))
    assert np.allclose(mean_shape(example),mean_shape(example*np.array([1,3,0.2,10])[:,None,None]))
    assert np.allclose(mean_shape(np.repeat(example[:1],4,axis=0)),normalize(example[:1])[0])
    OUTPUT.mkdir(exist_ok=True)
    metric=ShapeWD(5); results={}; plots={}
    for group in ('short','long'):
        dest=OUTPUT/group;dest.mkdir(exist_ok=True)
        with np.load(RUN/group/'pca_details.npz') as z:X=z['erps'];names=z['filenames'].tolist()
        # 3. Normalize each full 0–199-ms map, then average with equal recording weights.
        target=mean_shape(X)
        np.save(dest/f'mean_shape_{group}.npy',target)
        np.savez(dest/f'mean_shape_{group}.npz',csd=target,times_ms=np.arange(200),depth_bins=np.arange(30),metadata_json=json.dumps(dict(normalization='per-recording full-map L2 before arithmetic mean',units='dimensionless',source=str(RUN),filenames=names)))
        # 4. Held-out WD: build every comparison target from the other 12 recordings.
        import csv
        old=list(csv.DictReader((RUN/group/'wd_leave_one_recording_out.csv').open()))
        rows=[];loo=[]
        for i,name in enumerate(names):
            t=mean_shape(np.delete(X,i,axis=0));loo.append(corr(target,t))
            row=dict(file=name,target='mean_shape',**metric(X[i],t),correlation=corr(X[i],t))
            rows.append(row)
        write_csv(dest/'held_out.csv',rows)
        comparisons={}
        for kind in ('mean','median','pc1','pc1_shape'):
            ref={r['file']:float(r['wd']) for r in old if r['target']==kind}
            delta=np.array([r['wd']-ref[r['file']] for r in rows])
            comparisons[kind]=dict(normalized_mean_lower_count=int((delta<0).sum()),n=len(delta),paired_difference_quantiles=quantiles(delta),reference_wd=quantiles(list(ref.values())))
        # 5. Same random draws as existing diagnostics: bootstrap and landmark uncertainty.
        rng=np.random.default_rng(SEED);boot=[]
        for _ in range(DRAWS):boot.append(corr(target,mean_shape(X[rng.integers(0,len(X),len(X))])))
        physical=[];anchors=[]
        for name in names:
            with np.load(RUN/'recordings'/Path(name).stem/'erp.npz') as z:
                physical.append(z['physical_after'][:,z['times_ms']>=0]);anchors.append(z['anchors_csd'])
        rng=np.random.default_rng(SEED+1);jitter=[]
        for _ in range(DRAWS):
            maps=[]
            for p,a in zip(physical,anchors):
                for attempt in range(100):
                    shifted=a+rng.integers(-1,2,3)
                    if np.all(np.diff([0,*shifted,p.shape[0]-1])>0):break
                else:raise ValueError('Invalid landmarks')
                maps.append(align_laminar(p,shifted))
            jitter.append(corr(target,mean_shape(np.stack(maps))))
        results[group]=dict(n=len(X),held_out_wd=quantiles([r['wd'] for r in rows]),comparisons=comparisons,leave_one_out_correlation=quantiles(loo),minimum_leave_one_out_correlation=min(loo),bootstrap_correlation=quantiles(boot),landmark_jitter_correlation=quantiles(jitter),correlation_with_ordinary_mean=corr(target,X.mean(0)),normalization_tests='passed')
        save_json(dest/'diagnostics.json',results[group]);plots[group]=(X.mean(0),target,np.load(RUN/group/f'pc1_{group}.npy'),np.load(RUN/group/f'pc1_shape_{group}.npy'))
        print(group,json.dumps(results[group]),flush=True)
    # 6. Summary plots and provenance; display scale is separate from saved values.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
    for row,g in enumerate(('short','long')):
        for col,(label,a) in enumerate(zip(('ordinary mean','normalized mean','PC1','shape PC1'),plots[g])):
            ax=axes[row,col];ax.imshow(a/abs(a).max(),aspect='auto',cmap='RdBu',vmin=-1,vmax=1,extent=[0,200,29,0]);ax.set_title(g+': '+label);ax.set_xlabel('Time (ms)');ax.set_ylabel('Aligned depth bin')
            for d in (7,15,22):ax.axhline(d,color='k',ls=':',lw=.5)
    fig.suptitle('Each panel scaled to its own peak; red = sink, blue = source')
    fig.savefig(OUTPUT/'template_comparison.png',dpi=160);plt.close(fig)
    save_json(OUTPUT/'summary.json',results)
    save_json(OUTPUT/'manifest.json',dict(source_run=str(RUN),source_manifest_sha256=hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),draws=DRAWS,seed=SEED,wd_time_bin_ms=5,status='complete'))
if __name__=='__main__':main()
