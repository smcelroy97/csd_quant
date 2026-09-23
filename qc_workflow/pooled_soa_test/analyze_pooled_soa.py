"""Run directly in a Python editor. Editable settings; no command-line arguments."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
# 1. Configuration and existing, verified helper functions.
HELPERS=Path('/Users/scoot/dev/csd_quant/qc_workflow/normalized_mean_test')
sys.path.insert(0,str(HELPERS))
from test_paper_pca import paper_pca
from test_normalized_mean import mean_shape
from qc_templates import corr,align_laminar,save_json,write_csv
from validate_wd import ShapeWD,quantiles
RUN=Path('/Users/scoot/dev/csd_quant/qc_workflow/csd_channel_interpretation/qc_results/run_20260923_csd_one')
OUTPUT=Path(__file__).resolve().parent/'results'
DRAWS=100
SEED=314159

def targets(X):return {'mean':X.mean(0),'normalized_mean':mean_shape(X),'paper_pc1':paper_pca(X)[0]}
def similarity(a,b,kind):return abs(corr(a,b)) if kind=='paper_pc1' else corr(a,b)

def main():
    # 2. Pool retained recording means, giving each recording equal initial weight.
    arrays=[];names=[];groups=[]
    for g in ('short','long'):
        with np.load(RUN/g/'pca_details.npz') as z:
            arrays.append(z['erps']);names.extend(z['filenames'].tolist());groups.extend([g]*len(z['erps']))
    X=np.concatenate(arrays);groups=np.array(groups)
    pairs=np.array([n.split('@')[0][:-3] for n in names]);unique=np.unique(pairs)
    assert all(sorted(groups[pairs==p])==['long','short'] for p in unique)
    OUTPUT.mkdir(exist_ok=True);full=targets(X);_,evr,weights=paper_pca(X)
    for kind,a in full.items():
        np.save(OUTPUT/f'{kind}_pooled.npy',a)
        np.savez(OUTPUT/f'{kind}_pooled.npz',csd=a,times_ms=np.arange(200),depth_bins=np.arange(30),metadata_json=json.dumps(dict(source=str(RUN),kind=kind,filenames=names,soa_labels=groups.tolist(),weighting='one trial-averaged map per recording',anchor_convention='csd-one')))
    np.savez(OUTPUT/'pca_details.npz',erps=X,weights=weights,explained_variance_ratio=evr,filenames=names,pair_ids=pairs)
    # 3. Leave one recording out, then leave both inferred pair members out.
    metric=ShapeWD(5);rows=[];stability=[]
    for scheme in ('recording','pair'):
        identifiers=range(len(X)) if scheme=='recording' else unique
        for identifier in identifiers:
            held=np.arange(len(X))==identifier if scheme=='recording' else pairs==identifier
            train=X[~held];fit=targets(train)
            assert len(train)==(25 if scheme=='recording' else 24)
            for kind,t in fit.items():
                stability.append(dict(scheme=scheme,held=str(identifier),target=kind,correlation=similarity(full[kind],t,kind)))
                for i in np.flatnonzero(held):rows.append(dict(scheme=scheme,file=names[i],group=groups[i],pair=pairs[i],target=kind,**metric(X[i],t)))
        print('Completed',scheme,'validation',flush=True)
    write_csv(OUTPUT/'held_out_wd.csv',rows);write_csv(OUTPUT/'leave_out_stability.csv',stability)
    # 4. Bootstrap recording maps, then bootstrap complete pairs to preserve dependence.
    bootstrap={};rng=np.random.default_rng(SEED)
    for scheme in ('recording','pair'):
        values={k:[] for k in full}
        for _ in range(DRAWS):
            ids=rng.integers(0,len(X),len(X)) if scheme=='recording' else np.concatenate([np.flatnonzero(pairs==p) for p in rng.choice(unique,len(unique))])
            fit=targets(X[ids])
            for kind in full:values[kind].append(similarity(full[kind],fit[kind],kind))
        bootstrap[scheme]={k:quantiles(v) for k,v in values.items()}
    # 5. Independent per-recording ±1-contact landmark jitter (matches earlier diagnostics).
    physical=[];anchors=[]
    for name in names:
        with np.load(RUN/'recordings'/Path(name).stem/'erp.npz') as z:physical.append(z['physical_after'][:,z['times_ms']>=0]);anchors.append(z['anchors_csd'])
    rng=np.random.default_rng(SEED+1);jitter={k:[] for k in full}
    for _ in range(DRAWS):
        maps=[]
        for p,a in zip(physical,anchors):
            for attempt in range(100):
                shifted=a+rng.integers(-1,2,3)
                if np.all(np.diff([0,*shifted,p.shape[0]-1])>0):break
            else:raise ValueError('Invalid landmark jitter')
            maps.append(align_laminar(p,shifted))
        fit=targets(np.stack(maps))
        for kind in full:jitter[kind].append(similarity(full[kind],fit[kind],kind))
    # 6. Summarize by held-out SOA as well as pooled, and compare separate templates.
    summary=dict(n_recordings=len(X),n_inferred_pairs=len(unique),pc1_explained_variance=evr,bootstrap=bootstrap,landmark_jitter={k:quantiles(v) for k,v in jitter.items()},validation={},stability={},separate_template_correlations={})
    for scheme in ('recording','pair'):
        summary['validation'][scheme]={g:{k:quantiles([r['wd'] for r in rows if r['scheme']==scheme and r['target']==k and (g=='all' or r['group']==g)]) for k in full} for g in ('all','short','long')}
        summary['stability'][scheme]={k:dict(minimum=min(r['correlation'] for r in stability if r['scheme']==scheme and r['target']==k),**quantiles([r['correlation'] for r in stability if r['scheme']==scheme and r['target']==k])) for k in full}
    for g,A in zip(('short','long'),arrays):
        fit=targets(A);summary['separate_template_correlations'][g]={k:similarity(full[k],fit[k],k) for k in full}
    save_json(OUTPUT/'summary.json',summary)
    # 7. Plot pooled and separate targets on independent display scales.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
    for row,(label,A) in enumerate(zip(('short','long','pooled'),[*arrays,X])):
        for col,(kind,a) in enumerate(targets(A).items()):
            ax=axes[row,col];ax.imshow(a/abs(a).max(),cmap='RdBu',vmin=-1,vmax=1,aspect='auto',extent=[0,200,29,0]);ax.set_title(label+': '+kind);ax.set_xlabel('Time (ms)');ax.set_ylabel('Aligned depth bin')
            for d in (7,15,22):ax.axhline(d,color='k',ls=':',lw=.5)
    fig.suptitle('Each panel scaled to its own peak; red = sink, blue = source')
    fig.savefig(OUTPUT/'pooled_comparison.png',dpi=150);plt.close(fig)
    save_json(OUTPUT/'manifest.json',dict(status='complete',source=str(RUN),source_manifest_sha256=hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),draws=DRAWS,seed=SEED,pairing='filename-derived, not confirmed animal IDs',wd_time_bin_ms=5))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
