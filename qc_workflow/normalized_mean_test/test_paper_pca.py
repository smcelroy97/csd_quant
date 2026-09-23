"""Rimehaug Figure_2_and_3 PCA orientation, on our existing preprocessing."""
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from test_normalized_mean import RUN, OUTPUT, DRAWS, SEED, mean_shape
from qc_templates import corr, align_laminar, save_json, write_csv
from validate_wd import ShapeWD,quantiles

def paper_pca(X):
    flat=X.reshape(len(X),-1)
    centered=flat-flat.mean(axis=1,keepdims=True)
    val,vec=np.linalg.eigh(centered@centered.T)
    weights=vec[:,-1]
    # Resolve arbitrary PC sign using positive correlation with the training mean.
    target=(weights@flat).reshape(X.shape[1:])
    if corr(target,X.mean(0))<0:weights=-weights;target=-target
    evr=val[-1]/val.sum()
    return target*evr,float(evr),weights

def main():
    # Numerical agreement with sklearn fit(X.T), followed by their reconstruction.
    ex=np.random.default_rng(2).normal(size=(6,30,200));f=ex.reshape(6,-1)
    p=PCA().fit(f.T);a,e,w=paper_pca(ex)
    ref=(p.components_[0]@f).reshape(30,200)*p.explained_variance_ratio_[0]
    assert abs(corr(a,ref))>1-1e-10 and abs(e-p.explained_variance_ratio_[0])<1e-10
    metric=ShapeWD(5);results={}
    for group in ('short','long'):
        with np.load(RUN/group/'pca_details.npz') as z:X=z['erps'];names=z['filenames'].tolist()
        target,evr,weights=paper_pca(X);dest=OUTPUT/group
        np.savez(dest/f'paper_pc1_{group}.npz',csd=target,times_ms=np.arange(200),depth_bins=np.arange(30),weights=weights,explained_variance_ratio=evr,metadata_json=json.dumps(dict(method='PCA fit on flattened_CSD.T; weights @ original flattened_CSD * EVR',sign='positive correlation with training mean',preprocessing='our csd-one QC run; not full replication of paper')))
        rows=[];loo=[]
        for i,name in enumerate(names):
            t,_,_=paper_pca(np.delete(X,i,axis=0));loo.append(abs(corr(target,t)))
            rows.append(dict(file=name,**metric(X[i],t)))
        write_csv(dest/'paper_pc1_held_out.csv',rows)
        rng=np.random.default_rng(SEED);boot=[]
        for _ in range(DRAWS):boot.append(abs(corr(target,paper_pca(X[rng.integers(0,len(X),len(X))])[0])))
        physical=[];anchors=[]
        for name in names:
            with np.load(RUN/'recordings'/Path(name).stem/'erp.npz') as z:physical.append(z['physical_after'][:,z['times_ms']>=0]);anchors.append(z['anchors_csd'])
        rng=np.random.default_rng(SEED+1);jitter=[]
        for _ in range(DRAWS):
            arrays=[]
            for a,p in zip(anchors,physical):
                for attempt in range(100):
                    sh=a+rng.integers(-1,2,3)
                    if np.all(np.diff([0,*sh,p.shape[0]-1])>0):break
                else:raise ValueError('Invalid anchors')
                arrays.append(align_laminar(p,sh))
            jitter.append(abs(corr(target,paper_pca(np.stack(arrays))[0])))
        results[group]=dict(explained_variance=evr,weights=weights.tolist(),held_out_wd=quantiles([r['wd'] for r in rows]),minimum_loo_abs_correlation=min(loo),bootstrap_abs_correlation=quantiles(boot),landmark_jitter_abs_correlation=quantiles(jitter),correlation_with_mean=corr(target,X.mean(0)))
        print(group,results[group],flush=True)
    save_json(OUTPUT/'paper_pca_summary.json',results)
if __name__=='__main__':main()
