"""Run in an editor: paired 1-ms versus 5-ms held-out short-SOA mean benchmark."""
from pathlib import Path
import sys,json,time
import numpy as np
HELPERS=Path('/Users/scoot/dev/csd_quant/qc_workflow')
sys.path.insert(0,str(HELPERS))
from validate_wd import ShapeWD,quantiles
from qc_templates import write_csv,save_json
RUN=HELPERS/'csd_channel_interpretation/qc_results/run_20260923_csd_one'
OUTPUT=Path(__file__).resolve().parent/'results'
GROUP='short'
def main():
 OUTPUT.mkdir(exist_ok=True)
 with np.load(RUN/GROUP/'pca_details.npz') as z:X=z['erps'];names=z['filenames'].tolist();halves=z['halves']
 rows=[]
 for bin_ms in (5,1):
  metric=ShapeWD(bin_ms)
  for i,name in enumerate(names):
   for kind,a,b in [('held_out_mean',X[i],np.delete(X,i,axis=0).mean(0)),('split_half',halves[i,0],halves[i,1])]:
    start=time.monotonic();r=dict(file=name,comparison=kind,time_bin_ms=bin_ms,**metric(a,b),elapsed_s=time.monotonic()-start);rows.append(r)
    write_csv(OUTPUT/'distances.csv',rows)
    print(json.dumps(r),flush=True)
  del metric
 summary={kind:{str(ms):quantiles([r['wd'] for r in rows if r['comparison']==kind and r['time_bin_ms']==ms]) for ms in (1,5)} for kind in ('held_out_mean','split_half')}
 from scipy.stats import spearmanr
 for kind in summary:
  a=np.array([r['wd'] for r in rows if r['comparison']==kind and r['time_bin_ms']==1]);b=np.array([r['wd'] for r in rows if r['comparison']==kind and r['time_bin_ms']==5])
  summary[kind]['paired_1ms_minus_5ms']=quantiles(a-b)
  summary[kind]['spearman']=float(spearmanr(a,b).statistic)
 save_json(OUTPUT/'summary.json',summary)
 save_json(OUTPUT/'manifest.json',dict(status='complete',source=str(RUN),group=GROUP,window_ms=[0,200],depth_bins=30,solver='exact balanced EMD',notes='Short-SOA mean held out by recording; one recording per inferred pair within this SOA. Split-half same-recording benchmark. No simulation ranking tested.'))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
