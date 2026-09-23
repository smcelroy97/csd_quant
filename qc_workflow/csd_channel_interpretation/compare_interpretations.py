"""Run in an editor after qc_templates.py and validate_wd.py; no CLI options."""
from pathlib import Path
import json
import csv
import numpy as np
from qc_templates import corr
OLD = Path('/Users/scoot/dev/csd_quant/qc_workflow/qc_results/run_20260922_161348')
NEW = Path(__file__).resolve().parent/'qc_results/run_20260923_csd_one'

def main():
    rows=[]
    for group in ('short','long'):
        a=json.loads((OLD/group/'diagnostics.json').read_text())
        b=json.loads((NEW/group/'diagnostics.json').read_text())
        aw=json.loads((OLD/group/'wd_validation.json').read_text())
        bw=json.loads((NEW/group/'wd_validation.json').read_text())
        metrics={
          'PC1 explained variance':lambda d:d['after_qc']['pc1_explained_variance'],
          'PC1 correlation with mean':lambda d:d['after_qc']['pc1_mean_correlation'],
          'Maximum recording influence':lambda d:max(d['after_qc']['pc1_influence']),
          'Minimum leave-one-out PC1 absolute correlation':lambda d:min(d['after_qc']['loo_pc1_abs_correlation']),
          'Median landmark-jitter PC1 absolute correlation':lambda d:d['landmarks']['pc1_abs_correlation_quantiles'][1],
          'Median bootstrap PC1 absolute correlation':lambda d:d['after_qc']['bootstrap_pc1_abs_correlation_quantiles'][1],
        }
        for name,fn in metrics.items():rows.append(dict(group=group,metric=name,raw_one=fn(a),csd_one=fn(b)))
        for name in aw:
            if isinstance(aw[name],dict) and 'median' in aw[name]:
                rows.append(dict(group=group,metric='Median WD: '+name,raw_one=aw[name]['median'],csd_one=bw[name]['median']))
        for kind in ('pc1','mean'):
            print(group,kind,'old/new correlation',corr(np.load(OLD/group/f'{kind}_{group}.npy'),np.load(NEW/group/f'{kind}_{group}.npy')))
    with (NEW/'interpretation_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows: print(row)
    oldqc=list(csv.DictReader((OLD/'recording_qc.csv').open()))
    newqc=list(csv.DictReader((NEW/'recording_qc.csv').open()))
    for a,b in zip(oldqc,newqc):
        for key in ('file','n_kept','n_rejected','bad_contacts_zero','excluded'):assert a[key]==b[key],(key,a,b)
    print('QC decisions identical across all recordings.')
if __name__=='__main__':main()
