"""Assess the completed three-arm batch, then report paired patient uncertainty."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from baseline_core import LABELS,metrics,sha
from train_baseline import assess,node_check,write_json

def patient_indices(ids,rng):
    patients=np.unique(ids)
    groups={p:np.flatnonzero(ids==p) for p in patients}
    return np.concatenate([groups[p] for p in rng.choice(patients,len(patients),replace=True)])

def compare(batch,output):
    arrays={arm:np.load(Path(folder)/'assessment_predictions.npz',allow_pickle=False) for arm,folder in batch.items()}
    reference=arrays['B0']
    for arm,data in arrays.items():
        for key in ['domain','file_id','patient_id','role','y_true']:
            if not np.array_equal(reference[key],data[key]):raise ValueError('Paired prediction identities do not match: '+arm+'/'+key)
    thresholds={arm:json.loads(str(data['metadata_json']))['thresholds'] for arm,data in arrays.items()}
    observed={arm:metrics(data['y_true'],data['probabilities'],thresholds[arm]) for arm,data in arrays.items()}
    pairs=[('B1','B0'),('B2','B0'),('B2','B1')]
    fields=['f1_positive','auroc','average_precision','sensitivity','specificity']
    samples={f'{a}-{b}':{label:{k:[] for k in fields} for label in LABELS} for a,b in pairs}
    rng=np.random.default_rng(20260911)
    for _ in range(2000):
        idx=patient_indices(reference['patient_id'],rng)
        current={arm:metrics(data['y_true'][idx],data['probabilities'][idx],thresholds[arm]) for arm,data in arrays.items()}
        for a,b in pairs:
            for label in LABELS:
                for key in fields:
                    x,y=current[a][label][key],current[b][label][key]
                    samples[f'{a}-{b}'][label][key].append(None if x is None or y is None else x-y)
    differences={}
    for a,b in pairs:
        name=f'{a}-{b}';differences[name]={}
        for label in LABELS:
            differences[name][label]={}
            for key in fields:
                draws=samples[name][label][key];valid=[x for x in draws if x is not None]
                x,y=observed[a][label][key],observed[b][label][key]
                differences[name][label][key]={'difference':None if x is None or y is None else x-y,
                    'percentile_95':np.quantile(valid,[.025,.975]).tolist() if valid else None,
                    'valid_draws':len(valid),'undefined_draws':2000-len(valid)}
    result={'status':'seed-0 development assessment, not replicated or external evidence',
        'observed':observed,'paired_patient_differences':differences,'bootstrap_seed':20260911,
        'thresholds_fixed':thresholds,'images':len(reference['y_true']),'patients':len(np.unique(reference['patient_id'])),
        'prediction_sha256':{arm:sha(Path(folder)/'assessment_predictions.npz') for arm,folder in batch.items()},
        'limitations':'Intervals condition on selected trained models; do not account for all training or method-selection uncertainty.'}
    write_json(output,result)
    text=['**Seed-0 baseline development assessment**','','These are new development results, not a replicated gain or an untouched external test.','',
          '| Arm | DR positive F1 | ME positive F1 | DR AUROC | ME AUROC |','|---|---:|---:|---:|---:|']
    for arm,m in observed.items():
        text.append(f"| {arm} | {m[LABELS[0]]['f1_positive']:.4f} | {m[LABELS[1]]['f1_positive']:.4f} | {m[LABELS[0]]['auroc']:.4f} | {m[LABELS[1]]['auroc']:.4f} |")
    text+=['','B0: target-only; B1: joint; B2: source then target. Same total updates; domain exposure and staged learning rates differ as prespecified.',
        '','See the JSON for thresholds, both-label outcomes, paired patient intervals and undefined-draw counts. No winner is declared from one seed.']
    Path(output).with_suffix('.md').write_text('\n'.join(text)+'\n')

def main():
    node_check(True)
    parser=argparse.ArgumentParser();parser.add_argument('--batch-gate',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();batch=json.loads(Path(args.batch_gate).read_text())
    for arm,folder in batch.items():
        if (Path(folder)/'assessment_predictions.npz').exists():raise RuntimeError('Assessment already exists; do not silently rerun')
        assess(SimpleNamespace(output=folder,batch_gate=args.batch_gate,seed=0,workers=4))
    compare(batch,args.output)

if __name__=='__main__':main()
