"""Compare interrupted/resumed training against uninterrupted training on allocated GPU."""
import json
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from baseline_core import sha

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated node required')
    work=Path('/home/users/sthummala2/brset-convnextv2/research/codex/step2')
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='smoke');args=parser.parse_args()
    paths=[work/(args.prefix+'_full'),work/(args.prefix+'_resume')]
    states=[torch.load(p/'last.pth',map_location='cpu',weights_only=False) for p in paths]
    mismatches=[]
    for field in ['model','ema']:
        for key in states[0][field]:
            if not torch.equal(states[0][field][key],states[1][field][key]):mismatches.append(field+':'+key)
    for key in states[0]['optimizer']['state']:
        for field,a in states[0]['optimizer']['state'][key].items():
            b=states[1]['optimizer']['state'][key][field]
            if torch.is_tensor(a) and not torch.equal(a,b):mismatches.append(f'optimizer:{key}:{field}')
    for field in ['consumed_draws','phase_update','global_update','exposure','best']:
        if states[0][field]!=states[1][field]:mismatches.append(field)
    assert states[0]['global_update']==6
    for p in paths:
        if (p/'assessment_predictions.npz').exists():raise AssertionError('Smoke touched assessment')
    events=[json.loads(s) for s in (paths[0]/'events.jsonl').read_text().splitlines()]
    secs=[x['seconds'] for x in events if x['event']=='update' and x['global_update']>1]
    evalsecs=[x['inference_seconds'] for x in events if x['event']=='selection']
    completion=json.loads((paths[0]/'complete.json').read_text())
    result={'pass':not mismatches,'state_mismatches':mismatches,'updates':6,'effective_batch':64,
        'interruption_after_update':3,'assessment_accessed':False,'median_update_seconds':float(np.median(secs)),
        'median_selection_seconds':float(np.median(evalsecs)),
        'estimated_baseline_hours':float((np.median(secs)*5775+np.median(evalsecs)*26+np.median(completion['checkpoint_write_seconds'])*25)/3600),
        'gpu_peak_allocated_gib':completion['max_gpu_bytes']/1024**3,
        'checkpoint_hashes':{p.name:sha(p/'last.pth') for p in paths},
        'scope':'Short deterministic smoke including real images and full selection inference. Not convergence, performance or full-run stability evidence.'}
    result['resume_checkpoint_cleanup']='Successful smoke last.pth files removed after verification; hashes retained; best.pth retained.' if not mismatches else 'retained for diagnosis'
    (work/(args.prefix+'_checks.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if mismatches:raise SystemExit(1)
    for p in paths:(p/'last.pth').unlink()

if __name__=='__main__':main()
