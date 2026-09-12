"""Allocated-node-only baseline trainer. Assessment requires a separate command."""
import argparse
import copy
import json
import os
import random
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from baseline_core import (LABELS,DrawBatches,Images,sha,stream_seed,validate_manifest,
    mixup,focal,learning_rate,select_thresholds,metrics,flip_probabilities,phase_specs)

ROOT=Path(os.environ.get('BRSET_REPO','/home/users/sthummala2/brset-convnextv2'))
PROTOCOL=ROOT/'research/codex/step2/full_pool/protocol'
WORK=ROOT/'research/codex/step2/full_pool'

def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def save(path,state):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)

def node_check(gpu=False):
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('Compute is restricted to an allocated Slurm compute node')
    if gpu and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Protocol requires an allocated bf16-capable GPU')

def setup(seed):
    torch.set_num_threads(4)
    torch.manual_seed(stream_seed(seed,'model'));np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}

def restore_rng(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy']);random.setstate(s['python'])

def protocol():
    p=json.loads((PROTOCOL/'v1.json').read_text())
    d=validate_manifest(PROTOCOL/p['manifest'],p['manifest_sha256'])
    return p,d

def initialize(seed):
    node_check();setup(seed);p,_=protocol()
    folder=WORK/'initializations';folder.mkdir(exist_ok=True)
    path=folder/f'seed{seed}.pth';meta=folder/f'seed{seed}.json'
    if path.exists():
        m=json.loads(meta.read_text())
        if sha(path)!=m['sha256']:raise ValueError('Initialization hash mismatch')
        return path
    if sha(p['initialization']['weight_path'])!=p['initialization']['weight_sha256']:
        raise ValueError('External weight hash mismatch')
    torch.manual_seed(stream_seed(seed,'initialization'))
    model=timm.create_model(p['initialization']['model'],pretrained=False,num_classes=2,drop_path_rate=.3)
    weights=load_file(p['initialization']['weight_path'],device='cpu')
    weights.pop('head.fc.weight');weights.pop('head.fc.bias')
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if set(missing)!={'head.fc.weight','head.fc.bias'} or unexpected:raise ValueError((missing,unexpected))
    save(path,model.state_dict())
    write_json(meta,{'seed':seed,'sha256':sha(path),'external_weight_sha256':p['initialization']['weight_sha256'],
                     'initialization_stream_seed':stream_seed(seed,'initialization'),'model':p['initialization']['model']})
    return path

def loader(rows,training,seed,workers,start=0,stop=0):
    ds=Images(rows,training,seed)
    kwargs={'num_workers':workers,'pin_memory':True,'generator':torch.Generator().manual_seed(stream_seed(seed,'loader'))}
    if training:return DataLoader(ds,batch_sampler=DrawBatches(len(ds),16,start,stop,seed),**kwargs)
    return DataLoader(ds,batch_size=16,shuffle=False,**kwargs)

@torch.inference_mode()
def infer(model,rows,workers):
    model.eval();ys=[];ps=[];zs=[];ids=[];start=time.monotonic()
    for x,y,idx in loader(rows,False,0,workers):
        x=x.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):p,z=flip_probabilities(model,x)
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite inference')
        ys.append(y.numpy());ps.append(p.cpu().numpy());zs.append(z.cpu().numpy());ids.extend(idx.tolist())
    if ids!=list(range(len(rows))):raise ValueError('Prediction identity/order mismatch')
    torch.cuda.synchronize()
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(zs),time.monotonic()-start

def save_predictions(path,rows,y,p,z,metadata):
    # Store view logits: average sigmoid is not sigmoid of average logits.
    np.savez_compressed(path,domain=rows.domain.to_numpy(dtype=str),file_id=rows.file.to_numpy(dtype=str),
        patient_id=rows.patient_key.to_numpy(dtype=str),role=rows.role.to_numpy(dtype=str),
        y_true=y,probabilities=p,view_logits=z,metadata_json=np.array(json.dumps(metadata)))

def contract(args,p,init):
    return {'protocol_sha256':sha(PROTOCOL/'v1.json'),'manifest_sha256':p['manifest_sha256'],
        'trainer_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('baseline_core.py')),
        'initialization_sha256':sha(init),'arm':args.arm,'seed':args.seed,'smoke':args.smoke,
        'torch':torch.__version__,'timm':timm.__version__,'phases':phase_specs(args.arm,args.smoke)}

def train(args):
    node_check(True);setup(args.seed);p,d=protocol()
    integrity=json.loads((WORK/'image_integrity_summary.json').read_text())
    if not integrity['pass'] or integrity['manifest_sha256']!=p['manifest_sha256']:
        raise ValueError('Image integrity/duplicate gate failed')
    init=initialize(args.seed);setup(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=contract(args,p,init)
    if (out/'contract.json').exists():
        if json.loads((out/'contract.json').read_text())!=cfg:raise ValueError('Run contract changed')
        if not args.resume:raise ValueError('Existing run requires explicit --resume')
    else:write_json(out/'contract.json',cfg)
    model=timm.create_model(p['initialization']['model'],pretrained=False,num_classes=2,drop_path_rate=.3).cuda()
    model.load_state_dict(torch.load(init,map_location='cpu',weights_only=True))
    ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    selection=d[(d.role=='selection') & (d.domain=='mBRSET')].reset_index(drop=True)
    phases=phase_specs(args.arm,args.smoke)
    resume=None
    if args.resume:
        resume=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        if resume['contract']!=cfg:raise ValueError('Checkpoint contract mismatch')
        model.load_state_dict(resume['model']);ema.load_state_dict(resume['ema'])
        # Also restore at a completed phase boundary, where its loop is skipped.
        restore_rng(resume['rng'])
    best=-1.0 if resume is None else resume['best']
    global_update=0 if resume is None else resume['global_update']
    completed_phase=-1 if resume is None else resume['phase']
    log=open(out/'events.jsonl','a',buffering=1)
    def event(kind,**data):
        record={'event':kind,'time':time.time(),'global_update':global_update,**data}
        log.write(json.dumps(record,allow_nan=False)+'\n');print(json.dumps(record),flush=True)
    event('start',host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),resume=args.resume)
    checkpoint_seconds=[]
    for phase_idx,phase in enumerate(phases):
        if phase_idx<completed_phase:continue
        start_update=resume['phase_update'] if resume and phase_idx==completed_phase else 0
        if start_update==phase['updates']:continue
        entering_new_phase=phase_idx>0 and start_update==0
        if entering_new_phase:
            ema.load_state_dict(model.state_dict());best=-1.0
        optimizer=torch.optim.AdamW(model.parameters(),lr=phase['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.1)
        if resume and phase_idx==completed_phase:
            optimizer.load_state_dict(resume['optimizer']);restore_rng(resume['rng'])
        rows=d[d.role=='fit']
        if phase['domain']!='both':rows=rows[rows.domain==phase['domain']]
        rows=rows.reset_index(drop=True)
        draw_seed=stream_seed(args.seed,'phase_'+phase['name'])
        data=iter(loader(rows,True,draw_seed,args.workers,start_update*64,phase['updates']*64))
        durations=[];exposure=np.zeros((2,3),dtype=np.int64)
        if resume and phase_idx==completed_phase:exposure=np.asarray(resume['exposure'])
        event('phase',phase=phase['name'],fit_images=len(rows),phase_start_update=start_update)
        def checkpoint(u):
            t=time.monotonic()
            save(out/'last.pth',{'contract':cfg,'model':model.state_dict(),'ema':ema.state_dict(),
                'optimizer':optimizer.state_dict(),'scheduler':{'phase':phase,'next_update':u+1},
                'rng':rng_state(),'phase':phase_idx,'phase_update':u,'global_update':global_update,
                'consumed_draws':u*64,'exposure':exposure.tolist(),'best':best})
            checkpoint_seconds.append(time.monotonic()-t)
        for u in range(start_update+1,phase['updates']+1):
            torch.cuda.synchronize();t=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(u,phase['updates'],phase['warmup'],phase['lr'])
            for group in optimizer.param_groups:group['lr']=lr
            loss_sum=0.
            for micro in range(4):
                x,y,idx=next(data)
                sampled=rows.iloc[idx.tolist()]
                for di,dom in enumerate(['BRSET','mBRSET']):
                    sub=sampled[sampled.domain==dom]
                    exposure[di] += [len(sub),int(sub[LABELS[0]].sum()),int(sub[LABELS[1]].sum())]
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                x,y=mixup(x,y,draw_seed,(u-1)*4+micro)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=focal(model(x),y)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {u}/{micro}')
                (loss/4).backward();loss_sum+=loss.item()/4
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.state_dict().values(),model.state_dict().values()):
                    if a.is_floating_point():a.mul_(.999).add_(b,alpha=.001)
                    else:a.copy_(b)
            global_update+=1;torch.cuda.synchronize();elapsed=time.monotonic()-t;durations.append(elapsed)
            if args.smoke or u%25==0 or u==1:
                event('update',phase=phase['name'],phase_update=u,loss=loss_sum,lr=lr,
                      gradient_norm=float(grad_norm),seconds=elapsed,draws=u*64)
            if u in phase['eval_at']:
                y,pred,z,seconds=infer(ema,selection,args.workers)
                scored=metrics(y,pred,[.5,.5]);score=float(np.mean([scored[k]['auroc'] for k in LABELS]))
                if score>best:
                    best=score
                    save(out/'best.pth',{'model':ema.state_dict(),'contract':cfg,'global_update':global_update,
                        'phase':phase['name'],'selection_score':score})
                event('selection',score=score,best=best,inference_seconds=seconds,metrics_at_05=scored)
            if u%231==0 or u==phase['updates'] or (args.stop_after and global_update>=args.stop_after):checkpoint(u)
            if args.stop_after and global_update>=args.stop_after:
                event('paused',reason='explicit checkpoint/resume verification');log.close();return
        event('phase_complete',phase=phase['name'],exposure=exposure.tolist(),
              median_update_seconds=float(np.median(durations)) if durations else None)
        resume=None
    if not (out/'best.pth').exists():raise RuntimeError('No selected checkpoint')
    chosen=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    ema.load_state_dict(chosen['model']);y,pred,z,seconds=infer(ema,selection,args.workers)
    thresholds=select_thresholds(y,pred)
    meta={**cfg,'checkpoint_sha256':sha(out/'best.pth'),'thresholds':thresholds,'checkpoint_update':chosen['global_update']}
    save_predictions(out/'selection_predictions.npz',selection,y,pred,z,meta)
    write_json(out/'selection_summary.json',{'metadata':meta,'metrics':metrics(y,pred,thresholds),
        'warning':'Selection metrics are tuned development diagnostics, not assessment results.'})
    write_json(out/'complete.json',{'training_complete':True,'smoke':args.smoke,'updates':global_update,
        'selection_inference_seconds':seconds,'max_gpu_bytes':torch.cuda.max_memory_allocated(),
        'checkpoint_write_seconds':checkpoint_seconds,'contract':cfg})
    event('complete',assessment_performed=False);log.close()

def assess(args):
    node_check(True);setup(args.seed);p,d=protocol();out=Path(args.output)
    if not (out/'complete.json').exists():raise ValueError('Training incomplete')
    complete=json.loads((out/'complete.json').read_text())
    if complete['smoke']:raise ValueError('Smoke run cannot provide research assessment')
    batch=json.loads(Path(args.batch_gate).read_text())
    # A separate predeclared batch gate ensures all three baseline runs finished first.
    if set(batch)!= {'B0','B1','B2'}:raise ValueError('Baseline assessment gate must contain B0/B1/B2')
    for arm,folder in batch.items():
        other=json.loads((Path(folder)/'complete.json').read_text())
        if other['smoke'] or not other['training_complete'] or other['contract']['arm']!=arm:raise ValueError('Invalid batch gate')
        for key in ['protocol_sha256','manifest_sha256','trainer_sha256','core_sha256','initialization_sha256','seed']:
            if other['contract'][key]!=complete['contract'][key]:raise ValueError('Mismatched assessment batch: '+key)
    summary=json.loads((out/'selection_summary.json').read_text());meta=summary['metadata']
    if meta['manifest_sha256']!=p['manifest_sha256'] or meta['checkpoint_sha256']!=sha(out/'best.pth'):raise ValueError('Assessment provenance mismatch')
    rows=d[(d.role=='assessment')&(d.domain=='mBRSET')].reset_index(drop=True)
    model=timm.create_model(p['initialization']['model'],pretrained=False,num_classes=2,drop_path_rate=.3).cuda()
    model.load_state_dict(torch.load(out/'best.pth',map_location='cpu',weights_only=False)['model'])
    y,pred,z,seconds=infer(model,rows,args.workers)
    save_predictions(out/'assessment_predictions.npz',rows,y,pred,z,meta)
    write_json(out/'assessment_summary.json',{'metadata':meta,'metrics':metrics(y,pred,meta['thresholds']),
        'metrics_at_05':metrics(y,pred,[.5,.5]),'seconds':seconds,'interpretation':'Development assessment; not independent external confirmation'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['init','train','assess'])
    ap.add_argument('--arm',choices=['B0','B1','B2'],default='B1');ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--output');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--stop-after',type=int);ap.add_argument('--batch-gate')
    args=ap.parse_args()
    if args.command=='init':print(initialize(args.seed));return
    if not args.output:ap.error('--output required')
    if args.command=='assess' and not args.batch_gate:ap.error('--batch-gate required')
    try:train(args) if args.command=='train' else assess(args)
    except Exception as exc:
        out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
        write_json(out/f'failure_{int(time.time())}.json',{'error':repr(exc),'job':os.environ.get('SLURM_JOB_ID')})
        raise

if __name__=='__main__':main()
