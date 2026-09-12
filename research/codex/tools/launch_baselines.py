"""Lightweight launch gate: verify evidence, freeze source, submit one-GPU array."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path('/home/users/sthummala2/brset-convnextv2')
WORK=ROOT/'research/codex/step2'
TOOLS=ROOT/'research/codex/tools'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    if (WORK/'launch_seed0.json').exists():raise SystemExit('Batch already submitted; inspect its state before any resubmission')
    cpu=read(WORK/'cpu_checks.json');integrity=read(WORK/'image_integrity_summary.json')
    smoke=read(WORK/'smoke_checks.json');phase=read(WORK/'phase_smoke_checks.json')
    if not all(x['pass'] for x in [cpu,integrity,smoke,phase]):raise SystemExit('Execution gate failed')
    if cpu['source_sha256']!=sha(TOOLS/'baseline_core.py'):raise SystemExit('Core changed after CPU tests')
    cfg=read(WORK/'smoke_full/contract.json')
    for folder in ['smoke_resume','phase_smoke_full','phase_smoke_resume']:
        other=read(WORK/folder/'contract.json')
        for key in ['trainer_sha256','core_sha256','protocol_sha256','manifest_sha256','initialization_sha256']:
            if other[key]!=cfg[key]:raise SystemExit('Different sources across smoke checks')
    if cfg['trainer_sha256']!=sha(TOOLS/'train_baseline.py') or cfg['core_sha256']!=sha(TOOLS/'baseline_core.py'):
        raise SystemExit('Trainer changed after smoke tests')
    if cfg['protocol_sha256']!=sha(ROOT/'research/codex/protocol/v1.json'):raise SystemExit('Protocol changed')
    if integrity['manifest_sha256']!=cfg['manifest_sha256']:raise SystemExit('Integrity manifest changed')
    if smoke['estimated_baseline_hours']*1.35>12:raise SystemExit('Runtime plus buffer exceeds job limit; revise resource request first')
    frozen=WORK/'frozen';frozen.mkdir(exist_ok=True)
    sources=['baseline_core.py','train_baseline.py','assess_baselines.py','test_baseline.py','verify_smoke.py']
    for name in sources:shutil.copy2(TOOLS/name,frozen/name)
    (WORK/'runs').mkdir(exist_ok=True)
    batch={arm:str(WORK/'runs'/f'{arm}_seed0') for arm in ['B0','B1','B2']}
    (WORK/'batch_gate_seed0.json').write_text(json.dumps(batch,indent=2)+'\n')
    # Remove only disposable smoke model files, after successful verification.
    # Original research checkpoints and the shared baseline initialization stay intact.
    cleanup=[]
    for folder in ['smoke_full','smoke_resume','phase_smoke_full','phase_smoke_resume']:
        model=WORK/folder/'best.pth'
        if model.exists():
            cleanup.append({'file':str(model),'sha256':read(WORK/folder/'selection_summary.json')['metadata']['checkpoint_sha256']})
            model.unlink()
    free=shutil.disk_usage(WORK/'runs').free
    if free<16*1024**3:raise SystemExit('Need 16 GiB for baseline checkpoints and atomic-write headroom')
    estimate=smoke['estimated_baseline_hours']
    smoke_job=next(json.loads(line)['job'] for line in (WORK/'smoke_full/events.jsonl').read_text().splitlines() if json.loads(line)['event']=='start')
    record={'state':'prepared','source_sha256':{n:sha(frozen/n) for n in sources},
        'protocol_sha256':cfg['protocol_sha256'],'manifest_sha256':cfg['manifest_sha256'],
        'initialization_sha256':cfg['initialization_sha256'],'estimated_hours_per_run':estimate,
        'planning_hours_per_run_with_35pct_buffer':estimate*1.35,'expected_three_run_gpu_hours':estimate*3,
        'gpu_concurrency':1,'gpu_type':'A40','job_limit_hours_each':12,'smoke_job':smoke_job,'smoke_model_cleanup':cleanup,
        'checkpoint_storage':str((WORK/'runs').resolve()),'checkpoint_storage_free_bytes_at_launch':free,
        'tracked_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    # Persist before submission: even a later failure must not hide a submitted job.
    path=WORK/'launch_seed0.json';path.write_text(json.dumps(record,indent=2)+'\n')
    job=subprocess.check_output(['sbatch','--parsable','--kill-on-invalid-dep=yes',f'--dependency=afterok:{smoke_job}',str(WORK/'jobs/baselines.slurm')],text=True).strip().split(';')[0]
    record.update(state='training_submitted',baseline_array_job=job);path.write_text(json.dumps(record,indent=2)+'\n')
    assessment=subprocess.check_output(['sbatch','--parsable','--kill-on-invalid-dep=yes',f'--dependency=afterok:{job}',str(WORK/'jobs/assessment.slurm')],text=True).strip().split(';')[0]
    record.update(state='training_and_gated_assessment_submitted',assessment_job=assessment)
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
