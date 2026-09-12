import csv,json,os,hashlib
from pathlib import Path
from baseline_core import validate_manifest,sha,LABELS
R=Path('/home/users/sthummala2/brset-convnextv2'); W=R/'research/codex/step2/full_pool';P=W/'protocol'
assert os.environ.get('SLURM_JOB_ID')
old=list(csv.DictReader((R/'research/codex/protocol/development_manifest_v1.csv').open()))
roots={d:Path(next(x['image_path'] for x in old if x['domain']==d)).parent for d in ['BRSET','mBRSET']}
rows=[];provenance={}
for domain,filename,pid in [('BRSET','splits_multilabel.csv','patient_id'),('mBRSET','splits_mbrset_multilabel.csv','patient')]:
 source=R/'results'/filename;provenance[str(source)]=sha(source)
 for x in csv.DictReader(source.open()):
  # Source held-outs are excluded: all selection and assessment is target-domain.
  if domain=='BRSET' and x['split']!='train':continue
  rows.append(dict(domain=domain,file=x['file'],patient_key=domain+':'+x[pid],diabetic_retinopathy=int(x[LABELS[0]]),macular_edema=int(x[LABELS[1]]),original_split=x['split'],fold=-1,role={'train':'fit','val':'selection','test':'assessment'}[x['split']],image_path=str(roots[domain]/x['file'])))
path=P/'full_pool_manifest.csv'
with path.open('w') as f:
 wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
d=validate_manifest(path,sha(path));counts=[]
for (domain,role),g in d.groupby(['domain','role']):
 assert g.image_path.map(lambda x:Path(x).is_file()).all()
 assert all(g[k].nunique()==2 for k in LABELS)
 counts.append(dict(domain=domain,role=role,images=len(g),patients=g.patient_key.nunique(),positive_images={k:int(g[k].sum()) for k in LABELS},positive_patients={k:int(g.groupby('patient_key')[k].max().sum()) for k in LABELS}))
assert {(x['domain'],x['role']):x['images'] for x in counts}=={('BRSET','fit'):11372,('mBRSET','fit'):3402,('mBRSET','selection'):725,('mBRSET','assessment'):732}
p=json.loads((R/'research/codex/protocol/v1.json').read_text());p.update(version='2.0-full-original-pools',status='Full original training pools; supersedes reduced-pool v1 for Step 2',authoritative_specification='FULL_POOL_PROTOCOL.md',manifest=path.name,manifest_sha256=sha(path),roles={'fit':'original train','selection':'original mBRSET val','assessment':'original mBRSET test'},split_seed=None)
p['historical_test_limitation']='Original test used in previous project experiments; not pristine confirmation. No checkpoint, threshold or hyperparameter tuning on test in this batch.'
p['step_2_execution_checks']=['original split-role and patient separation','decoded duplicates including validation/test','CPU checks','GPU resume smoke','frozen code/config hashes']
(P/'v1.json').write_text(json.dumps(p,indent=2)+'\n')
(W/'preflight.json').write_text(json.dumps(dict(pass_checks=True,counts=counts,source_manifest_sha256=provenance,manifest_sha256=sha(path)),indent=2)+'\n')
print(json.dumps(counts,indent=2))
