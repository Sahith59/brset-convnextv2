"""Compute image/patient prevalence from fixed manifests on a compute node."""
import csv,json,os,hashlib
from pathlib import Path
ROOT=Path('/home/users/sthummala2/brset-convnextv2')
OUT=ROOT/'research/codex/step2'
def summarize(rows,name):
    patients={}
    for r in rows:
        patients[r['patient_key']]=max(patients.get(r['patient_key'],0),int(r['diabetic_retinopathy']))
    n=len(rows);p=len(patients);positive=sum(int(r['diabetic_retinopathy']) for r in rows);pp=sum(patients.values())
    return {'group':name,'images':n,'DR_positive_images':positive,'DR_positive_images_pct':100*positive/n,
            'patients':p,'patients_with_DR_positive_image':pp,'patients_with_DR_positive_image_pct':100*pp/p}
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use allocated compute node')
    path=ROOT/'research/codex/protocol/development_manifest_v1.csv'
    rows=list(csv.DictReader(path.open()));current=[];legacy=[]
    for domain in ['BRSET','mBRSET']:
        for role in ['fit','selection','assessment']:
            current.append(summarize([r for r in rows if r['domain']==domain and r['role']==role],domain+' '+role))
    current.append(summarize([r for r in rows if r['role']=='fit'],'joint fit'))
    for domain,file,pid in [('BRSET','splits_multilabel.csv','patient_id'),('mBRSET','splits_mbrset_multilabel.csv','patient')]:
        data=list(csv.DictReader((ROOT/'results'/file).open()))
        for r in data:r['patient_key']=domain+':'+r[pid]
        for split in ['train','val','test']:
            legacy.append(summarize([r for r in data if r['split']==split],domain+' legacy '+split))
    payload={'patient_definition':'At least one image with manifest DR label 1; descriptive aggregation, not a separately confirmed patient diagnosis.',
             'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'current':current,'legacy':legacy}
    (OUT/'prevalence_explanation.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
