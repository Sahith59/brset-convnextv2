"""Prepare development manifests and immutable evidence; does not train/evaluate models."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/codex/protocol'
SEED = 20260910
LABELS = ['diabetic_retinopathy', 'macular_edema']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    frames, summaries, input_hashes, stratification = [], [], {}, {}
    for domain, name, pid, folder in [
        ('BRSET', 'splits_multilabel.csv', 'patient_id', 'finetune_multilabel'),
        ('mBRSET', 'splits_mbrset_multilabel.csv', 'patient', 'finetune_mbrset_multilabel')]:
        path = ROOT / 'results' / name
        input_hashes[name] = sha(path)
        full = pd.read_csv(path)
        assert full.file.is_unique
        assert full.groupby(pid)['split'].nunique().max() == 1
        d = full[full.split == 'train'].copy().sort_values('file').reset_index(drop=True)
        assert d[LABELS].isin([0, 1]).all().all()
        patients = d.groupby(pid)[LABELS].max().sort_index()
        strata = patients[LABELS[0]].astype(str) + patients[LABELS[1]].astype(str)
        stratification[domain] = {'joint_patient_label_counts': strata.value_counts().to_dict()}
        if strata.value_counts().min() < 5:
            strata = patients[LABELS[0]].astype(str)
            stratification[domain]['rule'] = 'DR patient-level stratum; joint rare stratum has fewer than 5 patients'
        else:
            stratification[domain]['rule'] = 'joint patient-level maximum DR/ME labels'
        assigned = {}
        for fold, (_, idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(patients, strata)):
            assigned.update({p: fold for p in patients.index[idx]})
        d['domain'] = domain
        d['patient_key'] = domain + ':' + d[pid].astype(str)
        d['fold'] = d[pid].map(assigned)
        d['role'] = d.fold.map({0:'assessment', 1:'selection', 2:'fit', 3:'fit', 4:'fit'})
        d['original_split'] = d['split']
        d['image_path'] = [str((ROOT / 'data' / folder / 'train' / f).resolve(strict=True)) for f in d.file]
        assert d.image_path.is_unique
        labels = pd.read_csv(ROOT / 'data' / folder / 'train/labels.csv').set_index('file')
        assert labels.index.is_unique and set(labels.index) == set(d.file)
        assert (labels.loc[d.file, LABELS].to_numpy() == d[LABELS].to_numpy()).all()
        assert d.groupby('patient_key').role.nunique().max() == 1
        for role, rows in d.groupby('role'):
            assert all(0 < rows[c].sum() < len(rows) for c in LABELS)
            summaries.append({'domain':domain, 'role':role, 'images':len(rows),
                              'patients':rows.patient_key.nunique(),
                              'DR_positive_images':int(rows[LABELS[0]].sum()),
                              'ME_positive_images':int(rows[LABELS[1]].sum())})
        frames.append(d[['domain','file','patient_key',*LABELS,'original_split','fold','role','image_path']])
    manifest = pd.concat(frames, ignore_index=True)
    assert not manifest.duplicated(['domain','file']).any()
    assert manifest.image_path.is_unique
    manifest.to_csv(OUT/'development_manifest_v1.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT/'development_counts_v1.csv', index=False)
    cache = Path.home()/'.cache/huggingface/hub/models--timm--convnextv2_large.fcmae_ft_in22k_in1k_384'
    revision = (cache/'refs/main').read_text().strip()
    blob = cache/'blobs/075703d09f806c09597fa3d68d8c6733a18ba1126adfbc1de97f8eb096402c2f'
    digest = sha(blob)
    assert digest == blob.name
    env = {k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','numpy','pandas','scikit-learn','Pillow','scipy']}
    evidence = {'split_seed':SEED, 'input_manifest_sha256':input_hashes,
                'development_manifest_sha256':sha(OUT/'development_manifest_v1.csv'),
                'counts':summaries, 'stratification':stratification,
                'checks':{'patient_roles_disjoint_within_domain':True,'original_train_only':True,
                          'labels_match_training_csv':True,'all_image_paths_resolve':True,
                          'resolved_paths_unique':True,'both_classes_each_label_each_role':True,
                          'cross_dataset_person_identity_known':False,
                          'decoded_image_duplicate_scan_completed_here':False},
                'initialization':{'model':'convnextv2_large.fcmae_ft_in22k_in1k_384',
                                  'cached_hf_revision':revision,'weight_path':str(blob),
                                  'weight_sha256':digest,'weight_bytes':blob.stat().st_size,
                                  'head':'replace pretrained head with two outputs; save per-seed initialized state before any training'},
                'environment':env,
                'tracked_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'local_source_sha256':{n:sha(ROOT/n) for n in ['scripts/30_train_strong_baseline.py','scripts/55_degradation_ops.py','scripts/57_fit_degradation.py']}}
    (OUT/'preflight_v1.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'counts':summaries,'manifest_sha256':evidence['development_manifest_sha256'],'initialization_sha256':digest},indent=2))

if __name__ == '__main__':
    main()
