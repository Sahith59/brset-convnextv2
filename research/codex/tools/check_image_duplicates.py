"""Decode every development image; audit exact RGB duplicates across patient roles."""
import concurrent.futures as cf
import hashlib
import json
import os
import socket
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/codex/step2'

def inspect(row):
    try:
        with Image.open(row['image_path']) as image:
            image.load()
            rgb = image.convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            # Also check EXIF-normalized pixels, though training does not apply this.
            canonical = ImageOps.exif_transpose(image).convert('RGB')
            cdigest = hashlib.sha256(str(canonical.size).encode()+canonical.tobytes()).hexdigest()
        return {**row, 'rgb_sha256':digest, 'exif_rgb_sha256':cdigest, 'width':rgb.width,
                'height':rgb.height, 'error':''}
    except Exception as exc:
        return {**row, 'error':repr(exc)}

def main():
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('Image decoding must run on an allocated compute node')
    OUT.mkdir(exist_ok=True)
    manifest = ROOT/'research/codex/protocol/development_manifest_v1.csv'
    df = pd.read_csv(manifest)
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(inspect,df.to_dict('records'))):
            rows.append(row)
            if (i+1)%1000==0: print(f'decoded {i+1}/{len(df)}',flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'image_integrity_private.csv',index=False)
    groups=[]
    for kind in ['rgb_sha256','exif_rgb_sha256']:
        for digest,g in result[result.error==''].groupby(kind):
            if len(g)>1:
                groups.append({'kind':kind,'hash':digest,'cross_role':g.role.nunique()>1,
                    'cross_patient':g.patient_key.nunique()>1,
                    'members':g[['domain','file','patient_key','role']].to_dict('records')})
    summary={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'images':len(df),'decode_errors':result[result.error!=''][['file','error']].to_dict('records'),
             'duplicate_groups':groups,'cross_role_groups':sum(g['cross_role'] for g in groups),
             'scope':'Exact native RGB and EXIF-normalized RGB; near duplicates/recompressed/registered eyes are not excluded by this test.'}
    summary['pass']=not summary['decode_errors'] and not summary['cross_role_groups']
    (OUT/'image_integrity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['pass']: raise SystemExit(1)

if __name__=='__main__': main()
