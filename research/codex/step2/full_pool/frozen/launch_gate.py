"""Allocated lightweight pre-training provenance gate for the revised batch."""
import json,hashlib,shutil
from pathlib import Path
W=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
p=read(W/'protocol/v1.json')
for file in ['cpu_checks.json','image_integrity_summary.json','smoke_checks.json','phase_smoke_checks.json']:
 assert read(W/file)['pass'],file
assert read(W/'cpu_checks.json')['source_sha256']==sha(W/'frozen/baseline_core.py')
assert read(W/'image_integrity_summary.json')['manifest_sha256']==p['manifest_sha256']
assert sha(W/'protocol'/p['manifest'])==p['manifest_sha256']
for folder in ['smoke_full','smoke_resume','phase_smoke_full','phase_smoke_resume']:
 c=read(W/folder/'contract.json')
 assert c['protocol_sha256']==sha(W/'protocol/v1.json')
 assert c['manifest_sha256']==p['manifest_sha256']
 assert c['trainer_sha256']==sha(W/'frozen/train_baseline.py')
 assert c['core_sha256']==sha(W/'frozen/baseline_core.py')
 assert c['initialization_sha256']==read(W/'initializations/seed0.json')['sha256']
assert read(W/'smoke_checks.json')['estimated_baseline_hours']*1.35<12
assert shutil.disk_usage(W/'runs').free>16*1024**3
print('Full-pool execution and provenance gates passed',flush=True)
