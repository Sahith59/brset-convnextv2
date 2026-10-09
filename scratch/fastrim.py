import numpy as np, pandas as pd, sys
from PIL import Image
from pathlib import Path
from scipy import ndimage
def sharp(a, e):
    g=a.mean(axis=2); m=g>0.04
    if e: m=ndimage.binary_erosion(m, iterations=e)
    if m.sum()<1000: return np.nan,np.nan
    return float(ndimage.laplace(g)[m].var()), float(m.mean())
def load(split,n,size,seed):
    d=Path(split); f=pd.read_csv(d/"labels.csv").file.values
    f=np.random.default_rng(seed).choice(f,n,replace=False); out=[]
    for x in f:
        try: im=Image.open(d/x).convert("RGB")
        except Exception: continue
        if size: im=im.resize((size,size),Image.BILINEAR)
        out.append(np.asarray(im,dtype=np.float32)/255.)
    return out
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
for size in (0,512):
    B=load("data/finetune_multilabel/train",N,size,0)
    M=load("data/finetune_mbrset_multilabel/train",N,size,1)
    print(f"\n=== {'native' if not size else 'squash512'}  n={len(B)}/{len(M)}", flush=True)
    print(f"{'erode':>6} {'BRSET':>11} {'mBRSET':>11} {'RATIO':>7} {'Bmask':>7} {'Mmask':>7}", flush=True)
    for e in (0,1,2,5,10,20):
        bs=[sharp(a,e) for a in B]; ms=[sharp(a,e) for a in M]
        b=np.nanmean([x[0] for x in bs]); mm=np.nanmean([x[0] for x in ms])
        bf=np.nanmean([x[1] for x in bs]); mf=np.nanmean([x[1] for x in ms])
        print(f"{e:>6} {b:11.6f} {mm:11.6f} {mm/max(b,1e-12):7.2f} {bf:7.3f} {mf:7.3f}", flush=True)
