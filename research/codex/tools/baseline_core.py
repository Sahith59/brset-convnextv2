"""Protocol-v1 data ordering, transformations, loss, and metrics (no model fitting)."""
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

LABELS=['diabetic_retinopathy','macular_edema']
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stream_seed(seed,name,index=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{index}'.encode()).digest()[:8],'little')%(2**63-1)

def validate_manifest(path,expected_hash):
    if sha(path)!=expected_hash:raise ValueError('Manifest hash mismatch')
    d=pd.read_csv(path)
    if d.duplicated(['domain','file']).any():raise ValueError('Duplicate image identity')
    if not d.original_split.eq('train').all():raise ValueError('Legacy non-train data in manifest')
    if d.groupby('patient_key').role.nunique().max()!=1:raise ValueError('Patient crosses roles')
    if not d.role.isin(['fit','selection','assessment']).all():raise ValueError('Unknown role')
    if not d[LABELS].isin([0,1]).all().all():raise ValueError('Invalid labels')
    return d

class DrawBatches(Sampler):
    """Stateless shuffled passes; cursor is consumed draws, not DataLoader prefetch."""
    def __init__(self,n,batch_size,start,stop,seed):
        if n<1 or start%batch_size or stop%batch_size:raise ValueError('Invalid draw interval')
        self.n,self.batch_size,self.start,self.stop,self.seed=n,batch_size,start,stop,seed
    def __len__(self):return (self.stop-self.start)//self.batch_size
    def __iter__(self):
        previous_cycle,order=-1,None
        batch=[]
        for draw in range(self.start,self.stop):
            cycle,offset=divmod(draw,self.n)
            if cycle!=previous_cycle:
                order=np.random.default_rng(stream_seed(self.seed,'sampler',cycle)).permutation(self.n)
                previous_cycle=cycle
            batch.append((int(order[offset]),draw))
            if len(batch)==self.batch_size:yield batch;batch=[]

class Images(Dataset):
    def __init__(self,rows,training=False,seed=0):
        self.rows=rows.reset_index(drop=True);self.seed=seed;self.training=training
        common=[T.Resize((560,560),interpolation=InterpolationMode.BILINEAR,antialias=True)]
        if training:
            if not self.rows.role.eq('fit').all():raise ValueError('Training dataset contains non-fit images')
            common += [T.RandomCrop(512),T.RandomHorizontalFlip(0.5),
                T.RandomRotation(15,interpolation=InterpolationMode.NEAREST,expand=False,fill=0),
                T.ColorJitter(brightness=.2,contrast=.2,saturation=.1,hue=0)]
        else:common += [T.CenterCrop(512)]
        self.transform=T.Compose(common+[T.ToTensor(),T.Normalize(MEAN,STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        idx,draw=key if isinstance(key,tuple) else (key,0)
        row=self.rows.iloc[idx]
        with Image.open(row.image_path) as raw:image=raw.convert('RGB')
        if self.training:
            # Fork CPU RNG only: augmentation must not change model/dropout RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stream_seed(self.seed,'common_transform',draw))
                image=self.transform(image)
        else:image=self.transform(image)
        return image,torch.tensor(row[LABELS].to_numpy(dtype=np.float32)),idx

def mixup(images,targets,seed,microstep):
    rng=np.random.default_rng(stream_seed(seed,'mixup',microstep))
    lam=float(rng.beta(.2,.2));perm=torch.as_tensor(rng.permutation(len(targets)),device=images.device)
    return lam*images+(1-lam)*images[perm], .9*(lam*targets+(1-lam)*targets[perm])+.05

def focal(logits,targets):
    # fp32 loss calculation even when model forward uses bf16.
    b=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')
    return (((1-torch.exp(-b))**2)*b).mean()

def learning_rate(update,total,warmup,peak):
    """1-based update: 0.1*peak first, peak at warmup end, zero at final update."""
    if not 1<=update<=total or not 1<warmup<total:raise ValueError('Invalid scheduler position')
    if update<=warmup:return peak*(.1+.9*(update-1)/(warmup-1))
    return peak*.5*(1+math.cos(math.pi*(update-warmup)/(total-warmup)))

def select_thresholds(y,p):
    y,p=np.asarray(y),np.asarray(p)
    if y.shape!=p.shape or y.ndim!=2 or not np.isfinite(p).all():raise ValueError('Invalid predictions')
    grid=np.arange(1001,dtype=float)/1000
    result=[]
    for k in range(y.shape[1]):
        if len(np.unique(y[:,k]))!=2:raise ValueError('Threshold selection requires both classes')
        pred=p[:,k,None]>=grid
        tp=(pred & (y[:,k,None]==1)).sum(0)
        fp=(pred & (y[:,k,None]==0)).sum(0)
        fn=((~pred) & (y[:,k,None]==1)).sum(0)
        score=2*tp/np.maximum(2*tp+fp+fn,1)
        candidates=grid[np.abs(score-score.max())<=1e-12]
        closest=np.abs(candidates-.5).min()
        result.append(float(candidates[np.abs(np.abs(candidates-.5)-closest)<=1e-12].min()))
    return result

def metrics(y,p,thresholds):
    if not np.isfinite(p).all():raise ValueError('Nonfinite probabilities')
    result={}
    for k,label in enumerate(LABELS):
        truth=np.asarray(y[:,k],dtype=int);prediction=p[:,k]>=thresholds[k]
        tn,fp,fn,tp=confusion_matrix(truth,prediction,labels=[0,1]).ravel().tolist()
        ratio=lambda n,d:float(n/d) if d else None
        both=len(np.unique(truth))==2
        result[label]={'threshold':float(thresholds[k]),'f1_positive':float(f1_score(truth,prediction,zero_division=0)),
            'f1_class_macro':float(f1_score(truth,prediction,labels=[0,1],average='macro',zero_division=0)),
            'auroc':float(roc_auc_score(truth,p[:,k])) if both else None,
            'average_precision':float(average_precision_score(truth,p[:,k])) if truth.sum()>0 else None,
            'precision':ratio(tp,tp+fp),'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),
            'confusion_matrix':[[tn,fp],[fn,tp]],'n_images':len(truth)}
    return result

@torch.inference_mode()
def flip_probabilities(model,x):
    probabilities=[];logits=[]
    for dims in [(),(-1,),(-2,),(-2,-1)]:
        z=model(torch.flip(x,dims) if dims else x).float()
        logits.append(z);probabilities.append(z.sigmoid())
    return torch.stack(probabilities).mean(0),torch.stack(logits,dim=1)

def phase_specs(arm,smoke=False):
    if smoke and arm=='B2':return [
        {'name':'source','domain':'BRSET','updates':3,'warmup':2,'lr':3e-5,'eval_at':[]},
        {'name':'target','domain':'mBRSET','updates':3,'warmup':2,'lr':1e-5,'eval_at':[1,3]}]
    if smoke:return [{'name':'smoke','domain':'both' if arm=='B1' else 'mBRSET','updates':6,'warmup':2,'lr':3e-5,'eval_at':[3,6]}]
    if arm=='B2':return [
        {'name':'source','domain':'BRSET','updates':2887,'warmup':346,'lr':3e-5,'eval_at':[]},
        {'name':'target','domain':'mBRSET','updates':2888,'warmup':347,'lr':1e-5,'eval_at':[math.ceil(j*2888/25) for j in range(1,26)]}]
    return [{'name':'main','domain':'mBRSET' if arm=='B0' else 'both','updates':5775,'warmup':693,'lr':3e-5,'eval_at':list(range(231,5776,231))}]
