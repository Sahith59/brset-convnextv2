"""Training-only validation of frozen historical degradation parameters. No model training."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[3]; OUT=R/'research/codex/updates/2026-09-10'
spec=importlib.util.spec_from_file_location('historical_ops',R/'scripts/55_degradation_ops.py'); ops=importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
ST=['sharpness','brightness','contrast','falloff','saturation']; P=json.loads((R/'results/fitted_degradation_params.json').read_text())
paths={'BRSET':R/'data/finetune_multilabel/train','mBRSET':R/'data/finetune_mbrset_multilabel/train'}
frames={}; manifest=[]; exclusions={}
for dom,split,pid in [('BRSET','splits_multilabel.csv','patient_id'),('mBRSET','splits_mbrset_multilabel.csv','patient')]:
 files=pd.read_csv(paths[dom]/'labels.csv',usecols=['file']); m=pd.read_csv(R/'results'/split,usecols=['file',pid,'split']).query("split == 'train'")
 df=files.merge(m,on='file',validate='one_to_one').rename(columns={pid:'patient'})
 used=set(np.random.default_rng(0).choice(files.file.values,400,replace=False))
 if dom=='BRSET': used.update(np.random.default_rng(0).choice(files.file.values,25,replace=False))
 usedpatients=set(df.loc[df.file.isin(used),'patient']); available=df[~df.patient.isin(usedpatients)]
 pick=np.random.default_rng(20260910).choice(len(available),64,replace=False); chosen=available.iloc[pick].copy()
 assert not set(chosen.patient)&usedpatients
 frames[dom]=chosen; exclusions[dom]={'excluded_reference_files':len(used),'excluded_patients':len(usedpatients),'eligible_images':len(available),'evaluation_images':len(chosen),'evaluation_patients':chosen.patient.nunique()}
 for row in chosen.itertuples(): manifest.append({'domain':dom,'file':row.file,'patient':row.patient,'split':row.split})
pd.DataFrame(manifest).to_csv(OUT/'image_manifest_private.csv',index=False)
def load(dom,f,mode):
 with Image.open(paths[dom]/f) as im: im=im.convert('RGB')
 if mode=='legacy_512': return im.resize((512,512))
 # torchvision PIL default resize is bilinear; crop at 24 reproduces 560 -> 512.
 return im.resize((560,560),resample=Image.Resampling.BILINEAR).crop((24,24,536,536))
def meanrows(images,p,seed):
 rng=np.random.default_rng(seed); rows=[ops.image_stats(ops.degrade(im,p,rng) if p is not None else im) for im in images]
 assert all(x is not None and all(np.isfinite(list(x.values()))) for x in rows)
 return pd.DataFrame(rows)
allrows=[]; distances=[]
for mode in ['legacy_512','model_eval_560_crop512']:
 b=[load('BRSET',f,mode) for f in frames['BRSET'].file]; m=[load('mBRSET',f,mode) for f in frames['mBRSET'].file]
 target=meanrows(m,None,0); tm=target.mean(); ts=target.std().clip(lower=1e-6)
 for arm,params in [('BRSET_unaltered',None),('local_generic',ops.GDRNET_GENERIC),('fitted_frozen',P),('mBRSET_reference',None)]:
  for seed in ([0,1,2] if params is not None else [0]):
   x=meanrows(m if arm=='mBRSET_reference' else b,params,seed)
   for k in ST: allrows.append({'preprocessing':mode,'arm':arm,'degradation_seed':seed,'statistic':k,'mean':x[k].mean(),'std':x[k].std(),'n':len(x)})
   distances.append({'preprocessing':mode,'arm':arm,'degradation_seed':seed,'distance':float((((x.mean()-tm)/ts)**2).sum())})
# Re-evaluate saved parameters on original fit sample, with the actual saved-summary RNG.
original_files=np.random.default_rng(0).choice(pd.read_csv(paths['BRSET']/'labels.csv').file.values,25,replace=False)
imgs=[load('BRSET',f,'legacy_512') for f in original_files]; old=json.loads((R/'results/dataset_appearance_stats_preserve.json').read_text())
target=pd.Series({k:old[k]['mbrset_mean'] for k in ST}); scale=pd.Series({k:max(old[k]['mbrset_std'],1e-6) for k in ST})
rep={arm:float((((meanrows(imgs,p,seed).mean()-target)/scale)**2).sum()) for arm,p,seed in [('BRSET_unaltered',None,0),('local_generic',ops.GDRNET_GENERIC,1),('fitted_saved_summary_seed999',P,999)]}
pd.DataFrame(allrows).to_csv(OUT/'appearance_statistics.csv',index=False); dd=pd.DataFrame(distances); dd.to_csv(OUT/'appearance_distances.csv',index=False)
# Scientific figure from source images and actual operators, not generative illustration.
fig,axes=plt.subplots(3,4,figsize=(12,9)); titles=['BRSET original','Local generic transform','Frozen fitted transform','Real mBRSET (unpaired)']
for i in range(3):
 src=load('BRSET',frames['BRSET'].file.iloc[i],'model_eval_560_crop512'); real=load('mBRSET',frames['mBRSET'].file.iloc[i],'model_eval_560_crop512')
 ims=[src,ops.degrade(src,ops.GDRNET_GENERIC,np.random.default_rng(100+i)),ops.degrade(src,P,np.random.default_rng(200+i)),real]
 for j,im in enumerate(ims):
  axes[i,j].imshow(im); axes[i,j].axis('off')
  if i==0: axes[i,j].set_title(titles[j],fontsize=11)
fig.tight_layout(); fig.savefig(OUT/'degradation_examples.png',dpi=140,bbox_inches='tight'); fig.savefig(OUT/'degradation_examples.pdf',bbox_inches='tight'); plt.close(fig)
summary={'date':'2026-09-10','frozen_params':P,'cohorts':exclusions,'original_fit_sample_recheck':rep,'heldout_distances':dd.groupby(['preprocessing','arm']).distance.agg(['mean','min','max']).reset_index().to_dict('records'),'scope':'64 training images per dataset; patients disjoint from reconstructed original 400-image measurement sets and 25-image fit set. Three degradation seeds; no refitting, diagnosis labels, validation/test images, classifier inference or training. Actual-model preprocessing check is evaluation resize/crop before normalization; no random training augmentations. Figure uses first three random samples; target images are unrelated to source rows. Distances are descriptive standardized squared mean differences, not classifier accuracy or a formal distributional test.','hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'scripts/55_degradation_ops.py',R/'results/fitted_degradation_params.json',OUT/'image_manifest_private.csv']}}
(OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2,default=int)); print(json.dumps(summary,indent=2,default=int))
