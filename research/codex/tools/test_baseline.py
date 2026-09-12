"""Adversarial CPU checks. Run via Slurm, not on login nodes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from baseline_core import *

class CoreChecks(unittest.TestCase):
    def test_shuffle_coverage_wrap_and_resume(self):
        complete=list(DrawBatches(19,4,0,160,27))
        self.assertEqual(complete[7:],list(DrawBatches(19,4,28,160,27)))
        flat=[x for b in complete for x in b]
        for start in range(0,152,19):self.assertEqual(sorted(i for i,_ in flat[start:start+19]),list(range(19)))
        self.assertEqual([d for _,d in flat],list(range(160)))
    def test_bad_draw_interval(self):
        with self.assertRaises(ValueError):DrawBatches(19,4,3,100,0)
    def test_threshold_matches_independent_bruteforce(self):
        rng=np.random.default_rng(7);y=rng.integers(0,2,(41,2));p=np.round(rng.random((41,2)),2)
        actual=select_thresholds(y,p)
        for k in range(2):
            grid=np.arange(1001)/1000
            scores=np.array([f1_score(y[:,k],p[:,k]>=t,zero_division=0) for t in grid])
            candidates=grid[np.isclose(scores,scores.max(),rtol=0,atol=1e-12)]
            expected=sorted(candidates,key=lambda t:(round(abs(t-.5),12),t))[0]
            self.assertEqual(actual[k],expected)
    def test_threshold_ties_and_degenerate_input(self):
        y=np.array([[0,0],[1,1]]);p=np.array([[.1,.1],[.9,.9]])
        self.assertEqual(select_thresholds(y,p),[.5,.5])
        with self.assertRaises(ValueError):select_thresholds(np.zeros_like(y),p)
        p[0,0]=np.nan
        with self.assertRaises(ValueError):select_thresholds(y,p)
    def test_metric_definitions_not_macro_confusion(self):
        y=np.array([[1,0],[1,1],[0,0],[0,1],[0,0]])
        p=np.array([[.9,.2],[.1,.8],[.6,.4],[.2,.7],[.3,.1]])
        m=metrics(y,p,[.5,.5])[LABELS[0]]
        self.assertEqual(m['confusion_matrix'],[[2,1],[1,1]])
        self.assertEqual(m['f1_positive'],.5)
        self.assertAlmostEqual(m['f1_class_macro'],(2/3+.5)/2)
        self.assertIsNone(metrics(np.zeros((3,2)),np.zeros((3,2)),[.5,.5])[LABELS[0]]['auroc'])
    def test_focal_hard_labels_and_soft_extremes(self):
        z=torch.tensor([-2.,2.],requires_grad=True);y=torch.tensor([0.,1.])
        prob=z.sigmoid();pt=torch.where(y==1,prob,1-prob)
        self.assertTrue(torch.allclose(focal(z,y),(-((1-pt)**2)*pt.log()).mean()))
        extreme=torch.tensor([-1000.,1000.],requires_grad=True)
        loss=focal(extreme,torch.tensor([.05,.95]));loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(extreme.grad).all())
    def test_accumulation_equals_full_batch_gradient(self):
        torch.manual_seed(8);a=torch.nn.Linear(3,2);b=__import__('copy').deepcopy(a)
        x=torch.randn(16,3);y=torch.rand(16,2)
        focal(a(x),y).backward()
        for xx,yy in zip(x.chunk(4),y.chunk(4)):(focal(b(xx),yy)/4).backward()
        for aa,bb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(aa.grad,bb.grad,atol=1e-7))
    def test_mixup_independent_and_reproducible(self):
        x=torch.arange(12.).reshape(4,3);y=torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])
        a= mixup(x,y,0,8);np.random.seed(91);np.random.random(100);torch.rand(100)
        b= mixup(x,y,0,8)
        self.assertTrue(all(torch.equal(i,j) for i,j in zip(a,b)))
        self.assertTrue(torch.all(a[1]>=.05) and torch.all(a[1]<=.95))
    def test_schedule_endpoints_and_phase_counts(self):
        self.assertAlmostEqual(learning_rate(1,5775,693,3e-5),3e-6)
        self.assertAlmostEqual(learning_rate(693,5775,693,3e-5),3e-5,places=15)
        self.assertEqual(learning_rate(5775,5775,693,3e-5),0)
        for arm in ['B0','B1','B2']:
            phases=phase_specs(arm)
            self.assertEqual(sum(x['updates'] for x in phases),5775)
            self.assertEqual(sum(len(x['eval_at']) for x in phases),25)
        self.assertEqual(phase_specs('B2')[0]['eval_at'],[])
    def test_flip_averages_probabilities_not_logits(self):
        class M(torch.nn.Module):
            def forward(self,x):return x[:,:2,0,0]
        x=torch.tensor([[[[0.,1.],[2.,3.]],[[1.,2.],[3.,4.]]]])
        p,z=flip_probabilities(M(),x)
        self.assertTrue(torch.allclose(p,z.sigmoid().mean(1)))
        self.assertFalse(torch.allclose(p,z.mean(1).sigmoid()))
    def test_transform_draw_reproducibility_and_role_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.png';Image.fromarray(np.random.default_rng(1).integers(0,256,(600,800,3),dtype=np.uint8)).save(path)
            rows=pd.DataFrame([{'image_path':str(path),'role':'fit',LABELS[0]:1,LABELS[1]:0}])
            ds=Images(rows,True,4);before=torch.get_rng_state();a=ds[(0,9)][0]
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(torch.equal(a,ds[(0,9)][0]));self.assertFalse(torch.equal(a,ds[(0,10)][0]))
            rows['role']='assessment'
            with self.assertRaises(ValueError):Images(rows,True,4)
    def test_manifest_role_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.csv'
            d=pd.DataFrame([{'domain':'mBRSET','file':str(i),'patient_key':'m:1','role':role,'original_split':'train',LABELS[0]:0,LABELS[1]:1} for i,role in enumerate(['fit','selection'])]);d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Patient crosses'):validate_manifest(p,sha(p))
    def test_patient_bootstrap_keeps_images_and_multiplicity(self):
        from assess_baselines import patient_indices
        class FixedDraw:
            def choice(self,patients,n,replace):return np.array(['a','a','b'])
        ids=np.array(['a','a','b','c','c','c'])
        self.assertEqual(patient_indices(ids,FixedDraw()).tolist(),[0,1,0,1,2])

if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Use allocated Slurm node')
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoreChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path('/home/users/sthummala2/brset-convnextv2/research/codex/step2/cpu_checks.json')
    out.write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),'source_sha256':sha(Path(__file__).with_name('baseline_core.py'))},indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
