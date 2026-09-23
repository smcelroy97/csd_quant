"""Run directly in an editor; no command-line arguments required."""
import unittest
from pathlib import Path
import numpy as np
import utils
class ScoringTests(unittest.TestCase):
 def test_landmarks(self):
  depths=np.arange(100,2000,100.);csd=depths[:,None]*np.ones((1,200))
  out=utils.align_model_csd(csd,depths)
  np.testing.assert_allclose(out[[0,7,15,22,29],0],[100,475,1100,1625,1900]);self.assertEqual(out.shape,(30,200))
 def test_validation(self):
  with self.assertRaises(ValueError):utils.align_model_csd(np.ones((19,200)),np.arange(19))
  with self.assertRaises(ValueError):utils.wasserstein_csd(np.zeros((3,4)),np.zeros((4,4)))
  with self.assertRaises(ValueError):utils.wasserstein_csd(np.full((3,4),np.nan),np.zeros((3,4)))
 def test_pairwise(self):
  a=np.array([[1.,0,0],[-1,0,0]]);b=np.roll(a,1,axis=1)
  D=utils.pairwise_wd_csd([a,b]);self.assertGreater(D[0,1],0);np.testing.assert_allclose(D,D.T);np.testing.assert_allclose(np.diag(D),0,atol=1e-12)
 def test_wrapper_alignment(self):
  # Load staged wrapper while resolving its unchanged template path in the repo.
  namespace=dict(__file__='/Users/scoot/dev/csd_quant/wasserstein_dist.py',__name__='staged_wrapper',__package__='')
  exec(compile(Path(__file__).with_name('wasserstein_dist.py').read_text(),'staged_wrapper','exec'),namespace)
  captured=[]
  def score(a,b):captured.append((a,b));return (0.3,0.1,0.2)
  namespace['wasserstein_csd']=score
  depths=np.arange(100,2000,100.)
  result=namespace['wd_from_template'](np.repeat(depths[:,None],200,axis=1),np.arange(200),sim_depths_um=depths)
  self.assertEqual(result,(0.3,0.1,0.2))
  np.testing.assert_allclose(captured[0][1][[7,15,22],0],[475,1100,1625])
 def test_benchmark_and_scale(self):
  root=Path('/Users/scoot/dev/csd_quant/qc_workflow')
  import sys;sys.path.insert(0,str(root))
  from validate_wd import ShapeWD
  with np.load(root/'csd_channel_interpretation/qc_results/run_20260923_csd_one/short/pca_details.npz') as z:X=z['erps']
  a,b=X[0],X[1:].mean(0);before=a.copy()
  expected=ShapeWD(1)(a,b);actual=utils.wasserstein_csd(a,b)
  np.testing.assert_allclose(actual,[expected['wd'],expected['wd_sink'],expected['wd_source']],atol=1e-9)
  np.testing.assert_allclose(utils.wasserstein_csd(a*3,b),actual,atol=1e-9);np.testing.assert_array_equal(a,before)
if __name__=='__main__':unittest.main(argv=['test_scoring'],verbosity=2)
