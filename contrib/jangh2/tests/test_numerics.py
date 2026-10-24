import unittest
import numpy as np
from jangh2_bridge.model_checks import compare_logits, temporary_expert_forward
from types import SimpleNamespace

class NumericTests(unittest.TestCase):
    def test_identical_logits(self):
        x=np.array([[1.,2.,3.],[3.,1.,0.]])
        m=compare_logits(x,x)
        self.assertEqual(m['top1_agreement'],1.);self.assertEqual(m['max_abs'],0.)
    def test_detects_large_numerical_drift(self):
        with self.assertRaises(AssertionError):compare_logits(np.array([[4.,0.]]),np.array([[0.,4.]]))
    def test_nonfinite_and_shape_refused(self):
        for v in (np.array([[np.nan,0.]]),np.ones((2,3))):
            with self.assertRaises(AssertionError):compare_logits(v,np.ones((1,2)))
    def test_context_restores_expert_forward_even_on_exception(self):
        class Expert:
            is_jangtq2=True
            def __call__(self,x,ids):return 'original'
        m=SimpleNamespace(named_modules=lambda:[('e',Expert())])
        def alternate(self,x,ids):return 'alternate'
        with self.assertRaisesRegex(RuntimeError,'test'):
            with temporary_expert_forward(m,alternate):
                self.assertEqual(Expert()(None,None),'alternate')
                raise RuntimeError('test')
        self.assertEqual(Expert()(None,None),'original')
