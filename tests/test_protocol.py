"""Offline tests: no inference, no credentials, no service restart."""
import importlib.util, math, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify_results',ROOT/'scripts/verify_results.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class ProtocolTests(unittest.TestCase):
    def test_tie(self):self.assertEqual(v.probability(-1,-1),.5)
    def test_stability(self):self.assertAlmostEqual(v.probability(-901,-902),1/(1+math.exp(-1)))
    def test_false(self):self.assertLess(v.probability(-3,-1),.5)
    def test_true(self):self.assertGreater(v.probability(-1,-3),.5)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):v.probability(float('nan'),-1)
    def test_invalid_condition_rejected(self):
        with self.assertRaises(ValueError):v.require(False,'bad')
    def test_all_scripts_compile(self):
        for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'historical').glob('*.py')):
            compile(p.read_text(encoding='utf-8'),str(p),'exec')
    def test_full_record_integrity(self):
        r=v.verify();self.assertTrue(r['passed']);self.assertEqual(r['formal_API_responses'],2160)
if __name__=='__main__':unittest.main()
