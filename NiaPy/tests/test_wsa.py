# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import WolfSearchAlgorithm


class WSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = WolfSearchAlgorithm

	def test_type_parameters(self):
		tp = self.algo.typeParameters()
		self.assertTrue(tp['NP'](1))
		self.assertFalse(tp['NP'](0))
		self.assertFalse(tp['NP'](-1))
		self.assertFalse(tp['NP'](1.0))
		self.assertTrue(tp['r'](1.0))
		self.assertFalse(tp['r'](0.0))
		self.assertFalse(tp['r'](-1.0))
		self.assertFalse(tp['r'](1))
		self.assertTrue(tp['s'](1.0))
		self.assertFalse(tp['s'](0.0))
		self.assertFalse(tp['s'](-1.0))
		self.assertFalse(tp['s'](1))
		self.assertTrue(tp['alpha'](1.0))
		self.assertFalse(tp['alpha'](0.0))
		self.assertFalse(tp['alpha'](-1.0))
		self.assertFalse(tp['alpha'](1))
		self.assertTrue(tp['pa'](1.0))
		self.assertTrue(tp['pa'](1.0))
		self.assertTrue(tp['pa'](0.5))
		self.assertTrue(tp['pa'](0.0))
		self.assertFalse(tp['pa'](-1.0))
		self.assertFalse(tp['pa'](1))
		self.assertFalse(tp['pa'](-1.0))
		self.assertFalse(tp['pa'](1.1))
		self.assertFalse(tp['pa'](1))

	def test_works_fine(self):
		wsa = self.algo(NP=10, r=1, s=0.2, alpha=1, pa=0.2, seed=self.seed)
		wsac = self.algo(NP=10, r=1, s=0.2, alpha=1, pa=0.2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wsa, wsac, MyBenchmark())

	def test_griewank_works_fine(self):
		wsa_griewank = self.algo(NP=10, r=1, s=0.2, alpha=1, pa=0.2, seed=self.seed)
		wsa_griewankc = self.algo(NP=10, r=1, s=0.2, alpha=1, pa=0.2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wsa_griewank, wsa_griewankc)
