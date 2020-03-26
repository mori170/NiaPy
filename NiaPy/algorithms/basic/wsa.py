# encoding=utf8
import logging
from numpy import where, apply_along_axis, zeros, append, ndarray, delete, arange, argmin, absolute, int32
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import limit_repair

__all__ = ['WolfSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')


class WolfSearchAlgorithm(Algorithm):
	r"""Implementation of Wolf Search Algorithm.

	Algorithm:
		Wolf Search Algorithm

	Date:
		2020

	Authors:
		Rok Mori

	License:
		MIT

	Reference paper:
		Tang, Rui & Fong, Simon & Yang, Xin-She & Deb, Suash. (2012). Wolf search algorithm with ephemeral memory. 10.1109/ICDIM.2012.6360147.

	References URL:
		TODO

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		r (float): Visual radius parameter.
		s (float): Step size parameter.
		alpha ((float): Velocity parameter.
		pa (float): Enemy appearance threshold parameter.
	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['WolfSearchAlgorithm', 'WSA']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r"""
		Description: Wolf Search Algorithm is based on wolf preying behavior and it simultaneously possesses both individual local searching ability and autonomous flocking movement.
		Authors: Tang, Rui & Fong, Simon & Yang, Xin-She & Deb, Suash
		Year: 2012
		Main reference: Tang, Rui & Fong, Simon & Yang, Xin-She & Deb, Suash. (2012). Wolf search algorithm with ephemeral memory. 10.1109/ICDIM.2012.6360147.
		"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* r (Callable[[float], bool]): Checks if visual radius parameter has a proper value.
				* s (Callable[[float], bool]): Checks if step size parameter has a proper value.
				* alpha (Callable[[float], bool]): Checks if velocity parameter has a proper value.
				* pa (Callable[[float], bool]): Checks if enemy appearance threshold parameter has a proper value.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'r': lambda x: isinstance(x, float) and x > 0,
			's': lambda x: isinstance(x, float) and x > 0,
			'alpha': lambda x: isinstance(x, float) and x > 0,
			'pa': lambda x: isinstance(x, float) and 0 <= x <= 1,
		})
		return d

	def setParameters(self, NP=10, r=1, s=0.2, alpha=1, pa=0.2, **kwargs):
		r"""Set the parameters of the algorithm.

		Args:
			NP (Optional[int]): Population size.
			r (Optional[float]): Visual radius parameter.
			s (Optional[float]): Step size parameter.
			alpha (Optional[float]): Velocity parameter.
			pa (Optional[float]): Enemy appearance threshold parameter.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **kwargs)
		self.r, self.s, self.alpha, self.pa = r, s, alpha, pa

	def getParameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'r': self.r,
			's': self.s,
			'alpha': self.alpha,
			'pa': self.pa
		})
		return d

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Wolves, Evaluations, _ = Algorithm.initPopulation(self, task)
		return Wolves, Evaluations, {}

	def runIteration(self, task, Wolves, Evaluations, xb, fxb, **dparams):
		r"""Core function of Forest Optimization Algorithm.

		Args:
			task (Task): Optimization task.
			Wolves (numpy.ndarray): Current population.
			Evaluations (numpy.ndarray[float]): Current population function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individual fitness/function value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments.
		"""

		raise NotImplementedError
		# return Wolves, Evaluations, xb, fxb, {}
