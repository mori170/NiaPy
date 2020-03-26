# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import WolfSearchAlgorithm
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

task = StoppingTask(D=2, nFES=11, benchmark=Sphere())
algo = WolfSearchAlgorithm(NP=10, r=1, s=0.2, alpha=1, pa=0.2)
best = algo.run(task)
print('%s -> %s' % (best[0], best[1]))

# # we will run Wolf Search Algorithm for 5 independent runs
# for i in range(5):
#     task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
#     algo = ForestOptimizationAlgorithm(NP=20, lt=5, lsc=1, gsc=1, al=20, tr=0.35)
#     best = algo.run(task)
#     print('%s -> %s' % (best[0], best[1]))

