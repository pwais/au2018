import os

from au import util
from au.experiments.data_ablation import mnist as mnist_ablated
from au.fixtures.tf import mnist
from au.test import testconf
from au.test import testutils

import unittest

import imageio
import numpy as np
import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'test_mnist_ablation') 


def test_my_fun_tassst():
  from au.spark import Spark
  spark = Spark.getOrCreate()
  experiment = mnist_ablated.Experiment(run_name='default.2019-02-15-06_35_17.H8ONI')
  # experiment = mnist_ablated.Experiment(run_name='default.2019-02-03-07_25_48.GIBOB')
  # experiment._build_activations(spark=spark)
  # experiment.run(spark=spark)
  # df = experiment.as_df(spark)
  report = mnist_ablated.ExperimentReport(spark, experiment)
  report.save()
  # import ipdb; ipdb.set_trace()
  
  

# """

# df.createOrReplaceTempView('data')
# spark.sql("select t.keep_frac, avg(100. * t.acc), std(100. * t.acc), count(*) support from ( select TRAIN_TABLE_KEEP_FRAC keep_frac, params_hash, max(simple_value) acc from data where tag = 'accuracy'  group by TRAIN_TABLE_KEEP_FRAC, params_hash order by TRAIN_TABLE_KEEP_FRAC, params_hash) as t group by keep_frac order by keep_frac").show()

# +--------------------+--------------------------------+----------------------------------------+-------+
# |           keep_frac|avg((CAST(100 AS DOUBLE) * acc))|stddev_samp((CAST(100 AS DOUBLE) * acc))|support|
# +--------------------+--------------------------------+----------------------------------------+-------+
# |9.999999999998899E-5|              32.166999876499176|                      7.6488477810816695|     10|
# |4.999999999999449E-4|               62.59899973869324|                       4.393402116711541|     10|
# |0.001000000000000...|               73.41899991035461|                       3.094451494289516|     10|
# |0.005000000000000...|                 89.211110273997|                      0.8508874872703646|      9|
# |0.010000000000000009|                92.8199997970036|                     0.34477090306163716|      7|
# |0.050000000000000044|               97.08999991416931|                                     NaN|      1|
# +--------------------+--------------------------------+----------------------------------------+-------+
# """


# new results:
"""
+--------------------+--------------------------------+----------------------------------------+-------+
|           keep_frac|avg((CAST(100 AS DOUBLE) * acc))|stddev_samp((CAST(100 AS DOUBLE) * acc))|support|
+--------------------+--------------------------------+----------------------------------------+-------+
|9.999999999998899E-5|               32.25024297833443|                       7.656910928219753|     10|
|4.999999999999449E-4|               58.16700026392937|                      16.500533334416954|     10|
|0.001000000000000...|               73.92222219043308|                       2.858039140948358|      9|
|0.005000000000000...|               89.02199923992157|                      1.0356510705433526|     10|
|0.010000000000000009|                92.4399995803833|                      0.6090972777065516|     10|
|0.050000000000000044|               97.34399974346161|                      0.2977400461479309|     10|
| 0.09999999999999998|                98.2260000705719|                     0.13150381196266359|     10|
|                 0.5|                99.1430002450943|                     0.04498191889029135|     10|
|                 1.0|               99.39099967479706|                    0.023309430402744788|     10|
+--------------------+--------------------------------+----------------------------------------+-------+

"""

def test_gen_target_dists():
  dists = list(mnist_ablated.gen_ablated_dists([1, 2, 3], [0.5, 0.9]))
  assert {1: 0.5, 2: 1., 3: 1.} in dists

class TestMNISTAblatedDataset(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Use /tmp for test fixtures
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

    cls.params = mnist.MNIST.Params()
    cls.params.LIMIT = 10000

    table = mnist_ablated.AblatedDataset

    table.setup(params=cls.params)
    cls.spark = testutils.LocalSpark.getOrCreate()
    cls.std_class_freq = table.get_class_freq(cls.spark)
    cls.std_class_counts = table.get_class_freq(cls.spark, raw_counts=True)
  
  def test_uniform_ablate(self):
    class UniformAblated(mnist_ablated.AblatedDataset):
      UNIFORM_ABLATE = 0.5
    
    UniformAblated.setup(params=self.params)
    class_counts = UniformAblated.get_class_freq(self.spark, raw_counts=True)

    print 'Whole dataset counts:'
    self.std_class_counts.show()

    print 'Ablated counts:'
    class_counts.show()

    std_rows = sorted(self.std_class_counts.collect())
    test_rows = sorted(class_counts.collect())
    assert std_rows and test_rows

    std_counts = np.array([row.num for row in std_rows]).astype(np.float32)
    test_counts = np.array([row.num for row in test_rows]).astype(np.float32)
    assert std_counts.shape == test_counts.shape
    assert all(
            (0.45 <= v <= 0.55)
              # The sample approaches the target of 50%
              # as the table size grows
            for v in (test_counts / std_counts))

    


