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

    


