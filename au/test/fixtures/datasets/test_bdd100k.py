from au.test import testutils
from au.fixtures.datasets import bdd100k

import unittest
import os

import pytest

class BDD100kTests(unittest.TestCase):
  """Exercise utiltiies in the bdd100k module.  Allow soft failures
  if the user has none of the required zip files.  We assume exclusively
  one of:
     1) the user emplaced the fixtures correctly using aucli
     2) the user has no fixtures and does not need them
  """

  @classmethod
  def setUpClass(cls):
    cls.fixtures = None
    try:
      bdd100k.BDD100KFixtures.create_test_fixtures()

      class TestFixtures(bdd100k.BDD100KFixtures):
        ROOT = bdd100k.BDD100KFixtures.TEST_FIXTURE_DIR
      cls.fixtures = TestFixtures
    except Exception as e:
      print "Failed to create test fixtures: %s" % (e,)
  
  @pytest.mark.slow
  def test_info_dataset(self):
    if not self.fixtures:
      return
    
    class TestInfoDataset(bdd100k.BDD100kInfoDataset):
      NAMESPACE_PREFIX = 'test'
      FIXTURES = self.fixtures
    
    with testutils.LocalSpark.sess() as spark:
      ts_row_rdd = TestInfoDataset._dataset_from_zip(spark)
      df = ts_row_rdd#spark.createDataFrame(ts_row_rdd)
      df.printSchema()
      df.write.parquet('/tmp/yyyyyy', partitionBy=['split', 'video'], compression='gzip')
    
