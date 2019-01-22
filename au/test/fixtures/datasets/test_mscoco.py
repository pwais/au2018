from au.fixtures.datasets import mscoco
from au.test import testconf
from au.test import testutils

import os
import unittest

import pytest

class TestFixtures(mscoco.Fixtures):
  ROOT = mscoco.Fixtures.TEST_FIXTURE_DIR

class TestTrainAnnos(mscoco.TrainAnnos):
  FIXTURES = TestFixtures

class TestValAnnos(mscoco.ValAnnos):
  FIXTURES = TestFixtures

class TestMSCOCOImageTableTrain(mscoco.MSCOCOImageTableTrain):
  FIXTURES = TestFixtures
  ANNOS_CLS = TestTrainAnnos
  APPROX_MB_PER_SHARD = 10.

class TestMSCOCOImageTableVal(mscoco.MSCOCOImageTableVal):
  FIXTURES = TestFixtures
  ANNOS_CLS = TestValAnnos
  APPROX_MB_PER_SHARD = 10.

class TestMSCOCOImageTable(unittest.TestCase):
  """Exercise utiltiies in the mscoco module.  Allow soft failures
  if the user has none of the required zip files.  We assume exclusively
  one of:
     1) the user emplaced the fixtures correctly using aucli
     2) the user has no fixtures and does not need them
  """

  @classmethod
  def setUpClass(cls):
    cls.have_fixtures = False
    try:
      mscoco.Fixtures.create_test_fixtures()
      cls.have_fixtures = True
    except Exception as e:
      print "Failed to create test fixtures: %s" % (e,)
  
  @pytest.mark.slow
  def test_image_table(self):
    if not self.have_fixtures:
      return
    
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    TEST_TEMPDIR = os.path.join(
                        testconf.TEST_TEMPDIR_ROOT,
                        'test_mscoco')
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

    with testutils.LocalSpark.sess() as spark:
      TABLES = (
        TestMSCOCOImageTableTrain,
        TestMSCOCOImageTableVal,
      )

      for table in TABLES:
        table.setup(spark=spark)

