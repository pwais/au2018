from au import util
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

  EXPECTED_FNAMES = ( # NB: a subset
    'train2017/000000000078.jpg', 
    'train2017/000000000450.jpg', 
    'train2017/000000000328.jpg', 
    'train2017/000000000064.jpg', 
    'train2017/000000000384.jpg', 
    'train2017/000000000531.jpg',
  )

class TestMSCOCOImageTableVal(mscoco.MSCOCOImageTableVal):
  FIXTURES = TestFixtures
  ANNOS_CLS = TestValAnnos
  APPROX_MB_PER_SHARD = 10.

  EXPECTED_FNAMES = ( # NB: a subset
    'val2017/000000007511.jpg',
    'val2017/000000006894.jpg',
    'val2017/000000007278.jpg',
    'val2017/000000003553.jpg',
    'val2017/000000001532.jpg',
    'val2017/000000000872.jpg',
  )

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
        util.cleandir(table.table_root())
        table.setup(spark=spark)
        util.run_cmd('du -sh %s' % table.table_root())

        rows = table.as_imagerow_rdd(spark).collect()
        uris = [mscoco.ImageURI.from_uri(r.uri) for r in rows]
        fnames = set(u.image_fname for u in uris)
        assert len(fnames) == (TestFixtures.NUM_IMAGES_IN_TEST_ZIP - 1)
                                    # subtract zip folder entry
        assert set(table.EXPECTED_FNAMES) - fnames == set([])
        assert all(len(r.image_bytes) > 0 for r in rows)

