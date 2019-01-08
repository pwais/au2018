from au.test import testutils
from au.fixtures.datasets import bdd100k

import unittest
import os

import pytest

class TestFixtures(bdd100k.Fixtures):
  ROOT = bdd100k.Fixtures.TEST_FIXTURE_DIR

class TestInfoDataset(bdd100k.InfoDataset):
  FIXTURES = TestFixtures

  VIDEOS = set((
    'b9d24e81-a9679e2a.mov',
    'c2bc5a4c-b2bc828b.mov',
    'c6a4abc9-e999da65.mov',
    'b7f75fad-1c1c419b.mov',
    'b2752cd6-12ba5588.mov',
    'be986afd-f734d33e.mov',
    'c53c9807-1eadf674.mov'
  ))

class TestVideoDataset(bdd100k.VideoDataset):
  FIXTURES = TestFixtures
  INFO = TestInfoDataset

class BDD100kTests(unittest.TestCase):
  """Exercise utiltiies in the bdd100k module.  Allow soft failures
  if the user has none of the required zip files.  We assume exclusively
  one of:
     1) the user emplaced the fixtures correctly using aucli
     2) the user has no fixtures and does not need them
  """

  @classmethod
  def setUpClass(cls):
    cls.have_fixtures = False
    try:
      bdd100k.Fixtures.create_test_fixtures()
      cls.have_fixtures = True
    except Exception as e:
      print "Failed to create test fixtures: %s" % (e,)

  @pytest.mark.slow
  def test_info_dataset(self):
    if not self.have_fixtures:
      return
    
    with testutils.LocalSpark.sess() as spark:
      meta_rdd = TestInfoDataset.create_meta_rdd(spark)
      metas = meta_rdd.collect()
      videos = set(meta.video for meta in metas)
      assert videos == TestInfoDataset.VIDEOS

  @pytest.mark.slow
  def test_video_datset(self):
    if not self.have_fixtures:
      return

    EXPECTED_VIDEOS = set(TestInfoDataset.VIDEOS)
    EXPECTED_VIDEOS.add('video_with_no_info.mov')

    with testutils.LocalSpark.sess() as spark:

      ### Test VideoMeta
      videometa_df = TestVideoDataset.load_videometa_df(spark)
      videometa_df.show()

      rows = videometa_df.collect()
      assert set(r.video for r in rows) == EXPECTED_VIDEOS

      for row in rows:
        if row.video == 'video_with_no_info.mov':
          # We don't know the epoch time of this video (no BDD100k info) ...
          assert row.startTime == -1
          assert row.endTime == -1
          
          # ... but we can glean some data from the video itself.
          assert row.duration != float('nan')
          assert row.nframes > 0
          assert row.width > 0 and row.height > 0
        else:
          assert row.startTime > 0
          assert row.endTime > 0

      ### Test Videos
      video_rdd = TestVideoDataset.load_video_rdd(spark)
      videos = video_rdd.collect()
      for video in videos:
        if video.name == 'video_with_no_info.mov':
          assert video.timeseries == []
        elif video.name == 'b9d24e81-a9679e2a.mov':
          ts = video.timeseries
          assert len(ts) == 4059
          assert ts[0].t == 1506800843701
          assert ts[0].gyro.x == -0.36110000000000003
          assert ts[0].gyro.y == 0.19210000000000002
          assert ts[0].gyro.z == 0.0012000000000000001

    #   ts_row_rdd = TestInfoDataset._info_table_from_zip(spark)
    #   # df = ts_row_rdd#spark.createDataFrame(ts_row_rdd)
    #   # import ipdb; ipdb.set_trace()
    #   df = TestInfoDataset._ts_table_from_info_table(spark, ts_row_rdd)
    #   df.printSchema()
    #   df.registerTempTable('moof')
    #   spark.sql('select * from moof').show()
    #   # spark.sql('select * from moof').write.parquet('/tmp/yyyyyy', partitionBy=['split', 'video'], compression='gzip')
    
