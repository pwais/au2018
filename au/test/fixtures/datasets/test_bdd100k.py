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
    '0000f77c-6257be58.mov',
    '0000f77c-62c2a288.mov',
    '0000f77c-cb820c98.mov',
    '0001542f-5ce3cf52.mov',
    '0001542f-7c670be8.mov',
    '0001542f-ec815219.mov',
    '0004974f-05e1c285.mov'
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
          assert video.timeseries == [] # No info!

          # Test smoke since we know how many frames to expect
          rows = list(video.iter_imagerows())
          assert len(rows) == 30
          assert all(row.as_numpy().shape == (32, 32, 3) for row in rows)

        elif video.name == '0000f77c-cb820c98.mov':
          # Smoke test!
          ts = video.timeseries
          assert len(ts) == 4076
          assert ts[0].t == 1503828163000
          assert ts[0].location.latitude == 40.67684291865739
          assert ts[0].location.longitude == -73.83530301048778

          # Smoke test!
          w = bdd100k.VideoDebugWebpage(video)
          w.save()

    #   ts_row_rdd = TestInfoDataset._info_table_from_zip(spark)
    #   # df = ts_row_rdd#spark.createDataFrame(ts_row_rdd)
    #   # import ipdb; ipdb.set_trace()
    #   df = TestInfoDataset._ts_table_from_info_table(spark, ts_row_rdd)
    #   df.printSchema()
    #   df.registerTempTable('moof')
    #   spark.sql('select * from moof').show()
    #   # spark.sql('select * from moof').write.parquet('/tmp/yyyyyy', partitionBy=['split', 'video'], compression='gzip')
    
