from au.test import testconf
from au.test import testutils
from au.fixtures.datasets import bdd100k

import unittest
import os

import pytest



## Fixtures

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

class TestVideoFrameTable(bdd100k.VideoFrameTable):
  VIDEO = TestVideoDataset

  TARGET_VID = '0000f77c-6257be58.mov'
  N_FRAMES = 10000

  @classmethod
  def as_imagerow_rdd(cls, spark):
    # While the test set is a small number of vidoes, those videos still
    # contain ~12k frames.  So here we restrict to a small subset.

    row_rdd = super(TestVideoFrameTable, cls).as_imagerow_rdd(spark)

    # class IsTargetVideo(object):
    #   def __call__(self, row):
    #     return '0000f77c-6257be58.mov' in row.uri

    video_rows = row_rdd.filter(lambda row: cls.TARGET_VID in row.uri) #IsTargetVideo())
    rows = video_rows.take(cls.N_FRAMES)
    test_rdd = spark.sparkContext.parallelize(rows, numSlices=10)
    return test_rdd

## Tests

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
    
    # Test smoke
    meta = TestInfoDataset.get_meta_for_video('0000f77c-6257be58.mov')
    assert meta.video == '0000f77c-6257be58.mov'
    assert meta.startTime == 1503833985934

    assert set(TestInfoDataset.videonames()) == TestInfoDataset.VIDEOS

  @pytest.mark.slow
  def test_video_datset(self):
    if not self.have_fixtures:
      return

    EXPECTED_VIDEOS = set(TestInfoDataset.VIDEOS)
    EXPECTED_VIDEOS.add('video_with_no_info.mov')

    with testutils.LocalSpark.sess() as spark:

      ### Test Setup
      TestVideoDataset.setup(spark)

      assert set(TestVideoDataset.videonames()) == EXPECTED_VIDEOS

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
          assert len(rows) == 28
            # NB: actual video file is longer, but ffmpeg is :(
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
    
  @pytest.mark.slow
  def test_video_activations(self):
    if not self.have_fixtures:
      return

    # Use /tmp for test fixtures
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    TEST_TEMPDIR = os.path.join(
                        testconf.TEST_TEMPDIR_ROOT,
                        'test_bdd100k_mobilenet')
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
    

    from au.fixtures import nnmodel
    from au.fixtures.tf import mobilenet

    with testutils.LocalSpark.sess() as spark:
      TestVideoFrameTable.setup(spark=spark)

      class TestTable(nnmodel.ActivationsTable):
        TABLE_NAME = 'bdd100k_mobilenet_activations_test'
        NNMODEL_CLS = mobilenet.Mobilenet
        MODEL_PARAMS = mobilenet.Mobilenet.Small()
        IMAGE_TABLE_CLS = TestVideoFrameTable
      
      TestTable.MODEL_PARAMS.INFERENCE_BATCH_SIZE = 10
      TestTable.setup(spark=spark)


"""
 'MobilenetV2/Logits/output:0',
        'MobilenetV2/embedding:0',
         'MobilenetV2/expanded_conv_16/output:0',
        'MobilenetV2/Predictions/Reshape_1:0',
root@au1:/opt/au# du -sh /tmp/au_test/test_bdd100k_mobilenet/tables/bdd100k_mobilenet_test/dataset\=bdd100k.video/split\=train/*
128M    /tmp/au_test/test_bdd100k_mobilenet/tables/bdd100k_mobilenet_test/dataset=bdd100k.video/split=train/part-00002-f4bf9899-3b64-4f2c-814d-335e24f1487c.c000.gz.parquet
34M     /tmp/au_test/test_bdd100k_mobilenet/tables/bdd100k_mobilenet_test/dataset=bdd100k.video/split=train/part-00003-f4bf9899-3b64-4f2c-814d-335e24f1487c.c000.gz.parquet
62M     /tmp/au_test/test_bdd100k_mobilenet/tables/bdd100k_mobilenet_test/dataset=bdd100k.video/split=train/part-00006-f4bf9899-3b64-4f2c-814d-335e24f1487c.c000.gz.parquet
95M     /tmp/au_test/test_bdd100k_mobilenet/tables/bdd100k_mobilenet_test/dataset=bdd100k.video/split=train/part-00007-f4bf9899-3b64-4f2c-814d-335e24f1487c.c000.gz.parquet
root@au1:/opt/au# du -sh /tmp/au_test/test_bdd100k_mobilenet/
models/ tables/ 
root@au1:/opt/au# du -sh /tmp/au_test/test_                                                                            
test_bdd100k_mobilenet/ test_mobilenet/         
root@au1:/opt/au# du -sh cache/test/         
bdd100k/     yaymap.html  
root@au1:/opt/au# du -sh cache/test/bdd100k/
debug/  index/  videos/ zips/   
root@au1:/opt/au# du -sh cache/test/bdd100k/videos/*
136M    cache/test/bdd100k/videos/100k
root@au1:/opt/au# du -sh cache/test/bdd100k/videos/100k/train/*
20M     cache/test/bdd100k/videos/100k/train/0000f77c-6257be58.mov
20M     cache/test/bdd100k/videos/100k/train/0000f77c-62c2a288.mov
20M     cache/test/bdd100k/videos/100k/train/0000f77c-cb820c98.mov
20M     cache/test/bdd100k/videos/100k/train/0001542f-5ce3cf52.mov
19M     cache/test/bdd100k/videos/100k/train/0001542f-7c670be8.mov
20M     cache/test/bdd100k/videos/100k/train/0001542f-ec815219.mov
20M     cache/test/bdd100k/videos/100k/train/0004974f-05e1c285.mov
"""