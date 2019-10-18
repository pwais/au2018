from au import conf
from au import util
from au.fixtures.datasets import av
from au.test import testconf
from au.test import testutils

import os

import unittest

import pytest



###
### Data Structure Tests
###

class AVURITest(unittest.TestCase):

  def test_basic(self):

    def check_eq(uri, s):
      assert str(uri) == s
      
      # Also exercise URI.__eq__
      assert av.URI.from_str(s) == uri

    check_eq(av.URI(), av.URI.PREFIX)

    check_eq(
      av.URI(dataset='d', split='s', segment_id='s', timestamp=0, topic='t'),
      'avframe://dataset=d&split=s&segment_id=s&timestamp=0&topic=t')
    
    # String timestamps get converted
    check_eq(
      av.URI(dataset='d', split='s', segment_id='s', timestamp='0', topic='t'),
      'avframe://dataset=d&split=s&segment_id=s&timestamp=0&topic=t')

    # Special handling for extra
    check_eq(
      av.URI(dataset='d', extra={'a': 'foo', 'b': 'bar'}),
      'avframe://dataset=d&extra.a=foo&extra.b=bar')
    
  def test_sorting(self):
    # A less-complete URI is always less than a more-complete one
    assert av.URI() < av.URI(dataset='d', timestamp=0, topic='t')
    
    # Ties are broken using tuple-based encoding
    u1 = av.URI(dataset='d', timestamp=0, topic='t')
    u2 = av.URI(dataset='d', timestamp=1, topic='t')
    assert u1 < u2
    assert u1.as_tuple() < u2.as_tuple()
    assert str(u1) < str(u2)  # Usually true, but NB timestamps are NOT padded!



###
### StampedDatumTable Tests
###

def test_base_table_does_nothing(monkeypatch):
  TEST_TEMPDIR = os.path.join(
                    testconf.TEST_TEMPDIR_ROOT,
                    'test_base_table_does_nothing')
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  av.StampedDatumTableBase.setup()
  assert util.missing_or_empty(av.StampedDatumTableBase.table_root())


def test_stamped_datum_table_io(monkeypatch):
  TEST_TEMPDIR = os.path.join(
                    testconf.TEST_TEMPDIR_ROOT,
                    'test_stamped_datum_table_io')
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  class TestStampedDatumTable(av.StampedDatumTableBase):

    DATUMS = (
      av.StampedDatum(
        dataset='d1',
        split='train',
        segment_id='seg1',
        timestamp=1,

        topic='camera',
        camera_image=av.CAMERAIMAGE_PROTO,
      ),
      av.StampedDatum(
        dataset='d1',
        split='train',
        segment_id='seg1',
        timestamp=1,

        topic='point_cloud',
        point_cloud=av.POINTCLOUD_PROTO,
      ),
      av.StampedDatum(
        dataset='d1',
        split='train',
        segment_id='seg1',
        timestamp=1,

        topic='cuboids',
        cuboids=[av.CUBOID_PROTO, av.CUBOID_PROTO],
      ),

      av.StampedDatum(
        dataset='d1',
        split='test',
        segment_id='seg2',
        timestamp=10,

        topic='ego_pose',
        transform=av.TRANSFORM_PROTO,
      ),
    )

    @classmethod
    def _create_datum_rdds(cls, spark):
      return [spark.sparkContext.parallelize(cls.DATUMS)]
    
  assert util.missing_or_empty(TestStampedDatumTable.table_root())
  
  with testutils.LocalSpark.sess() as spark:
    TestStampedDatumTable.setup(spark=spark)

    # Test basic data consistency
    df = TestStampedDatumTable.as_df(spark)
    assert len(TestStampedDatumTable.DATUMS) == df.count()
    
    sd_rdd = TestStampedDatumTable.as_stamped_datum_rdd(spark)
    assert len(TestStampedDatumTable.DATUMS) == sd_rdd.count()
    
    # Records are preserved in full
    assert sorted(sd_rdd.collect()) == sorted(TestStampedDatumTable.DATUMS)



# class AVStampedDatumTableTest(unittest.TestCase):

#   TEST_TEMPDIR = os.path.join(
#                     testconf.TEST_TEMPDIR_ROOT,
#                     'test_AVStampedDatumTable')

  

    


#   @classmethod
#   def setUpClass(cls):
#     # Now monkeypatch au.conf and friends
#     from _pytest.monkeypatch import MonkeyPatch
#     monkeypatch = MonkeyPatch()
#     testconf.use_tempdir(monkeypatch, cls.TEST_TEMPDIR)

#   def test_base_table_does_nothing(self):
#     pass














# class Message(object):

#   def __init__(self, **kwargs):
#     for k, v in kwargs.items():
#       setattr(self, k, v)

# def test_av_yay():

#   from au import util
#   dest = '/tmp/test_messages'
#   util.cleandir(dest)

#   from au.spark import Spark
#   spark = Spark.getOrCreate()

#   from au.fixtures.datasets import av
#   from au.spark import RowAdapter

#   rows = [
#     Message(
#       unused=av.URI(),
#       k={'a': 10, 'b': 10},
#       dataset='test', 
#       segment_id='test_seg',
#       topic='cuboid',
#       timestamp=1,
#       body=av.CUBOID_PROTO),
#     Message(
#       unused=av.URI(),
#       k={'a': 10, 'b': 20},
#       dataset='test', segment_id='test_seg', timestamp=2,
#       topic='cuboid',
#       body=av.CUBOID_PROTO),
#   ]

#   proto = Message(
#       unused=av.URI_PROTO,
#       k={'a': 10},
#       dataset='test', segment_id='test_seg', timestamp=2,
#       topic='cuboid',
#       body=av.CUBOID_PROTO)

#   # rows2 = [
#   #   Message(
#   #     unused=av.URI(),
#   #     k={'a': 10, 'b': 40},
#   #     dataset='test', segment_id='test_seg', timestamp=3,
#   #     topic='pointcloud',
#   #     body=av.CUBOID_PROTO)
#   #     # body=RowAdapter.to_row(av.POINTCLOUD_PROTO)),
#   # ]

#   to_row = RowAdapter.to_row
#   schema = RowAdapter.to_schema(to_row(proto))
#   # row_rdd = spark..map(to_row)
#   df = spark.createDataFrame([to_row(r) for r in rows], schema=schema)
#   df.write.save(
#     format='parquet', partitionBy=['topic'], mode='append', path=dest)

#   dfr = spark.read.option("mergeSchema", "true").parquet(dest)
#   dfr.show()
#   dfr.printSchema()


#   # df = spark.createDataFrame(rows)
#   # df2 = spark.createDataFrame(rows2)
#   # dfc = Spark.union_dfs(df, df2)
#   # dfc.show()
#   # dfc.write.save(
#   #   format='parquet', partitionBy=['topic'], mode='append', path=dest)
#   # df.show(); df2.show()
#   # df.write.save(
#   #   format='parquet', partitionBy=['topic'], mode='append', path=dest)
#   # df2.write.save(
#   #   format='parquet', partitionBy=['topic'], mode='append', path=dest)

#   # dfr = spark.read.option("mergeSchema", "true").parquet(dest)
#   # dfr.show()
#   # dfr.printSchema()

#   # rrows = [RowAdapter.from_row(r) for r in dfr.collect()]
#   # print(rrows[2].body2.cloud)
  