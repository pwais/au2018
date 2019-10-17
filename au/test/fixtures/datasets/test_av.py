
import unittest

from au.fixtures.datasets import av

class AVURITest(unittest.TestCase):

  def test_basic(self):

    def check_eq(uri, s):
      assert str(uri) == s
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
  

class Message(object):

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

def test_av_yay():

  from au import util
  dest = '/tmp/test_messages'
  util.cleandir(dest)

  from au.spark import Spark
  spark = Spark.getOrCreate()

  from au.fixtures.datasets import av
  from au.spark import RowAdapter

  rows = [
    Message(
      k={'a': 10, 'b': 10},
      dataset='test', 
      segment_id='test_seg',
      topic='cuboid',
      timestamp=1,
      body=RowAdapter.to_row(av.CUBOID_PROTO)),
    Message(
      k={'a': 10, 'b': 20},
      dataset='test', segment_id='test_seg', timestamp=2,
      topic='cuboid',
      body=RowAdapter.to_row(av.CUBOID_PROTO)),
  ]

  rows2 = [
    Message(
      k={'a': 10, 'b': 40},
      dataset='test', segment_id='test_seg', timestamp=3,
      topic='pointcloud',
      body2=RowAdapter.to_row(av.POINTCLOUD_PROTO)),
  ]

  
  df = spark.createDataFrame(rows)
  df2 = spark.createDataFrame(rows2)
  dfc = Spark.union_dfs(df, df2)
  dfc.show()
  dfc.write.save(
    format='parquet', partitionBy=['k.b'], mode='append', path=dest)
  df.show(); df2.show()
  df.write.save(
    format='parquet', partitionBy=['k.b'], mode='append', path=dest)
  df2.write.save(
    format='parquet', partitionBy=['k.b'], mode='append', path=dest)

  dfr = spark.read.option("mergeSchema", "true").parquet(dest)
  dfr.show()
  dfr.printSchema()

  rrows = [RowAdapter.from_row(r) for r in dfr.collect()]
  print(rrows[2].body.cloud)
  