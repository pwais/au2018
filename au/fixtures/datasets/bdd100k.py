import itertools
import os

from au import conf
from au import util
from au.spark import Spark

class BDD100K(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  ZIPS = os.path.join(ROOT, 'zips')

  TELEMETRY = os.path.join(ZIPS, 'bdd100k_info.zip')

  VIDEO_DIR = os.path.join(ROOT, 'videos')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'bdd100k')

  @classmethod
  def test_fixture(cls, path):
    fname = os.path.split(path)[-1]
    return os.path.join(TEST_FIXTURE_DIR, fname)

  @classmethod
  def create_test_fixtures(cls):
    log = util.create_log()

    log.info("Creating bdd100k test fixtures ...")
    ZIPS_TO_COPY = (
      cls.TELEMETRY,
    )

    def _copy_n(src, dest, n):
      log.info("Copying %s of %s -> %s ..." % (n, src, dest))

      import zipfile
      with zipfile.ZipFile(src) as zin:
        with zipfile.ZipFile(dest, mode='w') as zout:
          for zinfo in itertools.islice(zin.infolist(), n):
            zout.write(zinfo)
      
      log.info("... done")

    util.cleandir(cls.TEST_FIXTURE_DIR)
    for path in ZIPS_TO_COPY:
      _copy_n(path, cls.test_fixture(path), 10)

  @classmethod
  def setup(cls, spark=None):
    cls.create_test_fixtures()

    ### Transform telemetry into Parquet table

### Utils and Data

class MutableTuple(object):
  __SLOTS__ = tuple()

  def __init__(self, **kwargs):
    for k in self.__SLOTS__:
      setattr(self, k, kwargs.get(k))

class Meta(MutableTuple):
  __SLOTS__ = (
    'startTime',
    'endTime',
    'id',
    'filename',
    'timelapse',
    'rideID',
  )

class GPSObs(MutableTuple):
  __SLOTS__ = (
    'altitude', 
    'longitude', 
    'vertical accuracy', 
    'horizontal accuracy', 
    'latitude', 
    'speed',

    'accuracy',
    'course',
  )

class Point3(MutableTuple):
  __SLOTS__ = ('x', 'y', 'z')
  

class TimeseriesRow(object):
  __SLOTS__ = (
    'namespace',
    'timestamp',
    'meta',
    
    'accelerometer',
    'gyro',
    'location',
    'gps',
  )



### Interface

class BDD100kInfoDataset(object):

  NAMESPACE_PREFIX = ''

  @classmethod
  def info_json_to_rows(cls, jobj):
  
    meta = Meta(**jobj)
    
    namespace = cls.NAMESPACE_PREFIX + '.' if cls.NAMESPACE_PREFIX else ''
    namespace = namespace + 'bdd100k.' + meta.filename
    
    for datum in jobj.get('gps', []):
      yield TimeseriesRow(
        namespace=namespace,
        timestamp=datum['timestamp'],
        meta=meta,
        gps=GPSObs(**datum),
      )
    
    for datum in jobj.get('location', []):
      yield TimeseriesRow(
        namespace=namespace,
        timestamp=datum['timestamp'],
        meta=meta,
        location=GPSObs(**datum),
      )
    
    for datum in jobj.get('accelerometer', []):
      yield TimeseriesRow(
        namespace=namespace,
        timestamp=datum['timestamp'],
        meta=meta,
        accelerometer=Point3(**datum),
      )
    
    for datum in jobj.get('gyro', []):
      yield TimeseriesRow(
        namespace=namespace,
        timestamp=datum['timestamp'],
        meta=meta,
        gyro=Point3(**datum),
      )

  @classmethod
  def ts_row_rdd(cls, spark):
    
    archive_rdd = Spark.archive_rdd(spark, BDD100K.TELEMETRY)

    def to_rows(entry):
      import json
      from pyspark.sql import Row
      
      fname = entry.name
      if 'train' in fname:
        split = 'train'
      elif 'val' in fname:
        split = 'val'
      else:
        split = ''
      
      json_bytes = entry.data
      jobj = json.loads(json_bytes)
      for row in cls.info_json_to_rows(jobj):
        yield Row(
                split=split,
                fname=fname,
                **row)
    
    ts_row_rdd = archive_rdd.map(to_rows)
    return ts_row_rdd



    
    
