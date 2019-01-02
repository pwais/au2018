import itertools
import os

from au import conf
from au import util
from au.spark import Spark

class BDD100KFixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'bdd100k')

  @classmethod
  def telemetry(cls):
    return os.path.join(cls.ROOT, 'zips', 'bdd100k_info.zip')

  # VIDEO_DIR = os.path.join(ROOT, 'videos')

  @classmethod
  def test_fixture(cls, path):
    relpath = os.path.relpath(path, cls.ROOT)
    return os.path.join(cls.TEST_FIXTURE_DIR, relpath)

  @classmethod
  def create_test_fixtures(cls):
    log = util.create_log()

    log.info("Creating bdd100k test fixtures ...")
    ZIPS_TO_COPY = (
      cls.telemetry(),
    )

    def _copy_n(src, dest, n):
      log.info("Copying %s of %s -> %s ..." % (n, src, dest))

      util.mkdir(os.path.split(dest)[0])

      import zipfile
      with zipfile.ZipFile(src) as zin:
        with zipfile.ZipFile(dest, mode='w') as zout:
          for name in itertools.islice(zin.namelist(), n):
            zout.writestr(name, zin.read(name))
      
      log.info("... done")

    util.cleandir(cls.TEST_FIXTURE_DIR)
    for path in ZIPS_TO_COPY:
      _copy_n(path, cls.test_fixture(path), 10)

  @classmethod
  def setup(cls, spark=None):
    cls.create_test_fixtures()

    ### Transform telemetry into Parquet table

### Utils and Data

# TODO make metaclass
# class MutableTuple(object):
#   # __slots__ = ('dummy',)

#   def __init__(self, **kwargs):
#     for k in self.__slots__:
#       setattr(self, k, kwargs.get(k))

class Meta(object):
  __slots__ = (
    'startTime',
    'endTime',
    'id',
    'filename',
    'timelapse',
    'rideID',
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))

class GPSObs(object):
  __slots__ = (
    'altitude', 
    'longitude', 
    'vertical_accuracy', 
    'horizontal_accuracy', 
    'latitude', 
    'speed',

    'accuracy',
    'course',
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      # FIXME
      if '_' in k:
        kwk = k.replace('_', ' ')
      else:
        kwk = k
      setattr(self, k, kwargs.get(kwk))

class Point3(object):
  __slots__ = ('x', 'y', 'z')

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))

class TimeseriesRow(object):
  __slots__ = (
    'namespace',
    'timestamp',
    'meta',
    
    'accelerometer',
    'gyro',
    'location',
    'gps',

    # # For pyspark https://github.com/apache/spark/blob/master/python/pyspark/sql/types.py#L1058
    # '__dict__', 
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      if k != '__dict__':
        setattr(self, k, kwargs.get(k))
  
  @property
  def __dict__(self):
    return dict((k, getattr(self, k)) for k in self.__slots__)



### Interface

class BDD100kInfoDataset(object):

  NAMESPACE_PREFIX = ''

  FIXTURES = BDD100KFixtures

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
    
    archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry())

    def to_rows(entry):
      import json
      from pyspark.sql import Row
      
      fname = entry.name
      if 'json' not in fname:
        return
      
      # Determine train / test split.  NB: Fisher has 20k info objects
      # on the eval server and is not sharing (hehe)
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
                **row.__dict__)
    
    # ts_row_rdd = archive_rdd.flatMap(to_rows)
    ts_row_rdd = spark.read.json(archive_rdd.map(lambda entry: entry.data))
    return ts_row_rdd



    
    
