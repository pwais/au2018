import itertools
import os

from au import conf
from au import util
from au.spark import Spark

class BDD100KFixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'bdd100k')

  @classmethod
  def telemetry_zip(cls):
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
      cls.telemetry_zip(),
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
  def _info_table_from_zip(cls, spark):
    archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

    def get_filename_split(entry):
      import json
      
      video_fname = ''
      json_bytes = entry.data
      if json_bytes:
        jobj = json.loads(json_bytes)
        video_fname = jobj.get('filename', '')

      # Determine train / test split.  NB: Fisher has 20k info objects
      # on the eval server and is not sharing (hehe)
      if 'train' in video_fname:
        split = 'train'
      elif 'val' in video_fname:
        split = 'val'
      else:
        split = ''
      
      return {'video': video_fname, 'split': split}

    filename_split_rdd = archive_rdd.map(get_filename_split)
    df = spark.createDataFrame(filename_split_rdd)
    df.registerTempTable('bdd100k_info_video_split')

    # Let Spark's json magic do schema deduction
    jobj_df = spark.read.json(archive_rdd.map(lambda entry: entry.data))
    jobj_df.registerTempTable('bdd100k_jobj')

    # Join to create dataset
    query = """
              SELECT *
              FROM
                bdd100k_info_video_split INNER JOIN bdd100k_jobj
              WHERE
                bdd100k_info_video_split.video = bdd100k_jobj.filename AND
                bdd100k_info_video_split.video != ''
            """
    info_table_df = spark.sql(query)
    return info_table_df

  @classmethod
  def _ts_table_from_info_table(cls, spark, info_table_df):
    INFO_TABLENAME = 'bdd100k_info'
    info_table_df.registerTempTable(INFO_TABLENAME)
    info_table_df.printSchema()

    TS_QUERIES = (
      # Accelerometer
      """
        SELECT 
          video,
          split,
          a.timestamp as t,
          a.x as accel_x,
          a.y as accel_y,
          a.z as accel_z
        FROM 
          (SELECT video, split, explode(accelerometer) a FROM {table})
      """,

      # Gyro
      """
        SELECT 
          video,
          split,
          gyro.timestamp as t,
          gyro.x as gyro_x,
          gyro.y as gyro_y,
          gyro.z as gyro_z
        FROM 
          (SELECT video, split, explode(gyro) gyro FROM {table})
      """,

      # Gps
      """
        SELECT 
          video,
          split,
          
          gps.timestamp as t,
          gps.longitude as gps_long,
          gps.latitude as gps_lat,
          gps.altitude as gps_alt,
          gps.speed as gps_v,

          gps.`horizontal accuracy` as gps_herr,
          gps.`vertical accuracy` as gps_verr
        FROM 
          (SELECT video, split, explode(gps) gps FROM {table})
      """
    )
  
    dfs = [
      spark.sql(query.format(table=INFO_TABLENAME))
      for query in TS_QUERIES
    ]

    all_cols = set(
      itertools.chain.from_iterable(
        df.schema.names for df in dfs))
    
    from pyspark.sql.functions import lit
    for i in range(len(dfs)):
      df = dfs[i]
      df_missing_cols = all_cols - set(df.schema.names)
      for colname in df_missing_cols:
        df = df.withColumn(colname, lit(None))
      dfs[i] = df
    
    ts_df = dfs[0]
    for df in dfs[1:]:
      ts_df = ts_df.union(df)


    # # NB: we must use RDD union because the columns between queries
    # # differ; Dataframe union uses SQL UNION, which requires each
    # # query to have the same columns.
    # ts_rdd = spark.sparkContext.union([df.rdd for df in dfs])
    # ts_df = spark.createDataFrame(ts_rdd, samplingRatio=1)
    return ts_df

    





  @classmethod
  def ts_row_rdd(cls, spark):
    
    archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

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



    
    
