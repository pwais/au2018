import itertools
import os
import threading

from au import conf
from au import util
from au.fixtures import dataset
from au.spark import Spark
from au.test import testutils

class Fixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'bdd100k')

  @classmethod
  def telemetry_zip(cls):
    return os.path.join(cls.ROOT, 'zips', 'bdd100k_info.zip')
  
  @classmethod
  def video_zip(cls):
    return os.path.join(cls.ROOT, 'zips', 'bdd100k_videos.zip')

  @classmethod
  def video_index_root(cls):
    return os.path.join(cls.ROOT, 'index', 'bdd100k_videos')

  @classmethod
  def video_dir(cls):
    return os.path.join(cls.ROOT, 'videos')

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
    
    # Videos: just create synthetic ones since the zipfile is 1.8TB
    log.info("Creating videos ...")
    fws = util.ArchiveFileFlyweight.fws_from(
                  cls.test_fixture(cls.telemetry_zip()))
    def get_video_fname(fw):
      if 'json' not in fw.name:
        return None
      return InfoDataset.json_fname_to_video_fname(fw.name)

    video_fnames = [get_video_fname(fw) for fw in fws]
    video_fnames = [f for f in video_fnames if f]
    video_fnames.append('video_with_no_info.mov')

    video_bytes = testutils.create_video(
                    n=10,
                    w=1280,
                    h=720,
                    format='mov',
                    fps=29.97)
    real_basedir = os.path.join(cls.video_dir(), '100k', 'train')
    basedir = cls.test_fixture(real_basedir)
    util.cleandir(basedir)
    for fname in video_fnames:
      dest = os.path.join(basedir, fname)
      with open(dest, 'wc') as f:
        f.write(video_bytes)
      log.info("... wrote %s ..." % dest)
    log.info("... done creating videos.")

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

# These defaults help explicitly define attribute types
_Meta_DEFAULTS = {
  'startTime': -1,
  'endTime': -1,
  'id': '',
  #'filename': '', -- this attribute is misleading; Fisher deprecated it
  'timelapse': False,
  'rideID': '',
  'split': '',
  'video': '',
}
class Meta(object):
  __slots__ = _Meta_DEFAULTS.keys()

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, _Meta_DEFAULTS[k]))

  @property
  def __dict__(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

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
    return dict((k, getattr(self, k, None)) for k in self.__slots__)



### Interface

class InfoDataset(object):

  NAMESPACE_PREFIX = ''

  FIXTURES = Fixtures

  @staticmethod
  def json_fname_to_video_fname(fname):
    """Unfortunately, the 'filename' attribute in the Info JSON files
    is complete bunk.  Fisher simply named the json and video files using
    the same hashes.  E.g.
      7b2a9ec9-7f4dd1a6.mov <-> 7b2a9ec9-7f4dd1a6.json
    """
    assert 'json' in fname
    if os.path.sep in fname:
      fname = os.path.split(fname)[-1]
    return fname.replace('.json', '.mov')

  @classmethod
  def create_meta_rdd(cls, spark):
    """Return an RDD of `Meta` instances for all info JSONs available"""
    archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

    def get_meta(entry):
      import json
      
      video_fname = ''
      json_bytes = entry.data
      if not json_bytes:
        return None
      
      jobj = json.loads(json_bytes)
      video = InfoDataset.json_fname_to_video_fname(entry.name)

      # Determine train / test split.  NB: Fisher has 20k info objects
      # on the eval server and is not sharing (hehe)
      if 'train' in video_fname:
        split = 'train'
      elif 'val' in video_fname:
        split = 'val'
      else:
        split = ''
      return Meta(video=video, split=split, **jobj)
    
    meta_rdd = archive_rdd.map(get_meta).filter(lambda m: m is not None)
    return meta_rdd




  # @classmethod
  # def video_to_meta(cls):
  #   if not hasattr(cls, '_video_to_meta'):
  #     path = cls.FIXTURES.telemetry_zip()
  #     fws = util.ArchiveFileFlyweight.fws_from(path)

  #     import json
  #     video_to_meta = {}
  #     for fw in fws:
  #       jobj = json.load(fw.data)
  #       meta = Meta(**jobj)
  #       video_to_meta[meta.filename] = meta

  #     # Single assignment is thread-robust
  #     self._video_to_meta = video_to_meta 
  #   return cls._video_to_meta

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



















# These defaults help explicitly define attribute types
_VideoMeta_DEFAULTS = dict(
  duration=float('nan'),
  fps=float('nan'),
  nframes=0,
  width=0,
  height=0,
  **_Meta_DEFAULTS
)
class VideoMeta(object):
  """A row in the index/bdd100k_videos table.  Designed to speed up Spark jobs
  against raw BDD100k videos"""

  __slots__ = _VideoMeta_DEFAULTS.keys()
    
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, _VideoMeta_DEFAULTS[k]))
  
  @property
  def __dict__(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class Video(object):
  """Flyweight for a single BDD100k video file"""

  __slots__ = ('name', 'video_meta', '_data', '_local')

  def __init__(self, name='', video_meta=None, data=None):
    self.name = name
    self.video_meta = video_meta
    self._data = data
    self._local = threading.local()

  @staticmethod
  def videoname(path):
    return os.path.split(path)[-1] if os.path.sep in path else path

  def get_frame(self, i):
    """Get the image frame at offset `i`"""
    return self.reader.get_data(i)
  
  def get_video_meta(self):
    imageio_meta = self.reader.get_meta_data()
    imageio_meta.update({
      'width': imageio_meta['size'][0],
      'height': imageio_meta['size'][1],
      'video': self.name,
    })
    video_meta = self.video_meta.__dict__ if self.video_meta else {}
    video_meta.update(**imageio_meta)
    return VideoMeta(**video_meta)

  @property
  def reader(self):
    """Return a thread-local imageio reader for the video"""
    if not hasattr(self._local, 'reader'):
      import imageio
      format = self.name.split('.')[-1]
      self._local.reader = imageio.get_reader(self.data, format=format)
    return self._local.reader

  @property
  def data(self):
    """Return raw video bytes"""
    assert self._data is not None
    if isinstance(self._data, util.ArchiveFileFlyweight):
      return self._data.data
    elif isinstance(self._data, basestring):
      # Read the whole movie file
      return open(self._data).read()
    else:
      raise ValueError("No idea what to do with %s" % (self._data,))

  def iter_imagerows(self):
    start_time = self.info_meta.startTime
    
    video_meta = self.video_meta
    frame_period_ms = (1. / video_meta['fps']) * 1e3
    n_frames = video_meta['nframes']

    for i in range(n_frames):
      frame_timestamp_ms = int(start_time + frame_period_ms)
      yield ImageRow(
              dataset='bdd100k.video',
              # TODO split
              uri='bdd100k.video:' + self.name + ':' + frame_timestamp_ms,
              _arr_factory=lambda: self.get_frame(i),
              attrs={
                'nanostamp': frame_timestamp_ms * 1e6,
                'bdd100k': {
                  'video': self.name,
                  'offset': i,
                  'meta': video_meta,
                  'frame_timestamp_ms': frame_timestamp_ms,
                }
              }
      )


class VideoDataset(object):
  FIXTURES = Fixtures

  INFO = InfoDataset

  @classmethod
  def create_video_rdd(cls, spark):
    """Return an RDD of `Video` instances for all known videos"""
    util.log.info("Finding videos ...")
    if os.path.exists(cls.FIXTURES.video_zip()):

      assert False, "TODO require zipfile be decompressed for better caching"

      # path = cls.FIXTURES.video_zip()
      # archive_rdd = Spark.archive_rdd(spark, path)
      # video_rdd = archive_rdd.map(
      #   lambda fw: \
      #     Video(name=Video.videoname(fw.name), data=fw))
      # return video_rdd
    
    elif os.path.exists(cls.FIXTURES.video_dir()):

      video_dir = cls.FIXTURES.video_dir()
      util.log.info("Using expanded dir of videos: %s" % video_dir)
      
      paths = list(util.all_files_recursive(video_dir))
      util.log.info("... found %s total videos." % len(paths))
      
      videos = [Video(name=Video.videoname(p), data=p) for p in paths]
      video_rdd = spark.sparkContext.parallelize(videos)
      return video_rdd
          
    else:
      raise ValueError(
        "Can't find videos for fixtures %s" % (cls.FIXTURES.__dict__))

  @classmethod
  def _create_videometa_rdd(cls, spark):
    meta_rdd = cls.INFO.create_meta_rdd(spark)
    video_rdd = cls.create_video_rdd(spark)

    filename_meta = meta_rdd.map(lambda m: (m.video, m)).cache()
    filename_video = video_rdd.map(lambda v: (v.name, v)).cache()
    joined = filename_meta.fullOuterJoin(filename_video)

    def to_video_meta(k_lr):
      key, (meta, video) = k_lr
      if not video:
        return VideoMeta(**meta.__dict__)
      video.video_meta = meta
      return video.get_video_meta()

    videometa_rdd = joined.map(to_video_meta)
    return videometa_rdd

  @classmethod
  def load_videometa_df(cls, spark):
    cls.setup(spark)

    import pandas as pd
    df = spark.read.parquet(cls.FIXTURES.video_index_root())
    return df

  @classmethod
  def setup(cls, spark):
    if not os.path.exists(cls.FIXTURES.video_index_root()):
      util.log.info("Creating video meta index ...")
      videometa_rdd = cls._create_videometa_rdd(spark)
      
      from pyspark.sql import Row
      row_rdd = videometa_rdd.map(lambda vm: Row(**vm.__dict__))
      df = spark.createDataFrame(row_rdd)
      
      dest = cls.FIXTURES.video_index_root()
      df.write.parquet(dest, compression='gzip')
      util.log.info("... wrote video meta index to %s ." % dest)


    # path = cls.FIXTURES.telemetry_zip()
    # fws = util.ArchiveFileFlyweight.fws_from(path)

    # def read_meta(fw):
    #   import json
    #   jobj = json.load(fw.data)
    #   return Meta(**jobj)

    # meta_rdd = 
    

class VideoFrameTable(dataset.ImageTable):

  TABLE_NAME = 'bdd100k_videos'

  FIXTURES = Fixtures

  INFO = InfoDataset
  
  @classmethod
  def setup(cls):
    util.log.info("User must manually set up files.  TODO instructions.")
    
  ## Utils

  




  # @classmethod
  # def videos(cls):
  #   """Return a cached map of video -> `Video` instances"""
    
  #   class VideoFlyweight(object):
  #     __slots__ = ('_data')

  #     def __init__(self, data=None):
  #       self._data = data

      

    

  #   # Load flyweights to videos
  #   if not hasattr(cls, '_videos'):
      
    
  #   return cls._videos

  # @staticmethod
  # def _video_to_rows(vidname, vidfw):
    
  #   # class LazyFrame(object):
  #   #   __slots__ = ('n', 'vidfw', '_reader')
  #   #   def __init__(self, n=-1, vidfw=None):
  #   #     self.n = n
  #   #     self.vidfw = vidfw
  #   #     self._reader = None
      
  #   #   @property
  #   #   def reader(self):
  #   #     if self._reader is None:
  #   #       import imageio
  #   #       self._reader = 
  #   #   def __call__(self):
  #   #     import imageio




  ## ImageTable API

  @classmethod
  def save_to_image_table(cls, rows):
    raise ValueError("The BDD100k Video Dataset is read-only")

  @classmethod
  def get_rows_by_uris(cls, uris):
    pass
    # import pandas as pd
    # import pyarrow.parquet as pq
    
    # pa_table = pq.read_table(cls.table_root())
    # df = pa_table.to_pandas()
    # matching = df[df.uri.isin(uris)]
    # return list(ImageRow.from_pandas(matching))

  @classmethod
  def iter_all_rows(cls):
    pass
    # """Convenience method (mainly for testing) using Pandas"""
    # import pandas as pd
    # import pyarrow.parquet as pq
    
    # pa_table = pq.read_table(cls.table_root())
    # df = pa_table.to_pandas()
    # for row in ImageRow.from_pandas(df):
    #   yield row
  
  @classmethod
  def as_imagerow_rdd(cls, spark):
    pass
    # df = spark.read.parquet(cls.table_root())
    # row_rdd = df.rdd.map(lambda row: ImageRow(**row.asDict()))
    # return row_rdd
 
    
