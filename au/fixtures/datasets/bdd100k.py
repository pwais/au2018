"""

Notes:
 * Expanding bdd100k_videos.zip can take days (!!!).
 * See /docs/bdd100k.md for more info.

"""

import itertools
import json
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

  ## Source Data

  @classmethod
  def telemetry_zip(cls):
    return os.path.join(cls.ROOT, 'zips', 'bdd100k_info.zip')
  
  @classmethod
  def video_zip(cls):
    return os.path.join(cls.ROOT, 'zips', 'bdd100k_videos.zip')

  @classmethod
  def video_dir(cls):
    return os.path.join(cls.ROOT, 'videos')


  ## Derived Data

  @classmethod
  def index_root(cls):
    return os.path.join(cls.ROOT, 'index')

  @classmethod
  def video_index_root(cls):
    return os.path.join(cls.index_root(), 'videos')

  @classmethod
  def video_debug_dir(cls):
    return os.path.join(cls.ROOT, 'debug', 'video')

  ## Test Data

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
          for name in itertools.islice(sorted(zin.namelist()), n):
            zout.writestr(name, zin.read(name))
      
      log.info("... done")

    util.cleandir(cls.TEST_FIXTURE_DIR)
    for path in ZIPS_TO_COPY:
      _copy_n(path, cls.test_fixture(path), 10)
    
    # Videos: just copy the ones that have INFO data
    log.info("Copying videos ...")
    fws = util.ArchiveFileFlyweight.fws_from(
                  cls.test_fixture(cls.telemetry_zip()))
    for fw in fws:
      if 'json' not in fw.name:
        continue
      
      relpath = InfoDataset.json_fname_to_video_fname(fw.name)
      relpath = relpath[len('bdd100k/info/'):]
      path = os.path.join(cls.video_dir(), relpath)
      dest = cls.test_fixture(path)
      util.mkdir(os.path.dirname(dest))
      util.run_cmd('cp -v ' + path + ' ' + dest)
    log.info("... done copying videos.")

    # For testing, create a video that has no INFO
    dest = cls.test_fixture(
              os.path.join(
                cls.video_dir(), '100k', 'train', 'video_with_no_info.mov'))
    codec = 'h264' # Chrome will not play `png` movies
    video_bytes = testutils.VideoFixture(codec=codec).get_bytes()
    with open(dest, 'wc') as f:
      f.write(video_bytes)
    log.info("Wrote synth video to %s ..." % dest)

  @classmethod
  def run_import(cls, spark=None):
    import argparse
    import pprint
  
    parser = argparse.ArgumentParser(
                    description=(
                      "Help import and set up the bdd100k dataset. "
                      "FMI see /docs/bdd100k.md"))
    parser.add_argument(
      '--src', default='/outer_root/tmp',
      help='Find zips in this dir [default %(default)s]')
    parser.add_argument(
      '--dest', default=cls.ROOT,
      help='Place files in this dir [default %(default)s]')
    parser.add_argument(
      '--num-videos', default=1000,
      help='Expand only this many video files [default %(default)s]; ')
    parser.add_argument(
      '--all-videos', default=False, action='store_true',
      help='Expand all videos (equivalent to --num-videos=-1)')
    parser.add_argument(
      '--skip-reindex', default=False, action='store_true',
      help='Skip extended setup / (re-)index phase')
    parser.add_argument(
      '--dry-run', default=False, action='store_true',
      help='Only show what would happen')

    args = parser.parse_args()

    EXPECTED_FILE_TO_DEST = {
      'bdd100k_info.zip': cls.telemetry_zip(),
      'bdd100k_videos.zip': cls.video_zip(),
      # 'bdd100k_drivable_maps.zip': ?,
      # 'bdd100k_images.zip': ?,
      # 'bdd100k_labels_release.zip': ?,
      # 'bdd100k_seg.zip': ?,
    }

    src_paths = list(util.all_files_recursive(args.src))
    found = (
        set(EXPECTED_FILE_TO_DEST.keys())
        & set(os.path.dirname(p) for p in src_paths))
    if found:
      print "Found the following files, which we will import:"
      pprint.pprint(list(found))

    spark = spark or Spark.getOrCreate()


    ### Emplace Data

    def get_path(fname, paths):
      for p in paths:
        if fname in paths:
          return p
    
    def run_safe(cmd):
      if args.dry_run:
        print "DRY RUN SKIPPED: " + cmd
      else:
        util.run_cmd(cmd)

    for fname, dest in sorted(EXPECTED_FILE_TO_DEST.iteritems()):
      src_path = get_path(fname, src_paths)
      if not src_path:
        continue
      
      if fname == 'bdd100k_videos.zip':
        cmd = 'ln -s %s %s' % (src_path, dest)
        run_safe(cmd)

        archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.video_zip())
        archive_rdd = archive_rdd.filter(lambda fw: 'mov' in fw.name)
        n_vids = archive_rdd.count()
        print "Found %s videos in %s ..." % (n_vids, src_path)
        
        if args.all_videos:
          max_videos = n_vids
        else:
          max_videos = min(n_vids, args.num_videes)
        
        vids = sorted(archive_rdd.map(lambda fw: fw.name).collect())
        vids = set(vids[:max_videos])
        vids_to_import = archive_rdd.filter(lambda fw: fw.name in vids)
        print "... importing %s videos ..." % len(vids_to_import.count())
        
        dry_run = args.dry_run
        dest_dir = cls.video_dir()
        def copy_vid(fw):
          vid_dest = os.path.join(dest_dir, fw.name)
          util.mkdir(os.path.dirname(vid_dest))
          if dry_run:
            print "DRY RUN SKIPPED: " + f.name
          else:
            with open(vid_dest, 'wc') as f:
              f.write(fw.data)
        vids_to_import.foreach(copy_vid)

        print "... import complete! Imported to %s ." % dest_dir  

      else:
        cmd = 'cp -v %s %s' % (src_path, dest)
        run_safe(cmd)
    
      
    ### Index Data
    if args.skip_reindex:
      print "Skipping (re-)index phase"
      return
    
    if args.dry_run:
      print "DRY RUN SKIPPED index & setup phase"
    else:
      print "Cleaning index and debug dirs ..."
      util.cleandir(cls.video_index_root())
      util.cleandir(cls.video_debug_dir())
      
      print "Running video setup ..."
      VideoDataset.setup(spark)




    

  # @classmethod
  # def setup(cls, spark=None):
  #   cls.create_test_fixtures()

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

  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)
  
  @staticmethod
  def from_zip_entry(entry):
    if 'json' not in entry.name:
        return None
    video = InfoDataset.json_path_to_video_fname(entry.name)

    # Determine train / test split.  NB: Fisher has 20k info objects
    # on the eval server and is not sharing (hehe)
    if 'train' in entry.name:
      split = 'train'
    elif 'val' in entry.name:
      split = 'val'
    else:
      split = ''
      
    json_bytes = entry.data
    if not json_bytes:
      return None
    else:
      jobj = json.loads(json_bytes)
    return Meta(video=video, split=split, **jobj)

  # def __getstate__(self):
  #   return self.__dict__
  
  # def __setstate__(self, d):
  #   for k in self.__slots__:
  #     setattr(self, k, d.get(k, _Meta_DEFAULTS[k]))


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

  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class Point3(object):
  __slots__ = ('x', 'y', 'z')

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))

class TimeseriesRow(object):
  __slots__ = (
    't',
    'accel',
    'gyro',
    'gps',
    'location',
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  @staticmethod
  def get_series(path, ts):
    
    def rgetattr(v, path):
      if not path:
        return v
      else:
        i = path.find('.')
        if i == -1:
          i = len(path)
        name = path[:i]
        subv = getattr(v, name, None)
        if subv:
          return rgetattr(subv, path[i+1:])
        else:
          return None

    vt = []
    for t in ts:
      v = rgetattr(t, path)
      if v is not None:
        vt.append((t.t, v))
    return vt

  # @staticmethod
  # def get_vt(path, ts):
  #   return [(p.t, getattr(p, name)) for p in pts]
  
  # @staticmethod
  # def get_all_ts(pts):
  # return {
  #     'x(t)': Point3.to_ts('x', pts),
  #     'y(t)': Point3.to_ts('y', pts),
  #     'z(t)': Point3.to_ts('z', pts),
  #   }

  # TODO: convert into a matrix row

#   @property
#   def __dict__(self):
#     return dict((k, getattr(self, k, None)) for k in self.__slots__)

# class TimeseriesRow(object):
#   __slots__ = (
#     'namespace',
#     'timestamp',
#     'meta',
    
#     'accelerometer',
#     'gyro',
#     'location',
#     'gps',
#   )

#   def __init__(self, **kwargs):
#     for k in self.__slots__:
#       if k != '__dict__':
#         setattr(self, k, kwargs.get(k))
  
#   @property
#   def __dict__(self):
#     return dict((k, getattr(self, k, None)) for k in self.__slots__)



### Interface

class InfoDataset(object):

  FIXTURES = Fixtures

  @staticmethod
  def json_path_to_video_fname(fname):
    assert 'json' in fname
    if os.path.sep in fname:
      fname = os.path.basename(fname)
    return InfoDataset.json_fname_to_video_fname(fname)

  @staticmethod
  def json_fname_to_video_fname(fname):
    """Unfortunately, the 'filename' attribute in the Info JSON files
    is complete bunk.  Fisher simply named the json and video files using
    the same hashes.  E.g.
      7b2a9ec9-7f4dd1a6.mov <-> 7b2a9ec9-7f4dd1a6.json
    """
    return fname.replace('.json', '.mov')

  # @classmethod
  # def create_meta_factory_rdd(cls, spark):
  #   """Return an RDD of `Meta` instances for all info JSONs available"""
  #   archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

  #   def get_meta_factory(entry):
  #     if 'json' not in entry.name:
  #       return 
  #     video = InfoDataset.json_path_to_video_fname(entry.name)

  #     # Determine train / test split.  NB: Fisher has 20k info objects
  #     # on the eval server and is not sharing (hehe)
  #     if 'train' in entry.name:
  #       split = 'train'
  #     elif 'val' in entry.name:
  #       split = 'val'
  #     else:
  #       split = ''
      
  #     def get_meta():
  #       import json
  #       json_bytes = entry.data
  #       if not json_bytes:
  #         return None
  #       else:
  #         jobj = json.loads(json_bytes)
  #       return Meta(video=video, split=split, **jobj)
      
  #     return (video, split, get_meta)
    
  #   rdd = archive_rdd.map(get_meta).filter(lambda m: m is not None)
  #   return rdd

  @classmethod
  def _video_to_fw_cache(cls):
    if not hasattr(cls, '_local'):
      cls._local = threading.local()
    
    if not hasattr(cls._local, 'video_to_fw_cache'):
      fws = util.ArchiveFileFlyweight.fws_from(cls.FIXTURES.telemetry_zip())
      vidname = lambda fw: InfoDataset.json_path_to_video_fname(fw.name)
      video_to_fw_cache = dict(
        (vidname(fw), fw)
        for fw in fws
        if 'json' in fw.name # Skip directory entries
      )
      cls._local.video_to_fw_cache = video_to_fw_cache
    return cls._local.video_to_fw_cache

  @classmethod
  def get_raw_info_for_video(cls, videoname):
    video_to_fw = cls._video_to_fw_cache()
    
    fw = video_to_fw.get(videoname, None)
    if fw is None or not fw.data:
      return {}
    jobj = json.loads(fw.data)
    jobj['videoname'] = videoname
    jobj['zip_path'] = fw.name
    return jobj

  @classmethod
  def videonames(cls):
    video_to_fw = cls._video_to_fw_cache()
    return video_to_fw.keys()

  @classmethod
  def get_meta_for_video(cls, videoname):
    jobj = cls.get_raw_info_for_video(videoname)

    zip_path = jobj.get('zip_path', '')
    if 'train' in zip_path:
      split = 'train'
    elif 'val' in zip_path:
      split = 'val'
    else:
      split = ''
    
    return Meta(video=videoname, split=split, **jobj)

  @staticmethod
  def info_json_to_timeseries(jobj):
    def _iter_rows(jobj):
      for datum in jobj.get('gps', []):
        yield TimeseriesRow(
          t=datum['timestamp'],
          gps=GPSObs(**datum),
        )
      
      for datum in jobj.get('locations', []):
        yield TimeseriesRow(
          t=datum['timestamp'],
          location=GPSObs(**datum),
        )
      
      for datum in jobj.get('accelerometer', []):
        yield TimeseriesRow(
          t=datum['timestamp'],
          accel=Point3(**datum),
        )
      
      for datum in jobj.get('gyro', []):
        yield TimeseriesRow(
          t=datum['timestamp'],
          gyro=Point3(**datum),
        )
    
    rows = sorted(_iter_rows(jobj), key=lambda row: row.t)
    return rows

  @classmethod
  def get_timeseries_for_video(cls, videoname):
    jobj = cls.get_raw_info_for_video(videoname)
    return InfoDataset.info_json_to_timeseries(jobj)


# meta = Meta(**jobj)
    
#     namespace = 'bdd100k.' + jobj['videoname']
    
#     for datum in jobj.get('gps', []):
#       yield TimeseriesRow(
#         namespace=namespace,
#         timestamp=datum['timestamp'],
#         meta=meta,
#         gps=GPSObs(**datum),
#       )
    
#     for datum in jobj.get('location', []):
#       yield TimeseriesRow(
#         namespace=namespace,
#         timestamp=datum['timestamp'],
#         meta=meta,
#         location=GPSObs(**datum),
#       )
    
#     for datum in jobj.get('accelerometer', []):
#       yield TimeseriesRow(
#         namespace=namespace,
#         timestamp=datum['timestamp'],
#         meta=meta,
#         accelerometer=Point3(**datum),
#       )
    
#     for datum in jobj.get('gyro', []):
#       yield TimeseriesRow(
#         namespace=namespace,
#         timestamp=datum['timestamp'],
#         meta=meta,
#         gyro=Point3(**datum),
#       )


  # @classmethod
  # def _info_table_from_zip(cls, spark):
  #   archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

  #   def get_filename_split(entry):
  #     import json
      
  #     video_fname = ''
  #     json_bytes = entry.data
  #     if json_bytes:
  #       jobj = json.loads(json_bytes)
  #       video_fname = jobj.get('filename', '')

  #     # Determine train / test split.  NB: Fisher has 20k info objects
  #     # on the eval server and is not sharing (hehe)
  #     if 'train' in video_fname:
  #       split = 'train'
  #     elif 'val' in video_fname:
  #       split = 'val'
  #     else:
  #       split = ''
      
  #     return {'video': video_fname, 'split': split}

  #   filename_split_rdd = archive_rdd.map(get_filename_split)
  #   df = spark.createDataFrame(filename_split_rdd)
  #   df.registerTempTable('bdd100k_info_video_split')

  #   # Let Spark's json magic do schema deduction
  #   jobj_df = spark.read.json(archive_rdd.map(lambda entry: entry.data))
  #   jobj_df.registerTempTable('bdd100k_jobj')

  #   # Join to create dataset
  #   query = """
  #             SELECT *
  #             FROM
  #               bdd100k_info_video_split INNER JOIN bdd100k_jobj
  #             WHERE
  #               bdd100k_info_video_split.video = bdd100k_jobj.filename AND
  #               bdd100k_info_video_split.video != ''
  #           """
  #   info_table_df = spark.sql(query)
  #   return info_table_df

  # @classmethod
  # def _ts_table_from_info_table(cls, spark, info_table_df):
  #   INFO_TABLENAME = 'bdd100k_info'
  #   info_table_df.registerTempTable(INFO_TABLENAME)
  #   info_table_df.printSchema()

  #   TS_QUERIES = (
  #     # Accelerometer
  #     """
  #       SELECT 
  #         video,
  #         split,
  #         a.timestamp as t,
  #         a.x as accel_x,
  #         a.y as accel_y,
  #         a.z as accel_z
  #       FROM 
  #         (SELECT video, split, explode(accelerometer) a FROM {table})
  #     """,

  #     # Gyro
  #     """
  #       SELECT 
  #         video,
  #         split,
  #         gyro.timestamp as t,
  #         gyro.x as gyro_x,
  #         gyro.y as gyro_y,
  #         gyro.z as gyro_z
  #       FROM 
  #         (SELECT video, split, explode(gyro) gyro FROM {table})
  #     """,

  #     # Gps
  #     """
  #       SELECT 
  #         video,
  #         split,
          
  #         gps.timestamp as t,
  #         gps.longitude as gps_long,
  #         gps.latitude as gps_lat,
  #         gps.altitude as gps_alt,
  #         gps.speed as gps_v,

  #         gps.`horizontal accuracy` as gps_herr,
  #         gps.`vertical accuracy` as gps_verr
  #       FROM 
  #         (SELECT video, split, explode(gps) gps FROM {table})
  #     """
  #   )
  
  #   dfs = [
  #     spark.sql(query.format(table=INFO_TABLENAME))
  #     for query in TS_QUERIES
  #   ]

  #   all_cols = set(
  #     itertools.chain.from_iterable(
  #       df.schema.names for df in dfs))
    
  #   from pyspark.sql.functions import lit
  #   for i in range(len(dfs)):
  #     df = dfs[i]
  #     df_missing_cols = all_cols - set(df.schema.names)
  #     for colname in df_missing_cols:
  #       df = df.withColumn(colname, lit(None))
  #     dfs[i] = df
    
  #   ts_df = dfs[0]
  #   for df in dfs[1:]:
  #     ts_df = ts_df.union(df)


  #   # # NB: we must use RDD union because the columns between queries
  #   # # differ; Dataframe union uses SQL UNION, which requires each
  #   # # query to have the same columns.
  #   # ts_rdd = spark.sparkContext.union([df.rdd for df in dfs])
  #   # ts_df = spark.createDataFrame(ts_rdd, samplingRatio=1)
  #   return ts_df

    





  # @classmethod
  # def ts_row_rdd(cls, spark):
    
  #   archive_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())

  #   def to_rows(entry):
  #     import json
  #     from pyspark.sql import Row
      
  #     fname = entry.name
  #     if 'json' not in fname:
  #       return
      
  #     # Determine train / test split.  NB: Fisher has 20k info objects
  #     # on the eval server and is not sharing (hehe)
  #     if 'train' in fname:
  #       split = 'train'
  #     elif 'val' in fname:
  #       split = 'val'
  #     else:
  #       split = ''
      
  #     json_bytes = entry.data
  #     jobj = json.loads(json_bytes)
  #     for row in cls.info_json_to_rows(jobj):
  #       yield Row(
  #               split=split,
  #               fname=fname,
  #               **row.__dict__)
    
  #   # ts_row_rdd = archive_rdd.flatMap(to_rows)
  #   ts_row_rdd = spark.read.json(archive_rdd.map(lambda entry: entry.data))
  #   return ts_row_rdd



















# These defaults help explicitly define attribute types
_VideoMeta_DEFAULTS = dict(
  duration=float('nan'),
  fps=float('nan'),
  nframes=-1,
  width=-1,
  height=-1,
  path='',
  **_Meta_DEFAULTS
)
class VideoMeta(object):
  """A row in the index/bdd100k_videos table.  Designed to speed up Spark jobs
  against raw BDD100k videos"""

  __slots__ = _VideoMeta_DEFAULTS.keys()
    
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, _VideoMeta_DEFAULTS[k]))
  
  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

  @staticmethod
  def from_meta(meta):
    return VideoMeta(**meta.to_dict())

  def update(self, **kwargs):
    for k in self.__slots__:
      if k in kwargs:
        setattr(self, k, kwargs[k])

  # def __getstate__(self):
  #   return self.__dict__
  
  # def __setstate__(self, d):
  #   for k in self.__slots__:
  #     setattr(self, k, d.get(k, _Meta_DEFAULTS[k]))


class Video(object):
  """Flyweight for a single BDD100k video file"""

  __slots__ = ('name', 'path', 'info', '_data', '_local')

  def __init__(self, name='', data=None, path='', info=None):
    self.name = name
    self.path = path
    self._data = data
    self._local = threading.local()
    self.info = info or InfoDataset

  @staticmethod
  def from_videometa(meta, info=None):
    return Video(
      name=meta.video,
      path=meta.path,
      info=info)

  @staticmethod
  def from_path(path, **kwargs):
    return Video(name=Video.videoname(path), path=path, **kwargs)

  @staticmethod
  def videoname(path):
    return os.path.split(path)[-1] if os.path.sep in path else path

  def get_frame(self, i):
    """Get the image frame at offset `i`"""
    return self.reader.get_data(i)
  
  def fill_video_meta(self, videometa):
    imageio_meta = {}
    if self.data:
      imageio_meta = self.reader.get_meta_data()    
      imageio_meta.update({
        'width': imageio_meta['size'][0],
        'height': imageio_meta['size'][1],
      })
      self._local.reader = None
    videometa.update(**imageio_meta)
  
  def get_video_meta(self):
    videometa = VideoMeta()
    meta = self.info.get_meta_for_video(self.name)
    videometa.update(**meta.to_dict())
    self.fill_video_meta(videometa)
    return videometa
    
    # assert self.meta is not None, (self.name, self.path)
    # video_meta = self.meta.to_dict() if self.meta else {}
    # video_meta.update(**imageio_meta)
    # video_meta.update({
    #   'video': self.name,
    #   'path': self.path,
    # })
    # # self._local.reader = None
    # return VideoMeta(**video_meta)

  @property
  def reader(self):
    """Return a thread-local imageio reader for the video.  Designed so 
    that Spark worker threads get individual readers to the same
    process video memory buffer."""
    if not hasattr(self._local, 'reader') or self._local.reader is None:
      format = self.name.split('.')[-1]

      # bdd100k videos have some odd dimension issue, so we silence:
      # "the frame size for reading (1280, 720) is different from the source frame size (720, 1280)"
      # import logging
      # logging.getLogger().setLevel(logging.FATAL)
      # with util.imageio_ignore_warnings():
      import imageio
      import imageio.plugins.ffmpeg
      imageio.plugins.ffmpeg.logging.warning = lambda m: True
      self._local.reader = imageio.get_reader(self.data, format=format)
      # logging.getLogger().setLevel(logging.INFO)
    return self._local.reader

  @property
  def data(self):
    """Return raw video bytes"""
    if isinstance(self._data, util.ArchiveFileFlyweight):
      return self._data.data
    elif self.path != '':
      # Read the whole movie file
      return open(self.path).read()
    else:
      return None
      # raise ValueError(
      #   "No idea what to do with %s %s %s" % (
      #     self._data, self.path, self.name))

  @property
  def timeseries(self):
    return self.info.get_timeseries_for_video(self.name)

  def iter_imagerows(self):
    video_meta = self.get_video_meta()    
    frame_period_ms = (1. / video_meta.fps) * 1e3
    
    # If we know nothing about when this video was recorded, assume it
    # started at epoch 0 rather than the VideoMeta default.
    start_time = max(0, video_meta.startTime)
    for i in range(video_meta.nframes):
      frame_timestamp_ms = int(start_time + frame_period_ms)
      yield dataset.ImageRow(
              dataset='bdd100k.video',
              split=video_meta.split,
              uri='bdd100k.video:' + self.name + ':' + str(frame_timestamp_ms),
              _arr_factory=lambda: self.get_frame(i),
              attrs={
                'nanostamp': int(frame_timestamp_ms * 1e6),
                'bdd100k': {
                  'video': self,
                  'offset': i,
                  'video_meta': video_meta,
                  'frame_timestamp_ms': frame_timestamp_ms,
                }
              }
      )
  

class VideoDebugWebpage(object):

  def __init__(self, video):
    self.video = video
  
  def save(self, dest=None):
    if not dest:
      fname = self.video.name + '.html'
      dest = os.path.join(self.video.info.FIXTURES.video_debug_dir(), fname)
      util.mkdir(self.video.info.FIXTURES.video_debug_dir())
    
    video = self._gen_video_html()
    map_path = self._save_map_html(dest)
    plot_paths = self._save_plots(dest)

    # We'll embed relative paths in the HTML
    map_fname = os.path.basename(map_path)
    plot_fnames = map(os.path.basename, plot_paths)

    map_html = ''
    if map_fname:
      map_html = (
      '<iframe width="40%%" height="40%%" src="%s"></iframe>' % map_fname)
    plots_html = ''.join(
      '<img src="%s" width="400px" object-fit="contain" />' % p
      for p in plot_fnames)

    PAGE = """
      <html>
      <head></head>
      <body>
        <div height="40%%">
          {video} {map}
        </div>
        <br>
        <div>
          {plots}
        </div>
      </body>
      </html>
      """
    
    html = PAGE.format(video=video, map=map_html, plots=plots_html)
    with open(dest, 'wc') as f:
      f.write(html)
    util.log.info("Saved page to %s" % dest)

  def _gen_video_html(self):
    VIDEO = """
    <video controls src="{path}" width="50%" object-fit="cover"></video>
    """
    meta = self.video.get_video_meta()
    if not meta.path:
      return '(no video for %s)' % (meta.to_dict(),)
    path = os.path.relpath(
              meta.path,
              self.video.info.FIXTURES.video_debug_dir())
    util.log.info("Saving page for video at %s" % path)
    return VIDEO.format(path=path)

  def _save_map_html(self, dest_base):
    ts = self.video.timeseries
    if not ts:
      return ''
    
    import numpy as np
    import gmplot

    gps = TimeseriesRow.get_series('gps', ts)
    locs = TimeseriesRow.get_series('location', ts)
    
    gps_lats = [v for t, v in TimeseriesRow.get_series('gps.latitude', ts)]
    gps_lons = [v for t, v in TimeseriesRow.get_series('gps.longitude', ts)]
    loc_lats = [v for t, v in TimeseriesRow.get_series('location.latitude', ts)]
    loc_lons = [v for t, v in TimeseriesRow.get_series('location.longitude', ts)]
    # loc_lats = [l.latitude for t, l in locs]
    # loc_lons = [l.longitude for t, l in locs]

    center_lat = np.mean(gps_lats or loc_lats)
    center_lon = np.mean(gps_lons or loc_lons)
    zoom_level = 17 # approx 1m/pixel https://groups.google.com/forum/#!topic/google-maps-js-api-v3/hDRO4oHVSeM
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, zoom_level)

    # By default, gmplot references images in the local filesytem
    # https://ezgif.com/image-to-datauri
    # https://images.emojiterra.com/twitter/v11/512px/1f3ce.png
    # NB: ends in '%s' because this variable is hard-coded as a template
    # string in gmplot.  In practice, the browser doesn't care about
    # the extra junk at the end of the data string.
    gmap.coloricon = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAAAXNSR0IArs4c6QAAAAlwSFlzAADR7AAA0ewB+NcmSwAAActpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx4bXA6Q3JlYXRvclRvb2w+d3d3Lmlua3NjYXBlLm9yZzwveG1wOkNyZWF0b3JUb29sPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KGMtVWAAAAlRJREFUKBXVUUtIVGEUPv9j7vU6Y8OMFk422gjj5CvSosiMEkKIAmvRLAyiVW3bRkVSRLSpgRatDNqFrWoXZA+KaMiQNMVsLHF0GN/MnUd37tz/P/0TGATVvgMf5/Wdw3kA/JdC/jY1DgBVOYXDSr0EaAEkURDK+bfg0Cn2J0a5ISKQkXO7Xcrmv3EQgOCLgV9B5WsKvKw3iOXiDfsnEYeAwaRCy5AgPVFHkTfND166GK/u7ihxI0uk45ldf5ds+Hz5CiGwaMYiESIWdvJyJxUo76IQBbwJ278EY3eT4ejxOW8ApAQoqiUiyQKIRXCv3Q49HE8b16s9/gRXhbgMUMVv1O5xedL7chL667RMe3z2K976XiwGsMR0zkQ6V6VH+YX+EI0dbarZ5rNsbZHjx5NbnKX1+yKVOuIypU71JcjPXRXE3gx9QaMiXOnA4xnLda3gkn1eg1Rq4GMkK+lMpJfnx6Y73X7jGK8oQX6VSzcFmcz1Mr27h5wJeRAyC4T666HrwzhtxEcoBQgHvAy9JnC+nmgvrtgwnAhbk6sedqI5xb7hXlyBSjTAAtMpok4RAnYSnk4lSJa2st6wibXBKck5Fd3LeR+cHwtXzHMqk6U6anTsguGJNRAmArpqIJVKQ9XsJ3i2cACeW3X4gE3Q0x1c/ZCwxow6Z5PuQBsv0VeiBRmptqCQp4MTNowSD9zT53CrPkLi3K2rzxONOsIWkOHoyLP1/iJrkOnWJ9PZyKGuztcFt3tJsy1NEMTRvEV2+NdEc+A93nnbdhDMrBHen3tjaM7yD5X8Bi3hKS9JAAAAAElFTkSuQmCC%s'

    # # Plot 'locations'
    # gmap.plot(loc_lats, loc_lons, '#edc169', edge_width=1)
    # for t, l in TimeseriesRow.get_series('location', ts):
    #   gmap.marker(l.latitude, l.longitude, title='loc')
    
    # Plot GPS readings
    gmap.plot(gps_lats, gps_lons, '#6495ed', edge_width=4)
    for t, g in TimeseriesRow.get_series('gps', ts):
      title = "\\n".join(
        k + ' = ' + str(v)
        for k , v in sorted(g.to_dict().iteritems())
        if v)
      gmap.marker(g.latitude, g.longitude, title=title)

    # NB: sadly gmplot can only target files for output
    dest = dest_base + '.map.html'
    gmap.draw(dest)
    util.log.info("Saved map to %s" % dest)
    return dest

  def _save_plots(self, dest_base):
    ts = self.video.timeseries
    if not ts:
      return []
    
    TSs = {
      'accel x(t)': TimeseriesRow.get_series('accel.x', ts),
      'accel y(t)': TimeseriesRow.get_series('accel.y', ts),
      'accel z(t)': TimeseriesRow.get_series('accel.z', ts),

      'gyro x(t)': TimeseriesRow.get_series('gyro.x', ts),
      'gyro y(t)': TimeseriesRow.get_series('gyro.y', ts),
      'gyro z(t)': TimeseriesRow.get_series('gyro.z', ts),

      'gps v(t)': TimeseriesRow.get_series('gps.speed', ts),
      'location course(t)': TimeseriesRow.get_series('location.course', ts),
    }

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for title, vts in sorted(TSs.iteritems()):

      # Vanilla plot of v(t)
      fig = plt.figure()
      ts = [vt[0] for vt in vts]
      vs = [vt[1] for vt in vts]
      plt.plot(ts, vs)
      plt.title(title)

      # Save as a png
      def to_filename(value):
        # Based upon: https://stackoverflow.com/a/295466
        import unicodedata
        import re
        value = unicode(value)
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
        value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
        value = unicode(re.sub('[-\s]+', '-', value))
        return value

      dest = dest_base + '.' + to_filename(title) + '.png'
      plt.savefig(dest)
      util.log.info("Saved plot %s" % dest)
      paths.append(dest)
    return paths

_setup_thruput = None
class VideoDataset(object):
  FIXTURES = Fixtures

  INFO = InfoDataset

  @classmethod
  def _videoname_to_video(cls):
    if not hasattr(cls, '_local'):
      cls._local = threading.local()
    
    if not hasattr(cls._local, 'videoname_to_reader_cache'):
      util.log.info("Finding videos ...")
      videoname_to_reader_cache = {}
      if os.path.exists(cls.FIXTURES.video_zip()):
        assert False, "TODO focus on decompressed videos"
        path = cls.FIXTURES.video_zip()
      
        # archive_rdd = Spark.archive_rdd(spark, path)
        # video_rdd = archive_rdd.map(
        #   lambda fw: \
        #     Video(name=Video.videoname(fw.name), data=fw))
        # return video_rdd
    
      # Prefer to reference files (better disk caching)
      if os.path.exists(cls.FIXTURES.video_dir()):
        video_dir = cls.FIXTURES.video_dir()
        util.log.info("Using expanded dir of videos: %s" % video_dir)
      
        paths = list(util.all_files_recursive(video_dir))
        util.log.info("... found %s total video files." % len(paths))
        videoname_to_reader_cache.update(
          dict(
            (Video.videoname(p), Video.from_path(p))
            for p in paths
            if (
              '.mov' in p and # Skip directories
              not util.is_stupid_mac_file(p))
        ))
      
      cls._local.videoname_to_reader_cache = videoname_to_reader_cache
      util.log.info(
        "... have %s total videos." % len(videoname_to_reader_cache))
    return cls._local.videoname_to_reader_cache

  @classmethod
  def videonames(cls):
    return cls._videoname_to_video().keys()

  @classmethod
  def get_video(cls, videoname):
    return cls._videoname_to_video().get(videoname)

  # @classmethod
  # def _create_video_rdd(cls, spark):
  #   """Return an RDD of `Video` instances for all known videos"""
  #   util.log.info("Finding videos ...")
  #   if os.path.exists(cls.FIXTURES.video_zip()):

  #     assert False, "TODO require zipfile be decompressed for better caching"

  #     # path = cls.FIXTURES.video_zip()
  #     # archive_rdd = Spark.archive_rdd(spark, path)
  #     # video_rdd = archive_rdd.map(
  #     #   lambda fw: \
  #     #     Video(name=Video.videoname(fw.name), data=fw))
  #     # return video_rdd
    
  #   elif os.path.exists(cls.FIXTURES.video_dir()):

  #     video_dir = cls.FIXTURES.video_dir()
  #     util.log.info("Using expanded dir of videos: %s" % video_dir)
      
  #     paths = list(util.all_files_recursive(video_dir))
  #     util.log.info("... found %s total videos." % len(paths))
      
  #     videos = [
  #       Video(name=Video.videoname(p), path=p)
  #       for p in paths
  #       if '.mov' in p
  #     ]
  #     video_rdd = spark.sparkContext.parallelize(videos)
  #     return video_rdd
          
  #   else:
  #     raise ValueError(
  #       "Can't find videos for fixtures %s" % (cls.FIXTURES.__dict__))

  # @classmethod
  # def _create_videometa_rdd(cls, spark):
  #   info_entry_rdd = Spark.archive_rdd(spark, cls.FIXTURES.telemetry_zip())
  #   def to_video_entry(partition_entries):
  #     for entry in partition_entries:
  #       if 'json' not in entry.name:
  #         continue
        
  #       video = InfoDataset.json_path_to_video_fname(entry.name)
  #       yield (video, entry)
  #   video_info_rdd = info_entry_rdd.mapPartitions(to_video_entry)
    
  #   video_rdd = cls._create_video_rdd(spark)
  #   filename_video = video_rdd.map(lambda v: (v.name, v)).cache()
  #   joined = video_info_rdd.fullOuterJoin(filename_video)
    
  #   import multiprocessing
  #   joined = joined.repartition(10 * multiprocessing.cpu_count())
  #   joined = joined.cache()
    
  #   def to_video_meta(k_lr):
  #     key, (meta_entry, video) = k_lr
  #     meta = Meta.from_zip_entry(meta_entry)
  #     if not video:
  #       return VideoMeta(**meta.to_dict())
  #     video.meta = meta
  #     print 'row'
  #     return video.get_video_meta()

  #   videometa_rdd = joined.map(to_video_meta)
  #   return videometa_rdd

  @classmethod
  def load_videometa_df(cls, spark):
    # cls.setup(spark)
    df = spark.read.parquet(cls.FIXTURES.video_index_root())
    return df

  @classmethod
  def load_video_rdd(cls, spark):
    # cls.setup(spark)
    df = cls.load_videometa_df(spark)
    video_rdd = df.rdd.map(
      lambda meta: \
        Video.from_videometa(meta, info=cls.INFO))
    return video_rdd

  @classmethod
  def setup(cls, spark):
    VIDS_PER_PARTITION = 10

    video_index_dir = cls.FIXTURES.video_index_root()
    if util.missing_or_empty(video_index_dir):
      util.log.info("Creating video meta index ...")

      vids_with_info = cls.INFO.videonames()[:50]
      vids_with_movs = cls.videonames()[:50]
      all_vids = set(vids_with_info).union(set(vids_with_movs))
      util.log.info("... have %s total videos to index ..." % len(all_vids))

      # Use mapPartitions below to limit json / imageio memory usage
      # by partition size
      global _setup_thruput
      _setup_thruput = Spark.thruput_accumulator(spark)
      def gen_videometas(vidnames):
        t = util.ThruputObserver()
        for vidname in vidnames:
          with t.observe(n=1):
            meta = cls.INFO.get_meta_for_video(vidname)
            videometa = VideoMeta.from_meta(meta)
            video = cls.get_video(vidname)
            if video:
              video.fill_video_meta(videometa)
            yield videometa
        global _setup_thruput
        _setup_thruput += t
      
      vids_rdd = spark.sparkContext.parallelize(all_vids)
  
      n_partitions = max(1, int(len(all_vids) / VIDS_PER_PARTITION))
      vids_rdd = vids_rdd.repartition(n_partitions)
      videometa_rdd = vids_rdd.mapPartitions(gen_videometas)
      
      from pyspark.sql import Row
      row_rdd = videometa_rdd.map(lambda vm: Row(**vm.to_dict()))
      util.log.info("Video meta index:")
      spark.createDataFrame(row_rdd.sample(False, 0.1, 0)).show()
      
      util.log.info("Writing meta index to %s ..." % video_index_dir)
      df = spark.createDataFrame(row_rdd)
      df.write.parquet(video_index_dir, mode='overwrite', compression='gzip')
      util.log.info("... wrote video meta index to %s ." % video_index_dir)

      t_end = _setup_thruput.value
      t_end.num_bytes = os.path.getsize(cls.FIXTURES.telemetry_zip())
      util.log.info("Stats:")
      util.log.info(str(t_end))
    

    video_debug_dir = cls.FIXTURES.video_debug_dir()
    if util.missing_or_empty(video_debug_dir):
      util.log.info("Creating video debug webpages ...")
      video_rdd = cls.load_video_rdd(spark)

      # Don't OOM creating webpages
      n_partitions = max(1, int(video_rdd.count() / VIDS_PER_PARTITION))
      video_rdd = video_rdd.repartition(n_partitions)
      video_rdd.foreach(lambda v: VideoDebugWebpage(v).save())

    
class VideoFrameTable(dataset.ImageTable):

  TABLE_NAME = 'bdd100k_videos'

  VIDEO = VideoDataset
  
  @classmethod
  def setup(cls):
    util.log.info("User must manually set up files.  TODO instructions.")

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
 
    
