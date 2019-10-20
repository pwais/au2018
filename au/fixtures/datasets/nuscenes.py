
import itertools
import os

import pandas as pd
import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av



## Utils

def transform_from_record(rec, src_frame='', dest_frame=''):
  from pyquaternion import Quaternion
  return av.Transform(
          rotation=Quaternion(rec['rotation']).rotation_matrix,
          translation=np.array(rec['translation']),
          src_frame=src_frame,
          dest_frame=dest_frame)

def get_camera_normal(K, extrinsic):
    """FMI see au.fixtures.datasets.auargoverse.get_camera_normal()"""

    # Build P
    # P = |K 0| * | R |T|
    #             |000 1|
    K_h = np.zeros((3, 4))
    K_h[:3, :3] = K
    P = K_h.dot(extrinsic)

    # Zisserman pg 161 The principal axis vector.
    # P = [M | p4]; M = |..|
    #                   |m3|
    # pv = det(M) * m3
    pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
    pv_hat = pv / np.linalg.norm(pv)
    return pv_hat

def to_nanostamp(timestamp_micros):
  return int(timestamp_micros) * 1000

import shelve
from nuscenes.nuscenes import NuScenes
class AUNuScenes(NuScenes):

  ### Memory Efficiency
  # The base NuScenes object uses 8GB resident RAM (each instance) due to
  # the "tables" of JSON data that it loads.  Below we replace these "tables"
  # with disk-based `shelve`s in order to dramatically reduce memory usage.
  # This change is needed in order to support instantiating multiple
  # NuScenes readers per machine (e.g. for Spark)

  CACHE_ROOT = os.path.join(conf.AU_DATA_CACHE, 'nuscenes_table_cache')

  SAMPLE_DATA_TS_CACHE_NAME = 'sample_data_ts_df'

  ALL_CAMERAS = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
  )

  def _get_cache_path(self, cache_name):
    return os.path.join(self.CACHE_ROOT, self.version, cache_name)

  def __init__(self, **kwargs):
    self.version = kwargs['version']
      # Base ctor does this, but we'll do it here for convenience
    
    if util.missing_or_empty(self._get_cache_path('')):
      util.log.info("Creating shelve caches.  Reading source JSON ...")
      nusc = NuScenes(**kwargs)
        # NB: The above ctor call not only loads all JSON but also runs
        # 'reverse indexing', which edits the data loaded into memory.  We'll
        # then write the edited data below using `shelve` so that we don't have
        # to try to make `AUNuScenes` support reverse indexing itself.
      util.log.info("... done loading JSON data ...")
      
      for table_name in nusc.table_names:
        util.log.info("Building shelve cache file %s ..." % table_name)
        
        cache_path = self._get_cache_path(table_name)
        util.mkdir(os.path.dirname(cache_path))
        
        import pickle
        d = shelve.open(cache_path, protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name)
        d.update((r['token'], r) for r in rows)
        d.close()
      util.log.info("... done.")
      del nusc # Free several GB memory
    
    super(AUNuScenes, self).__init__(**kwargs)

  def _get_table(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):
      cache_path = self._get_cache_path(table_name)
      util.log.info("Using shelve cache %s" % cache_path)
      d = shelve.open(cache_path)
      setattr(self, attr, d)
    return getattr(self, attr)

  def __load_table__(self, table_name):
    return self._get_table(table_name).values()
      # Despite the type annotation, the parent class actually returns a list
      # of dicts.  This return type is a Values View (a generator-like thing)
      # and does not break any core NuScenes functionality.
  
  def __make_reverse_index__(self, verbose):
    # NB: Shelve data files have, built-in, the reverse indicies that the
    # base `NuScenes` creates.  See above.
    
    # Build a timestamp index over `sample_data`s to support efficient
    # interpolation.
    cache_path = self._get_cache_path(self.SAMPLE_DATA_TS_CACHE_NAME)
    if not os.path.exists(cache_path):
      util.log.info("Building sample_data / ego_pose timestamp cache ...")

      sample_to_scene = {}
      for sample in self.sample:
        scene = self.get('scene', sample['scene_token'])
        sample_to_scene[sample['token']] = scene['token']
    
      def to_ts_rows(sample_data):
        row = dict(sample_data)
        row['timestamp_ns'] = to_nanostamp(row['timestamp'])
        row['scene_token'] = sample_to_scene[row['sample_token']]
        row['scene_name'] = self.get('scene', row['scene_token'])['name']
        SKIP = ('fileformat', 'next', 'prev', 'height', 'width')
        for skip in SKIP:
          row.pop(skip)
        yield row

        ego_pose = self.get('ego_pose', row['ego_pose_token'])
        ego_row = dict(ego_pose)
        ego_row['channel'] = 'ego_pose'
        ego_row['timestamp_ns'] = to_nanostamp(ego_row['timestamp'])
        ego_row['scene_token'] = row['scene_token']
        ego_row['scene_name'] = row['scene_name']
        SKIP = ('rotation', 'translation')
        for skip in SKIP:
          ego_row.pop(skip)
        
        yield ego_row

      # This job takes 1-2 minutes and is I/O bound on the shelve DB
      # backing `sample_data`.
      t = util.ThruputObserver(
        name='build_rows', n_total=len(self.sample_data))
      df_rows = []
      for sample_datas in util.ichunked(self.sample_data, 1000):
        with t.observe(n=len(sample_datas)):
          df_rows.extend(
            itertools.chain.from_iterable(
              (to_ts_rows(sd) for sd in sample_datas)))
        t.maybe_log_progress(every_n=100000)
      
      util.log.info("... to pandas ...")
      df = pd.DataFrame(df_rows)

      util.log.info("... writing ...")
      df.to_parquet(
        cache_path, partition_cols=['scene_name'], compression=None)
      
      util.log.info("... done.")

    return
  
  def get(self, table_name, token):
    assert table_name in self.table_names, \
      "Table {} not found".format(table_name)
    return self._get_table(table_name)[token]
  
  def getind(self, table_name, token):
    raise ValueError("Unsupported / unnecessary; provided by shelve")



  ### AU-added Utils

  @property
  def sample_data_ts_df(self):
    if not hasattr(self, '_sample_data_ts_df'):
      cache_path = self._get_cache_path(self.SAMPLE_DATA_TS_CACHE_NAME)
      util.log.info("Using sample_data timestamp cache at %s ..." % cache_path)
      self._sample_data_ts_df = pd.read_parquet(cache_path)
      # import pyarrow.parquet as pq
      # dataset = pq.ParquetDataset(cache_path)
      # self._sample_data_ts_df_cached = dataset.read(use_threads=False).to_pandas()
      util.log.info("... read %s ." % cache_path)
    return self._sample_data_ts_df

  def get_nearest_sample_data(self, scene_name, timestamp_ns, channel=None):
    """Get the nearest `sample_data` record and return the record as well
    as the time diff (in nanoseconds)"""
    import time
    s = time.time()
    print('doing slow linear find nearest')

    sample_to_scene = {}
    for sample in self.sample:
      scene = self.get('scene', sample['scene_token'])
      sample_to_scene[sample['token']] = scene['token']

    diff, best_sd = min(
      (sd['timestamp_ns'] - timestamp_ns, sd)
      for sd in self.sample_data
      if (
        self.get('scene', sample_to_scene[sd['sample_token']])['name'] == scene_name
        and (not channel or sd['channel'] == channel)))
    print('done %s' % (time.time() - s))
    return best_sd, diff

    # df = self.sample_data_ts_df
    # # First narrow down to the relevant scene / car and (maybe) sensor
    # df = df[df['scene_name'] == scene_name]
    # if channel:
    #   df = df[df['channel'] == channel]
    
    # nearest = df.iloc[  
    #   (df['timestamp_ns'] - timestamp_ns).abs().argsort()[:1]  ]
    # if len(nearest) > 0:
    #   row = nearest.to_dict(orient='records')[0]
    #   return row, row['timestamp_ns'] - timestamp_ns
    # else:
    #   return None, 0
  
    # def get_row(self, scene_name, timestamp_ns, channel):
    #   """Get the record that exactly matches the given timestamp and channel"""
    #   df = self.sample_data_ts_df
    #   # First narrow down to the relevant scene / car and (maybe) sensor
    #   df = df[df['scene_name'] == scene_name]
    #   df = df[df['channel'] == channel]
    #   df = df[df['timestamp_ns'] == timestamp_ns]
    #   if not len(df):
    #     raise KeyError(
    #             "Can't find record for %s %s %s" % (
    #               scene_name, timestamp_ns, channel))
    #   elif len(df) != 1:
    #     raise ValueError("Multiple results for %s %s %s %s" % (
    #               scene_name, timestamp_ns, channel, df))
    #   else:
    #     return df.iloc[0].to_dict()

  def get_sample_data_for_scene(self, scene_name):
    print('expensive get rows')

    sample_to_scene = {}
    for sample in self.sample:
      scene = self.get('scene', sample['scene_token'])
      sample_to_scene[sample['token']] = scene['token']

    for sd in self.sample_data:
      if self.get('scene', sample_to_scene[sd['sample_token']])['name'] == scene_name:
        yield sd
    # df = self.sample_data_ts_df
    # df = df[df['scene_name'] == scene_name]
    # return df.to_dict(orient='records')



  #### Adhoc Utils

  def get_all_sensors(self):
    return set(itertools.chain.from_iterable(
      s['data'].keys() for s in self.sample))
    # NuScenes:
    # (TODO)
    # Lyft Level 5:
    # 'CAM_FRONT_ZOOMED', 'CAM_BACK', 'LIDAR_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    # 'CAM_BACK_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'LIDAR_TOP',
    # 'LIDAR_FRONT_LEFT', 'CAM_BACK_RIGHT'
  
  def get_all_classes(self):
    return set(anno['category_name'] for anno in self.sample_annotation)
    # NuScenes:
    # (TODO)
    # Lyft Level 5:
    # 'other_vehicle', 'bus', 'truck', 'car', 'bicycle', 'pedestrian', 
    # 'animal', 'emergency_vehicle', 'motorcycle'

  def print_sensor_sample_rates(self, scene_names=None):
    """Print a report to stdout describing the sample rates of all sensors,
    labels, and localization objects."""

    if not scene_names:
      scene_names = [s['name'] for s in self.scene]
    scene_names = set(scene_names)

    scene_tokens = set(
      s['token'] for s in self.scene if s['name'] in scene_names)
    for scene_token in scene_tokens:
      scene_samples = [
        s for s in self.sample if s['scene_token'] == scene_token
      ]

      def to_sec(timestamp):
        # NuScenes and Lyft Level 5 timestamps are in microseconds
        return timestamp * 1e-6

      # Samples are supposed to be 'best groupings' of lidar and camera data.
      # Let's measure their sample rate.
      from collections import defaultdict
      name_to_tss = defaultdict(list)
      for sample in scene_samples:
        name_to_tss['sample (annos)'].append(to_sec(sample['timestamp']))
      
      # Now measure individual sensor sample rates.
      sample_toks = set(s['token'] for s in scene_samples)
      scene_sample_datas = [
        sd for sd in self.sample_data if sd['sample_token'] in sample_toks
      ]
      for sd in scene_sample_datas:
        name_to_tss[sd['channel']].append(to_sec(sd['timestamp']))
        ego_pose = self.get('ego_pose', sd['ego_pose_token'])
        name_to_tss['ego_pose'].append(to_sec(ego_pose['timestamp']))
        name_to_tss['sample_data'].append(to_sec(sd['timestamp']))
      
      annos = [
        a for a in self.sample_annotation if a['sample_token'] in sample_toks
      ]
      num_tracks = len(set(a['instance_token'] for a in annos))

      import itertools
      all_ts = sorted(itertools.chain.from_iterable(name_to_tss.values()))
      from datetime import datetime
      dt = datetime.utcfromtimestamp(all_ts[0])
      start = dt.strftime('%Y-%m-%d %H:%M:%S')
      duration = (all_ts[-1] - all_ts[0])

      # Print a report
      print('---')
      print('---')

      scene = self.get('scene', scene_token)
      print('Scene %s %s' % (scene['name'], scene['token']))
      print('Start %s \tDuration %s sec' % (start, duration))
      print('Num Annos %s (Tracks %s)' % (len(annos), num_tracks))
      import pandas as pd
      from collections import OrderedDict
      rows = []
      for name in sorted(name_to_tss.keys()):
        def get_series(name):
          return np.array(sorted(name_to_tss[name]))
        
        series = get_series(name)
        freqs = series[1:] - series[:-1]

        lidar_series = get_series('LIDAR_TOP')
        diff_lidar_ms = 1e3 * np.mean(
          [np.abs(lidar_series - t).min() for t in series])

        rows.append(OrderedDict((
          ('Series',              name),
          ('Freq Hz',             1. / np.mean(freqs)),
          ('Diff Lidar (msec)',   diff_lidar_ms),
          ('Duration',            series[-1] - series[0]),
          ('Support',             len(series)),
        )))
      print(pd.DataFrame(rows))

      print()
      print()

      # NuScenes:
      # ---
      # ---
      # Scene scene-1000 09f67057dd8346388b28f79d9bb1cf04
      # Start 2018-11-14 11:01:41       Duration 19.922956943511963 sec
      # Num Annos 493 (Tracks 27)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.738035          10.554540  19.850000      234
      # 1       CAM_BACK_LEFT   11.536524           0.918133  19.850000      230
      # 2      CAM_BACK_RIGHT   11.788413          20.070969  19.850000      235
      # 3           CAM_FRONT   11.637280          14.986247  19.850000      232
      # 4      CAM_FRONT_LEFT   11.536524           7.586523  19.850000      230
      # 5     CAM_FRONT_RIGHT   11.536524          22.528135  19.850000      230
      # 6           LIDAR_TOP   19.747937           0.000000  19.850175      393
      # 7     RADAR_BACK_LEFT   12.593898          12.635898  19.850883      251
      # 8    RADAR_BACK_RIGHT   13.211602          12.786931  19.831054      263
      # 9         RADAR_FRONT   12.976290          12.728528  19.882416      259
      # 10   RADAR_FRONT_LEFT   13.510020          12.710849  19.911148      270
      # 11  RADAR_FRONT_RIGHT   13.724216          12.571387  19.891846      274
      # 12           ego_pose  155.599393          11.128198  19.922957     3101
      # 13     sample (annos)    2.015096           0.000000  19.850175       41
      # 14        sample_data  155.599393          11.128198  19.922957     3101
      # ---
      # ---
      # Scene scene-0293 6308d6d934074a028fc3145eedf3e65f
      # Start 2018-08-31 15:25:42       Duration 19.525898933410645 sec
      # Num Annos 3548 (Tracks 277)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.773779          10.441362  19.450000      230
      # 1       CAM_BACK_LEFT   11.825193           1.573582  19.450000      231
      # 2      CAM_BACK_RIGHT   11.928021          19.624993  19.450000      233
      # 3           CAM_FRONT   11.876607          14.872485  19.450000      232
      # 4      CAM_FRONT_LEFT   11.876607           7.329719  19.450000      232
      # 5     CAM_FRONT_RIGHT   11.979434          22.875966  19.450000      234
      # 6           LIDAR_TOP   19.844216           0.000000  19.451512      387
      # 7     RADAR_BACK_LEFT   13.157897          12.698666  19.455997      257
      # 8    RADAR_BACK_RIGHT   13.490288          12.647669  19.421380      263
      # 9         RADAR_FRONT   13.082394          12.616018  19.491845      256
      # 10   RADAR_FRONT_LEFT   13.305139          12.709008  19.466163      260
      # 11  RADAR_FRONT_RIGHT   13.209300          12.723777  19.455989      258
      # 12           ego_pose  157.329504          11.144872  19.525899     3073
      # 13     sample (annos)    2.004986           0.000000  19.451512       40
      # 14        sample_data  157.329504          11.144872  19.525899     3073
      # ---
      # ---
      # Scene scene-1107 89f20737ec344aa48b543a9e005a38ca
      # Start 2018-11-21 11:59:53       Duration 19.820924997329712 sec
      # Num Annos 496 (Tracks 47)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.696203          11.014590  19.750000      232
      # 1       CAM_BACK_LEFT   11.848101           1.382666  19.750000      235
      # 2      CAM_BACK_RIGHT   11.746835          20.483792  19.750000      233
      # 3           CAM_FRONT   11.848103          14.481831  19.749997      235
      # 4      CAM_FRONT_LEFT   11.898734           7.063236  19.750000      236
      # 5     CAM_FRONT_RIGHT   11.898734          22.159666  19.750000      236
      # 6           LIDAR_TOP   19.797377           0.000000  19.750091      392
      # 7     RADAR_BACK_LEFT   13.578124          12.646209  19.811279      270
      # 8    RADAR_BACK_RIGHT   13.271911          12.721395  19.740940      263
      # 9         RADAR_FRONT   13.198245          12.553021  19.775357      262
      # 10   RADAR_FRONT_LEFT   13.730931          12.645984  19.736462      272
      # 11  RADAR_FRONT_RIGHT   13.778595          12.488227  19.740764      273
      # 12           ego_pose  158.317536          11.102567  19.820925     3139
      # 13     sample (annos)    2.025307           0.000000  19.750091       41
      # 14        sample_data  158.317536          11.102567  19.820925     3139

      # Lyft Level 5:
      # ---
      # ---
      # Scene host-a015-lidar0-1235423635198474636-1235423660098038666 755e4564756ad5c92243b7f77039d07ab1cce40662a6a19b67c820647666a3ef
      # Start 2019-02-28 21:13:55       Duration 24.99979877471924 sec
      # Num Annos 1637 (Tracks 44)
      #               Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0           CAM_BACK   5.020080          98.882582  24.900000      126
      # 1      CAM_BACK_LEFT   5.020080          16.919276  24.900000      126
      # 2     CAM_BACK_RIGHT   5.020080          82.887411  24.900000      126
      # 3          CAM_FRONT   5.020080          50.027272  24.900000      126
      # 4     CAM_FRONT_LEFT   5.020080          33.542296  24.900000      126
      # 5    CAM_FRONT_RIGHT   5.020080          66.427920  24.900000      126
      # 6   CAM_FRONT_ZOOMED   5.020080          50.096270  24.900000      126
      # 7          LIDAR_TOP   5.020156           0.000000  24.899626      126
      # 8           ego_pose  40.442375           0.000000  24.899626     1008
      # 9     sample (annos)   5.020156           0.000000  24.899626      126
      # 10       sample_data  40.280324          49.847878  24.999799     1008
      # ---
      # ---
      # Scene host-a004-lidar0-1233947108297817786-1233947133198765096 114b780b2efd6f73f134fc3a8f9db628e43131dc47f90e9b5dfdb886400d70f2
      # Start 2019-02-11 19:05:08       Duration 25.000741004943848 sec
      # Num Annos 4155 (Tracks 137)
      #               Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0           CAM_BACK   5.020080          98.790201  24.900000      126
      # 1      CAM_BACK_LEFT   5.020080          17.030725  24.900000      126
      # 2     CAM_BACK_RIGHT   5.020080          83.033195  24.900000      126
      # 3          CAM_FRONT   5.020080          50.151564  24.900000      126
      # 4     CAM_FRONT_LEFT   5.020080          33.667718  24.900000      126
      # 5    CAM_FRONT_RIGHT   5.020080          66.649443  24.900000      126
      # 6   CAM_FRONT_ZOOMED   5.020080          50.265691  24.900000      126
      # 7          LIDAR_TOP   5.019934           0.000000  24.900724      126
      # 8           ego_pose  40.440592           0.000000  24.900724     1008
      # 9     sample (annos)   5.019934           0.000000  24.900724      126
      # 10       sample_data  40.278806          49.948567  25.000741     1008
      # ---
      # ---
      # Scene host-a101-lidar0-1241886983298988182-1241887008198992182 7b4640d63a9c62d07a8551d4b430d0acd88eaba8249c843248feb888f4630070
      # Start 2019-05-14 16:36:23       Duration 25.002139806747437 sec
      # Num Annos 4777 (Tracks 173)
      #                Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   5.020080          93.825297  24.900000      126
      # 1       CAM_BACK_LEFT   5.020080          85.205165  24.900000      126
      # 2      CAM_BACK_RIGHT   5.020080          19.099243  24.900000      126
      # 3           CAM_FRONT   5.020080          52.394347  24.900000      126
      # 4      CAM_FRONT_LEFT   5.020080          68.799780  24.900000      126
      # 5     CAM_FRONT_RIGHT   5.020080          35.769232  24.900000      126
      # 6    CAM_FRONT_ZOOMED   5.020080          52.394347  24.900000      126
      # 7    LIDAR_FRONT_LEFT   5.020060           0.000000  24.900101      126
      # 8   LIDAR_FRONT_RIGHT   5.020060           0.000000  24.900101      126
      # 9           LIDAR_TOP   5.020060           0.000000  24.900101      126
      # 10           ego_pose  50.562044           0.000000  24.900101     1260
      # 11     sample (annos)   5.020060           0.000000  24.900101      126
      # 12        sample_data  50.355690          40.748741  25.002140     1260



## Data

class Fixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'nuscenes')

  TARBALLS = (
    'v1.0-mini.tar',
    
    'v1.0-test_meta.tar',
    'v1.0-test_blobs.tar',

    'v1.0-trainval01_blobs.tar',
    'v1.0-trainval02_blobs.tar',
    'v1.0-trainval03_blobs.tar',
    'v1.0-trainval04_blobs.tar',
    'v1.0-trainval05_blobs.tar',
    'v1.0-trainval06_blobs.tar',
    'v1.0-trainval07_blobs.tar',
    'v1.0-trainval08_blobs.tar',
    'v1.0-trainval09_blobs.tar',
    'v1.0-trainval10_blobs.tar',

    'nuScenes-map-expansion.zip',
  )

  MINI_TARBALL = 'v1.0-mini.tar'

  SPLITS = ('train', 'val', 'test', 'mini')
  
  TRAIN_TEST_SPLITS = ('train', 'val')


  ## Source Data

  @classmethod
  def tarballs_dir(cls):
    return os.path.join(cls.ROOT, 'tarballs')

  @classmethod
  def tarball_path(cls, fname):
    return os.path.join(cls.tarballs_dir(), fname)


  ## Derived Data
  
  @classmethod
  def dataroot(cls):
    return '/outer_root/media/seagates-ext4/au_datas/nuscenes_root'
    # return '/outer_root/media/seagates-ext4/au_datas/lyft_level_5_root/train'

  # @classmethod
  # def dataroot(cls):
  #   return os.path.join(cls.ROOT, 'nuscenes_dataroot')

  @classmethod
  def index_root(cls):
    return os.path.join(cls.ROOT, 'index')


  ## Setup

  @classmethod
  def run_import(cls, only_mini=False):
    pass


  ## Public API

  @classmethod
  def get_loader(cls, version='v1.0-trainval'):
    """Create and return a `NuScenes` object for the given `version`."""
    nusc = AUNuScenes(version=version, dataroot=cls.dataroot(), verbose=True)
    return nusc
  
  @classmethod
  def get_split_for_scene(cls, scene):
    if not hasattr(cls, '_scene_to_split'):
      from nuscenes.utils.splits import create_splits_scenes
      split_to_scenes = create_splits_scenes()

      scene_to_split = {}
      for split, scenes in split_to_scenes.items():
        # Ignore mini splits because they duplicate train/val
        if 'mini' not in split:
          for s in scenes:
            scene_to_split[s] = split
      cls._scene_to_split = scene_to_split
    # return cls._scene_to_split[scene]
    return 'train' # for Lyft Level 5, we assume all train for now
    # TODO use lyft constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        

class StampedDatumTable(av.StampedDatumTableBase):

  FIXTURES = Fixtures

  NUSC_VERSION = 'v1.0-trainval' # E.g. v1.0-mini, v1.0-trainval, v1.0-test

  SENSORS_KEYFRAMES_ONLY = False
    # NuScenes: If enabled, throttles sensor data to about 2Hz, in tune with
    #   samples; if disabled, samples at full res.
    # Lyft Level 5: all sensor data is key frames.
    # FMI see print_sensor_sample_rates() above.
  
  LABELS_KEYFRAMES_ONLY = True
    # If enabled, samples only raw annotations.  If disabled, will motion-
    # correct cuboids to every sensor reading.


  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/nusc_sd_table'
  
  @classmethod
  def _create_datum_rdds(cls, spark):

    PARTITIONS_PER_SEGMENT = 4 * os.cpu_count()
    PARTITIONS_PER_TASK = os.cpu_count()

    datum_rdds = []
    for segment_id in cls.get_segment_ids():
      for partitions in util.ichunked(range(PARTITIONS_PER_SEGMENT), PARTITIONS_PER_TASK):
        task_rdd = spark.sparkContext.parallelize(
          [(segment_id, partition) for partition in partitions])

        def gen_partition_datums(task):
          segment_id, partition = task
          for i, uri in enumerate(cls.iter_uris_for_segment(segment_id)):
            if (i % PARTITIONS_PER_SEGMENT) == partition:
              yield cls.create_stamped_datum(uri)
        
        datum_rdd = task_rdd.flatMap(gen_partition_datums)
        datum_rdds.append(datum_rdd)
    return datum_rdds

  
  ## Public API

  @classmethod
  def get_nusc(cls):
    if not hasattr(cls, '_nusc'):
      cls._nusc = cls.FIXTURES.get_loader(version=cls.NUSC_VERSION)
    return cls._nusc
  
  @classmethod
  def get_segment_ids(cls):
    nusc = cls.get_nusc()
    return sorted(s['name'] for s in nusc.scene)

  @classmethod
  def iter_uris_for_segment(cls, segment_id):
    nusc = cls.get_nusc()

    scene_split = cls.FIXTURES.get_split_for_scene(segment_id)

    ## Get sensor data and ego pose
    for sd in nusc.get_sample_data_for_scene(segment_id):
      # Note all poses
      yield av.URI(
              dataset='nuscenes', # use fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              split=scene_split,
              segment_id=segment_id,
              timestamp=to_nanostamp(sd['timestamp']),
              topic='ego_pose',
              extra={'nuscenes-token': 'ego_pose|' + sd['ego_pose_token']})

      # Maybe skip the sensor data if we're only doing keyframes
      if cls.SENSORS_KEYFRAMES_ONLY:
        if sd['sensor_modality'] and not sd['is_key_frame']:
          continue

      yield av.URI(
              dataset='nuscenes', # use fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              split=scene_split,
              segment_id=segment_id,
              timestamp=to_nanostamp(sd['timestamp']),
              topic=sd['sensor_modality'] + '|' + sd['channel'],
              extra={'nuscenes-token': 'sample_data|' + sd['token']})

      # Get labels (non-keyframes; interpolated one per track)
      if not cls.LABELS_KEYFRAMES_ONLY:
        yield av.URI(
                dataset='nuscenes',  # use fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                split=scene_split,
                segment_id=segment_id,
                timestamp=to_nanostamp(row['timestamp']),
                topic='labels|cuboids',
                extra={
                  'nuscenes-token': 'sample_data|' + sd['token'],
                })

    ## Get labels (keyframes only)
    if cls.LABELS_KEYFRAMES_ONLY:

      # Get annos for *only* samples, which are keyframes
      scene_tokens = [
        s['token'] for s in nusc.scene if s['name'] == segment_id]
      assert scene_tokens
      scene_token = scene_tokens[0]

      scene_samples = [
        s for s in nusc.sample if s['scene_token'] == scene_token
      ]

      for sample in scene_samples:
        for channel, sample_data_token in sample['data'].items():
          sd = nusc.get('sample_data', sample_data_token)
          yield av.URI(
                  dataset='nuscenes',
                  split=scene_split,
                  segment_id=segment_id,
                  timestamp=to_nanostamp(sd['timestamp']),
                  topic='labels|cuboids',
                  extra={
                    'nuscenes-token': 'sample_data|' + sd['token'],
                  })

  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic.startswith('camera'):
      sample_data = cls.__get_row(uri)
      return cls.__create_camera_image(uri, sample_data)
    elif uri.topic.startswith('lidar') or uri.topic.startswith('radar'):
      sample_data = cls.__get_row(uri)
      return cls.__create_point_cloud(uri, sample_data)
    elif uri.topic == 'ego_pose':
      pose_record = cls.__get_row(uri)
      return cls.__create_ego_pose(uri, pose_record)
    elif uri.topic == 'labels|cuboids':
      sample_data = cls.__get_row(uri)
      # nusc = cls.get_nusc()
      # best_sd, diff_ns = nusc.get_nearest_sample_data(
      #                           uri.segment_id,
      #                           uri.timestamp)
      # assert best_sd
      # assert diff_ns < .01 * 1e9
      return cls.__create_cuboids_in_ego(uri, sample_data['token'])
    else:
      raise ValueError(uri)


  ## Support

  @classmethod
  def __get_row(cls, uri):
    if 'nuscenes-token' in uri.extra:
      record = uri.extra['nuscenes-token']
      table, token = record.split('|')
      nusc = cls.get_nusc()
      return nusc.get(table, token)
    raise ValueError
    # nusc = cls.get_nusc()
    # return nusc.get_row(uri.segment_id, uri.timestamp, uri.topic)
  
  @classmethod
  def __create_camera_image(cls, uri, sample_data):
    nusc = cls.get_nusc()

    camera_token = sample_data['token']
    cs_record = nusc.get(
      'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])

    data_path, _, cam_intrinsic = nusc.get_sample_data(camera_token)
      # Ignore box_list, we'll get boxes in ego frame later
    
    viewport = uri.get_viewport()
    w, h = sample_data['width'], sample_data['height']
    if not viewport:
      from au.fixtures.datasets import common
      viewport = common.BBox.of_size(w, h)

    timestamp = sample_data['timestamp']

    ego_from_cam = transform_from_record(
                      cs_record,
                      dest_frame='ego',
                      src_frame=sample_data['channel'])
    cam_from_ego = ego_from_cam.get_inverse()
    RT_h = cam_from_ego.get_transformation_matrix(homogeneous=True)
    principal_axis_in_ego = get_camera_normal(cam_intrinsic, RT_h)

    ego_pose = transform_from_record(
                      pose_record,
                      dest_frame='ego',
                      src_frame='city')
    ci = av.CameraImage(
            camera_name=sample_data['channel'],
            image_jpeg=bytearray(open(data_path, 'rb').read()),
            height=h,
            width=w,
            viewport=viewport,
            timestamp=to_nanostamp(timestamp),
            ego_pose=ego_pose,
            cam_from_ego=cam_from_ego,
            K=cam_intrinsic,
            principal_axis_in_ego=principal_axis_in_ego,
    )
    return av.StampedDatum.from_uri(uri, camera_image=ci)
  
  @classmethod
  def __create_point_cloud(cls, uri, sample_data):
    # Based upon nuscenes.py#map_pointcloud_to_image()

    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.data_classes import RadarPointCloud

    nusc = cls.get_nusc()

    target_pose_token = sample_data['ego_pose_token']

    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    if sample_data['sensor_modality'] == 'lidar':
      pc = LidarPointCloud.from_file(pcl_path)
    else:
      pc = RadarPointCloud.from_file(pcl_path)

    # Step 1: Points live in the point sensor frame.  First transform to
    # world frame:
    # 1a transform to ego
    # First step: transform the point-cloud to the ego vehicle frame for the
    # timestamp of the sweep.
    cs_record = nusc.get(
                  'calibrated_sensor', sample_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # 1b transform to the global frame.
    poserecord = nusc.get('ego_pose', sample_data['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Step 2: Send points into the ego frame at the target timestamp
    poserecord = nusc.get('ego_pose', target_pose_token)
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    n_xyz = pc.points[:3, :].T
      # Throw out intensity (lidar) and ... other stuff (radar)
    
    ego_pose = transform_from_record(
                      nusc.get('ego_pose', sample_data['ego_pose_token']),
                      dest_frame='ego',
                      src_frame='city')
    ego_to_sensor = transform_from_record(
                      cs_record,
                      src_frame='ego',
                      dest_frame=sample_data['channel'])

    motion_corrected= (sample_data['ego_pose_token'] != target_pose_token)

    pc = av.PointCloud(
            sensor_name=sample_data['channel'],
            timestamp=to_nanostamp(sample_data['timestamp']),
            cloud=n_xyz,
            motion_corrected=motion_corrected,
            ego_to_sensor=ego_to_sensor,
            ego_pose=ego_pose,
    )
    return av.StampedDatum.from_uri(uri, point_cloud=pc)
  
  @classmethod
  def __create_cuboids_in_ego(cls, uri, sample_data_token):
    nusc = cls.get_nusc()

    # NB: This helper always does motion correction (interpolation) unless
    # `sample_data_token` refers to a keyframe.
    boxes = nusc.get_boxes(sample_data_token)
  
    # Boxes are in world frame.  Move all to ego frame.
    from pyquaternion import Quaternion
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for box in boxes:
      # Move box to ego vehicle coord system
      box.translate(-np.array(pose_record['translation']))
      box.rotate(Quaternion(pose_record['rotation']).inverse)

    ego_pose = transform_from_record(
      pose_record, dest_frame='ego', src_frame='city')
    from au.fixtures.datasets.av import NUSCENES_CATEGORY_TO_AU_AV_CATEGORY
    cuboids = []
    for box in boxes:
      cuboid = av.Cuboid()

      # Core
      sample_anno = nusc.get('sample_annotation', box.token)
      cuboid.track_id = \
        'nuscenes_instance_token:' + sample_anno['instance_token']
      cuboid.category_name = box.name
      cuboid.timestamp = to_nanostamp(sd_record['timestamp'])
      
      cuboid.au_category = NUSCENES_CATEGORY_TO_AU_AV_CATEGORY[box.name]
      
      # Try to give bikes riders
      # NB: In Lyft Level 5, they appear to *not* label bikes without riders
      attribs = [
        nusc.get('attribute', attrib_token)['name']
        for attrib_token in sample_anno['attribute_tokens']
      ]
      if 'cycle.with_rider' in attribs:
        if cuboid.au_category == 'bike_no_rider':
          cuboid.au_category = 'bike_with_rider'
        elif cuboid.au_category == 'motorcycle_no_rider':
          cuboid.au_category = 'motorcycle_with_rider'
        else:
          raise ValueError(
            "Don't know how to give a rider to %s %s" % (cuboid, attribs))

      cuboid.extra = {
        'nuscenes_token': box.token,
        'nuscenes_attribs': '|'.join(attribs),
      }

      # Points
      cuboid.box3d = box.corners().T
      cuboid.motion_corrected = (not sd_record['is_key_frame'])
      cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
      
      # Pose
      cuboid.width_meters = float(box.wlh[0])
      cuboid.length_meters = float(box.wlh[1])
      cuboid.height_meters = float(box.wlh[2])

      cuboid.obj_from_ego = av.Transform(
          rotation=box.orientation.rotation_matrix,
          translation=box.center,
          src_frame='ego',
          dest_frame='obj')
      cuboid.ego_pose = ego_pose
      cuboids.append(cuboid)
    return av.StampedDatum.from_uri(uri, cuboids=cuboids)

  @classmethod
  def __create_ego_pose(cls, uri, pose_record):
    nusc = cls.get_nusc()
    pose_record = nusc.get('ego_pose', pose_record['token'])
    ego_pose = transform_from_record(
                      pose_record,
                      dest_frame='ego',
                      src_frame='city')
    return av.StampedDatum.from_uri(uri, transform=ego_pose)

  @classmethod
  def _get_():
    
    uris = []
    for sample_data in sample_datas:
      if sample_data['sensor_modality'] != 'camera':
        continue
        
      sample = nusc.get('sample', sample_data['sample_token'])
      scene_record = nusc.get('scene', sample['scene_token'])
      scene_split = cls.FIXTURES.get_split_for_scene(scene_record['name'])
      if scene_split not in splits:
        continue

      uris.append(av.URI(
                    dataset='nuscenes',
                    split=scene_split,
                    timestamp=to_nanostamp(sample_data['timestamp']),
                    segment_id=scene_record['name'],
                    camera=sample_data['channel']))

    return uris












# class FrameTable(av.FrameTableBase):

#   FIXTURES = Fixtures

#   NUSC_VERSION = 'v1.0-trainval' # E.g. v1.0-mini, v1.0-trainval, v1.0-test

#   PROJECT_CLOUDS_TO_CAM = True
#   PROJECT_CUBOIDS_TO_CAM = True
#   IGNORE_INVISIBLE_CUBOIDS = True

#   KEYFRAMES_ONLY = True#False
#     # When disabled, will use motion-corrected points
  
#   ## Subclass API

#   @classmethod
#   def table_root(cls):
#     return '/outer_root/media/seagates-ext4/au_datas/nusc_frame_table'

#   @classmethod
#   def create_frame(cls, uri):
#     f = av.Frame(uri=uri)
#     cls._fill_ego_pose(f)
#     cls._fill_camera_images(f)
#     return f

#   @classmethod
#   def _create_frame_rdds(cls, spark):
#     uris = cls._get_camera_uris()
#     print('len(uris)', len(uris))

#     # TODO fixmes
#     uri_rdd = spark.sparkContext.parallelize(uris)

#     # Try to group frames from segments together to make partitioning easier
#     # and result in fewer files
#     uri_rdd = uri_rdd.sortBy(lambda uri: uri.segment_id)
    
#     frame_rdds = []
#     uris = uri_rdd.toLocalIterator()
#     for uri_chunk in util.ichunked(uris, cls.SETUP_URIS_PER_CHUNK):
#       chunk_uri_rdd = spark.sparkContext.parallelize(uri_chunk)
#       # create_frame = util.ThruputObserver.wrap_func(
#       #                       cls.create_frame,
#       #                       name='create_frame',
#       #                       log_on_del=True)

#       frame_rdd = chunk_uri_rdd.map(cls.create_frame)

#       frame_rdds.append(frame_rdd)
#     return frame_rdds
  
#   ## Public API

#   @classmethod
#   def get_nusc(cls):
#     if not hasattr(cls, '_nusc'):
#       cls._nusc = cls.FIXTURES.get_loader(version=cls.NUSC_VERSION)
#     return cls._nusc


#   ## Support

#   @classmethod
#   def _get_camera_uris(cls, splits=None):
#     nusc = cls.get_nusc()

#     if not splits:
#       splits = cls.FIXTURES.TRAIN_TEST_SPLITS
#     if cls.KEYFRAMES_ONLY:
#       import itertools
#       sample_datas = itertools.chain.from_iterable(
#         (nusc.get('sample_data', token)
#           for sensor, token in sample['data'].items())
#         for sample in nusc.sample)
#     else:
#       sample_datas = iter(nusc.sample_data)
    
#     uris = []
#     for sample_data in sample_datas:
#       if sample_data['sensor_modality'] != 'camera':
#         continue
        
#       sample = nusc.get('sample', sample_data['sample_token'])
#       scene_record = nusc.get('scene', sample['scene_token'])
#       scene_split = cls.FIXTURES.get_split_for_scene(scene_record['name'])
#       if scene_split not in splits:
#         continue

#       uris.append(av.URI(
#                     dataset='nuscenes',
#                     split=scene_split,
#                     timestamp=to_nanostamp(sample_data['timestamp']),
#                     segment_id=scene_record['name'],
#                     camera=sample_data['channel']))

#     return uris
  
#   # @classmethod
#   # def _scene_to_ts_to_sample_token(cls):
#   #   if not hasattr(cls, '__scene_to_ts_to_sample_token'):
#   #     nusc = cls.get_nusc()
#   #     scene_name_to_token = dict(
#   #       (scene['name'], scene['token']) for scene in nusc.scene)
    
#   #     from collections import defaultdict
#   #     scene_to_ts_to_sample_token = defaultdict(dict)
#   #     for sample in nusc.sample:
#   #       scene_name = nusc.get('scene', sample['scene_token'])['name']
#   #       timestamp_micros = sample['timestamp']
#   #       token = sample['token']
#   #       scene_to_ts_to_sample_token[scene_name][timestamp_micros] = token
      
#   #     cls.__scene_to_ts_to_sample_token = scene_to_ts_to_sample_token
#   #   return cls.__scene_to_ts_to_sample_token

#   # @classmethod
#   # def _create_frame_from_sample(cls, uri, sample):
#   #   f = av.Frame(uri=uri)
#   #   cls._fill_ego_pose(f)
#   #   cls._fill_camera_images(f)
#   #   return f
  
#   @classmethod
#   def _fill_ego_pose(cls, f):
#     nusc = cls.get_nusc()
#     uri = f.uri

#     # Every sample has a pose, so we should get an exact match
#     best_sd, diff = nusc.get_nearest_sample_data(
#                                 uri.segment_id,
#                                 uri.timestamp)
#     assert best_sd and diff == 0, "Can't interpolate pose"

#     token = best_sd['ego_pose_token']
#     pose_record = nusc.get('ego_pose', best_sd['ego_pose_token'])
#     f.world_to_ego = transform_from_record(pose_record)

#     # # For now, always set ego pose using the *lidar* timestamp, as is done
#     # # in nuscenes.  (They probably localize mostly from lidar anyways).
#     # token = sample['data']['LIDAR_TOP']
#     # sd_record = nusc.get('sample_data', token)
    

#   @classmethod
#   def _fill_camera_images(cls, f):
#     nusc = cls.get_nusc()
#     uri = f.uri
    
#     cameras = list(nusc.ALL_CAMERAS)
#     if uri.camera:
#       cameras = [uri.camera]
    
#     for camera in cameras:
#       best_sd, diff_ns = nusc.get_nearest_sample_data(
#                                 uri.segment_id,
#                                 uri.timestamp,
#                                 channel=camera)
#       assert best_sd
#       assert diff_ns < .01 * 1e9

#       ci = cls._get_camera_image(uri, best_sd)
#       f.camera_images.append(ci)
  
#   @classmethod
#   def _get_camera_image(cls, uri, sd_record):
#     nusc = cls.get_nusc()
#     # sd_record = nusc.get('sample_data', camera_token)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     camera_token = sd_record['token']
#     cs_record = nusc.get(
#       'calibrated_sensor', sd_record['calibrated_sensor_token'])
#     sensor_record = nusc.get('sensor', cs_record['sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

#     data_path, _, cam_intrinsic = nusc.get_sample_data(camera_token)
#       # Ignore box_list, we'll get boxes in ego frame later
    
#     viewport = uri.get_viewport()
#     w, h = sd_record['width'], sd_record['height']
#     if not viewport:
#       from au.fixtures.datasets import common
#       viewport = common.BBox.of_size(w, h)

#     timestamp = sd_record['timestamp']

#     ego_from_cam = transform_from_record(cs_record)
#     cam_from_ego = ego_from_cam.get_inverse()
#     RT_h = cam_from_ego.get_transformation_matrix(homogeneous=True)
#     principal_axis_in_ego = get_camera_normal(cam_intrinsic, RT_h)
    
#     ci = av.CameraImage(
#           camera_name=uri.camera,
#           image_jpeg=bytearray(open(data_path, 'rb').read()),
#           height=h,
#           width=w,
#           viewport=viewport,
#           timestamp=to_nanostamp(timestamp),
#           cam_from_ego=cam_from_ego,
#           K=cam_intrinsic,
#           principal_axis_in_ego=principal_axis_in_ego,
#         )
    
#     if cls.PROJECT_CLOUDS_TO_CAM:
#       # TODO fuse
#       # pc = None
#       # for sensor in ('LIDAR_TOP',):
#       for sensor in ('LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT'): # lyft
#         # sample = nusc.get('sample', sd_record['sample_token']) ~~~~~~~~~~~~~~~~
#         target_pose_token = sd_record['ego_pose_token']
#         pc = cls._get_point_cloud_in_ego(uri, sensor, target_pose_token)
#         if not pc:
#           continue
#         # if pts:
#         #   pts = np.concatenate((pts, pc))
#         # else:
#         #   pts = pc

#         # Project to image
#         pc.cloud = ci.project_ego_to_image(pc.cloud, omit_offscreen=True)
#         pc.sensor_name = pc.sensor_name + '_in_cam'
#         ci.clouds.append(pc)
      
#     if cls.PROJECT_CUBOIDS_TO_CAM:
#       sample_data_token = sd_record['token']
#       cuboids = cls._get_cuboids_in_ego(sample_data_token)
#       for cuboid in cuboids:
#         bbox = ci.project_cuboid_to_bbox(cuboid)
#         if cls.IGNORE_INVISIBLE_CUBOIDS and not bbox.is_visible:
#           continue
#         ci.bboxes.append(bbox)
    
#     return ci

#   @classmethod
#   def _get_point_cloud_in_ego(cls, uri, sensor, target_pose_token):
#     # Based upon nuscenes.py#map_pointcloud_to_image()
    
#     from pyquaternion import Quaternion
#     from nuscenes.utils.data_classes import LidarPointCloud
#     from nuscenes.utils.data_classes import RadarPointCloud

#     nusc = cls.get_nusc()

#     # Get the cloud closest to the uri time
#     pointsensor, diff = nusc.get_nearest_sample_data(
#                                 uri.segment_id,
#                                 uri.timestamp,
#                                 channel=sensor)
#     if not pointsensor:
#       # Perhaps this scene does not have `sensor`
#       return None
    
#     #pointsensor_token = sample['data'][sensor]
#     #pointsensor = nusc.get('sample_data', pointsensor_token)
#     pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
#     if pointsensor['sensor_modality'] == 'lidar':
#       pc = LidarPointCloud.from_file(pcl_path)
#     else:
#       pc = RadarPointCloud.from_file(pcl_path)

#     # Step 1: Points live in the point sensor frame.  First transform to
#     # world frame:
#     # 1a transform to ego
#     # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
#     cs_record = nusc.get(
#                   'calibrated_sensor', pointsensor['calibrated_sensor_token'])
#     pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
#     pc.translate(np.array(cs_record['translation']))

#     # 1b transform to the global frame.
#     poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
#     pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
#     pc.translate(np.array(poserecord['translation']))

#     # Step 2: Send points into the ego frame at the target timestamp
#     poserecord = nusc.get('ego_pose', target_pose_token)
#     pc.translate(-np.array(poserecord['translation']))
#     pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

#     n_xyz = pc.points[:3, :].T
#       # Throw out intensity (lidar) and ... other stuff (radar)
#     return av.PointCloud(
#       sensor_name=sensor,
#       timestamp=to_nanostamp(pointsensor['timestamp']),
#       cloud=n_xyz,
#       ego_to_sensor=transform_from_record(cs_record),
#       motion_corrected=(pointsensor['ego_pose_token'] != target_pose_token),
#     )
    
#   @classmethod
#   def _get_cuboids_in_ego(cls, sample_data_token):
#     nusc = cls.get_nusc()

#     # NB: This helper always does motion correction (interpolation) unless
#     # `sample_data_token` refers to a keyframe.
#     boxes = nusc.get_boxes(sample_data_token)
  
#     # Boxes are in world frame.  Move all to ego frame.
#     from pyquaternion import Quaternion
#     sd_record = nusc.get('sample_data', sample_data_token)
#     pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
#     for box in boxes:
#       # Move box to ego vehicle coord system
#       box.translate(-np.array(pose_record['translation']))
#       box.rotate(Quaternion(pose_record['rotation']).inverse)

#     from au.fixtures.datasets.av import NUSCENES_CATEGORY_TO_AU_AV_CATEGORY
#     cuboids = []
#     for box in boxes:
#       cuboid = av.Cuboid()

#       # Core
#       sample_anno = nusc.get('sample_annotation', box.token)
#       cuboid.track_id = \
#         'nuscenes_instance_token:' + sample_anno['instance_token']
#       cuboid.category_name = box.name
#       cuboid.timestamp = to_nanostamp(sd_record['timestamp'])
      
#       cuboid.au_category = NUSCENES_CATEGORY_TO_AU_AV_CATEGORY[box.name]
      
#       # Try to give bikes riders
#       # NB: In Lyft Level 5, they appear to *not* label bikes without riders
#       attribs = [
#         nusc.get('attribute', attrib_token)['name']
#         for attrib_token in sample_anno['attribute_tokens']
#       ]
#       if 'cycle.with_rider' in attribs:
#         if cuboid.au_category == 'bike_no_rider':
#           cuboid.au_category = 'bike_with_rider'
#         elif cuboid.au_category == 'motorcycle_no_rider':
#           cuboid.au_category = 'motorcycle_with_rider'
#         else:
#           raise ValueError(
#             "Don't know how to give a rider to %s %s" % (cuboid, attribs))

#       cuboid.extra = {
#         'nuscenes_token': box.token,
#         'nuscenes_attribs': '|'.join(attribs),
#       }

#       # Points
#       cuboid.box3d = box.corners().T
#       cuboid.motion_corrected = (not sd_record['is_key_frame'])
#       cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
      
#       # Pose
#       cuboid.width_meters = float(box.wlh[0])
#       cuboid.length_meters = float(box.wlh[1])
#       cuboid.height_meters = float(box.wlh[2])

#       cuboid.obj_from_ego = av.Transform(
#           rotation=box.orientation.rotation_matrix,
#           translation=box.center.reshape((3, 1)))
#       cuboids.append(cuboid)
#     return cuboids
