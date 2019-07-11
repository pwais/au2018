
import itertools
import os

from au import conf
from au import util
from au.fixtures.datasets import common

import imageio
import math
import numpy as np
import six

from argoverse.data_loading.argoverse_tracking_loader import \
  ArgoverseTrackingLoader


###
### Utils
###

AV_OBJ_CLASS_TO_COARSE = {
  "VEHICLE":            'car',
  "PEDESTRIAN":         'ped',
  "ON_ROAD_OBSTACLE":   'other',
  "LARGE_VEHICLE":      'car',
  "BICYCLE":            'bike',
  "BICYCLIST":          'ped',
  "BUS":                'car',
  "OTHER_MOVER":        'other',
  "TRAILER":            'car',
  "MOTORCYCLIST":       'ped',
  "MOPED":              'bike',
  "MOTORCYCLE":         'bike',
  "STROLLER":           'other',
  "EMERGENCY_VEHICLE":  'car',
  "ANIMAL":             'ped',
  "WHEELCHAIR":         'ped',
  "SCHOOL_BUS":         'car',
}

def get_image_width_height(camera):
  from argoverse.utils import camera_stats
  if camera in camera_stats.RING_CAMERA_LIST:
    return camera_stats.RING_IMG_WIDTH, camera_stats.RING_IMG_HEIGHT
  elif camera in camera_stats.STEREO_CAMERA_LIST:
    return camera_stats.STEREO_IMG_WIDTH, camera_stats.STEREO_IMG_HEIGHT
  else:
    raise ValueError("Unknown camera: %s" % camera)

class FrameURI(object):
  __slots__ = (
    'tarball_name', # E.g. tracking_sample.tar.gz
    'log_id',       # E.g. c6911883-1843-3727-8eaa-41dc8cda8993
    'split',        # Official Argoverse split (see Fixtures.SPLITS)
    'camera',       # E.g. ring_front_center
    'timestamp',    # E.g. 315975652303331336, yes this is GPS time :P :P
    'track_id')     # Optional; a UUID of a specific track / annotation
  
  OPTIONAL = ('track_id',)

  PREFIX = 'argoverse://'

  def __init__(self, **kwargs):
    # Use kwargs, then fall back to args
    for i, k in enumerate(self.__slots__):
      setattr(self, k, kwargs.get(k, ''))
    if self.timestamp is not '':
      self.timestamp = int(self.timestamp)
  
  def to_str(self):
    path = '&'.join(
      attr + '=' + str(getattr(self, attr))
      for attr in self.__slots__
      if getattr(self, attr))
    return self.PREFIX + path
  
  def __str__(self):
    return self.to_str()
  
  def to_dict(self):
    return dict((k, getattr(self, k, '')) for k in self.__slots__)

  @staticmethod
  def from_str(s):
    assert s.startswith(FrameURI.PREFIX)
    toks_s = s[len(FrameURI.PREFIX):]
    toks = toks_s.split('&')
    assert len(toks) >= (len(FrameURI.__slots__) - len(FrameURI.OPTIONAL))
    iu = FrameURI(**dict(tok.split('=') for tok in toks))
    return iu

class BBox(common.BBox):
  __slots__ = tuple(
    list(common.BBox.__slots__) + [
      # From ObjectLabelRecord
      'occlusion',
      'track_id',
      'length_meters',
      'width_meters',
      'height_meters',

      # Inferred from object pose relative to robot
      'distance_meters',  # Dist to closest cuboid point
      'relative_yaw_radians',
      'has_offscreen',
      'is_visible',

      # Has the ObjectLabelRecord been motion-corrected?
      'motion_corrected',
      'cuboid_pts',       # In robot ego frame
      'ego_to_obj',       # Translation vector in ego frame
    ]
  )

  @staticmethod
  def from_argoverse_label(
        uri,
        object_label_record,
        motion_corrected=True,
        fixures_cls=None):
    """Construct and return a single `BBox` instance from the given
    Argoverse ObjectLabelRecord instance.  Labels are in lidar space-time
    and *not* camera space-time; therefore, transforming labels into
    the camera domain requires (to be most precise) correction for the
    egomotion of the robot.  This correction can be substantial (~20cm)
    at high robot speed.  Apply this correction only if
    `motion_corrected`.
    """
    
    if not fixures_cls:
      fixures_cls = Fixtures
    
    loader = fixures_cls.get_loader(uri)
    calib = loader.get_calibration(uri.camera)

    def fill_cuboid_pts(bbox):
      cuboid_pts = object_label_record.as_3d_bbox()
        # Points in robot frame
      if motion_corrected:
        cuboid_pts = loader.get_motion_corrected_pts(
                                    cuboid_pts,
                                    object_label_record.timestamp,
                                    uri.timestamp)
      bbox.cuboid_pts = cuboid_pts
      bbox.motion_corrected = motion_corrected

    def fill_extra(bbox):
      bbox.track_id = object_label_record.track_id
      bbox.occlusion = object_label_record.occlusion
      bbox.length_meters = object_label_record.length
      bbox.width_meters = object_label_record.width
      bbox.height_meters = object_label_record.height

      bbox.distance_meters = \
        float(np.min(np.linalg.norm(bbox.cuboid_pts, axis=-1)))

      from argoverse.utils.transform import quat2rotmat
        # NB: must use quat2rotmat due to Argo-specific quaternions
      rotmat = quat2rotmat(object_label_record.quaternion)
      bbox.relative_yaw_radians = math.atan2(rotmat[2, 1], rotmat[1, 1])
      bbox.ego_to_obj = object_label_record.translation

    def fill_bbox_core(bbox):
      bbox.category_name = object_label_record.label_class

      bbox.im_width, bbox.im_height = get_image_width_height(uri.camera)
      uv = calib.project_ego_to_image(bbox.cuboid_pts)

      x1, x2 = np.min(uv[:, 0]), np.max(uv[:, 0])
      y1, y2 = np.min(uv[:, 1]), np.max(uv[:, 1])
      z = float(np.max(uv[:,2]))

      num_onscreen = sum(
        1
        for x, y in ((x1, y1), (x2, y2))
        if (0 <= x < bbox.im_width) and (0 <= y < bbox.im_height))

      bbox.has_offscreen = ((z <= 0) or (num_onscreen < 2))
      bbox.is_visible = all((
        z > 0,
        num_onscreen > 0,
        object_label_record.occlusion < 100))

      # Clamp to screen
      x1 = np.clip(round(x1), 0, bbox.im_width - 1)
      x2 = np.clip(round(x2), 0, bbox.im_width - 1)
      y1 = np.clip(round(y1), 0, bbox.im_height - 1)
      y2 = np.clip(round(y2), 0, bbox.im_height - 1)

      # x1, x2 = uv[0, 0], uv[1, 0]
      # y1, y2 = uv[0, 1], uv[1, 1]

      bbox.x = int(x1)
      bbox.y = int(y1)
      bbox.width = int(x2 - x1)
      bbox.height = int(y2 - y1)

    bbox = BBox()
    fill_cuboid_pts(bbox)
    fill_bbox_core(bbox)
    fill_extra(bbox)
    return bbox



class AVFrame(object):

  __slots__ = (
    # Meta
    'uri',      # type: FrameURI
    'FIXTURES', # type: au.datasets.argoverse.Fixtures
    '_loader',  # type: AUTrackingLoader

    # Labels
    '_image_bboxes',

    # Vision
    '_image',
    'image_width',
    'image_height',
    
    # # Lidar
    # '_cloud',
    # 'cloud_interpolated',
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
    
    if isinstance(self.uri, six.string_types):
      self.uri = FrameURI.from_str(self.uri)
    
    # Fill context if needed
    if not self.FIXTURES:
      self.FIXTURES = Fixtures

    if not (self.image_width and self.image_height):
      self.image_width, self.image_height = \
        get_image_width_height(self.uri.camera)
    
  @property
  def loader(self):
    if not self._loader:
      self._loader = self.FIXTURES.get_loader(self.uri)
    return self._loader # type: AUTrackingLoader
  
  @property
  def image(self):
    if not self._image:
      path = self.loader.get_nearest_image_path(
                      self.uri.camera, self.uri.timestamp)
      self._image = imageio.imread(path)
    return self._image
  
  @property
  def image_bboxes(self):
    if not self._image_bboxes:
      t = self.uri.timestamp
      av_label_objects = self.loader.get_nearest_label_object(t)
      self._image_bboxes = [
        BBox.from_argoverse_label(self.uri, object_label_record)
        for object_label_record in av_label_objects
      ]
    return self._image_bboxes

  def get_debug_image(self):
    img = np.copy(self.image)
    
    if self.uri.track_id:
      # Draw a gold box first; then the draw() calls below will draw over
      # the box.
      WHITE = (225, 225, 255)
      for bbox in self.image_bboxes:
        if bbox.track_id == self.uri.track_id:
          bbox.draw_in_image(img, color=WHITE, thickness=8)

    for bbox in self.image_bboxes:
      if bbox.is_visible:
        bbox.draw_in_image(img)
    
    return img



class AUTrackingLoader(ArgoverseTrackingLoader):
  """This class makes several modifications to `ArgoverseTrackingLoader`:
   * By default, `ArgoverseTrackingLoader` wants to scan an entire
      directory of logs at init time, which is exceptionally costly.
      This subclass is designed to avoid that cost and work on a 
      *SINGLE* log directory.
   * `ArgoverseTrackingLoader` provides lidar-synced images in the form
      of filtering 30Hz images to those with timestamps that best match
      10Hz lidar sweeps.  We additionally provide a means to interpolate
      point clouds to the full 30Hz image streams.  See also
      `demo_usage/cuboids_to_bboxes.py` in Argoverse.
  """

  def __init__(self, root_dir, log_name):
    """Create a new loader.
    
    Args:
      root_dir: string, path to a directory containing log directories,
        e.g. /media/data/path/to/argoverse/argoverse-tracking/train1
      log_name: string, the name of the log to load,
        e.g. 5ab2697b-6e3e-3454-a36a-aba2c6f27818
    """

    assert os.path.exists(os.path.join(root_dir, log_name)), "Sanity check"

    # Sadly both the superclass and the `SynchronizationDB` thing do huge
    # directory scans, so we must use a symlink to save us:
    # root_dir/log_name -> virtual_root/log_name
    import tempfile
    virtual_root = os.path.join(
                    tempfile.gettempdir(),
                    'argoverse_loader',
                    log_name)
    util.mkdir(virtual_root)
    try:
      os.symlink(
        os.path.join(root_dir, log_name),
        os.path.join(virtual_root, log_name))
    except FileExistsError:
      pass

    super(AUTrackingLoader, self).__init__(virtual_root)
  
  def get_nearest_image_path(self, camera, timestamp):
    """Return a path to the image from `camera` at `timestamp`;
    provide either an exact match or choose the closest available."""
    ts_to_path = self.timestamp_image_dict[camera]
    if timestamp not in ts_to_path:
      # Find the nearest timestamp
      diff, timestamp = min((abs(timestamp - t), t) for t in ts_to_path.keys())
      assert diff < 1e9, \
          "Could not find timestamp within 1 sec of %s" % timestamp

    path = ts_to_path[timestamp]
    return path
  
  def get_nearest_lidar_sweep(self, timestamp):
    """Return the index of the lidar sweep in this log that either
    matches exactly or is closest to `timestamp`."""
    diff, idx = min(
              (abs(timestamp - t), idx)
              for idx, t in enumerate(self.lidar_timestamp_list))
    assert diff < 1e9, "Could not find a cloud within 1 sec of %s" % timestamp
    return idx

  def get_nearest_label_object(self, timestamp):
    """Load and return the `ObjectLabelRecord`s nearest to `timestamp`;
    provide either an exact match or choose the closest available."""

    idx = self.get_nearest_lidar_sweep(timestamp)
    import argoverse.data_loading.object_label_record as object_label
    objs = object_label.read_label(self.label_list[idx])
      # NB: actually reads a list of labels :P
    
    # We hide the object timestamp in the label; I guess the Argoverse
    # authors didn't think the timestamp was important :P
    label_t = self.lidar_timestamp_list[idx]
    for obj in objs:
      obj.timestamp = label_t
    
    return objs

  def get_motion_corrected_pts(self, pts, pts_timestamp, dest_timestamp):
    
    # Similar to project_lidar_to_img_motion_compensated, but:
    #  * do not project to image
    #  * do not fail silently
    #  * do not have an extremely poor interface
    # We transform the points through the city / world frame:
    #  pt_ego_dest_t = ego_dest_t_SE3_city * city_SE3_ego_pts_t * pt_ego_pts_t

    from argoverse.data_loading.pose_loader import \
      get_city_SE3_egovehicle_at_sensor_t

    city_SE3_ego_dest_t = get_city_SE3_egovehicle_at_sensor_t(
                              dest_timestamp,
                              self.root_dir,
                              self.current_log)
    assert city_SE3_ego_dest_t

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when the LiDAR sweep was recorded.
    city_SE3_ego_pts_t = get_city_SE3_egovehicle_at_sensor_t(
                              pts_timestamp,
                              self.root_dir,
                              self.current_log)
    assert city_SE3_ego_pts_t

    
    # Argoverse SE3 does not want homogenous coords
    pts = np.copy(pts)
    if pts.shape[-1] == 4:
      pts = pts.T[:, :3]

    ego_dest_t_SE3_ego_pts_t = \
      city_SE3_ego_dest_t.inverse().right_multiply_with_se3(city_SE3_ego_pts_t)
    pts = ego_dest_t_SE3_ego_pts_t.transform_point_cloud(pts)
    
    from argoverse.utils.calibration import point_cloud_to_homogeneous
    # pts = point_cloud_to_homogeneous(pts)
    return pts
    


###
### Data
###

class Fixtures(object):

  # All Argoverse tarballs served from here
  BASE_TARBALL_URL = "https://s3.amazonaws.com/argoai-argoverse"

  TRACKING_SAMPLE = "tracking_sample.tar.gz"

  SAMPLE_TARBALLS = (
    TRACKING_SAMPLE,
    "forecasting_sample.tar.gz",
  )

  TRACKING_TARBALLS = (
    "tracking_train1.tar.gz",
    "tracking_train2.tar.gz",
    "tracking_train3.tar.gz",
    "tracking_train4.tar.gz",
    "tracking_val.tar.gz",
    "tracking_test.tar.gz",
  )

  PREDICTION_TARBALLS = (
    "forecasting_train.tar.gz",
    "forecasting_val.tar.gz",
    "forecasting_test.tar.gz",
  )

  MAP_TARBALLS = (
    "hd_maps.tar.gz",
  )

  SPLITS = ('train', 'test', 'val', 'sample')

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'argoverse')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'argoverse')


  ## Source Data

  @classmethod
  def tarballs_dir(cls):
    return os.path.join(cls.ROOT, 'tarballs')

  @classmethod
  def tarball_path(cls, fname):
    return os.path.join(cls.tarballs_dir(), fname)

  @classmethod
  def tarball_dir(cls, fname):
    """Get the directory for an uncompressed tarball with `fname`"""
    dirname = fname.replace('.tar.gz', '')
    return cls.tarball_path(dirname)

  @classmethod
  def all_tarballs(cls):
    return list(
      itertools.chain.from_iterable(
        getattr(cls, attr, [])
        for attr in dir(cls)
        if attr.endswith('_TARBALLS')))

  @classmethod
  def all_tracking_tarballs(cls):
    return [t for t in cls.all_tarballs() if 'tracking' in t]


  ## Argoverse-specific Utils

  @classmethod
  def get_log_dirs(cls, base_path):
    # Log dirs have calibration JSON files
    # See e.g. https://github.com/argoai/argoverse-api/blob/16dec1ba51479a24b14d935e7873b26bfd1a7464/argoverse/data_loading/argoverse_tracking_loader.py#L121
    calib_paths = util.all_files_recursive(
                      base_path,
                      pattern="**/vehicle_calibration_info.json")
    return [os.path.dirname(cpath) for cpath in calib_paths]

  @classmethod
  def get_loader(cls, uri):
    """Return a (maybe cached) `AUTrackingLoader` for the given `uri`"""
    if isinstance(uri, six.string_types):
      uri = FrameURI.from_str(uri)
    
    if not hasattr(cls, '_tarball_to_log_id_to_loader'):
      cls._tarball_to_log_id_to_loader = {}
    
    if not uri.tarball_name in cls._tarball_to_log_id_to_loader:
      cls._tarball_to_log_id_to_loader[uri.tarball_name] = {}
    
    log_id_to_loader = cls._tarball_to_log_id_to_loader[uri.tarball_name]
    
    loader = None
    if not uri.log_id in log_id_to_loader:
      base_path = cls.tarball_dir(uri.tarball_name)
      for log_dir in cls.get_log_dirs(base_path):
        log_id = os.path.split(log_dir)[-1]
        if log_id == uri.log_id:
          log_id_to_loader[log_id] = AUTrackingLoader(
                                        os.path.dirname(log_dir),
                                        log_id)

    assert uri.log_id in log_id_to_loader, "Could not find log %s" % uri.log_id
    return log_id_to_loader[uri.log_id]

  @classmethod
  def iter_image_uris(cls, split):
    assert split in cls.SPLITS
    tarballs = cls.all_tracking_tarballs()
    tarballs = [t for t in tarballs if split in t]
    for tarball in tarballs:
      base_path = cls.tarball_dir(tarball)
      for log_dir in cls.get_log_dirs(base_path):
        log_id = os.path.split(log_dir)[-1]
        loader = cls.get_loader(FrameURI(tarball_name=tarball, log_id=log_id))
        for camera, ts_to_path in loader.timestamp_image_dict.items():
          for ts in ts_to_path.keys():
            yield FrameURI(
              tarball_name=tarball,
              log_id=log_id,
              split=split,
              camera=camera,
              timestamp=ts)

  @classmethod
  def _impute_rider_for_bikes(cls, spark, df):
    BIKE = ["BICYCLE", "MOPED", "MOTORCYCLE"]
    RIDER = ["BICYCLIST", "MOTORCYCLIST"]
    BIKE_AND_RIDER = BIKE + RIDER

    # Sub-select just bike and rider rows from `df` to improve performance
    bikes_df = df.filter(
      df.category_name.isin(BIKE_AND_RIDER) &
      df.is_visible)
    bikes_df = bikes_df.select(
                  'uri',
                  'track_id',
                  'category_name',
                  'ego_to_obj')

    def iter_nearest_rider(uri_rows):
      uri, rows = uri_rows
      rows = list(rows) # Spark gives us a generator
      bike_rows = [r for r in rows if r.category_name in BIKE]
      rider_rows = [r for r in rows if r.category_name in RIDER]
      from six.moves.urllib import parse
      print(
        'rider_rows',
        [parse.urlencode({'uri': r.uri}) for r in rider_rows],
        'bike_rows',
        [parse.urlencode({'uri': r.uri}) for r in bike_rows])
      
      # The best pair has smallest euclidean distance between centroids
      def l2_dist(r1, r2):
        a1 = np.array([r1.ego_to_obj.x, r1.ego_to_obj.y, r1.ego_to_obj.z])
        a2 = np.array([r2.ego_to_obj.x, r2.ego_to_obj.y, r2.ego_to_obj.z])
        return float(np.linalg.norm(a2 - a1))

      # Bikes may not have riders, so loop over bikes looking for riders
      for bike in bike_rows:
        if rider_rows:
          best_rider, distance = min(
                          (rider, l2_dist(bike, rider))
                          for rider in rider_rows)
          nearest_rider = dict(
            uri=bike.uri,
            track_id=bike.track_id,
            best_rider_track_id=best_rider.track_id,
            best_rider_distance=distance,
          )

          from pyspark.sql import Row
          yield Row(**nearest_rider)
    
    # We'll group all rows in our DF by URI, then do bike<->rider
    # for each URI (i.e. all the rows for a single URI).  The matching
    # will spit out a new DataFrame, which we'll join against the 
    # original `df` in order to "add" the columns encoding the
    # bike<->rider matchings.
    uri_chunks_rdd = bikes_df.rdd.groupBy(lambda r: r.uri)
    nearest_rider = uri_chunks_rdd.flatMap(iter_nearest_rider)
    nearest_rider_df = spark.createDataFrame(nearest_rider)
    
    joined = df.join(nearest_rider_df, ['uri', 'track_id'], 'outer')

    # Don't allow nulls; those can't be compared and/or written to Parquet
    joined = joined.na.fill({
                  'best_rider_distance': float('inf'),
                  'best_rider_track_id': ''
    })
    import pdb; pdb.set_trace()
    print('moof')
    return joined

  @classmethod
  def label_df(cls, spark, splits=None):
    if not splits:
      splits = cls.SPLITS
    
    split_rdd = spark.sparkContext.parallelize(splits, numSlices=len(splits))
    uri_rdd = split_rdd.flatMap(lambda split: cls.iter_image_uris(split))
    
    # Read frames in parallel
    uri_rdd = uri_rdd.repartition(1000)
    
    def iter_label_rows(uri):
      from collections import namedtuple
      pt = namedtuple('pt', 'x y z')

      frame = AVFrame(uri=uri)
      for box in frame.image_bboxes:
        row = {}

        # Obj
        # TODO make spark accept numpy and numpy float64 things
        row = box.to_dict()
        IGNORE = ('cuboid_pts', 'ego_to_obj')
        for attr in IGNORE:
          v = row.pop(attr)
          if hasattr(v, 'shape'):
            if len(v.shape) == 1:
              row[attr] = pt(*v.tolist())
            else:
              row[attr] = [pt(*v[r, :3].tolist()) for r in range(v.shape[0])]
        
        # Context
        import copy
        obj_uri = copy.deepcopy(uri)
        obj_uri.track_id = box.track_id
        row.update(uri=str(obj_uri), **obj_uri.to_dict())
        row.update(
          city=cls.get_loader(uri).city_name,
          coarse_category=AV_OBJ_CLASS_TO_COARSE.get(box.category_name, ''))
        
        from pyspark.sql import Row
        yield Row(**row)
    
    P = '/tmp/yay_label_df_cache'
    if not os.path.exists('/tmp/yay_label_df_cache'):
      df = spark.createDataFrame(uri_rdd.flatMap(iter_label_rows))
      df.write.parquet(P, mode='overwrite')
    print('fixme')
    df = spark.read.parquet(P)
    df = cls._impute_rider_for_bikes(spark, df)
    return df
  

  ## Setup

  @classmethod
  def download_all(cls):
    util.mkdir(cls.tarball_path(''))
    for tarball in cls.all_tarballs():
      uri = cls.BASE_TARBALL_URL + '/' + tarball
      util.download(uri, cls.zip_dir(fname), try_expand=True)

  @classmethod
  def run_import(cls):
    cls.download_all()



###
### Mining
###

class HistogramWithExamples(object):
  
  def run(self, df):
    df = df[df.is_visible == True]

    MACRO_FACETS = (
      'camera',
      'city',
      'split',
    )

    METRICS = (
      'distance_meters',
      'height_meters',
      'width_meters',
      'length_meters',
      'height',
      'width',
      'relative_yaw_radians',
      'occlusion',
    )

    MICRO_FACETS = (
      'category_name',
      # 'coarse_category',
    )
    # class + any occlusion
    # class + edge of image

    # centroid distance between bike and rider?

    def make_panel(df, metric, micro_facet):
      from bokeh import plotting
      from bokeh.models import ColumnDataSource
      import pandas as pd

      df = df.dropna() # FIXME SOME METRICS HAVE NANS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ## Organize Data
      micro_facets_values = list(df[micro_facet].unique())
      micro_facets_values.append('all')
      legend_to_panel_df = {}
      for mf in micro_facets_values:
        if mf == 'all':
          mf_data = pd.DataFrame(df)
          mf_uris = df['uri']
        else:
          mf_data = df[df[micro_facet] == mf]
          mf_uris = df[df[micro_facet] == mf]['uri']
        hist, edges = np.histogram(mf_data[metric], bins=100)
        mf_df = pd.DataFrame(dict(
          count=hist, proportion=hist / np.sum(hist),
          left=edges[:-1], right=edges[1:],
          uris=[df.loc[inds]['uri'] for inds in h_inds],
        ))
        mf_df['legend'] = mf
        from bokeh.colors import RGB
        mf_df['color'] = RGB(*util.hash_to_rbg(mf))

        h_inds = np.digitize(mf_data[metric], edges)
        mf_df['uri'] = [
          df.loc[h_inds == i]['uri'][:10]
          for i in range(min(h_inds), max(h_inds) + 1)
        ]
        assert False
        print([df.loc[h_inds == i]['uri'][:10]
          for i in range(min(h_inds), max(h_inds) + 1)])

        legend_to_panel_df[mf] = mf_df
      
      # # Make checkbox group that can filter data
      # plot_src = ColumnDataSource(pd.DataFrame({}))
      # def update_plot_data(legends_to_plot): # type: str
      #   plot_df = pd.concat(
      #                 df
      #                 for legend, df in legend_to_panel_df.items()
      #                 if legend in legends_to_plot)
      #   plot_df.sort_values(['legend', 'left'])
      #   plot_src.data.update(ColumnDataSource(plot_df).data)
      
      # # Initially only plot the 'all' series
      # update_plot_data(micro_facets_values)
      
      # def update(attr, old, new):
      #     to_plot = [checkbox_group.labels[i] for i in checkbox_group.active]
      #     update_plot_data(to_plot)

      # from bokeh.models.widgets import CheckboxGroup
      # checkbox_group = CheckboxGroup(
      #                   labels=sorted(micro_facets_values),
      #                   active=[1] * len(micro_facets_values))
      # checkbox_group.on_change('active', update)

      ## Make the plot
      title = metric + ' vs ' + micro_facet
      fig = plotting.figure(
              title=title,
              tools='tap',
              plot_width=1200,
              x_axis_label=metric,
              y_axis_label='Count')
      for _, plot_src in legend_to_panel_df.items():
        plot_src = ColumnDataSource(plot_src)
        r = fig.quad(
          source=plot_src, bottom=0, top='count', left='left', right='right',
          color='color', fill_alpha=0.5,
          hover_fill_color='color', hover_fill_alpha=1.0,
          legend='legend')
        from bokeh.models import HoverTool
        fig.add_tools(
          HoverTool(
            renderers=[r],
            mode='vline',
            tooltips=[
              ('Facet', '@legend'),
              ('Count', '@count'),
              ('Proportion', '@proportion'),
              ('Value', '@left'),
            ]))
        
        plot_src

      fig.legend.click_policy = 'hide'

      
      
      # fig.add_tools(
      #   HoverTool(tooltips=,
      #     mode='vline'))
      # fig.legend.click_policy = 'hide'

      # from bokeh.layouts import WidgetBox
      # controls = WidgetBox(checkbox_group)
      from bokeh.models.widgets import Div
      ctxbox = Div(text="Placeholder")

      from bokeh.models import TapTool
      taptool = fig.select(type=TapTool)

      from bokeh.models import CustomJS
      taptool.callback = CustomJS(
        args=dict(ctxbox=ctxbox),
        code="""console.log(cb_data);ctxbox.text="" + cb_data.source.data.uris """)



      from bokeh.layouts import row
      layout = row(fig, ctxbox)

      from bokeh.models.widgets import Panel
      panel = Panel(child=layout, title=title)
      return panel
    
    panels = []
    for mf in MICRO_FACETS:
      panels.extend(
        make_panel(df, metric, mf)
        for metric in METRICS)
    
    from bokeh.models.widgets import Tabs
    t = Tabs(tabs=panels)
    def save_plot(tabs):
      from bokeh import plotting
      if tabs is None:
        return
      
      dest = '/opt/au/yay_plot.html'
      plotting.output_file(dest, title='my title', mode='inline')
      plotting.save(tabs)
      util.log.info("Wrote to %s" % dest)
    save_plot(t)

    




if __name__ == '__main__':
  Fixtures().run_import()