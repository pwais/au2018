
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


### Utils

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
    'timestamp')    # E.g. 315975652303331336, yes this is GPS time :P :P
  
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
      for attr in self.__slots__)
    return self.PREFIX + path
  
  def __str__(self):
    return self.to_str()

  @staticmethod
  def from_str(s):
    assert s.startswith(FrameURI.PREFIX)
    toks_s = s[len(FrameURI.PREFIX):]
    toks = toks_s.split('&')
    assert len(toks) == len(FrameURI.__slots__)
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

      bbox.distance_meters = np.min(np.linalg.norm(bbox.cuboid_pts, axis=-1))

      from argoverse.utils.transform import quat2rotmat
        # NB: must use quat2rotmat due to Argo-specific quaternions
      rotmat = quat2rotmat(object_label_record.quaternion)
      bbox.relative_yaw_radians = math.atan2(rotmat[2, 1], rotmat[1, 1])

    def fill_bbox_core(bbox):
      bbox.category_name = object_label_record.label_class

      bbox.im_width, bbox.im_height = get_image_width_height(uri.camera)
      uv = calib.project_ego_to_image(bbox.cuboid_pts)

      x1, x2 = np.min(uv[:, 0]), np.max(uv[:, 0])
      y1, y2 = np.min(uv[:, 1]), np.max(uv[:, 1])

      num_onscreen = sum(
        1
        for x, y in ((x1, y1), (x2, y2))
        if (0 <= x < bbox.im_width) and (0 <= y < bbox.im_height))

      bbox.has_offscreen = (num_onscreen < 2)
      bbox.is_visible = (
        (num_onscreen == 0) or
        (object_label_record.occlusion == 100))


      # Clamp to screen
      def iround(v):
        return int(math.round(v))
      x1 = np.clip(iround(x1), 0, bbox.im_width - 1)
      x2 = np.clip(iround(x2), 0, bbox.im_width - 1)
      y1 = np.clip(iround(y1), 0, bbox.im_height - 1)
      y2 = np.clip(iround(y2), 0, bbox.im_height - 1)

      bbox.x = x1
      bbox.y = y1
      bbox.width = x2 - x1
      bbox.height = y2 - y1

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
    
    # Lidar
    '_cloud',
    'cloud_interpolated',
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

  @property
  def debug_image(self):
    img = np.copy(self.image)
    
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
    pts = point_cloud_to_homogeneous(pts).T
    assert False, (pts, pts.shape)
    return pts
    




### Data

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
        loader = cls.get_loader(
                      FrameURI(tarball_name=tarball, log_id=log_id))
        for camera, ts_to_path in loader.timestamp_image_dict.items():
          for ts in ts_to_path.keys():
            yield FrameURI(
              tarball_name=tarball,
              log_id=log_id,
              split=split,
              camera=camera,
              timestamp=ts)

  
  

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






if __name__ == '__main__':
  Fixtures().run_import()