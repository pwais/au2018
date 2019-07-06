
import itertools
import os

from au import conf
from au import util

import imageio
import numpy as np
import six

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

### Utils

class FrameURI(object):
  __slots__ = (
    'tarball_name', # E.g. tracking_sample.tar.gz
    'log_id',       # E.g. c6911883-1843-3727-8eaa-41dc8cda8993
    'split',        # Official Argoverse split {train,test,val,sample}
    'camera',       # E.g. ring_front_center
    'timestamp')    # E.g. 315975652303331336, yes this is GPS time :P :P
  PREFIX = 'argoverse://'

  def __init__(self, **kwargs):
    # Use kwargs, then fall back to args
    for i, k in enumerate(self.__slots__):
      setattr(self, k, kwargs.get(k, ''))
  
  def to_str(self):
    path = '&'.join(
      attr + '=' + getattr(self, attr)
      for attr in self.__slots__)
    return self.PREFIX + path
  
  def __str__(self):
    return self.to_str()

  @staticmethod
  def from_str(s):
    assert s.startswith(FrameURI.PREFIX)
    toks_s = s[len(FrameURI.PREFIX):]
    toks = toks_s.split('&')
    assert len(toks) == len(self.__slots__)
    iu = FrameURI(**dict(tok.split('=') for tok in toks))
    return iu

class AVFrame(object):

  __slots__ = (
    # Meta
    'uri',      # type: FrameURI
    'FIXTURES', # type: au.datasets.argoverse.Fixtures
    '_loader',  # type: AUTrackingLoader

    # Labels
    '_av_label_objects',
    '_image_bboxes',

    # Vision
    '_image',
    'image_width',
    'image_height',
    
    # Lidar
    '_cloud',
    'cloud_interpolated',
  )

  def __init__(self, **kwarg):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
    
    if isinstance(self.uri, six.string_types):
      self.uri = Frame.from_str(self.uri)
    
    # Fill context if needed
    if not self.FIXTURES:
      self.FIXTURES = Fixtures

    if not (self.image_width and self.image_height):
      from argoverse.utils import camera_stats
      if self.uri.camera in camera_stats.RING_CAMERA_LIST:
        self.image_width = camera_stats.RING_IMG_WIDTH
        self.image_height = camera_stats.RING_IMG_HEIGHT
      elif self.uri.camera in camera_stats.STEREO_CAMERA_LIST:
        self.image_width = camera_stats.STEREO_IMG_WIDTH
        self.image_height = camera_stats.STEREO_IMG_HEIGHT
      else:
        raise ValueError("Unknown camera: %s" % self.uri.camera)
    
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
  def av_label_objects(self):
    if not self._av_label_objects:
      t = self.uri.timestamp
      self._av_label_objects = self.loader.get_nearest_label_object(t)
    return self._av_label_objects

  @property
  def debug_image(self):
    im = np.array(self.image)
    return im





class LabeledObject(object):
  __slots__ = (
    # BBox
    'x', 'y', 'width', 'height',
    'im_width', 'im_height',
    # TODO cuboid
    'category_name',
  )

  def __getstate__(self):
    return self.to_dict()
  
  def __setstate__(self, d):
    for k in self.__slots__:
      setattr(self, k, d.get(k, ''))

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class AVImage(object):

  __slots__ = (
    'uri'
    '_loader',
  )
  
  def __init__(self, loader):
    self._loader = loader
  
  def get_labeled_objects(self):
    pass

  


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
    
    :param root_dir: string, path to a directory containing log directories,
      e.g. /media/data/path/to/argoverse/argoverse-tracking/train1
    :param log_name: string, the name of the log to load,
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
    return object_label.read_label(self.label_list[idx])



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

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'argoverse')

  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'argoverse')


  ## Argoverse-specific Utils

  @classmethod
  def _get_log_id_to_dir(cls, tarballs=None):
    """Return a map of Log ID -> path to the log files for all
    available logs; optionally filter by the `tarballs` containing
    log file(s)."""

    if not tarballs:
      tarballs = [cls.TRACKING_SAMPLE] + list(TRACKING_TARBALLS)
    
    log_id_to_dir = {}
    for tarball in tarballs:
      base_path = cls.tarball_dir(tarball)
      # Log dirs have calibration JSON files
      # See e.g. https://github.com/argoai/argoverse-api/blob/16dec1ba51479a24b14d935e7873b26bfd1a7464/argoverse/data_loading/argoverse_tracking_loader.py#L121
      calib_paths = util.all_files_recursive(
                      base_path,
                      pattern="**/vehicle_calibration_info.json")
      
      for cpath in calib_paths:
        log_dir = os.path.dirname(cpath)
        log_id = os.path.split(log_dir)[-1]
        log_id_to_dir[log_id] = log_dir
    return log_id_to_dir
  


  @classmethod
  def get_loader(cls, uri):
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
      # Log dirs have calibration JSON files
      # See e.g. https://github.com/argoai/argoverse-api/blob/16dec1ba51479a24b14d935e7873b26bfd1a7464/argoverse/data_loading/argoverse_tracking_loader.py#L121
      calib_paths = util.all_files_recursive(
                      base_path,
                      pattern="**/vehicle_calibration_info.json")
      
      for cpath in calib_paths:
        log_dir = os.path.dirname(cpath)
        log_id = os.path.split(log_dir)[-1]
        if log_id == uri.log_id:
          log_id_to_loader[log_id] = AUTrackingLoader(
                                        os.path.dirname(log_dir),
                                        log_id)

    assert uri.log_id in log_id_to_loader, "Could not find log %s" % uri.log_id
    return log_id_to_loader[uri.log_id]

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
  

  ## Setup

  @classmethod
  def download_all(cls):
    util.mkdir(cls.tarball_path(''))

    all_tarballs = itertools.chain.from_iterable(
      getattr(cls, attr, [])
      for attr in dir(cls)
      if attr.endswith('_TARBALLS'))
    for tarball in all_tarballs:
      uri = cls.BASE_TARBALL_URL + '/' + tarball
      util.download(uri, cls.zip_dir(fname), try_expand=True)

  @classmethod
  def run_import(cls):
    cls.download_all()






if __name__ == '__main__':
  Fixtures().run_import()