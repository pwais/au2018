"""
NB: every track is visible for at least a few frames:
  (where data is `ImageAnnoTable`)
spark.sql('''
  select track_id, sum(if(is_visible, 1, 0)) num_invisible, count(*) total
  from data
  group by track_id having total - num_invisible < 10
''').show()
+--------+-------------+-----+                                                  
|track_id|num_invisible|total|
+--------+-------------+-----+
+--------+-------------+-----+

Having total - num_invisible < 100: 232 tracks.

Total tracks: 8894
"""

import copy
import itertools
import os

from au import conf
from au import util
from au.fixtures import dataset
from au.fixtures.datasets import common
from au.spark import Spark

import imageio
import math
import numpy as np
import six

from pyspark.sql import Row

import klepto # For a cache of imageio Readers / Argoverse Loaders

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

BIKE = ["BICYCLE", "MOPED", "MOTORCYCLE"]
RIDER = ["BICYCLIST", "MOTORCYCLIST"]

class MissingPose(ValueError):
  pass

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

    ## Optional
    'track_id',     # A UUID of a specific track / annotation in the frame
    
    'crop_x', 'crop_y',
    'crop_w', 'crop_h',
                    # A specific viewport / crop of the frame
  )
  
  OPTIONAL = ('track_id', 'crop_x', 'crop_y', 'crop_w', 'crop_h',)

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

  def update(self, **kwargs):
    for k in self.__slots__:
      if k in kwargs:
        setattr(self, k, kwargs[k])

  def set_crop(self, bbox):
    self.update(
      crop_x=bbox.x,
      crop_y=bbox.y,
      crop_w=bbox.width,
      crop_h=bbox.height)

  def has_crop(self):
    return all(
      getattr(self, 'crop_%s' % a) is not ''
      for a in ('x', 'y', 'w', 'h'))

  def get_crop_bbox(self):
    return BBox(
            x=self.crop_x, y=self.crop_y,
            width=self.crop_w, height=self.crop_h)

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
      'distance_meters',      # Dist to closest cuboid point
      'relative_yaw_radians', # Yaw vs ego pose
      'relative_yaw_to_camera_radians',
      'has_offscreen',
      'is_visible',
      'z',

      # Has the ObjectLabelRecord been motion-corrected?
      'motion_corrected',
      'cuboid_pts',         # In robot ego frame
      'cuboid_pts_image',   # In image space
      'ego_to_obj',         # Translation vector in ego frame
    ]
  )

  def draw_cuboid_in_image(self, img, base_color=None, alpha=0.3, thickness=2):
    """Draw `cuboid_pts_image` in `img`.  Similar to argoverse
    render_clip_frustum_cv2(), but much simpler; in particular,
    the code below does not confound camera calibration.
    """
    
    if not hasattr(self.cuboid_pts, 'shape'):
      return

    ## Pick colors to draw
    if not base_color:
      from au import plotting as aupl
      base_color = aupl.hash_to_rbg(self.category_name)
    base_color = np.array(base_color)

    def color_to_opencv(color):
      r, g, b = np.clip(color, 0, 255).astype(int).tolist()
      return b, g, r
    
    front_color = color_to_opencv(base_color + 0.3 * 255)
    back_color = color_to_opencv(base_color - 0.3 * 255)
    center_color = color_to_opencv(base_color)

    import cv2
    # OpenCV can't draw transparent colors, so we use the 'overlay image' trick
    overlay = img.copy()

    front = self.cuboid_pts_image[:4].astype(int)
    cv2.polylines(
      overlay,
      [front],
      True, # is_closed
      front_color,
      thickness)

    back = self.cuboid_pts_image[4:].astype(int)
    cv2.polylines(
      overlay,
      [back],
      True, # is_closed
      back_color,
      thickness)
    
    for start, end in zip(front.tolist(), back.tolist()):
      cv2.line(overlay, tuple(start), tuple(end), center_color, thickness)

    # Now blend!
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

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
      bbox.cuboid_pts = object_label_record.as_3d_bbox()
      bbox.motion_corrected = False
        # Points in robot frame
      if motion_corrected:
        try:
          bbox.cuboid_pts = loader.get_motion_corrected_pts(
                                    bbox.cuboid_pts,
                                    object_label_record.timestamp,
                                    uri.timestamp)
          bbox.motion_corrected = True
        except MissingPose:
          # Garbage!
          pass

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

      camera_yaw = math.atan2(calib.R[2, 1], calib.R[1, 1])
      bbox.relative_yaw_to_camera_radians = (
        bbox.relative_yaw_radians + camera_yaw) % (2. * math.pi)

      bbox.ego_to_obj = object_label_record.translation

    def fill_bbox_core(bbox):
      bbox.category_name = object_label_record.label_class

      bbox.im_width, bbox.im_height = get_image_width_height(uri.camera)
      uv = calib.project_ego_to_image(bbox.cuboid_pts)
      
      bbox.cuboid_pts_image = np.array([uv[:, 0] , uv[:, 1]]).T

      x1, x2 = np.min(uv[:, 0]), np.max(uv[:, 0])
      y1, y2 = np.min(uv[:, 1]), np.max(uv[:, 1])
      z = float(np.max(uv[:, 2]))

      bbox.set_x1_y1_x2_y2(x1, y1, x2, y2)

      num_onscreen = bbox.get_num_onscreen_corners()
      bbox.has_offscreen = ((z <= 0) or (num_onscreen < 4))
      bbox.is_visible = (
        z > 0 and
        num_onscreen > 0 and
        object_label_record.occlusion < 100)

      bbox.clamp_to_screen()
      bbox.z = float(z)

    bbox = BBox()
    fill_cuboid_pts(bbox)
    fill_bbox_core(bbox)
    fill_extra(bbox)
    return bbox



class AVFrame(object):

  __slots__ = (
    # Meta
    'uri',            # type: FrameURI
    'FIXTURES',       # type: au.datasets.argoverse.Fixtures
    '_loader',        # type: AUTrackingLoader

    # Labels
    '_image_bboxes',  # type: List[BBox]

    # Vision
    '_image',         # type: np.ndarray
    'viewport',       # type: BBox (used to express a crop)
    
    # Lidar
    '_cloud',         # type: np.ndarray
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
    
    if isinstance(self.uri, six.string_types):
      self.uri = FrameURI.from_str(self.uri)
    
    # Fill context if needed
    if not self.FIXTURES:
      self.FIXTURES = Fixtures
  
    if not self.viewport:
      if self.uri.has_crop():
        self.viewport = self.uri.get_crop_bbox()
      else:
        self.viewport = BBox.of_size(*get_image_width_height(self.uri.camera))
    
  @property
  def loader(self):
    if not self._loader:
      self._loader = self.FIXTURES.get_loader(self.uri)
    return self._loader # type: AUTrackingLoader
  
  @staticmethod
  @klepto.lru_cache(maxsize=1000)
  def __load_image(path):
    return imageio.imread(path)

  @property
  def image(self):
    if not self._image:
      path = self.loader.get_nearest_image_path(
                      self.uri.camera, self.uri.timestamp)
      self._image = AVFrame.__load_image(path)
      # if not self.viewport.is_full_image():
      #   c, r, w, h = (
      #     self.viewport.x, self.viewport.y,
      #     self.viewport.width, self.viewport.height)
      #   self._image = self._image[r:r+h, c:c+w, :]
    return self._image
  
  @property
  def cloud(self):
    if not self._cloud:
      self._cloud, motion_corrected = \
        self.loader.get_maybe_motion_corrected_cloud(self.uri.timestamp)
        # We can ignore motion_corrected failures since the Frame will already
        # have this info embedded in `image_bboxes`.
    return self._cloud
  
  def get_cloud_in_image(self):
    cloud = self.cloud
    calib = self.loader.get_calibration(self.uri.camera)

    # Per the argoverse recommendation, this should be safe:
    # https://github.com/argoai/argoverse-api/blob/master/demo_usage/argoverse_tracking_tutorial.ipynb
    x, y, w, h = (
      self.viewport.x, self.viewport.y,
      self.viewport.width, self.viewport.height)
    uv = calib.project_ego_to_image(cloud).T
    idx_ = np.where(
            np.logical_and.reduce((
              # Filter offscreen points
              x <= uv[0, :], uv[0, :] < x + w - 1.0,
              y <= uv[1, :], uv[1, :] < y + h - 1.0,
              # Filter behind-screen points
              uv[2, :] > 0)))
    idx_ = idx_[0]
    uv = uv[:, idx_]
    return uv.T

  @property
  def image_bboxes(self):
    if not self._image_bboxes:
      
      t = self.uri.timestamp
      av_label_objects = self.loader.get_nearest_label_object(t)

      # Some of the labels are complete junk
      av_label_objects = [
        olr for olr in av_label_objects
        if not (
          np.isnan(olr.quaternion).any() or 
          np.isnan(olr.translation).any())
      ]

      bboxes = [
        BBox.from_argoverse_label(self.uri, olr) for olr in av_label_objects
      ]

      # Ingore invisible things
      self._image_bboxes = [
        bbox for bbox in bboxes
        if bbox.is_visible and self.viewport.overlaps_with(bbox)
      ]

    return self._image_bboxes

  def get_debug_image(self):
    img = np.copy(self.image)
    
    from au import plotting as aupl
    xyd = self.get_cloud_in_image()
    aupl.draw_xy_depth_in_image(img, xyd)

    if self.uri.track_id:
      # Draw a gold box first; then the draw() calls below will draw over
      # the box.
      WHITE = (225, 225, 255)
      for bbox in self.image_bboxes:
        if bbox.track_id == self.uri.track_id:
          bbox.draw_in_image(img, color=WHITE, thickness=20)

    for bbox in self.image_bboxes:
      bbox.draw_cuboid_in_image(img)
      bbox.draw_in_image(img)
    
    if not self.viewport.is_full_image():
      c, r, w, h = (
        self.viewport.x, self.viewport.y,
        self.viewport.width, self.viewport.height)
      img = img[r:r+h, c:c+w, :]
    return img

  def get_cropped(self, bbox):
    """Create and return a new AVFrame instance that contains the data in this
    frame cropped down to the viewport of just `bbox`."""

    uri = copy.deepcopy(self.uri)
    uri.set_crop(bbox)

    frame = AVFrame(uri=uri, FIXTURES=self.FIXTURES)
    return frame

class HardNegativeMiner(object):
  SEED = 1337
  MAX_FRACTION_ANNOTATED = 0.2
  WIDTH_PIXELS_MU_STD = (121, 50)
  HEIGHT_PIXELS_MU_STD = (121, 50)

  def __init__(self, frame):
    self._frame = frame
    
    # Build a binary mask where a pixel has an indicator value of 1 only if
    # one or more annotation bboxes covers that pixel
    mask = np.zeros((self._frame.viewport.height, self._frame.viewport.width))
    for bbox in frame.image_bboxes:
      mask[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width] = 1

    import imageio
    imageio.imwrite('/opt/au/mask.png', mask, format='png')

    # We'll use the integral image trick to make rejection sampling efficient
    class IntegralImage(object):
      def __init__(self, img):
        self.ii = img.cumsum(axis=0).cumsum(axis=1)
      
      def get_sum(self, r1, c1, r2, c2):
        return (
          self.ii[r2, c2]
          - self.ii[r1, c2] - self.ii[r2, c1]
          + self.ii[r1, c1])

    self._ii = IntegralImage(mask)
    import random
    self._random = random.Random(self.SEED)

  def next_sample(self, max_attempts=1000):
    rand = self._random
    for _ in range(max_attempts):
      v = self._frame.viewport
      
      # Pick a center
      c_x = rand.randint(v.x, v.x + v.width)
      c_y = rand.randint(v.y, v.y + v.height)

      # Pick a size
      c_w = rand.normalvariate(*self.WIDTH_PIXELS_MU_STD)
      c_h = rand.normalvariate(*self.HEIGHT_PIXELS_MU_STD)
      if c_w <= 0 or c_h <= 0:
        # Immediately reject boxen that have area 0 or are invalid
        continue

      # Snap to a valid box
      x1 = c_x - .5 * c_w
      y1 = c_y - .5 * c_h
      x2 = c_x + .5 * c_w
      y2 = c_y + .5 * c_h
      proposal = BBox.from_x1_y1_x2_y2(x1, y1, x2, y2)
      proposal.quantize()
      sample = v.get_intersection_with(proposal)

      num_anno_pixels = self._ii.get_sum(y1, x1, y2, x2)
      sample = BBox.from_x1_y1_x2_y2(x1, y1, x2, y2)
      
      # Do we have enough non-annotated pixels to accept?
      if num_anno_pixels / sample.get_area() <= self.MAX_FRACTION_ANNOTATED:
        print(num_anno_pixels / sample.get_area()) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return sample
    util.log.warn(
      "Tried %s times and could not sample an unannotated box" % max_attempts)
    return None

class RingMiner(HardNegativeMiner):
  # We could perhaps choose these using camera intrinsics, but instead
  # we use the empircal distributon from existing annotations
  # See cache/data/argoverse/index/image_annos/Size_stats_by_Camera.html
  WIDTH_PIXELS_MU_STD = (107.842119, 123.289099)
  HEIGHT_PIXELS_MU_STD = (92.435195, 107.863304)

class StereoMiner(HardNegativeMiner):
  # We could perhaps choose these using camera intrinsics, but instead
  # we use the empircal distributon from existing annotations
  # See cache/data/argoverse/index/image_annos/Size_stats_by_Camera.html
  WIDTH_PIXELS_MU_STD = (201.600639, 189.607279)
  HEIGHT_PIXELS_MU_STD = (196.776626, 198.066231)



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
                    conf.AU_CACHE_TMP,
                    'argoverse_loader',
                    log_name)
    util.mkdir(virtual_root)
    try:
      os.symlink(
        os.path.join(root_dir, log_name),
        os.path.join(virtual_root, log_name))
    except FileExistsError:
      pass

    util.log.info(
      "Creating loader for log %s with root dir %s" % (log_name, root_dir))
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
  
  def get_nearest_lidar_sweep_id(self, timestamp):
    """Return the index of the lidar sweep and its timestamp in this log that
    either matches exactly or is closest to `timestamp`."""
    diff, idx = min(
              (abs(timestamp - t), idx)
              for idx, t in enumerate(self.lidar_timestamp_list))
    assert diff < 1e9, \
      "Could not find a cloud within 1 sec of %s, diff %s" % (timestamp, diff)
    return idx, self.lidar_timestamp_list[idx]

  def get_nearest_label_object(self, timestamp):
    """Load and return the `ObjectLabelRecord`s nearest to `timestamp`;
    provide either an exact match or choose the closest available."""

    idx, _ = self.get_nearest_lidar_sweep_id(timestamp)
    if idx >= len(self.label_list):
      util.log.error(
        "Log %s has %s labels but %s lidar sweeps; idx %s out of range" % (
          self.current_log, len(self.label_list),
          len(self.lidar_timestamp_list), idx))
      return []

    import argoverse.data_loading.object_label_record as object_label
    objs = object_label.read_label(self.label_list[idx])
      # NB: the above actually reads a *list* of label objects :P
    
    # We hide the object timestamp in the label; I guess the Argoverse
    # authors didn't think the timestamp was important :P
    label_t = self.lidar_timestamp_list[idx]
    for obj in objs:
      obj.timestamp = label_t
    
    return objs

  # @klepto.lru_cache(maxsize=100) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIXME
  def get_maybe_motion_corrected_cloud(self, timestamp):
    """Similar to `get_lidar()` but motion-corrects the entire cloud
    to (likely camera-time) `timestamp`.  Return also True if
    motion corrected."""
    idx, lidar_t = self.get_nearest_lidar_sweep_id(timestamp)
    cloud = self.get_lidar(idx)
    try:
      return self.get_motion_corrected_pts(cloud, lidar_t, timestamp), True
    except MissingPose:
      return cloud, False

  def get_motion_corrected_pts(self, pts, pts_timestamp, dest_timestamp):
    """Similar to project_lidar_to_img_motion_compensated(), but:
      * do not project to image
      * do not fail silently
      * do not have an extremely poor interface
    We transform the points through the city / world frame:
      pt_ego_dest_t = ego_dest_t_SE3_city * city_SE3_ego_pts_t * pt_ego_pts_t
    """

    from argoverse.data_loading.pose_loader import \
      get_city_SE3_egovehicle_at_sensor_t

    city_SE3_ego_dest_t = get_city_SE3_egovehicle_at_sensor_t(
                              dest_timestamp,
                              self.root_dir,
                              self.current_log)
    if city_SE3_ego_dest_t is None:
      raise MissingPose()

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when the LiDAR sweep was recorded.
    city_SE3_ego_pts_t = get_city_SE3_egovehicle_at_sensor_t(
                              pts_timestamp,
                              self.root_dir,
                              self.current_log)
    if city_SE3_ego_pts_t is None:
      raise MissingPose()
    
    # Argoverse SE3 does not want homogenous coords
    pts = np.copy(pts)
    if pts.shape[-1] == 4:
      pts = pts.T[:, :3]

    ego_dest_t_SE3_ego_pts_t = \
      city_SE3_ego_dest_t.inverse().right_multiply_with_se3(city_SE3_ego_pts_t)
    pts = ego_dest_t_SE3_ego_pts_t.transform_point_cloud(pts)
    
    from argoverse.utils.calibration import point_cloud_to_homogeneous
    return pts
    


###
### Data
###

class Fixtures(object):

  # All Argoverse tarballs served from here
  # BASE_TARBALL_URL = "https://s3.amazonaws.com/argoai-argoverse"

  # If you happen to have a local copy of the tarballs, use this:
  BASE_TARBALL_URL = "file:///outer_root/tmp/argotars"

  # # If you happen to have a local copy of the tarballs, use this:
  # BASE_TARBALL_URL = "file:///tmp/argotars"

  ###
  ### NB: we omit the forecasting tarballs because they appear to exclude
  ### sensor data (and are therefore not useful for research).
  ###

  TRACKING_SAMPLE = "tracking_sample.tar.gz"

  SAMPLE_TARBALLS = (
    TRACKING_SAMPLE,
    # "forecasting_sample.tar.gz",
  )

  TRACKING_TARBALLS = (
    "tracking_train1.tar.gz",
    "tracking_train2.tar.gz",
    "tracking_train3.tar.gz",
    "tracking_train4.tar.gz",
    "tracking_val.tar.gz",
    "tracking_test.tar.gz",
  )

  PREDICTION_TARBALLS = tuple()
  # (
  #   "forecasting_train.tar.gz",
  #   "forecasting_val.tar.gz",
  #   "forecasting_test.tar.gz",
  # )

  MAP_TARBALLS = (
    "hd_maps.tar.gz",
  )

  SPLITS = ('train', 'test', 'val', 'sample')

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'argoverse')

  # TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'argoverse') ~~~~~~~~~~~


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


  ## Derived Data
  
  @classmethod
  def index_root(cls):
    return os.path.join(cls.ROOT, 'index')

  @classmethod
  def image_annos_reports_root(cls):
    return os.path.join(cls.index_root(), 'image_annos')


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
    return cls._get_loader(uri.tarball_name, uri.log_id)
  
  @classmethod
  @klepto.inf_cache(ignore=(0,))
  def _get_loader(cls, tarball_name, log_id):
    # Need to find the dir corresponding to log_id
    base_path = cls.tarball_dir(tarball_name)
    for log_dir in cls.get_log_dirs(base_path):
      cur_log_id = os.path.split(log_dir)[-1]
      if cur_log_id == log_id:
        return AUTrackingLoader(os.path.dirname(log_dir), log_id)
    
    raise ValueError("Could not find log %s in %s" % (log_id, base_path))

  # @classmethod
  # def _get_loader(cls, tarball_name, log_id):
  #   # NB: We tried to use @klepto.inf_cache(ignore=(0,)) here, but there
  #   # appeared to be GIL contention somehow ...
    
  #   if not hasattr(cls, '_key_to_loader'):
  #     cls._key_to_loader = {}
    
  #   if (tarball_name, log_id) not in cls._key_to_loader:
  #     loader = None # Build this

  #     # Need to find the dir corresponding to log_id
  #     base_path = cls.tarball_dir(tarball_name)
  #     for log_dir in cls.get_log_dirs(base_path):
  #       cur_log_id = os.path.split(log_dir)[-1]
  #       if cur_log_id == log_id:
  #         loader = AUTrackingLoader(os.path.dirname(log_dir), log_id)
    
  #     assert loader, "Could not find log %s" % log_id
  #     cls._key_to_loader[(tarball_name, log_id)] = loader
  #   return cls._key_to_loader[(tarball_name, log_id)]

    # if not hasattr(cls, '_tarball_log_id_to_loader'):
    #   cls._tarball_log_id_to_loader = {}
    
    # key = (uri.tarball_name, uri.log_id)
    # if key not in cls._tarball_log_id_to_loader:
      
      
    #   cls._tarball_log_id_to_loader[key] = loader
    # # else:
    # #   print('using cached loader') # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # return cls._tarball_log_id_to_loader[key]

  # @classmethod
  # def iter_frame_uris(cls, split):
  #   assert split in cls.SPLITS
  #   tarballs = cls.all_tracking_tarballs()
  #   tarballs = [t for t in tarballs if split in t]
  #   for tarball in tarballs:
  #     base_path = cls.tarball_dir(tarball)
  #     for log_dir in cls.get_log_dirs(base_path):
  #       log_id = os.path.split(log_dir)[-1]
  #       loader = cls.get_loader(FrameURI(tarball_name=tarball, log_id=log_id))
  #       for camera, ts_to_path in loader.timestamp_image_dict.items():
  #         for ts in ts_to_path.keys():
  #           yield FrameURI(
  #             tarball_name=tarball,
  #             log_id=log_id,
  #             split=split,
  #             camera=camera,
  #             timestamp=ts)
  
  @classmethod
  def get_log_uris(cls, split):
    assert split in cls.SPLITS
    tarballs = cls.all_tracking_tarballs()
    tarballs = [t for t in tarballs if split in t]
    for tarball in tarballs:
      base_path = cls.tarball_dir(tarball)
      for log_dir in cls.get_log_dirs(base_path):
        log_id = os.path.split(log_dir)[-1]
        yield FrameURI(split=split, tarball_name=tarball, log_id=log_id)

  @classmethod
  def get_frame_uris(cls, log_frame_uri):
    loader = cls.get_loader(log_frame_uri)
    for camera, ts_to_path in loader.timestamp_image_dict.items():
      for ts in ts_to_path.keys():
        uri = log_frame_uri.to_dict()
        uri.update(camera=camera, timestamp=ts)
        yield FrameURI(**uri)

  

  ## Setup

  @classmethod
  def download_all(cls, spark=None):
    util.mkdir(cls.tarball_path(''))
    util.log.info(
      'Downloading %s tarballs in parallel' % len(cls.all_tarballs()))
    with Spark.sess(spark) as spark:
      Spark.run_callables(
        spark,
        (
          lambda: (
            util.download(
              cls.BASE_TARBALL_URL + '/' + tarball,
              cls.tarball_dir(tarball),
              try_expand=True))
          for tarball in cls.all_tarballs()
        ))

  @classmethod
  def run_import(cls, spark=None):
    cls.download_all(spark=spark)
    ImageAnnoTable.setup(spark=spark)
    ImageAnnoTable.save_anno_reports(spark)



class ImageAnnoTable(object):
  """A table of argoverse annotations projected into image space."""

  FIXTURES = Fixtures

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, 'argoverse_image_annos')

  @classmethod
  def setup(cls, spark=None):
    if not os.path.exists(cls.table_root()):
      with Spark.sess(spark) as spark:
        df = cls.build_anno_df(spark)
        df.write.parquet(cls.table_root(), compression='gzip')

  @classmethod
  def as_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    return df

  ## Utils

  @classmethod
  def get_stats_dfs(cls, spark, df=None):
    if not df:
      df = cls.as_df(spark)
    df.createOrReplaceTempView("nms")
    title_queries = (
      ("Size stats by Split", """
              SELECT
                split,
                AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
                AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
                COUNT(*) AS num_annos,
                COUNT(DISTINCT frame_uri) AS num_frames
              FROM nms
              GROUP BY split"""
      ),
      ("Size stats by City", """
              SELECT
                  city,
                  AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
                  AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
                  COUNT(*) AS num_annos,
                  COUNT(DISTINCT frame_uri) AS num_frames
                FROM nms
                GROUP BY city"""
      ),
      ("Size stats by Category", """
              SELECT
                  category_name,
                  AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
                  AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
                  COUNT(*) AS num_annos,
                  COUNT(DISTINCT frame_uri) AS num_frames
                FROM nms
                GROUP BY category_name"""
      ),
      ("Size stats by Camera", """
              SELECT
                  camera,
                  AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
                  AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
                  COUNT(*) AS num_annos,
                  COUNT(DISTINCT frame_uri) AS num_frames
                FROM nms
                GROUP BY camera"""
      ),
      ("Anno Counts by Camera", """
              SELECT
                  camera,
                  AVG(num_annos) AS num_annos_mu,
                  STD(num_annos) AS num_annos_std,
                  SUM(num_annos) AS num_annos_total,
                  COUNT(*) AS num_frames_total
                FROM (
                  SELECT
                    camera, frame_uri, COUNT(*) AS num_annos
                  FROM nms
                  GROUP BY camera, frame_uri
                )
                GROUP BY camera
                ORDER BY camera"""
      ),
      ("Pedestrian stats by Distance", """
              SELECT
                  camera,
                  10 * MOD(ROUND(distance_meters), 10) AS distance_m_bucket,
                  AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
                  AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
                  COUNT(*) AS num_annos,
                  COUNT(DISTINCT frame_uri) AS num_frames
                FROM nms
                WHERE
                  category_name = 'PEDESTRIAN' AND 
                  camera in ('ring_front_center', 'stereo_front_left')
                GROUP BY camera, distance_m_bucket
                ORDER BY camera ASC, distance_m_bucket ASC"""
      ),
    )
    for title, query in title_queries:
      util.log.info("Running %s ..." % title)
      yield title, spark.sql(query).toPandas()

  @classmethod
  def _impute_rider_for_bikes(cls, spark, df):
    util.log.info("Imputing rider <-> bike matchings ...")

    # Sub-select just bike and rider rows from `df` to improve performance
    bikes_df = df.filter(df.category_name.isin(BIKE + RIDER))
    bikes_df = bikes_df.select(
                  'uri',
                  'frame_uri',
                  'track_id',
                  'category_name',
                  'ego_to_obj')

    def iter_nearest_bike(uri_rows):
      uri, rows = uri_rows
      rows = list(rows) # Spark gives us a generator
      bike_rows = [r for r in rows if r.category_name in BIKE]
      rider_rows = [r for r in rows if r.category_name in RIDER]
      
      # The best pair has smallest euclidean distance between centroids
      def l2_dist(r1, r2):
        a1 = np.array([r1.ego_to_obj.x, r1.ego_to_obj.y, r1.ego_to_obj.z])
        a2 = np.array([r2.ego_to_obj.x, r2.ego_to_obj.y, r2.ego_to_obj.z])
        return float(np.linalg.norm(a2 - a1))

      # Each rider gets assigned the nearest bike.  Note that bikes may not
      # have riders.
      for rider in rider_rows:
        if bike_rows:
          distance, best_bike = min(
                          (l2_dist(rider, bike), bike)
                          for bike in bike_rows)
          nearest_bike = dict(
            uri=rider.uri,
            track_id=rider.track_id,
            ridden_bike_track_id=best_bike.track_id,
            ridden_bike_distance=distance,
          )

          yield Row(**nearest_bike)
    
    # We'll group all rows in our DF by URI, then do bike<->rider
    # for each URI (i.e. all the rows for a single URI).  The matching
    # will spit out a new DataFrame, which we'll join against the 
    # original `df` in order to "add" the columns encoding the
    # bike<->rider matchings.
    uri_chunks_rdd = bikes_df.rdd.groupBy(lambda r: r.frame_uri)
    nearest_bike = uri_chunks_rdd.flatMap(iter_nearest_bike)
    if nearest_bike.isEmpty():
      util.log.info("... no matchings!")
      return df
    matched = spark.createDataFrame(nearest_bike)
    util.log.info("... matched %s bikes." % matched.count())
    
    joined = df.join(matched, ['uri', 'track_id'], 'outer')

    # Don't allow nulls; those can't be compared and/or written to Parquet
    joined = joined.na.fill({
                  'ridden_bike_distance': float('inf'),
                  'ridden_bike_track_id': ''
    })
    return joined

  @classmethod
  def _create_uri_rdd(cls, spark, splits=None):
    if not splits:
      splits = cls.FIXTURES.SPLITS

    util.log.info("Building anno df for splits %s" % (splits,))

    # Be careful to hint to Spark how to parallelize reads
    log_uris = list(
              itertools.chain.from_iterable(
                    cls.FIXTURES.get_log_uris(split)
                    for split in splits))
    util.log.info("... reading from %s logs ..." % len(log_uris))
    log_uri_rdd = spark.sparkContext.parallelize(
                            log_uris, numSlices=len(log_uris))
    uri_rdd = log_uri_rdd.flatMap(cls.FIXTURES.get_frame_uris)
    uri_rdd = uri_rdd.repartition(1000)
    util.log.info("... read %s URIs ..." % uri_rdd.count())
    return uri_rdd

  @classmethod
  def build_anno_df(cls, spark, splits=None):
    uri_rdd = cls._create_uri_rdd(spark, splits=splits)

    def iter_anno_rows(uri):
      from collections import namedtuple
      pt = namedtuple('pt', 'x y z')

      frame = AVFrame(uri=uri, FIXTURES=cls.FIXTURES)
      for box in frame.image_bboxes:
        row = {}

        # Obj
        # TODO make spark accept numpy and numpy float64 things
        row = box.to_dict()
        IGNORE = ('cuboid_pts', 'cuboid_pts_image', 'ego_to_obj')
        for attr in IGNORE:
          v = row.pop(attr)
          if attr == 'cuboid_pts_image':
            continue
          if hasattr(v, 'shape'):
            if len(v.shape) == 1:
              row[attr] = pt(*v.tolist())
            else:
              row[attr] = [pt(*v[r, :3].tolist()) for r in range(v.shape[0])]
        
        # Anno Context
        obj_uri = copy.deepcopy(frame.uri)
        obj_uri.track_id = box.track_id
        row.update(
          frame_uri=str(uri),
          uri=str(obj_uri),
          **obj_uri.to_dict())
        row.update(
          city=cls.FIXTURES.get_loader(uri).city_name,
          coarse_category=AV_OBJ_CLASS_TO_COARSE.get(box.category_name, ''))
        
        from pyspark.sql import Row
        yield Row(**row)
    
    row_rdd = uri_rdd.flatMap(iter_anno_rows)
    df = spark.createDataFrame(row_rdd)
    df = cls._impute_rider_for_bikes(spark, df)
    return df

  @classmethod
  def save_anno_reports(cls, spark=None, dest_dir=None):
    spark = spark or Spark.getOrCreate()
    dest_dir = dest_dir or cls.FIXTURES.image_annos_reports_root()
    util.mkdir(dest_dir)

    util.log.info("Creating image annotation reports in %s ..." % dest_dir)

    ## First do overall stats reports
    for title, pdf in cls.get_stats_dfs(spark):
      fname = title.replace(' ', '_') + '.html'
      with open(os.path.join(dest_dir, fname), 'w') as f:
        f.write(pdf.to_html())
      util.log.info("Saved simple report: \n%s\n%s\n\n" % (title, pdf))

    ## Histogram reports
    # For each of these metrics in ImageAnnoTable, generate a distinct plot
    # for each sub-pivot column
    SPLIT_AND_CITY = ['split', 'city']
    SPLIT_CITY_CATEGORY = SPLIT_AND_CITY + ['category_name']
    SPLIT_CITY_CAMERA = SPLIT_AND_CITY + ['camera']
    CATEGORY_AND_CAMERA = ['category_name', 'coarse_category', 'camera']
    ALL_SUB_PIVOTS = SPLIT_AND_CITY + CATEGORY_AND_CAMERA
    METRIC_AND_SUB_PIVOTS = (
      ('distance_meters',       ALL_SUB_PIVOTS),
      ('height_meters',         SPLIT_CITY_CATEGORY),
      ('width_meters',          SPLIT_CITY_CATEGORY),
      ('length_meters',         SPLIT_AND_CITY),
      ('height',                SPLIT_CITY_CAMERA),
      ('width',                 SPLIT_CITY_CAMERA),
      ('relative_yaw_radians',  SPLIT_AND_CITY),
      ('relative_yaw_to_camera_radians',  ALL_SUB_PIVOTS),
      ('occlusion',             SPLIT_AND_CITY),

      # Special handling! See below
      ('ridden_bike_distance',   ALL_SUB_PIVOTS),
    )

    num_plots = sum(len(spvs) for metric, spvs in METRIC_AND_SUB_PIVOTS)
    util.log.info("Going to generate %s plots ..." % num_plots)
    t = util.ThruputObserver(name='plotting', n_total=num_plots)
    
    # Generate plots!
    anno_df = cls.as_df(spark)
    for metric, sub_pivots in METRIC_AND_SUB_PIVOTS:
      for sub_pivot in sub_pivots:
        df = anno_df
        if metric == 'ridden_bike_distance':
          # We need to filter out Infinity for histograms to work
          df = df.filter(df.ridden_bike_distance < float('inf')).cache()
          if df.count() == 0:
            util.log.warn("... skipping %s, no data! ..." % plot_dest) 
            continue

        plot_name = metric + ' by ' + sub_pivot
        plot_fname = plot_name.replace(' ', '_') + '.html'
        plot_dest = os.path.join(dest_dir, plot_fname)
        if os.path.exists(plot_dest):
          util.log.info("... skipping %s ..." % plot_dest)
          continue
        util.log.info("... plotting %s ..." % plot_name)
        
        from au import plotting as aupl
        class AVHistogramPlotter(aupl.HistogramWithExamplesPlotter):
          NUM_BINS = 20
          SUB_PIVOT_COL = sub_pivot
          WIDTH = 1400
          TITLE = plot_name

          # Show only this many examples for each bucket.  More
          # examples -> more images -> larger plot files.
          EXAMPLES_PER_BUCKET = 10

          def display_bucket(self, sub_pivot, bucket_id, irows):
            util.log.info("Displaying bucket %s %s" % (sub_pivot, bucket_id))

            # Try to sample examples from distinct logs for higher
            # variance in examples.
            def sample_rows(n):
              from collections import defaultdict
              log_id_to_rows = defaultdict(list)
              for r in irows:
                log_id_to_rows[r.log_id].append(r)

                if len(log_id_to_rows.keys()) >= n:
                  # We'll have at least one log per sample, so it's safe to
                  # bail early (and thus stop consuming rows from Spark)
                  break
              
              # Now we can sample from log_ids round-robin
              round_robin_uris = util.roundrobin(*log_id_to_rows.values())
              rows = list(itertools.islice(round_robin_uris, n))
              return rows
            
            def disp_row(title, row):
              from six.moves.urllib import parse
              TEMPLATE = """<a href="{href}">{title} {img_tag} {uri}</a>"""
              BASE = "/view?"
              href = BASE + parse.urlencode({'uri': row.uri})

              frame = AVFrame(uri=row.uri, FIXTURES=cls.FIXTURES)
              debug_img = frame.get_debug_image()
              
              if row.ridden_bike_track_id:
                # Highlight the rider's bike if possible. Rather than draw a
                # new box in the image, it's easiest to just fetch a debug
                # image for the rider and blend using OpenCV
                best_bike_uri = FrameURI.from_str(row.uri)
                best_bike_uri.track_id = row.ridden_bike_track_id
                dframe = AVFrame(uri=best_bike_uri, FIXTURES=cls.FIXTURES)
                debug_img_bike = dframe.get_debug_image()
                import cv2
                debug_img[:] = cv2.addWeighted(
                  debug_img, 0.5, debug_img_bike, 0.5, 0)

              img_tag = aupl.img_to_img_tag(
                          debug_img,
                          jpeg_quality=50,
                          display_viewport_hw=(300, 300))
              s = TEMPLATE.format(
                              href=href,
                              title=title,
                              img_tag=img_tag,
                              uri=str(row.uri))
              return s

            rows = sample_rows(self.EXAMPLES_PER_BUCKET)
            disp_htmls = [
              disp_row('Example %s' % i, row)
              for i, row in enumerate(rows)
            ]
            disp_str = sub_pivot + '<br/><br/>' + '<br/><br/>'.join(disp_htmls)
            return bucket_id, disp_str
        
        t.start_block()
        plotter = AVHistogramPlotter()
        fig = plotter.run(df.cache(), metric)
        aupl.save_bokeh_fig(fig, plot_dest)

        # Show ETA
        t.stop_block(n=1)
        t.maybe_log_progress(every_n=1)


###
### Image Tables
###

from collections import namedtuple
cropattrs = namedtuple('cropattrs', 'anno cloud_npz viewport_annos')

class CroppedObjectImageTable(dataset.ImageTable):

  # Center the object in a viewport of this size (pixels)
  VIEWPORT_WH = (160, 160)
  
  # Pad the object by this many pixels against the viewport edges
  PADDING_PIXELS = 10

  # TODO mebbe make more rigorous check stats ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NEGATIVE_SAMPLES_PER_FRAME = 20

  ANNOS = ImageAnnoTable

  TABLE_NAME = (
    'argoverse_cropped_object_%s_%s' % (VIEWPORT_WH[0], VIEWPORT_WH[1]))

  @classmethod
  def setup(cls, **kwargs):
    if os.path.exists(cls.table_root()):
      util.log.info(
        "Skipping setup for %s, %s exists." % (
          cls.TABLE_NAME, cls.table_root()))
      return
    
    cls._create_eligable_annos(kwargs.get('spark'))
    

  ### Utils

  @classmethod
  def get_crop_viewport(cls, bbox):
    """Compute and return a viewport with the aspect ratio defined in
    `VIEWPORT_WH` (and respecting `PADDING_PIXELS`) from which we can
    create a crop of the object annotated in `bbox`.  Note that the
    returned bbox may extend beyond the image edges."""
    
    radius = .5 * max(bbox.width, bbox.height)
    padding_xy = 2. * cls.PADDING_PIXELS / np.array(cls.VIEWPORT_WH)
    radius_xy = radius + .5 * padding_xy
    box_center = np.array(
      [bbox.x + .5 * bbox.width, bbox.y + .5 * bbox.height])
    
    x1, y1 = (box_center - radius_xy).tolist()
    x2, y2 = (box_center + radius_xy).tolist()
    cropped = copy.deepcopy(bbox)
    cropped.set_x1_y1_x2_y2(x1, y1, x2, y2)
    cropped.update(im_width=bbox.im_width, im_height=bbox.im_height)
    cropped.quantize()
    return cropped

  @classmethod
  def _create_crop_spec_df(cls, spark):
    anno_df = cls.ANNOS.as_df(spark)

    # Filter
    CONDS = (
      # Motion correction almost always succeeds
      anno_df.motion_corrected == True,

      # Ignore very small things
      anno_df.height >= 50, # pixels
      anno_df.width >= 50,  # pixels

      # Skip sample set, and the test set has no labels
      anno_df.split != 'sample',
      anno_df.split != 'test',
    )
    for cond in CONDS:
      anno_df = anno_df.filter(cond)
    
    def anno_row_to_crop_row(anno):
      bbox = BBox(**anno.asDict())
      cropbox = cls.get_crop_viewport(bbox)
      has_offscreen = (cropbox.get_num_onscreen_corners() != 4)
      # Skip anything that is too near (or off) the edge of the image
      if has_offscreen:
        return None
      
      return Row(**cropbox.to_dict())

    crop_spec_rdd = anno_df.rdd.map(anno_row_to_crop_row)
    crop_spec_rdd = crop_spec_rdd.filter(lambda x: x is not None)
    crop_spec_df = spark.createDataFrame(crop_spec_rdd)


    print(crop_spec_df.count())
    crop_spec_df.show()
    import pdb; pdb.set_trace()

    return crop_spec_df
  
  @classmethod
  def _save_positives(cls, spark):
    def anno_row_to_imagerow(crop_spec_row):
      import cv2
      frame = AVFrame(uri=anno.uri, FIXTURES=cls.FIXTURES)
      cropbox = BBox(**crop_spec_row.asDict())
      cropped = frame.get_crop(cropbox)

      # TODO clean up dataset.ImageRow so we can use it here with jpeg
      def to_jpg_bytes(arr):
        import imageio
        import io
        buf = io.BytesIO()
        imageio.imwrite(buf, arr, 'jpeg', quality=100)
        return bytearray(buf.getvalue())

      def to_npz_bytes(arr):
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, arr)
        return bytearray(buf.getvalue())

      crop_img = cv2.resize(cropped.image, cls.VIEWPORT_WH)
      cloud = cropped.get_cloud_in_image()
      # TODO: expose other labels as cols, add a shard key ~~~~~~~~~~~~~~~~~~~~~~~~
      row = Row(
        uri=cropped.uri,
        split=cropped.split,
        dataset='argoverse',
        img_byte_jpeg=to_jpg_bytes(crop_img),
        label=cropbox.category_name,
        attrs=cropattrs(
          anno=crop_spec_row,
          cloud_npz=to_npz_bytes(cloud),
          viewport_annos=''))# TODO ~~~~~~also add other labels~~~~~~~~~~~~~~~~~~~~~~cropped.image_bboxes))
      return row
    
    crop_spec_df = cls._create_crop_spec_df(spark)
    imagerow_rdd = crop_spec_df.rdd.map(anno_row_to_imagerow)
    imagerow_df = spark.createDataFrame(imagerow_rdd)
    imagerow_df.write.parquet(
      cls.table_root(),
      mode='append',        # Write positives and negatives in separate steps
      compression='snappy') # TODO pyarrow / lz4
  
  @classmethod
  def _save_negatives(cls, spark):
    anno_df = cls.ANNOS.as_df(spark)
    frame_uris = anno_df.select('frame_uri').distinct()
    frame_uri_rdd = frame_uris.rdd.flatMap(lambda r: r)
    
    def iter_samples(uri):
      frame = AVFrame(uri=uri, FIXTURES=cls.FIXTURES)

      # Get a miner
      from argoverse.utils import camera_stats
      if frame.uri.camera in camera_stats.RING_CAMERA_LIST:
        miner = RingMiner(frame)
      elif frame.uri.camera in camera_stats.STEREO_CAMERA_LIST:
        miner = StereoMiner(frame)
      else:
        raise ValueError("Unknown camera: %s" % frame.uri.camera)
      
      for n in range(cls.NEGATIVE_SAMPLES_PER_FRAME):
        bbox = miner.next_sample()
        cropbox = cls.get_crop_viewport(bbox)
        has_offscreen = (cropbox.get_num_onscreen_corners() != 4)
        # Skip anything that is too near (or off) the edge of the image
        if has_offscreen:
          continue
        
        cropbox.category_name = 'background'
        yield frame.get_crop(cropbbox) # ~~~~ combine with other row encoding above


              
    
"""
TODO
 #* draw bike associations, wtf with very very far motorcycle
 # file:///Users/pwais/Downloads/imag_annos_new/best_rider_distance_by_category_name.html
 * report on number of invisible things (e.g. top of file comment)
    * report on number of motion-corrected frames
    * report on bikes no riders
 * do kl divergence or some sort of tests in 'split' plots
 * log-scale option for? plots
 * debug draw lider pts
 * try to measure occlusion / clutter by overlaping cuboids? raytrace / z-buffer
"""

