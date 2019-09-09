"""A set of utilities and objects defining the data schema for AV-oriented
datasets, e.g. Argoverse, nuScenes, Waymo Open. etc.
"""

import copy
import math

import numpy as np
import six

from au import conf
from au import util
from au.fixtures.datasets import common

def _set_defaults(obj, vals, defaults, DEFAULT_FOR_MISSING=''):
  for k in obj.__slots__:
    v = vals.get(k, defaults.get(k, DEFAULT_FOR_MISSING))
    setattr(obj, k, v)

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy array `pts` to Homogeneous coordinates of target `dim`
  if necessary"""
  if len(pts.shape) != dim + 1:
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts

class Transform(object):
  """An SE(3) / ROS Transform-like object"""

  slots__ = ('rotation', 'translation')
  
  def __init__(self, **kwargs):
    # Defaults to identity transform
    self.rotation = kwargs.get('rotation', np.eye(3, 3))
    self.translation = kwargs.get('translation', np.zeros((3, 1)))
  
  def apply(self, pts):
    """Apply this transform (i.e. right-multiply) to `pts` and return
    tranformed *homogeneous* points."""
    transform = np.eye(4)
    transform[:3, :3] = self.rotation
    transform[:3, 3] = self.translation
    pts = maybe_make_homogeneous(pts)
    return transform.dot(pts.T)

  def __str__(self):
    return 'Transform(rotation=%s;translation=%s)' % (
      self.rotation, self.translation)

class URI(object):
  __slots__ = (
    # All parameters are optional; more parameters address a more
    # specific piece of all Frame data available.
    
    # Frame-level selection
    'split',        # E.g. 'train'
    'dataset',      # E.g. 'argoverse'
    'segment_id',     # String identifier for a drive segment, e.g. a UUID
    'timestamp',    # Some integer; either Unix or GPS time

    # Sensor-level selection
    'camera',       # Address an image from a specific camera

                    # Address a specific viewport / crop of the image
    'crop_x', 'crop_y',
    'crop_w', 'crop_h',
                    

    # Object-level selection
    'track_id',     # A string identifier of a specific track    
  )

  # Partition all frames by Drive
  PARTITION_KEYS = ('dataset', 'split', 'segment_id')

  PREFIX = 'avframe://'

  def __init__(self, **kwargs):
    for k in self.__slots__:
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

  # def to_dict(self):
  #   return dict((k, getattr(self, k, '')) for k in self.__slots__)~~~~~~~~~~~~~~

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

  def get_viewport(self):
    if self.has_crop():
      return self.get_crop_bbox()

  @staticmethod
  def from_str(s):
    if isinstance(s, URI):
      return s
    assert s.startswith(URI.PREFIX)
    toks_s = s[len(URI.PREFIX):]
    toks = toks_s.split('&')
    uri = URI(**dict(tok.split('=') for tok in toks))
    return uri

class BBox(common.BBox):
  __slots__ = tuple(
    list(common.BBox.__slots__) + [
      'cuboid',             # Reference parent cuboid, if available
      
      'cuboid_pts',         # Points of parent cuboid projected into image;
                            #   array of n-by-(x, y, d) points
      'has_offscreen',      # Does the cuboid have off-screen points?
      'is_visible',         # Is at least one point of the cuboid visible?
      
      'cuboid_from_cam',    # Transform from camera center to cuboid pose
      
      'ypr_camera_local',   # Pose (in yaw, pitch roll) of object relative to a
                            #   ray cast from camera center to object centroid
    ]
  )
  def __init__(self, **kwargs):
    super(BBox, self).__init__(**kwargs)

    DEFAULTS = {
      'cuboid': None,
      'cuboid_pts': None,
      'cuboid_from_cam': Transform(),
      'ypr_camera_local': None
    }
    _set_defaults(self, kwargs, DEFAULTS)

    self.has_offscreen = bool(self.has_offscreen)
    self.is_visible = bool(self.is_visible)

  def draw_in_image(
        self,
        img,
        color=None,
        thickness=2,
        cuboid_alpha=0.3,
        cuboid_thickness=2):
    """Draw this BBox in `img`, and optionally include visualization of the
    box's cuboid if available"""
    
    super(BBox, self).draw_in_image(img, color=color, thickness=thickness)

    if hasattr(self.cuboid_pts, 'shape'):
      # Use category color
      from au import plotting as aupl
      base_color = aupl.hash_to_rbg(self.category_name)
      
      pts = self.cuboid_pts[:, :2]

      aupl.draw_cuboid_xy_in_image(
        img, pts, base_color, alpha=cuboid_alpha, thickness=cuboid_thickness)

  def to_html(self):
    import tabulate
    table = [
      [attr, getattr(self, attr)]
      for attr in self.__slots__
    ]
    print('check ypr_camera_local')
    return tabulate.tabulate(table, tablefmt='html')

class Cuboid(object):
  """TODO describe point order"""
  __slots__ = (
    ## Core
    'track_id',             # String identifier; same object across many frames
                            #   has same track_id
    'category_name',        # String category name
    'timestamp',            # Lidar timestamp associated with this cuboid

    ## Points
    'box3d',                # Points in ego / robot frame defining the cuboid.
                            # Given in order described above.
    'motion_corrected',     # Is `3d_box` corrected for ego motion?

    ## In robot / ego frame
    'length_meters',        # Cuboid frame: +x forward
    'width_meters',         #               +y left
    'height_meters',        #               +z up
    'distance_meters',      # Dist from ego to closest cuboid point
    
    # TODO
    # 'yaw',                  # +yaw to the left (right-handed)
    # 'pitch',                # +pitch up from horizon
    # 'roll',                 # +roll towards y axis (?); usually 0

    'obj_from_ego',         # type: Transform from ego / robot frame to object
    
    'extra',                # type: string -> ?  Extra metadata
  )

class PointCloud(object):
  __slots__ = (
    'sensor_name',          # type: string
    'timestamp',            # type: int (GPS or unix time)
    'cloud',                # type: np.array of points
    'motion_corrected',     # type: bool; is `cloud` corrected for ego motion?
    'ego_to_sensor',        # type: Transform
  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'timestamp': 0,
      'cloud': np.array([]),
      'ego_to_sensor': Transform(),
    }
    _set_defaults(self, kwargs, DEFAULTS)

    self.motion_corrected = bool(self.motion_corrected)
  
  def to_html(self):
    import tabulate
    table = [
      [attr, getattr(self, attr)]
      for attr in (
        'sensor_name',
        'timestamp',
        'motion_corrected',
        'ego_to_sensor')
    ]

    # TODO: BEV / RV cloud
    table.extend([
      ['Cloud', ''],
      [len(self.cloud), '']
    ])
    return tabulate.tabulate(table, tablefmt='html')

class CameraImage(object):
  __slots__ = (
    'camera_name',            # type: string
    'image_jpeg',             # type: bytearray
    'height',                 # type: int
    'width',                  # type: int
    'timestamp',              # type: int (GPS or unix time)

    # Optional Point Cloud (e.g. Lidar projected to camera)
    'cloud',                  # type: PointCloud
    
    # Optional BBoxes (e.g. Cuboids projected to camera)
    'bboxes',                 # type: List[BBox]

    # Context
    'cam_from_ego',           # type: Transform
    'K',                      # type: np.ndarray, Camera matrix
    # 'P',                      # type: np.ndarray, Camera projective matrix
    'principal_axis_in_ego',  # type: np.ndarray, pose of camera *device* in
                              #   ego frame; may be different from
                              #   `cam_from_ego`, which often has axis change

  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'image_jpeg': bytearray(b''),
      'timestamp': 0,
      'bboxes': [],
      'cam_from_ego': Transform(),
      'K': np.array([]),
      'principal_axis_in_ego': np.array([])
    }
    _set_defaults(self, kwargs, DEFAULTS)
  
  @property
  def image(self):
    if self.image_jpeg:
      import imageio
      from io import BytesIO
      return imageio.imread(BytesIO(self.image_jpeg))
    return img
  
  def project_ego_to_image(self, pts, omit_offscreen=True):
    """ TODO return 3 x n, where 3 = (x, y, depth)"""
    pts_from_cam = self.cam_from_ego.apply(pts)
    uvd = self.K.dot(pts_from_cam)
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T

    if omit_offscreen:
      x, y, w, h = 0, 0, self.width, self.height
        
      uv = uvd.T
      idx_ = np.where(
              np.logical_and.reduce((
                # Filter offscreen points
                x <= uv[0, :], uv[0, :] < x + w - 1.0,
                y <= uv[1, :], uv[1, :] < y + h - 1.0,
                # Filter behind-screen points
                uv[2, :] > 0)))
      idx_ = idx_[0]
      uv = uv[:, idx_]
      uvd = uv.T

    return uvd
  
  def project_cuboid_to_bbox(self, cuboid):
    bbox = BBox(
            im_width=self.width,
            im_height=self.height,
            category_name=cuboid.category_name,
            cuboid=cuboid)
    
    ## Fill Points
    cuboid_points = copy.deepcopy(cuboid.box3d)
    uvd = self.project_ego_to_image(cuboid.box3d, omit_offscreen=False)

    bbox.cuboid_pts = uvd

    x1, x2 = np.min(uvd[:, 0]), np.max(uvd[:, 0])
    y1, y2 = np.min(uvd[:, 1]), np.max(uvd[:, 1])
    bbox.set_x1_y1_x2_y2(x1, y1, x2, y2)

    z = float(np.max(uvd[:, 2]))
    num_onscreen = bbox.get_num_onscreen_corners()
    bbox.has_offscreen = ((z <= 0) or (num_onscreen < 4))
    bbox.is_visible = (z > 0 and num_onscreen > 0)

    bbox.clamp_to_screen()

    ## Fill Pose
    bbox.cuboid_from_cam = \
      cuboid.obj_from_ego.translation - self.cam_from_ego.translation

    cuboid_from_cam_hat = \
      bbox.cuboid_from_cam / np.linalg.norm(bbox.cuboid_from_cam)
    
    from scipy.spatial.transform import Rotation as R
    X_HAT = np.array([1, 0, 0])
    obj_normal = cuboid.obj_from_ego.rotation.dot(X_HAT)
    cos_theta = cuboid_from_cam_hat.dot(obj_normal)
    rot_axis = np.cross(cuboid_from_cam_hat, obj_normal)
    obj_from_ray = R.from_rotvec(
          math.acos(cos_theta) * rot_axis / np.linalg.norm(rot_axis))
    bbox.ypr_camera_local = obj_from_ray.as_euler('zxy')

    return bbox

  def to_html(self):
    import tabulate
    from au import plotting as aupl
    table = [
      [attr, '<pre>' + str(getattr(self, attr)) + '</pre>']
      for attr in (
        'camera_name',
        'timestamp',
        'cam_from_ego',
        'K',
        'principal_axis_in_ego')
    ]
    html = tabulate.tabulate(table, tablefmt='html')

    image = self.image
    if util.np_truthy(image):
      table = [
        ['<b>Image</b>'],
        [aupl.img_to_img_tag(image, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    if self.cloud:
      debug_img = np.copy(image)
      aupl.draw_xy_depth_in_image(debug_img, self.cloud.cloud, alpha=0.7)
      table = [
        ['<b>Image With Cloud</b>'],
        [aupl.img_to_img_tag(debug_img, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')
    
    if self.bboxes:
      debug_img = np.copy(image)
      for bbox in self.bboxes:
        bbox.draw_in_image(debug_img)
      table = [
        ['<b>Image With Boxes</b>'],
        [aupl.img_to_img_tag(debug_img, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

      table = [
        [aupl.img_to_img_tag(
            bbox.get_crop(image),
            image_viewport_hw=(300, 300)),
         bbox.to_html()]
        for bbox in self.bboxes
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    return html

class CroppedCameraImage(CameraImage):
  __slots__ = tuple(list(CameraImage.__slots__) + [
    # Viewport of camera; this image is potentially a crop of a (maybe shared)
    # image buffer
    'viewport',               # type: common.BBox
  ])

  def __init__(self, **kwargs):
    super(CroppedCameraImage, self).__init__(**kwargs)
    self.viewport = self.viewport or None

  @property
  def image(self):
    img = super(CroppedCameraImage, self).image
    if util.np_truthy(img):
      img = self.viewport.get_crop(img)
    return img
  
  def project_ego_to_image(self, pts, omit_offscreen=True):
    uvd = super(CroppedCameraImage, self).project_ego_to_image(
      pts, omit_offscreen=omit_offscreen)
    
    if omit_offscreen:
      x, y, w, h = (
        self.viewport.x, self.viewport.y,
        self.viewport.width, self.viewport.height)
      
      uv = uvd.T
      idx_ = np.where(
              np.logical_and.reduce((
                # Filter offscreen points
                x <= uv[0, :], uv[0, :] < x + w - 1.0,
                y <= uv[1, :], uv[1, :] < y + h - 1.0,
                # Filter behind-screen points
                uv[2, :] > 0)))
      idx_ = idx_[0]
      uv = uv[:, idx_]
      uvd = uv.T

    # Correct for moved image origin
    uvd -= np.array([self.viewport.x, self.viewport.y, 0])
    return uvd

class Frame(object):

  __slots__ = (
    'uri',                  # type: URI or str
    'camera_images',        # type: List[CameraImage]
    'clouds',               # type: List[PointCloud]
    'cuboids',              # type: List[Cuboid]
    'world_to_ego',         # type: Transform; the pose of the robot in the
                            #   global frame (typicaly the city frame)
  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'camera_images': [],
      'clouds': [],
      'cuboids': [],
      'world_to_ego': Transform(),
    }
    _set_defaults(self, kwargs, DEFAULTS)
    
    if isinstance(self.uri, six.string_types):
      self.uri = URI.from_str(self.uri)
    
  def to_html(self):
    import tabulate
    import pprint
    table = [
      ['URI', str(self.uri)],
      ['Num Labels', len(self.cuboids)],
      ['Ego Pose', pprint.pformat(self.world_to_ego)]
    ]
    html = tabulate.tabulate(table, tablefmt='html')
    table = [['<h2>Camera Images</h2>']]
    for c in self.camera_images:
      table += [[c.to_html()]]
    
    table += [['<h2>Point Clouds</h2>']]
    for c in self.clouds:
      table += [[c.to_html()]]

    html += tabulate.tabulate(table, tablefmt='html')
    return html

    # if not self.viewport:
    #   self.viewport = self.uri.get_viewport()
    
  # @property
  # def loader(self):
  #   if not self._loader:
  #     self._loader = self.FIXTURES.get_loader(self.uri)
  #   return self._loader # type: AUTrackingLoader
  
  # @staticmethod
  # @klepto.lru_cache(maxsize=100)
  # def __load_image(path):
  #   return imageio.imread(path)

  # @property
  # def image(self):
  #   if not util.np_truthy(self._image):
  #     path = self.loader.get_nearest_image_path(
  #                     self.uri.camera, self.uri.timestamp)
  #     self._image = AVFrame.__load_image(path)
  #     if not self.viewport.is_full_image():
  #       c, r, w, h = (
  #         self.viewport.x, self.viewport.y,
  #         self.viewport.width, self.viewport.height)
  #       self._image = self._image[r:r+h, c:c+w, :]
  #   return self._image
  
  # @property
  # def cloud(self):
  #   if not util.np_truthy(self._cloud):
  #     self._cloud, motion_corrected = \
  #       self.loader.get_maybe_motion_corrected_cloud(self.uri.timestamp)
  #       # We can ignore motion_corrected failures since the Frame will already
  #       # have this info embedded in `image_bboxes`.
  #   return self._cloud
  
  # def get_cloud_in_image(self):
  #   cloud = self.cloud
  #   calib = self.loader.get_calibration(self.uri.camera)

  #   # Per the argoverse recommendation, this should be safe:
  #   # https://github.com/argoai/argoverse-api/blob/master/demo_usage/argoverse_tracking_tutorial.ipynb
  #   x, y, w, h = (
  #     self.viewport.x, self.viewport.y,
  #     self.viewport.width, self.viewport.height)
  #   uv = calib.project_ego_to_image(cloud).T
  #   idx_ = np.where(
  #           np.logical_and.reduce((
  #             # Filter offscreen points
  #             x <= uv[0, :], uv[0, :] < x + w - 1.0,
  #             y <= uv[1, :], uv[1, :] < y + h - 1.0,
  #             # Filter behind-screen points
  #             uv[2, :] > 0)))
  #   idx_ = idx_[0]
  #   uv = uv[:, idx_]
  #   uv = uv.T

  #   # Correct for image origin if this frame is a crop
  #   uv -= np.array([self.viewport.x, self.viewport.y, 0])
  #   return uv

  # @property
  # def image_bboxes(self):
  #   if not self._image_bboxes:
  #     bboxes = self.loader.get_nearest_label_bboxes(self.uri)

  #     # Ingore invisible things
  #     self._image_bboxes = [
  #       bbox for bbox in bboxes
  #       if bbox.is_visible and self.viewport.overlaps_with(bbox)
  #     ]

  #     # Correct for image origin if this frame is a crop
  #     for bbox in self._image_bboxes:
  #       bbox.translate(-np.array(self.viewport.get_x1_y1()))
  #       bbox.im_width = self.viewport.width
  #       bbox.im_height = self.viewport.height

  #   return self._image_bboxes

  # def get_target_bbox(self):
  #   if self.uri.track_id:
  #     for bbox in self.image_bboxes:
  #         if bbox.track_id == self.uri.track_id:
  #           return bbox
  #   return None

  # def get_debug_image(self):
  #   img = np.copy(self.image)
    
  #   from au import plotting as aupl
  #   xyd = self.get_cloud_in_image()
  #   aupl.draw_xy_depth_in_image(img, xyd)

  #   target_bbox = self.get_target_bbox()
  #   if target_bbox:
  #     # Draw a highlight box first; then the draw() calls below will draw over
  #     # the box.
  #     # WHITE = (225, 225, 255)
  #     # target_bbox.draw_in_image(img, color=WHITE, thickness=20)

  #   # for bbox in self.image_bboxes:
  #     bbox = target_bbox
  #     bbox.draw_cuboid_in_image(img)
  #     # bbox.draw_in_image(img)
    
  #   return img

  # def get_cropped(self, bbox):
  #   """Create and return a new AVFrame instance that contains the data in this
  #   frame cropped down to the viewport of just `bbox`."""

  #   uri = copy.deepcopy(self.uri)
  #   uri.set_crop(bbox)
  #   if hasattr(bbox, 'track_id') and bbox.track_id:
  #     uri.track_id = bbox.track_id

  #   frame = self.FIXTURES.get_frame(uri)
  #   return frame

class FrameTableBase(object):

  ## Public API

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, 'av_frames')

  @classmethod
  def setup(cls, spark=None):
    if util.missing_or_empty(cls.table_root()):
      with Spark.sess(spark) as spark:
        df = cls.create_frame_df(spark)
        df.write.parquet(
          cls.table_root(),
          partitionBy=URI.PARTITION_KEYS,
          mode='append',
          compression='lz4')

  @classmethod
  def as_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    return df
  
  @classmethod
  def create_frame_rdd(cls, spark):
    """Subclasses should create and return a pyspark RDD containing `Frame`
    instances."""
    return spark.parallelize([Frame()])