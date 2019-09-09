"""A set of utilities and objects defining the data schema for AV-oriented
datasets, e.g. Argoverse, nuScenes, Waymo Open. etc.
"""

from au import conf
from au import util
from au.fixtures.datasets import common

import numpy as np

import six

def _set_default(attr, default):
  if not util.np_truthy(attr):
    attr = default

class Transform(object):
  """An SE(3) / ROS Transform-like object"""

  slots__ = ('rotation', 'translation')
  
  def __init__(self, **kwargs):
    # Defaults to identity transform
    self.rotation = kwargs.get('rotation', np.eye(3, 3))
    self.translation = kwargs.get('translation', np.zeros((3, 1)))
  
  def __str__(self):
    return 'Transform(rotation=%s;translation=%s)' % (
      self.rotation, self.translation)

class Cuboid(object):
  """TODO describe point order"""
  __slots__ = (
    'track_id',
    'category_name',

    ## Points
    'box3d',                # Points in ego / robot frame defining the cuboid.
                            # Given in order described above.
    'motion_corrected',     # Is `3d_box` corrected for ego motion?

    ## In robot / ego frame
    'length_meters',        # Cuboid frame: +x forward
    'width_meters',         #               +y left
    'height_meters',        #               +z up
    'distance_meters',      # Dist from ego to closest cuboid point
    
    # 'yaw',                  # +yaw to the left (right-handed)
    # 'pitch',                # +pitch up from horizon
    # 'roll',                 # +roll towards y axis (?); usually 0

    'obj_from_ego',         # type: Transform from ego / robot frame to object
    
    'extra',                # type: string -> ?  Extra metadata
  )

class BBox(common.BBox):
  __slots__ = tuple(
    list(common.BBox.__slots__) + [
      # Reference parent cuboid, if available
      'cuboid',
    ]
  )



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
    else:
      return BBox.of_size(*get_image_width_height(self.camera))

  @staticmethod
  def from_str(s):
    if isinstance(s, URI):
      return s
    assert s.startswith(URI.PREFIX)
    toks_s = s[len(URI.PREFIX):]
    toks = toks_s.split('&')
    uri = URI(**dict(tok.split('=') for tok in toks))
    return uri

class CameraImage(object):
  __slots__ = (
    'camera_name',            # type: string
    'image_jpeg',             # type: bytearray
    'timestamp',              # type: int (GPS or unix time)
    
    # Optional Point Cloud (e.g. Lidar)
    'cloud',                  # type: np.array of points
                              #   [pixel_x, pixel_y, depth]
    'cloud_motion_corrected', # type: bool; is `cloud` corrected for ego motion?
    
    'ego_to_camera',          # type: Transform
    'K',                      # type: np.ndarray, Camera matrix
    # 'P',                      # type: np.ndarray, Camera projective matrix
    'principal_axis_in_ego',  # type: np.ndarray, pose of camera *device* in
                              #   ego frame; may be different from
                              #   `ego_to_camera`, which often has axis change
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, ''))
        
    _set_default(self.camera_name, '')
    _set_default(self.cloud, np.array([]))
    _set_default(self.image_jpeg, bytearray(b''))
    _set_default(self.timestamp, 0)
    self.cloud_motion_corrected = bool(self.cloud_motion_corrected)

    _set_default(self.ego_to_camera, Transform())
    _set_default(self.K, np.array([]))
    _set_default(self.principal_axis_in_ego, np.array([]))
  
  @property
  def image(self):
    # TODO: cache
    if self.image_jpeg:
      import imageio
      from io import BytesIO
      return imageio.imread(BytesIO(self.image_jpeg))

  def to_html(self):
    import tabulate
    from au import plotting as aupl
    table = [
      [attr, '<pre>' + str(getattr(self, attr)) + '</pre>']
      for attr in (
        'camera_name',
        'timestamp',
        'ego_to_camera',
        'K',
        'principal_axis_in_ego')
    ]
    html = tabulate.tabulate(table, tablefmt='html')

    if util.np_truthy(self.image):
      table = [
        ['<b>Image</b>'],
        [aupl.img_to_img_tag(self.image, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    if util.np_truthy(self.cloud):
      debug_img = np.copy(self.image)
      aupl.draw_xy_depth_in_image(debug_img, self.cloud, alpha=0.5)
      table = [
        ['<b>Image With Cloud</b>'],
        [aupl.img_to_img_tag(debug_img, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')
    
    return html

class PointCloud(object):
  __slots__ = (
    'sensor_name',          # type: string
    'timestamp',            # type: int (GPS or unix time)
    'cloud',                # type: np.array of points
    'motion_corrected',     # type: bool; is `cloud` corrected for ego motion?
    'ego_to_sensor',        # type: Transform
  )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, ''))
    
    _set_default(self.sensor_name, '')
    _set_default(self.timestamp, 0)
    _set_default(self.cloud, np.array([]))
    _set_default(self.ego_to_sensor, Transform())
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

class Frame(object):

  __slots__ = (
    'uri',                  # type: URI or str
    'camera_images',        # type: List[CameraImage]
    'clouds',               # type: List[PointCloud]
    'cuboids',              # type: List[Cuboid]
    'world_to_ego',         # type: Transform; the pose of the robot in the
                            #   global frame (typicaly the city frame)
  )


  #   '_loader',        # type: AUTrackingLoader

  #   # Labels
  #   '_image_bboxes',  # type: List[BBox]

  #   # Vision
  #   '_image',         # type: np.ndarray
  #   'viewport',       # type: BBox (used to express a crop)
    
  #   # Lidar
  #   '_cloud',         # type: np.ndarray
  # )

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
    
    if isinstance(self.uri, six.string_types):
      self.uri = URI.from_str(self.uri)
    
    self.camera_images = self.camera_images or []

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