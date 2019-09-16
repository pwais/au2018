"""A set of utilities and objects defining the data schema for AV-oriented
datasets, e.g. Argoverse, nuScenes, Waymo Open. etc.
"""

import copy
import math
import os

import numpy as np
import six

from au import conf
from au import util
from au.fixtures.datasets import common
from au.spark import RowAdapter
from au.spark import Spark



###
### Utils
###

def _set_defaults(obj, vals, defaults, DEFAULT_FOR_MISSING=None):
  for k in obj.__slots__:
    v = vals.get(k, defaults.get(k, DEFAULT_FOR_MISSING))
    setattr(obj, k, v)

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy array `pts` to Homogeneous coordinates of target `dim`
  if necessary"""
  if len(pts.shape) != dim + 1:
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts



###
### Core Data Structures
###

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

  def is_identity(self):
    return (
      np.array_equal(self.rotation, np.eye(3, 3)) and
      np.array_equal(self.translation, np.zeros((3, 1))))

class URI(object):
  __slots__ = (
    # All parameters are optional; more parameters address a more
    # specific piece of all Frame data available.
    
    # Frame-level selection
    'dataset',      # E.g. 'argoverse'
    'split',        # E.g. 'train'
    'segment_id',   # String identifier for a drive segment, e.g. a UUID
    'timestamp',    # Some integer in nanoseconds; either Unix or GPS time

    # Sensor-level selection
    'camera',       # Address an image from a specific camera
    'camera_timestamp',

                    # Address a specific viewport / crop of the image
    'crop_x', 'crop_y',
    'crop_w', 'crop_h',
                    

    # Object-level selection
    'track_id',     # A string identifier of a specific track
  )

  PREFIX = 'avframe://'

  # DEFAULTS = {
  #   'timestamp': 0,
  #   'camera_timestamp': 0,
  #   'crop_x': -1, 'crop_y': -1,
  #   'crop_w': -1, 'crop_h': -1,
  # }

  def __init__(self, **kwargs):
    _set_defaults(self, kwargs, {})#self.DEFAULTS)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if isinstance(self.timestamp, six.string_types):
      self.timestamp = int(self.timestamp)
    if isinstance(self.camera_timestamp, six.string_types):
      self.camera_timestamp = int(self.camera_timestamp)
  
  def to_str(self):
    kvs = ((attr, getattr(self, attr)) for attr in self.__slots__)
    path = '&'.join((k + '=' + str(v)) for (k, v) in kvs if v)
    return self.PREFIX + path
  
  def __str__(self):
    return self.to_str()

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
      getattr(self, 'crop_%s' % a)
      for a in ('x', 'y', 'w', 'h'))

  def get_crop_bbox(self):
    return BBox(
            x=self.crop_x, y=self.crop_y,
            width=self.crop_w, height=self.crop_h)

  def get_viewport(self):
    if self.has_crop():
      return self.get_crop_bbox()

  @staticmethod
  def from_str(s, **overrides):
    if isinstance(s, URI):
      return s
    assert s.startswith(URI.PREFIX), "Missing %s in %s" % (URI.PREFIX, s)
    toks_s = s[len(URI.PREFIX):]
    toks = toks_s.split('&')
    assert all('=' in tok for tok in toks), "Bad token in %s" % (toks,)
    kwargs = dict(tok.split('=') for tok in toks)
    kwargs.update(**overrides)
    return URI(**kwargs)

class Cuboid(object):
  """An 8-vertex cuboid"""
  __slots__ = (
    ## Core
    'track_id',             # String identifier; same object across many frames
                            #   has same track_id
    'category_name',        # String category name
    'timestamp',            # Lidar timestamp associated with this cuboid

    ## Points
    'box3d',                # Points in ego / robot frame defining the cuboid.
                            # Given in order:
                            #   (+x +y +z)  [Front face CW about +x axis]
                            #   (+x -y +z)
                            #   (+x -y -z)
                            #   (+x +y -z)
                            #   (-x +y +z)  [Rear face CW about +x axis]
                            #   (-x -y +z)
                            #   (-x -y -z)
                            #   (-x +y -z)
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
    
    'extra',                # type: string -> string extra metadata
  )

  def __init__(self, **kwargs):
    _set_defaults(self, kwargs, {})
      # Default all to None

class BBox(common.BBox):
  __slots__ = tuple(
    list(common.BBox.__slots__) + [
      'cuboid',             # Reference parent cuboid, if available
      
      'cuboid_pts',         # Points of parent cuboid projected into image;
                            #   array of n-by-(x, y, d) points
      'has_offscreen',      # Does the cuboid have off-screen points?
      'is_visible',         # Is at least one point of the cuboid visible?
      
      'cuboid_from_cam',    # Vector from camera center to cuboid pose
      
      'ypr_camera_local',   # Pose (in yaw, pitch roll) of object relative to a
                            #   ray cast from camera center to object centroid
    ]
  )
  def __init__(self, **kwargs):
    super(BBox, self).__init__(**kwargs)
    _set_defaults(self, kwargs, {})
      # Default all to None

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
    return tabulate.tabulate(table, tablefmt='html')

class PointCloud(object):
  __slots__ = (
    'sensor_name',          # type: string
    'timestamp',            # type: int (GPS or unix time)
    'cloud',                # type: np.array of points
    'motion_corrected',     # type: bool; is `cloud` corrected for ego motion?
    'ego_to_sensor',        # type: Transform
  )

  def __init__(self, **kwargs):
    _set_defaults(self, kwargs, {})
      # Default all to None
  
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
    'principal_axis_in_ego',  # type: np.ndarray, A 3d Vector expressing the
                              #   pose of camera *device* in ego frame; may be
                              #   different from `cam_from_ego`, which often
                              #   has an embedded axis change.
  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'bboxes': [],
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

  def get_uri(self, uri):
    uri = copy.deepcopy(uri)
    uri.camera = self.camera_name
    uri.camera_timestamp = self.timestamp
    return uri

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
    self.viewport = \
      self.viewport or common.BBox.of_size(self.width, self.height)

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
    'world_to_ego',         # type: Transform; the pose of the robot in the
                            #   global frame (typicaly the city frame)
  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'uri': URI(),
      'camera_images': [],
      'clouds': [],
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



###
### Prototypes
###

# Spark (and `RowAdapter`) can automatically deduce schemas from object
# heirarchies, but these tools need non-null, non-empty members to deduce
# proper types.  Creating a DataFrame with an explicit schema can also
# improve efficiently dramatically, because then Spark can skip row sampling
# and parallelized auto-deduction.  The Prototypes below serve to provide
# enough type information for `RowAdapter` to deduce the full av.Frame schema.
# In the future, Spark may perhaps add support for reading Python 3 type
# annotations, in which case the Protoypes will be obviated.

URI_PROTO = URI(
  # Core spec; most URIs will have these set
  dataset='proto',
  split='train',
  segment_id='proto_segment',
  timestamp=int(100 * 1e9), # In nanoseconds
  
  # Uris can identify more specific things in a Frame
  camera='camera_1',
  camera_timestamp=int(100 * 1e9), # In nanoseconds
  
  crop_x=0, crop_y=0,
  crop_w=10, crop_h=10,
  
  track_id='track-001',
)

CUBOID_PROTO = Cuboid(
  track_id='track-01',
  category_name='vehicle',
  timestamp=int(100 * 1e9), # In nanoseconds

  box3d=np.array([
    [1.,  1.,  1.],
    [1., -1.,  1.],
    [1., -1., -1.],
    [1.,  1., -1.],

    [-1.,  1.,  1.],
    [-1., -1.,  1.],
    [-1., -1., -1.],
    [-1.,  1., -1.],
  ]),
  motion_corrected=True,
  length_meters=2.,
  width_meters=2.,
  height_meters=2.,
  distance_meters=10.,

  obj_from_ego=Transform(),
  extra={
    'key': 'value',
  },
)

BBOX_PROTO = BBox(
  x=0, y=0,
  width=10, height=10,
  im_width=100, im_height=100,
  category_name='vehicle',

  cuboid=CUBOID_PROTO,
  cuboid_pts=np.ones((8, 3)),

  has_offscreen=False,
  is_visible=True,

  cuboid_from_cam=np.array([1., 0., 1.]),
  ypr_camera_local=np.ones((1, 3)),
)

POINTCLOUD_PROTO = PointCloud(
  sensor_name='lidar',
  timestamp=int(100 * 1e9), # In nanoseconds
  cloud=np.ones((10, 10, 10)),
  motion_corrected=True,
  ego_to_sensor=Transform(),
)

CAMERAIMAGE_PROTO = CameraImage(
  camera_name='front_center',
  image_jpeg=bytearray(b''),
  height=0,
  width=0,
  timestamp=int(100 * 1e9), # In nanoseconds
  
  cloud=POINTCLOUD_PROTO,
  
  bboxes=[BBOX_PROTO],

  cam_from_ego=Transform(),
  K=np.zeros((3, 3)),
  principal_axis_in_ego=np.array([0., 0., 0.]),
)

FRAME_PROTO = Frame(
  uri=URI_PROTO,
  camera_images=[CAMERAIMAGE_PROTO],
  clouds=[POINTCLOUD_PROTO],
  world_to_ego=Transform(),
)

###
### Tables
###

class FrameTableBase(object):

  ## Public API

  PARTITION_KEYS = ('dataset', 'split', 'shard')

  @classmethod
  def get_shard(cls, uri):
    if isinstance(uri, six.string_types):
      uri = URI.from_str(uri)
    return uri.segment_id + '|' + str(int(uri.timestamp * 1e9))

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, 'av_frames')

  @classmethod
  def setup(cls, spark=None):
    if util.missing_or_empty(cls.table_root()):
      with Spark.sess(spark) as spark:
        frame_rdds = cls._create_frame_rdds(spark)
        class FrameDFThunk(object):
          def __init__(self, frame_rdd):
            self.frame_rdd = frame_rdd
          def __call__(self):
            return cls._frame_rdd_to_frame_df(spark, self.frame_rdd)
        df_thunks = [FrameDFThunk(frame_rdd) for frame_rdd in frame_rdds]
        Spark.save_df_thunks(
          df_thunks,
          path=cls.table_root(),
          format='parquet',
          partitionBy=cls.PARTITION_KEYS,
          compression='lz4')

  @classmethod
  def as_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    return df
  
  @classmethod
  def as_frame_rdd(cls, spark):
    df = cls.as_df(spark)
    return df.rdd.map(RowAdapter.from_row)

  @classmethod
  def _create_frame_rdds(cls, spark):
    """Subclasses should create and return a list of RDD[Frame]s"""
    return []

  @classmethod
  def _frame_rdd_to_frame_df(cls, spark, frame_rdd):
    from pyspark import StorageLevel
    from pyspark.sql import Row
    # frame_rdd = cls._create_frame_rdd(spark)
    # def add_id(f):
    #   f.uri = str(f.uri)
    #   return f
    # frame_rdd = frame_rdd.map(stringify_uri)
    
    # frame_row_rdd = frame_rdd.map(RowAdapter.to_row)
    def to_pkey_row(f):
      from collections import OrderedDict
      row = RowAdapter.to_row(f)
      row = row.asDict()
      
      row['id'] = str(f.uri)
      row['dataset'] = f.uri.dataset
      row['split'] = f.uri.split
      row['shard'] = cls.get_shard(f.uri)
      # partition = OrderedDict(
      #   (k, getattr(f.uri, k))
      #   for k in URI.PARTITION_KEYS)
      # partition_key = tuple(partition.values())
      # row.update(**partition)
      # return partition_key, Row(**row)
      return Row(**row)

    # print('frame_rdd size', frame_rdd.count())
    pkey_row_rdd = frame_rdd.map(to_pkey_row)
    # pkey_row_rdd = pkey_row_rdd.partitionBy(1000)
    pkey_row_rdd = pkey_row_rdd.persist(StorageLevel.DISK_ONLY)
    row_rdd = pkey_row_rdd#.map(lambda pkey_row: pkey_row[-1])
    
    schema = RowAdapter.to_schema(to_pkey_row(FRAME_PROTO))

    df = spark.createDataFrame(row_rdd, schema=schema)
    return df


###
### Tensorflow Interop
###

def camera_image_to_tf_example(
    frame_uri,
    camera_image,
    label_map_dict):
  """TODO TODO

  Based upon tensorflow/models
   * research/object_detection/dataset_tools/create_coco_tf_record.py
   * research/object_detection/dataset_tools/create_pet_tf_record.py
  """

  import hashlib
  key = hashlib.sha256(camera_image.image_jpeg).hexdigest()

  camera_uri = camera_image.get_uri(frame_uri)
  width = camera_image.width
  height = camera_image.height

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  for bbox in camera_image.bboxes:
    xmin, ymin, xmax, ymax = bbox.get_fractional_xmin_ymin_xmax_ymax(clip=True)
    xmins.append(xmin)
    ymins.append(ymin)
    xmaxs.append(xmax)
    ymaxs.append(ymax)

    c = bbox.category_name
    classes.append(int(label_map_dict[c]))
    classes_text.append(c.encode('utf-8'))
  n_annos = len(xmins)

  # From tensorflow/models
  from object_detection.utils import dataset_util
  feature_dict = {
    # Image
    'image/height':
        dataset_util.int64_feature(camera_image.height),
    'image/width':
        dataset_util.int64_feature(camera_image.width),
    'image/encoded':
        dataset_util.bytes_feature(bytes(camera_image.image_jpeg)),
    'image/format':
        dataset_util.bytes_feature('jpeg'.encode('utf8')),

    # Annos
    'image/object/bbox/xmin':
        dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax':
        dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin':
        dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax':
        dataset_util.float_list_feature(ymaxs),
    'image/object/class/label': 
        dataset_util.int64_list_feature(classes),
    'image/object/class/text':
        dataset_util.bytes_list_feature(classes_text),

    # Context
    'image/filename':
        dataset_util.bytes_feature(str(camera_uri).encode('utf8')),
    'image/source_id':
        dataset_util.bytes_feature(str(frame_uri).encode('utf8')),
    'image/key/sha256':
        dataset_util.bytes_feature(key.encode('utf8')),
    
    # Required(?) Junk
    'image/object/is_crowd':
        dataset_util.int64_list_feature([False] * n_annos),
    'image/object/area':
        dataset_util.float_list_feature([0.] * n_annos),
  }
  
  import tensorflow as tf
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def frame_df_to_tf_example_ds(frame_df, label_map_dict):
  from au.spark import spark_df_to_tf_dataset

  SHARD_COL = 'shard'

  class RowToTFExamples(object):
    def __init__(self, label_map_dict):
      self.label_map_dict = label_map_dict
    def __call__(self, row):
      frame = RowAdapter.from_row(row)
      ret = [
        camera_image_to_tf_example(
                          frame.uri,
                          ci,
                          self.label_map_dict).SerializeToString()
        for ci in frame.camera_images
      ]
      return (ret[0],) # NB: we must tuple-ize for Tensorflow
      # return (camera_image_to_tf_example(
      #                     frame.uri,
      #                     frame.camera_images[0],
      #                     self.label_map_dict).SerializeToString()[:100],)
  
  import tensorflow as tf
  ds = spark_df_to_tf_dataset(
          frame_df,
          SHARD_COL,
          RowToTFExamples(label_map_dict),
          (tf.string,),
          tf_output_shapes=(tf.TensorShape([]),))
  # ds = ds.apply(tf.data.experimental.unbatch())
  return ds


  
