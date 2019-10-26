"""A set of utilities and objects defining the data schema for AV-oriented
datasets, e.g. Argoverse, nuScenes, Waymo Open. etc.
"""

import copy
import itertools
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
### Constants
###

AU_AV_CATEGORIES = (
  'background',
  'car',
  'ped',
  'bike_with_rider',
  'bike_no_rider',
  'motorcycle_with_rider',
  'motorcycle_no_rider',
  'obstacle',
)
AU_AV_CATEGORY_TO_ID = dict((c, i) for i, c in enumerate(AU_AV_CATEGORIES))

ARGOVERSE_CATEGORY_TO_AU_AV_CATEGORY = {
  # Cars
  "BUS":                'car',
  "EMERGENCY_VEHICLE":  'car',
  "LARGE_VEHICLE":      'car',
  "SCHOOL_BUS":         'car',
  "TRAILER":            'car',
  "VEHICLE":            'car',

  # Peds
  "ANIMAL":             'ped',
  "PEDESTRIAN":         'ped',
  "STROLLER":           'ped',
  "WHEELCHAIR":         'ped',

  # These labels get transformed / merged
  "BICYCLE":            'car',
  "BICYCLIST":          'ped',
  "MOTORCYCLE":         'car',
  "MOTORCYCLIST":       'ped',
  "MOPED":              'car',

  # Misc
  "ON_ROAD_OBSTACLE":   'obstacle',
  "OTHER_MOVER":        'obstacle',
}

NUSCENES_CATEGORY_TO_AU_AV_CATEGORY = {

  # Cars
  'vehicle.car': 'car',
  'vehicle.bus.bendy': 'car',
  'vehicle.bus.rigid': 'car',
  'vehicle.truck': 'car',
  'vehicle.construction': 'car',
  'vehicle.emergency.ambulance': 'car',
  'vehicle.emergency.police': 'car',
  'vehicle.trailer': 'car',
  
  # Cars: Lyft Level 5
  'car': 'car',
  'other_vehicle': 'car', 
  'bus': 'car',
  'truck': 'car',
  'emergency_vehicle': 'car',


  # Peds
  'human.pedestrian.adult': 'ped',
  'human.pedestrian.child': 'ped',
  'human.pedestrian.wheelchair': 'ped',
  'human.pedestrian.stroller': 'ped',
  'human.pedestrian.personal_mobility': 'ped',
  'human.pedestrian.police_officer': 'ped',
  'human.pedestrian.construction_worker': 'ped',
  'animal': 'ped',

  # Peds: Lyft Level 5
  'pedestrian': 'ped',
  # Also: 'animal': 'ped',

  # Bikes
  # Assume no rider unless nuscenes attribute says otherwise
  'vehicle.motorcycle': 'motorcycle_no_rider',
  'vehicle.bicycle': 'bike_no_rider',
  
  # Bikes: Lyft Level 5
  'motorcycle': 'motorcycle_with_rider',
  'bicycle': 'bike_with_rider',

  # Misc
  'movable_object.barrier': 'obstacle',
  'movable_object.trafficcone': 'obstacle',
  'movable_object.pushable_pullable': 'obstacle',
  'movable_object.debris': 'obstacle',

  # Ignore
  'static_object.bicycle_rack': 'background',
}

WAYMO_OD_CATEGORY_TO_AU_AV_CATEGORY = {
  'TYPE_UNKNOWN':     'obstacle',
  'TYPE_VEHICLE':     'car',
  'TYPE_PEDESTRIAN':  'ped',
  'TYPE_SIGN':        'background',
  'TYPE_CYCLIST':     'bike_with_rider',
}



###
### Utils
###

def _set_defaults(obj, vals, defaults, DEFAULT_FOR_MISSING=None):
  for k in obj.__slots__:
    v = vals.get(k, defaults.get(k, DEFAULT_FOR_MISSING))
    setattr(obj, k, v)

def _slotted_eq(obj1, obj2):
  def is_eq(v1, v2):
    if isinstance(v1, np.ndarray):
      return np.array_equal(v1, v1)
    else:
      return v1 == v2
  def attrs_eq(attr):
    return is_eq(getattr(obj1, attr), getattr(obj2, attr))
  return (
    type(obj1) is type(obj2) and
    obj1.__slots__ == obj2.__slots__ and
    all(attrs_eq(attr) for attr in obj1.__slots__))

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy array `pts` to Homogeneous coordinates of target `dim`
  if necessary"""
  if len(pts.shape) != dim + 1:
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts

def l2_normalized(v):
  if len(v.shape) > 1:
    # Normalize row-wise
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]
  else:
    return v / np.linalg.norm(v)

def theta_signed(axis, v):
  return np.arctan2(np.cross(axis, v), np.dot(axis, v.T))

###
### Core Data Structures
###

class Transform(object):
  """An SE(3) / ROS Transform-like object"""

  __slots__ = ('rotation', 'translation', 'src_frame', 'dest_frame')
  
  def __init__(self, **kwargs):
    # Defaults to identity transform
    DEFAULTS = {
      'rotation': np.eye(3, 3),
      'translation': np.zeros((3, 1)),
      'src_frame': '',
      'dest_frame': '',
    }
    _set_defaults(self, kwargs, DEFAULTS)

    # Ensure Translation is a vector
    self.translation = np.reshape(self.translation, (3, 1))
  
  def __eq__(self, other):
    return _slotted_eq(self, other)

  def apply(self, pts):
    """Apply this transform (i.e. right-multiply) to `pts` and return
    tranformed *homogeneous* points."""
    transform = self.get_transformation_matrix()
    pts = maybe_make_homogeneous(pts)
    return transform.dot(pts.T)

  def get_transformation_matrix(self, homogeneous=False):
    if homogeneous:
      RT = np.eye(4, 4)
    else:
      RT = np.eye(3, 4)
    RT[:3, :3] = self.rotation
    RT[:3, 3] = self.translation.reshape(3)
    return RT

  def get_inverse(self):
    return Transform(
      rotation=self.rotation.T,
      translation=self.rotation.T.dot(-self.translation),
      src_frame=self.dest_frame,
      dest_frame=self.src_frame)

  def __str__(self):
    return 'Transform(\nrotation=%s;\ntranslation=%s)' % (
      self.rotation, self.translation)

  def is_identity(self):
    return (
      np.array_equal(self.rotation, np.eye(3, 3)) and
      np.array_equal(self.translation, np.zeros((3, 1))))



class URI(object):
  __slots__ = (
    # All parameters are optional; more parameters address a more
    # specific piece of all Frame data available.
    
    # Core selection
    'dataset',      # E.g. 'argoverse'
    'split',        # E.g. 'train'
    'segment_id',   # String identifier for a drive segment, e.g. a UUID
    'timestamp',    # Some integer in nanoseconds; typically Unix time
    'topic',        # Name for a series of messages, e.g. '/ego_pose'

    'extra',        # str -> str map for extra context

    # TODO deleteme ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

  def __init__(self, **kwargs):
    DEFAULTS = {
      'extra': {},
    }
    _set_defaults(self, kwargs, DEFAULTS)

    for k, v in kwargs.items():
      if k.startswith('extra.'):
        self.extra[k[len('extra.'):]] = str(v)

    if isinstance(self.timestamp, six.string_types):
      self.timestamp = int(self.timestamp)
    if isinstance(self.camera_timestamp, six.string_types):
      self.camera_timestamp = int(self.camera_timestamp)
  
  def as_tuple(self):
    def to_tokens(k, v):
      if v is not None:
        if k == 'extra':
          for ek, ev in sorted(v.items()):
            yield ('extra.%s' % ek, ev)
        else:
          yield (k, v)

    toks = itertools.chain.from_iterable(
      to_tokens(attr, getattr(self, attr)) for attr in self.__slots__)
    return tuple(toks)

  def to_str(self):
    tup = self.as_tuple()
    toks = ('%s=%s' % (k, v) for k, v in tup)
    return '%s%s' % (self.PREFIX, '&'.join(toks))
  
  def __str__(self):
    return self.to_str()

  def __repr__(self):
    kvs = ((attr, getattr(self, attr)) for attr in self.__slots__)
    kwargs_str = ', '.join('%s=%s' % (k, repr(v)) for k, v in kvs)
    return 'URI(%s)' % kwargs_str

  def __eq__(self, other):
    if type(other) is type(self):
      return all(
        getattr(self, attr) == getattr(other, attr)
        for attr in self.__slots__)
    return False

  def __lt__(self, other):
    assert type(other) is type(self)
    return self.as_tuple() < other.as_tuple()

  def __hash__(self):
    return hash(self.as_tuple())

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
    if not toks_s:
      return URI()
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
    'au_category',          # AU AV Category (coarser)
    'timestamp',            # Lidar timestamp associated with this cuboid

    ## Points # TODO keep ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    'ego_pose',             # type: Transform (ego from world)
    
    'extra',                # type: string -> string extra metadata
  )

  def __init__(self, **kwargs):
    _set_defaults(self, kwargs, {})
      # Default all to None

  def __eq__(self, other):
    return _slotted_eq(self, other)

  def to_html(self):
    import tabulate
    import pprint
    table = [
      [attr, '<pre>' + pprint.pformat(getattr(self, attr)) + '</pre>']
      for attr in self.__slots__
    ]
    return tabulate.tabulate(table, tablefmt='html')
  
  @staticmethod
  def get_merged(c1, c2):
    ## Find new box3d, maintaining orientation of old box.
    # Step 1: Compute mean centroid and pose

    merged_translation = .5 * (
      c1.obj_from_ego.translation + c2.obj_from_ego.translation)
    
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    rots = R.from_dcm([
      c1.obj_from_ego.rotation,
      c2.obj_from_ego.rotation,
    ])
    slerp = Slerp([0, 1], rots)
    merged_rot = slerp([0.5]).as_dcm()

    merged_transform = Transform(
      rotation=merged_rot,
      translation=merged_translation.reshape((3, 1)),
    )

    # Step 2: Compute cuboid bounds given pose
    cube_in_merged_frame = merged_transform.apply(
      # Send the unit cube into the object frame
      np.array([
        [1,  1.,  1.],
        [1, -1.,  1.],
        [1, -1., -1.],
        [1,  1., -1.],

        [-1,  1.,  1.],
        [-1, -1.,  1.],
        [-1, -1., -1.],
        [-1,  1., -1.],
      ])
    )
    cube_in_merged_frame = cube_in_merged_frame.T

    # Stretch the cuboid to fit all points
    all_pts = np.concatenate((c1.box3d, c2.box3d))
    merged_box3d = []
    for i in range(8):
      corner = cube_in_merged_frame[i, :3]
      corner /= np.linalg.norm(corner)
      merged_box3d.append(
        # Scale corner by the existing point with the greatest projection
        corner * all_pts.dot(corner).max()
      )
    merged_box3d = np.array(merged_box3d)

    width = np.linalg.norm(merged_box3d[1] - merged_box3d[0])
    length = np.linalg.norm(merged_box3d[4] - merged_box3d[0])
    height = np.linalg.norm(merged_box3d[3] - merged_box3d[0])

    return Cuboid(
      track_id=c1.track_id + '+' + c2.track_id,
      category_name=c1.category_name,
      au_category=c1.au_category,
      timestamp=c1.timestamp,
      box3d=merged_box3d,
      motion_corrected=c1.motion_corrected or c2.motion_corrected,
      length_meters=float(length),
      width_meters=float(width),
      height_meters=float(height),
      distance_meters=float(np.min(np.linalg.norm(merged_box3d, axis=-1))),
      obj_from_ego=merged_transform,
      extra=dict(list(c1.extra.items()) + list(c2.extra.items())),
    )



class BBox(common.BBox):
  __slots__ = tuple(
    list(common.BBox.__slots__) + [
      'au_category',        # Coarser than category_name
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

  def __eq__(self, other):
    return _slotted_eq(self, other)

  def draw_in_image(
        self,
        img,
        color=None,
        thickness=2,
        cuboid_alpha=0.3,
        cuboid_thickness=2):
    """Draw this BBox in `img`, and optionally include visualization of the
    box's cuboid if available"""
    
    super(BBox, self).draw_in_image(
      img, color=color, thickness=thickness, category=self.au_category)

    if hasattr(self.cuboid_pts, 'shape'):
      # Use category color
      from au import plotting as aupl
      base_color = aupl.hash_to_rbg(self.au_category)
      
      pts = self.cuboid_pts[:, :2]

      aupl.draw_cuboid_xy_in_image(
        img, pts, base_color, alpha=cuboid_alpha, thickness=cuboid_thickness)

  def to_html(self):
    import tabulate
    
    def to_display(v):
      if hasattr(v, 'to_html'):
        return v.to_html()
      else:
        return '<pre>' + str(v) + '</pre>'
    
    table = [
      [attr, to_display(getattr(self, attr))]
      for attr in self.__slots__
    ]
    return tabulate.tabulate(table, tablefmt='html')



class PointCloud(object):
  __slots__ = (
    'sensor_name',          # type: string
    'timestamp',            # type: int (GPS or unix time)
    'cloud',                # type: np.array of points,
                            #    **typically in *ego* frame**
    'motion_corrected',     # type: bool; is `cloud` corrected for ego motion?
    'ego_to_sensor',        # type: Transform
    'ego_pose',             # type: Transform (ego from world)
  )

  def __init__(self, **kwargs):
    _set_defaults(self, kwargs, {})
      # Default all to None
  
  def __eq__(self, other):
    return _slotted_eq(self, other)

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
    'ego_pose',               # type: Transform (ego from world)

    # Optional Point Cloud (e.g. Lidar projected to camera)
    'clouds',                 # type: List[PointCloud]
    
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
      'clouds': [],
    }
    _set_defaults(self, kwargs, DEFAULTS)
  
  def __eq__(self, other):
    return _slotted_eq(self, other)

  @property
  def image(self):
    if self.image_jpeg:
      import imageio
      from io import BytesIO
      return imageio.imread(BytesIO(self.image_jpeg))
    return img
  
  def get_fov(self):
    """Return the horizontal and verticle Fields of View in radians:
    (FoV_h, FoV_v)"""
    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    fov_h = 2. * math.atan(.5 * self.width / f_x)
    fov_v = 2. * math.atan(.5 * self.height / f_y)
    return fov_h, fov_v


  # def project_ego_to_image(self, pts, omit_offscreen=True):
  #   """Given a cloud of `pts` of shape n x (x, y, z) in the ego frame, project
  #   the points to the camera image plane and return points of the form 
  #   n x (x, y, depth)"""
  #   pts_from_cam = self.cam_from_ego.apply(pts)
  #   uvd = self.K.dot(pts_from_cam)
  #   uvd[0:2, :] /= uvd[2, :]
  #   uvd = uvd.T

  #   if omit_offscreen:
  #     x, y, w, h = 0, 0, self.width, self.height
        
  #     uv = uvd.T
  #     idx_ = np.where(
  #             np.logical_and.reduce((
  #               # Filter offscreen points
  #               x <= uv[0, :], uv[0, :] < x + w - 1.0,
  #               y <= uv[1, :], uv[1, :] < y + h - 1.0,
  #               # Filter behind-screen points
  #               uv[2, :] > 0)))
  #     idx_ = idx_[0]
  #     uv = uv[:, idx_]
  #     uvd = uv.T

  #   return uvd

  def project_ego_to_image(self, pts, omit_offscreen=True):
    """Given a cloud of `pts` of shape n x (x, y, z) in the ego frame, project
    the points to the camera image plane and return points of the form 
    n x (x, y, depth)"""
    pts_in_cam = self.cam_from_ego.apply(pts).T

    if omit_offscreen:
      fov_h, fov_v = self.get_fov()
      half_fov_h, half_fov_v = .5 * fov_h, .5 * fov_v

      Z_HAT = np.array([0, 1]) # Principal axis in X-Z and Y-Z planes
      pts_xz = pts_in_cam[:, (0, 2)]
      theta_h = theta_signed(l2_normalized(pts_xz), Z_HAT)
      pts_yz = pts_in_cam[:, (1, 2)]
      theta_v = theta_signed(l2_normalized(pts_yz), Z_HAT)

      PADDING_RADIANS = math.pi / 8
      idx_ = np.where(
              np.logical_and.reduce((
                # Filter off-the-edge points
                np.abs(theta_h) <= half_fov_h + PADDING_RADIANS,
                np.abs(theta_v) <= half_fov_v + PADDING_RADIANS)))
                # # Filter behind-screen points
                # uv[2, :] > 0)))
      idx_ = idx_[0]
      pts_in_cam = pts_in_cam[idx_, :]

    uvd = self.K.dot(pts_in_cam.T)
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T

    return uvd
  
  def _has_edge_in_fov(self, cuboid):
    
    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    fov_h = 2. * math.atan(.5 * self.width / f_x)
    fov_v = 2. * math.atan(.5 * self.height / f_y)

    

    def intervals_overlap(i1, i2):
      (s1, e1), (s2, e2) = (i1, i2)
      return max(s1, s2) <= min(e1, e2)

    # Check in x-y (horizontal) plane
    cuboid_pts_h_hat = l2_normalized(cuboid.box3d[:, :2])
    camera_pov_h_hat = l2_normalized(self.principal_axis_in_ego[:2])
    theta_h = theta_signed(camera_pov_h_hat, cuboid_pts_h_hat)
    is_in_fov_h = intervals_overlap(
                    (-.5 * fov_h, .5 * fov_h),
                    (theta_h.min(), theta_h.max()))

    # Check in x-z (vertical) plane
    XZ = np.array([0, 2])
    cuboid_pts_v_hat = l2_normalized(cuboid.box3d[:, XZ])
    camera_pov_v_hat = l2_normalized(self.principal_axis_in_ego[XZ])
    theta_v = theta_signed(camera_pov_v_hat, cuboid_pts_v_hat)
    is_in_fov_v = intervals_overlap(
                    (-.5 * fov_v, .5 * fov_v),
                    (theta_v.min(), theta_v.max()))

    # if cuboid.track_id == 'df33e853-f5d1-4e49-b0c7-b5523cfe75cd':
    #   print('offscreen', is_in_fov_h, is_in_fov_v)
    #   print(cuboid.box3d)
    #   import pdb; pdb.set_trace()
    # elif cuboid.track_id == '79f92a80-93dc-442b-8cce-1c8da11fbe3b':
    #   print('ON', is_in_fov_h, is_in_fov_v)
    #   print(cuboid.box3d)
    #   import pdb; pdb.set_trace()
    # return True
    # if cuboid.track_id == 'nuscenes_instance_token:e91afa15647c4c4994f19aeb302c7179':
    #   import pdb; pdb.set_trace()
    return is_in_fov_h and is_in_fov_v

  def project_cuboid_to_bbox(self, cuboid):
    bbox = BBox(
            im_width=self.width,
            im_height=self.height,
            category_name=cuboid.category_name,
            au_category=cuboid.au_category,
            cuboid=cuboid)
    
    ## Fill Points
    pts_in_cam = self.cam_from_ego.apply(cuboid.box3d).T
    centroid = np.mean(pts_in_cam, axis=0)

    # Since the cuboid could be behind or alongside the camera, not all
    # of the cuboid faces may be visible.  If the object is very large,
    # perhaps only a single edge is visible.  To find the image-space 
    # 2-D axis-aligned bounding box that bounds all cuboid points, we find
    # the horizonal and vertical angles relative to the camera principal
    # axis (Z in the camera frame) that fits all cuboid points.  Then
    # if the object is partially out of view (or even behind the camera),
    # it is easy to clip the bounding box to the camera field of view.

    def l2_normalized(v):
      if len(v.shape) > 1:
        # Normalize row-wise
        return v / np.linalg.norm(v, axis=1)[:, np.newaxis]
      else:
        return v / np.linalg.norm(v)

    def to_0_2pi(thetas):
      return (thetas + 2 * math.pi) % 2 * math.pi

    def theta_signed(cam_h, cuboid_h):
      thetas = np.arctan2(np.cross(cam_h, cuboid_h), np.dot(cam_h, cuboid_h.T))
      return thetas
      # return to_0_2pi(thetas)

    Z_HAT = np.array([0, 1]) # Principal axis in X-Z and Y-Z planes
    pts_xz = pts_in_cam[:, (0, 2)]
    theta_h = theta_signed(l2_normalized(pts_xz), Z_HAT)
    pts_yz = pts_in_cam[:, (1, 2)]
    theta_v = theta_signed(l2_normalized(pts_yz), Z_HAT)

    # center_h = theta_signed(Z_HAT, l2_normalized(centroid[(0, 2)]))
    # center_v = theta_signed(Z_HAT, l2_normalized(centroid[(1, 2)]))

    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    c_x = self.K[0, 2]
    c_y = self.K[1, 2]
    fov_h, fov_v = self.get_fov()

    t_h_min, t_h_max = theta_h.min(), theta_h.max()
    t_v_min, t_v_max = theta_v.min(), theta_v.max()

    def to_pixel(theta, fov, length):
      half_fov = .5 * fov
      # p = np.clip(theta, -half_fov, half_fov) / half_fov
      p = theta / half_fov
      p = (p + 1) / 2
      return length * p

    x1 = to_pixel(t_h_min, fov_h, self.width)
    x2 = to_pixel(t_h_max, fov_h, self.width)
    y1 = to_pixel(t_v_min, fov_v, self.height)
    y2 = to_pixel(t_v_max, fov_v, self.height)

    focal_pixel_h = (.5 * self.width) / math.tan(fov_h * .5)
    focal_pixel_v = (.5 * self.height) / math.tan(fov_v * .5)

    uvd = self.K.dot(pts_in_cam.T)
    uvd[0:2, :] /= (uvd[2, :])
    uvd = uvd.T

    # import pdb; pdb.set_trace()
    uvt_good = np.stack([
      np.sin(theta_h) * np.linalg.norm(pts_xz, axis=1) * focal_pixel_h,
      np.sin(theta_v) * np.linalg.norm(pts_yz, axis=1) * focal_pixel_v,
      uvd[:,2],
    ]).T

    # def to_point(theta, dist, fov, focal_l, pts):
    #   # disp = (theta > 0) * dist
    #   p_prime = 2. * np.tan(np.abs(theta) * .5) * focal_l * pts[:,0]
    #   return p_prime / np.abs(pts[:,1]) + .5 * dist

    # uvt = np.stack([
    #   to_point(theta_h, self.width, fov_h, f_x, pts_xz),
    #   to_point(theta_v, self.height, fov_v, f_y, pts_yz),
    #   # np.sin(theta_h - fov_h * .5) * f_x + .5 * self.width+ self.width, #np.linalg.norm(pts_xz, axis=1) * focal_pixel_h,
    #   # np.sin(theta_v - fov_v * .5) * f_y + , #np.linalg.norm(pts_yz, axis=1) * focal_pixel_v,
    #   uvd[:,2],
    # ]).T

    pts_xy = pts_in_cam[:, :2]
    theta_xy = np.arctan2(pts_xy[:, 1], pts_xy[:, 0])
    uvt = np.stack([
      np.cos(theta_xy) * f_x * (1 / .001),
      np.sin(theta_xy) * f_y * (1 / .001),
      uvd[:,2],
    ]).T

    for r in range(8):
      # if abs(theta_h[r]) > fov_h * .5 or abs(theta_v[r]) > fov_v * .5:
      if uvd[r, 2] <= 0:
        uvd[r, :] = uvt[r, :]
    # uvd = uvt

    # if cuboid.track_id == 'nuscenes_instance_token:df8a0ce6d79446369952166553ede088':
    #   import pdb; pdb.set_trace()












    # print('')
    # uvd = self.project_ego_to_image(cuboid.box3d, omit_offscreen=False)

    bbox.cuboid_pts = uvd
    # print('uvd')
    # print(uvd)
    # print()
    # if cuboid.track_id == 'nuscenes_instance_token:e91afa15647c4c4994f19aeb302c7179':
    #   import pdb; pdb.set_trace()

    x1, x2 = np.min(uvd[:, 0]), np.max(uvd[:, 0])
    y1, y2 = np.min(uvd[:, 1]), np.max(uvd[:, 1])
    bbox.set_x1_y1_x2_y2(x1, y1, x2, y2)

    z = float(np.max(uvd[:, 2]))
    num_onscreen = bbox.get_num_onscreen_corners()
    bbox.has_offscreen = ((z <= 0) or (num_onscreen < 4))

    # While none of the points or cuboid points may be onscreen, if the object
    # is very close to the camera then a single edge of the cuboid or bbox
    # may intersect the screen.  TODO: proper frustum clipping for objects
    # that are beyond FoV and yet very slightly in front of the image plane.
    bbox.is_visible = (z > 0 and self._has_edge_in_fov(cuboid))
      # bbox.overlaps_with(common.BBox.of_size(self.width, self.height)))

    bbox.clamp_to_screen()

    ## Fill Pose
    bbox.cuboid_from_cam = \
      cuboid.obj_from_ego.translation - self.cam_from_ego.translation

    cuboid_from_cam_hat = \
      bbox.cuboid_from_cam / np.linalg.norm(bbox.cuboid_from_cam)
    
    cuboid_from_cam_hat = cuboid_from_cam_hat.reshape(3)

    from scipy.spatial.transform import Rotation as R
    X_HAT = np.array([1, 0, 0])
    obj_normal = cuboid.obj_from_ego.rotation.dot(X_HAT)
    cos_theta = cuboid_from_cam_hat.dot(obj_normal.reshape(3))
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

    if self.clouds:
      debug_img = np.copy(image)
      for pc in self.clouds:
        aupl.draw_xy_depth_in_image(debug_img, pc.cloud, alpha=0.7)
      table = [
        ['<b>Image With Clouds</b>'],
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

      html += '<br /><b>Boxes</b><br />'
      table = [
        [aupl.img_to_img_tag(
            bbox.get_crop(image),
            image_viewport_hw=(300, 300)),
         bbox.to_html() + '<br /><hr />']
        for bbox in self.bboxes
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    return html



# class CroppedCameraImage(CameraImage):
#   __slots__ = tuple(list(CameraImage.__slots__) + [
#     # Viewport of camera; this image is potentially a crop of a (maybe shared)
#     # image buffer
#     'viewport',               # type: common.BBox
#   ])

#   def __init__(self, **kwargs):
#     super(CroppedCameraImage, self).__init__(**kwargs)
#     self.viewport = \
#       self.viewport or common.BBox.of_size(self.width, self.height)

#   @property
#   def image(self):
#     img = super(CroppedCameraImage, self).image
#     if util.np_truthy(img):
#       img = self.viewport.get_crop(img)
#     return img
  
#   def project_ego_to_image(self, pts, omit_offscreen=True):
#     uvd = super(CroppedCameraImage, self).project_ego_to_image(
#       pts, omit_offscreen=omit_offscreen)
    
#     # Correct for moved image origin
#     uvd -= np.array([self.viewport.x, self.viewport.y, 0])
#     return uvd


class StampedDatum(URI):
  """A single piece of data associated with a specific time and a specific
  robot (since we don't have robot IDs, we use segment_ids to imply robot).
  In practice, we stamp the datum with other context, such as dataset.
  Represents a single row in a `StampedDatumTable`."""

  __slots__ = tuple(list(URI.__slots__) + [
    # Inherit everything from a URI; we'll use URIs to address StampedDatums.
    # Python users can access URI attributes directly or thru the `uri`
    # property below.
    # Parquet users can partition data using URI attributes.

    # The actual Data
    'camera_image',       # type: CameraImage
    'point_cloud',        # type: PointCloud
    'cuboids',            # type: List[Cuboid]
    # 'bboxes',             # type: List[BBox]
    'transform',          # type: Transform
  ])

  def __init__(self, **kwargs):
    super(StampedDatum, self).__init__(**kwargs)
    DEFAULTS = {
      'cuboids': [],
    }
    _set_defaults(self, kwargs, DEFAULTS)

  @staticmethod
  def from_uri(uri, **init_kwargs):
    kwargs = dict(uri.as_tuple())
    kwargs.update(init_kwargs)
    return StampedDatum(**kwargs)

  def __str__(self):
    return 'StampedDatum[%s]' % self.uri

  def __repr__(self):
    kvs = ((attr, getattr(self, attr)) for attr in self.__slots__)
    kwargs_str = ', '.join('%s=%s' % (k, repr(v)) for k, v in kvs)
    return 'StampedDatum(%s)' % kwargs_str

  def __eq__(self, other):
    return _slotted_eq(self, other)

  def __lt__(self, other):
    assert type(other) is type(self)
    return self.uri < other.uri

  @property
  def uri(self):
    return URI(**dict((attr, getattr(self, attr)) for attr in URI.__slots__))




class Frame(object):

  __slots__ = (
    'uri',                  # type: URI or str
    'camera_images',        # type: List[CameraImage]
    'clouds',               # type: List[PointCloud]
    'world_to_ego',         # type: Transform; the pose of the robot in the
                            #   global frame (typicaly the city frame)
    'extra',                # type: string -> string extra metadata
  )

  def __init__(self, **kwargs):
    DEFAULTS = {
      'uri': URI(),
      'camera_images': [],
      'clouds': [],
      'world_to_ego': Transform(),
      'extra': {},
    }
    _set_defaults(self, kwargs, DEFAULTS)
    
    if isinstance(self.uri, six.string_types):
      self.uri = URI.from_str(self.uri)
    
  def to_html(self):
    from datetime import datetime
    import tabulate
    import pprint
    table = [
      ['URI', str(self.uri)],
      ['Timestamp', 
        datetime.utcfromtimestamp(self.uri.timestamp * 1e-9).strftime('%Y-%m-%d %H:%M:%S')],
      ['Ego Pose', '<pre>' + str(self.world_to_ego) + '</pre>'],
      ['Extra', '<pre>' + pprint.pformat(self.extra) + '</pre>'],
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

URI_PROTO_KWARGS = dict(
  # Core spec; most URIs will have these set
  dataset='proto',
  split='train',
  segment_id='proto_segment',
  topic='topic',
  timestamp=int(100 * 1e9), # In nanoseconds
  
  # Uris can identify more specific things in a Frame
  camera='camera_1',
  camera_timestamp=int(100 * 1e9), # In nanoseconds
  
  crop_x=0, crop_y=0,
  crop_w=10, crop_h=10,
  
  track_id='track-001',

  extra={'key': 'value'},
)
URI_PROTO = URI(**URI_PROTO_KWARGS)

TRANSFORM_PROTO = Transform(
  rotation=np.eye(3, 3),
  translation=np.zeros((3, 1)),
  src_frame='world',
  dest_frame='ego',
)

CUBOID_PROTO = Cuboid(
  track_id='track-01',
  category_name='vehicle',
  au_category='car',
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

  obj_from_ego=TRANSFORM_PROTO,
  ego_pose=TRANSFORM_PROTO,
  extra={
    'key': 'value',
  },
)

BBOX_PROTO = BBox(
  x=0, y=0,
  width=10, height=10,
  im_width=100, im_height=100,
  category_name='vehicle',
  au_category='car',

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
  cloud=np.ones((10, 3)),
  motion_corrected=True,
  ego_to_sensor=TRANSFORM_PROTO,
  ego_pose=TRANSFORM_PROTO,
)

CAMERAIMAGE_PROTO = CameraImage(
  camera_name='front_center',
  image_jpeg=bytearray(b''),
  height=0,
  width=0,
  timestamp=int(100 * 1e9), # In nanoseconds
  ego_pose=TRANSFORM_PROTO,
  
  clouds=[POINTCLOUD_PROTO],
  
  bboxes=[BBOX_PROTO],

  cam_from_ego=Transform(),
  K=np.zeros((3, 3)),
  principal_axis_in_ego=np.array([0., 0., 0.]),
)

STAMPED_DATUM_PROTO = StampedDatum(
  camera_image=CAMERAIMAGE_PROTO,
  point_cloud=POINTCLOUD_PROTO,
  cuboids=[CUBOID_PROTO],
  
  transform=TRANSFORM_PROTO,
  **URI_PROTO_KWARGS,
)

FRAME_PROTO = Frame(
  uri=URI_PROTO,
  camera_images=[CAMERAIMAGE_PROTO],
  clouds=[POINTCLOUD_PROTO],
  world_to_ego=Transform(),
  extra={
    'key': 'value',
  },
)



###
### Tables
###

class StampedDatumTableBase(object):

  ## Public API

  PARTITION_KEYS = ('dataset', 'split', 'segment_id')

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, 'av_stamped_data')
  
  @classmethod
  def setup(cls, spark=None):
    if util.missing_or_empty(cls.table_root()):
      with Spark.sess(spark) as spark:
        sd_rdds = cls._create_datum_rdds(spark)
        class StampedDatumDFThunk(object):
          def __init__(self, sd_rdd):
            self.sd_rdd = sd_rdd
          def __call__(self):
            return cls._sd_rdd_to_sd_df(spark, self.sd_rdd)
        df_thunks = [StampedDatumDFThunk(sd_rdd) for sd_rdd in sd_rdds]
        Spark.save_df_thunks(
          df_thunks,
          path=cls.table_root(),
          format='parquet',
          partitionBy=cls.PARTITION_KEYS,
          compression='lz4')

  @classmethod
  def as_df(cls, spark):
    df = spark.read.option("mergeSchema", "true").parquet(cls.table_root())
    return df

  @classmethod
  def as_stamped_datum_rdd(cls, spark):
    df = cls.as_df(spark)
    sd_rdd = df.rdd.map(RowAdapter.from_row)
    return sd_rdd

  ## Subclass API - Each dataset should provide ETL to a StampedDatumTable

  @classmethod
  def _create_datum_rdds(cls, spark):
    """Subclasses should create and return a list of `RDD[StampedDatum]`s"""
    return []

  ## Support

  @classmethod
  def _sd_rdd_to_sd_df(cls, spark, sd_rdd):
    to_row = RowAdapter.to_row
    schema = RowAdapter.to_schema(to_row(STAMPED_DATUM_PROTO))
    row_rdd = sd_rdd.map(to_row)
    df = spark.createDataFrame(row_rdd, schema=schema)
    return df


class FrameTableBase(object):

  ## Public API

  PARTITION_KEYS = ('dataset', 'split', 'segment_id')

  # @classmethod
  # def get_shard(cls, uri):
  #   if isinstance(uri, six.string_types):
  #     uri = URI.from_str(uri)
  #   return uri.segment_id + '|' + str(int(uri.timestamp * 1e9))

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
  def row_to_frame(cls, frame_df_row):
    return RowAdapter.from_row(frame_df_row)

  @classmethod
  def create_frame(cls, uri):
    """Subclasses should create and return a `Frame` for the given `uri`."""
    return av.Frame(uri=uri)

  @classmethod
  def _create_frame_rdds(cls, spark):
    """Subclasses should create and return a list of `RDD[Frame]`s"""
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
      row['segment_id'] = f.uri.segment_id
      row['timestamp'] = f.uri.timestamp
      # row['shard'] = cls.get_shard(f.uri)
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
    # pkey_row_rdd = pkey_row_rdd.persist(StorageLevel.DISK_ONLY)
    row_rdd = pkey_row_rdd #.map(lambda pkey_row: pkey_row[-1])
    
    schema = RowAdapter.to_schema(to_pkey_row(FRAME_PROTO))

    df = spark.createDataFrame(row_rdd, schema=schema)
    return df


###
### Tensorflow Interop
###

def camera_image_to_tf_example(
    frame_uri,
    camera_image,
    label_map_dict=AU_AV_CATEGORY_TO_ID):
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

    c = bbox.au_category
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
        dataset_util.bytes_feature(
          str(util.stable_hash(frame_uri) % (1 << 32)).encode('utf8')),
            # COCO wants a numeric ID and sadly Tensorflow convention is
            # to us a string-typed field here.  Also, we need for the int
            # to be a reasonable size verus a bigint
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

def frame_df_to_tf_example_ds(frame_df, label_map_dict=AU_AV_CATEGORY_TO_ID):
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



def frame_table_to_object_detection_tfrecords(
        spark,
        frame_table, 
        output_base_dir,
        label_map_dict=AU_AV_CATEGORY_TO_ID):
  
  def partition_to_tfrecords(partition_id, iter_rows):
    import tensorflow as tf
    
    t = util.ThruputObserver(name="tf_record_writer_%s" % partition_id)

    dest_to_writer = {}
    for row in iter_rows:
      with t.observe():
        frame = frame_table.row_to_frame(row)
        dest_fname = 'part-%s.tfrecords' % partition_id
        dest = os.path.join(
                  output_base_dir,
                  frame.uri.segment_id,
                  dest_fname)
        if dest not in dest_to_writer:
          if 'gs://' not in dest:
            util.mkdir(os.path.dirname(dest))
          dest_to_writer[dest] = tf.io.TFRecordWriter(dest)
        
        writer = dest_to_writer[dest]
        for ci in frame.camera_images:
          tf_example = camera_image_to_tf_example(frame.uri, ci, label_map_dict)
          example_str = tf_example.SerializeToString()
          writer.write(example_str)
          t.update_tallies(n=1, num_bytes=len(example_str))
        writer.flush()
      t.maybe_log_progress()
    
    for writer in dest_to_writer.values():
      writer.close()
    if t.n > 0:
      util.log.info(
        "Partition %s complete with thruput: \n %s" % (partition_id, str(t)))
    return [t.n]
  
  util.log.info("Writing TFRecords to %s ..." % output_base_dir)
  frame_df = frame_table.as_df(spark)
  num_written_rdd = frame_df.rdd.mapPartitionsWithIndex(partition_to_tfrecords)
  num_written = num_written_rdd.sum()
  util.log.info("Wrote %s records to %s ." % (num_written, output_base_dir))


  
###
### WebUI
###

class WebUI(object):

  def __init__(self):
    pass

  def _get_frame_df(self):
    pass

  def _spark(self):
    pass

  def get_frame(self, uri):
    uri = URI.from_str(uri)
    frame_df = self._get_frame_df()
    res = frame_df.select('*').where(
      frame_df.uri == str(uri),
      frame_df.dataset == uri.dataset,
      frame_df.split == uri.split)
    rows = res.collect()
    assert rows, "Frame not found %s" % uri
    f = FrameTable.row_to_frame(rows[0])
    return f.to_html()

  def list_segments(self, uri):
    uri = URI.from_str(uri)
    frame_df = self._get_frame_df()
    
    QUERY = """
      SELECT *
      FROM
        frame f INNER JOIN
          (
            SELECT
              segment_id,
              FIRST(uri),
              MIN(timestamp) min_ts
            FROM frame
            WHERE {where}
            GROUP BY segment_id
            HAVING timestamp = min_ts
          ) segs
        ON f.uri = segs.uri
    """

    where = "1"
    if uri.dataset:
      where += " AND dataset = '%s'" % uri.dataset
    if uri.split:
      where += " AND split = '%s'" % uri.split
    
    query = QUERY.format(where=where)
    df = self._spark.sql(query)

    def to_disp(row):
      f = FrameTable.row_to_frame(row)
      return Row(
        segment_id=f.uri.segment_id,
        dataset=f.uri.dataset,
        split=f.uri.split,
        debug_image=f.get_debug_image(),
      )
    
    disp_df = self._spark.createDataFrame(df.rdd.map(to_disp))
    disp_df = disp_df.orderBy('dataset', 'split', 'segment_id')
    return disp_df.toPandas()


    segs = frame_df
    if uri.dataset:
      segs = segs.where(segs.dataset == uri.dataset)
    if uri.split:
      segs = segs.where(segs.split == uri.split)
    
    segs = segs.groupBy('segment_id')
    segment_ids = segs.collect()

    util.log.info("Found %s segments" % len(segment_ids))

    from pyspark.sql import functions as F
    seg_rows = frame_df.where()


