"""

*** track ID df33e853-f5d1-4e49-b0c7-b5523cfe75cd is used for two
 different annotations :( in the same image

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
import random
import os
import sys

from au import conf
from au import util
from au.fixtures import dataset
from au.fixtures.datasets import av
from au.fixtures.datasets import common
from au.spark import Spark
from au.spark import NumpyArray

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
  "ANIMAL":             'ped',
  "BICYCLE":            'bike',
  "BICYCLIST":          'ped',
  "BUS":                'car',
  "EMERGENCY_VEHICLE":  'car',
  "LARGE_VEHICLE":      'car',
  "MOPED":              'bike',
  "MOTORCYCLE":         'bike',
  "MOTORCYCLIST":       'ped',
  "ON_ROAD_OBSTACLE":   'other',
  "OTHER_MOVER":        'other',
  "PEDESTRIAN":         'ped',
  "SCHOOL_BUS":         'car',
  "STROLLER":           'other',
  "TRAILER":            'car',
  "VEHICLE":            'car',
  "WHEELCHAIR":         'ped',
}

BIKE = ["BICYCLE", "MOPED", "MOTORCYCLE"]
RIDER = ["BICYCLIST", "MOTORCYCLIST"]

class MissingPose(ValueError):
  pass

def get_nanostamp_from_json_path(path):
  nanostamp = int(path.split('_')[-1].rstrip('.json'))
  return nanostamp

def get_lidar_extrinsic_matrix(config, lidar_name='vehicle_SE3_up_lidar_'):
  """Similar to argoverse.utils.calibration.get_camera_extrinsic_matrix()
  except this utility reads the *lidar* extrinsics from the calibration JSON
  file.  Argoverse does not actually provide any tool for doing this (!!!).

  Since Argoverse fuses the top and bottom lidars into one cloud, in practice
  one might want to simply use the 'up' lidar extrinsics. (From the JSON
  values, the up lidar appears to have a unit rotation, and the down lidar
  has slightly-off-from-unit rotation, so perhaps the down lidar is calibrated
  to the up one).

  Note that the Argoverse lidar points included with the dataset are already
  in ego frame.

  Observed valid `lidar_name` values are:
    * vehicle_SE3_up_lidar_
    * vehicle_SE3_down_lidar_

  Returns a standard transformation matrix of lidar frame from ego frame.
  """
  from argoverse.utils.se3 import SE3
  from argoverse.utils.transform import quat2rotmat

  vehicle_SE3_sensor = config[lidar_name]
  egovehicle_t_lidar = np.array(vehicle_SE3_sensor["translation"])
  egovehicle_q_lidar = vehicle_SE3_sensor["rotation"]["coefficients"]
  egovehicle_R_lidar = quat2rotmat(egovehicle_q_lidar)
  egovehicle_T_lidar = SE3(
                  rotation=egovehicle_R_lidar, translation=egovehicle_t_lidar)
  # NB: camera extrinsics require an inverse(), and comments in argoverse code
  # are confusing.  Manual inspection of the JSON data suggests no inverse()
  # is needed.
  return egovehicle_T_lidar.transform_matrix

def get_image_width_height(camera):
  from argoverse.utils import camera_stats
  if camera in camera_stats.RING_CAMERA_LIST:
    return camera_stats.RING_IMG_WIDTH, camera_stats.RING_IMG_HEIGHT
  elif camera in camera_stats.STEREO_CAMERA_LIST:
    return camera_stats.STEREO_IMG_WIDTH, camera_stats.STEREO_IMG_HEIGHT
  else:
    raise ValueError("Unknown camera: %s" % camera)

def get_camera_normal(calib):
  """Compute and return a 3d unit vector in the ego (car) frame that is
  normal to the principal plane of the camera with Calibration `calib`.

  In ROS land (e.g. a robot URDF), we typically give cameras two frames:
  one for the physical device (whose frame axes semantically match
  the robot's) and one for the CCD sensor (which typically has different
  frame axes, e.g. Z is depth [out of the camera forward] instead of
  height [or "up" in the robot frame]). This design allows us to
  disambiguate the sensor's pose on the robot (as e.g. a human would
  see the sensor) and the sensor's internal frame. Knowing the sensor's
  pose can facilitate easy estimates of blindspots, calibration checks,
  etc.

  BARF: Unfortunately, Argoverse does not use this convention, and
  instead their calibration confounds sensor pose and frame.  Moreover,
  they appear to use different sensors on differnet drives.  For example: 
    * Log f9fa3960 has pitch of 0.51 rad for front center camera
    * Log 53037376 has pitch of 0.008 rad for front center camera
  The above two logs have yaws / rolls that are off by Pi.

  Solution: We can estimate the camera normal in the robot's frame
  from the full camera projection matrix P = K[R|T], which is provided
  through `calib`.  We use a method described in Zisserman "Multiple
  View Geometry".

  First, we form the projection matrix P:
      P = |K 0| * | R  T|
                  |000 1|
  
  Via `argoverse.utils.calibration.get_camera_intrinsic_matrix`,
  we see that:
      calib.K = |fx s  cx 0|
                |0  fy cy 0|
                |0  0   1 0|

  and `argoverse.utils.calibration.Calibration.__init__` as well as
  `argoverse.utils.se3.SE3.__init__` show that:
      calib.extrinsic = | R  T|
                        |000 1|
  
  Thus 
      P = calib.K.dot(calib.extrinsic)

  Zisserman pg "Multiple View Gemoetric (2nd ed.) pg. 161
  http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
  The principal axis vector $pv$ is a ray that points along the
  principal axis of the camera (in the world frame) and is computed
  as follows:
      Let P = [M | p4] and
      Let M = |..|
              |m3|
      Then pv = det(M) * m3
  
  Zisserman cites 'proof by studying the P matrix' :P

  In practice, we observe the follow normalized vectors for Argoverse
  cars, which agree with Argoverse's published sensor placement
  diagrams as well as the frames documented
  in `argoverse.utils.calibration.Calibration`:
    Log Prefix            Camera                             Normal
    22160544   ring_front_center     [0.999917, -0.01263, 0.002472]
    53037376   ring_front_center   [0.999962, -0.008618, -0.001102]
    15c802a9   ring_front_center    [0.999958, -0.00647, -0.006419]
    5c251c22   ring_front_center   [0.999962, -0.008618, -0.001102]
    1d676737   ring_front_center     [0.999955, 0.009325, -0.00156]
    64c12551   ring_front_center   [0.999957, -0.002238, -0.008979]
    53037376     ring_front_left    [0.722965, 0.690788, -0.011564]
    1d676737     ring_front_left      [0.703746, 0.7104, -0.008523]
    cb0cba51    ring_front_right   [0.704961, -0.709232, -0.004577]
    c6911883    ring_front_right   [0.704096, -0.709917, -0.016312]
    2c07fcda      ring_rear_left    [-0.864428, 0.502756, 0.000217]
    f9fa3960     ring_rear_right   [-0.869722, -0.493458, 0.009077]
    02cf0ce1     ring_rear_right  [-0.871229, -0.490875, -0.001238]
    e17eed4f      ring_side_left   [-0.172404, 0.984978, -0.009765]
    e17eed4f     ring_side_right  [-0.177154, -0.984156, -0.007321]
    c6911883   stereo_front_left    [0.999996, -0.002041, 0.001755]
    3138907e   stereo_front_left    [0.999984, -0.001388, 0.005503]
    70d2aea5   stereo_front_left    [0.999996, -0.002041, 0.001755]
    043aeba7  stereo_front_right     [0.999957, -0.00716, 0.005925]

  """
  # Build P
  # P = |K 0| * | R |T|
  #             |000 1|
  P = calib.K.dot(calib.extrinsic)

  # Zisserman pg 161 The principal axis vector.
  # P = [M | p4]; M = |..|
  #                   |m3|
  # pv = det(M) * m3
  pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
  pv_hat = pv / np.linalg.norm(pv)
  return pv_hat

# class FrameURI(object):
#   __slots__ = (
#     'tarball_name', # E.g. tracking_sample.tar.gz
#     'log_id',       # E.g. c6911883-1843-3727-8eaa-41dc8cda8993
#     'split',        # Official Argoverse split (see Fixtures.SPLITS)
#     'camera',       # E.g. ring_front_center
#     'timestamp',    # E.g. 315975652303331336, yes this is GPS time :P :P

#     ## Optional
#     'track_id',     # A UUID of a specific track / annotation in the frame
    
#     'crop_x', 'crop_y',
#     'crop_w', 'crop_h',
#                     # A specific viewport / crop of the frame
#   )
  
#   OPTIONAL = ('track_id', 'crop_x', 'crop_y', 'crop_w', 'crop_h',)

#   PREFIX = 'argoverse://'

#   def __init__(self, **kwargs):
#     # Use kwargs, then fall back to args
#     for i, k in enumerate(self.__slots__):
#       setattr(self, k, kwargs.get(k, ''))
#     if self.timestamp is not '':
#       self.timestamp = int(self.timestamp)
  
#   def to_str(self):
#     path = '&'.join(
#       attr + '=' + str(getattr(self, attr))
#       for attr in self.__slots__
#       if getattr(self, attr))
#     return self.PREFIX + path
  
#   def __str__(self):
#     return self.to_str()

#   def to_dict(self):
#     return dict((k, getattr(self, k, '')) for k in self.__slots__)

#   def update(self, **kwargs):
#     for k in self.__slots__:
#       if k in kwargs:
#         setattr(self, k, kwargs[k])

#   def set_crop(self, bbox):
#     self.update(
#       crop_x=bbox.x,
#       crop_y=bbox.y,
#       crop_w=bbox.width,
#       crop_h=bbox.height)

#   def has_crop(self):
#     return all(
#       getattr(self, 'crop_%s' % a) is not ''
#       for a in ('x', 'y', 'w', 'h'))

#   def get_crop_bbox(self):
#     return BBox(
#             x=self.crop_x, y=self.crop_y,
#             width=self.crop_w, height=self.crop_h)

#   def get_viewport(self):
#     if self.has_crop():
#       return self.get_crop_bbox()
#     else:
#       return BBox.of_size(*get_image_width_height(self.camera))

#   @staticmethod
#   def from_str(s):
#     if isinstance(s, FrameURI):
#       return s
#     assert s.startswith(FrameURI.PREFIX)
#     toks_s = s[len(FrameURI.PREFIX):]
#     toks = toks_s.split('&')
#     assert len(toks) >= (len(FrameURI.__slots__) - len(FrameURI.OPTIONAL))
#     uri = FrameURI(**dict(tok.split('=') for tok in toks))
#     return uri

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
      # 'relative_yaw_to_camera_radians',
      'has_offscreen',
      'is_visible',
      'z',

      'motion_corrected',   # Has the ObjectLabelRecord
                            # been motion-corrected?
      'cuboid_pts',         # In robot ego frame
      'cuboid_pts_image',   # In image space
      'ego_to_obj',         # Translation vector in ego frame
      'city_to_ego',        # Transform of city to car
      # 'ego_to_camera',      # Transform from car to camera frame
      'camera_norm',        
      'camera_to_obj',
      'obj_ypr_in_ego',
      'obj_ypr_camera_local',
    ]
  )

  # TODO just make Spark support numpy
  def _adapt(v):
    if isinstance(v, np.ndarray):
      return NumpyArray(v)
    # elif isinstance(v, NumpyArray):
    #   return v.arr
    else:
      return v

  def to_row_dict(self):
    return dict((k, BBox._adapt(v)) for k, v in self.to_dict().items())

  @staticmethod
  def from_row_dict(row):
    adapted = dict((k, BBox._adapt(v)) for k, v in row.items())
    return BBox(**adapted)

  def translate(self, *args):
    super(BBox, self).translate(*args)
    if len(args) == 1:
      x, y = args[0].tolist()
    else:
      x, y = args
    self.cuboid_pts_image += np.array([x, y])

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

  # @staticmethod
  # def from_argoverse_label(
  #       uri,
  #       object_label_record,
  #       motion_corrected=True,
  #       fixures_cls=None):
  #   """Construct and return a single `BBox` instance from the given
  #   Argoverse ObjectLabelRecord instance.  Labels are in lidar space-time
  #   and *not* camera space-time; therefore, transforming labels into
  #   the camera domain requires (to be most precise) correction for the
  #   egomotion of the robot.  This correction can be substantial (~20cm)
  #   at high robot speed.  Apply this correction only if
  #   `motion_corrected`.
  #   """
    
  #   if not fixures_cls:
  #     fixures_cls = Fixtures
    
  #   loader = fixures_cls.get_loader(uri)
  #   calib = loader.get_calibration(uri.camera)

  #   def fill_cuboid_pts(bbox):
  #     bbox.cuboid_pts = object_label_record.as_3d_bbox()
  #     bbox.motion_corrected = False
  #       # Points in robot frame
  #     if motion_corrected:
  #       try:
  #         bbox.cuboid_pts = loader.get_motion_corrected_pts(
  #                                   bbox.cuboid_pts,
  #                                   object_label_record.timestamp,
  #                                   uri.timestamp)
  #         bbox.motion_corrected = True
  #       except MissingPose:
  #         # Garbage!
  #         pass

  #   def fill_extra(bbox):
  #     bbox.track_id = object_label_record.track_id
  #     bbox.occlusion = object_label_record.occlusion
  #     bbox.length_meters = object_label_record.length
  #     bbox.width_meters = object_label_record.width
  #     bbox.height_meters = object_label_record.height

  #     bbox.distance_meters = \
  #       float(np.min(np.linalg.norm(bbox.cuboid_pts, axis=-1)))

  #     from scipy.spatial.transform import Rotation as R
  #     from argoverse.utils.transform import quat2rotmat
  #     rotmat = quat2rotmat(object_label_record.quaternion)
  #       # NB: must use quat2rotmat due to Argo-specific quaternion encoding
  #     # bbox.relative_yaw_radians = math.atan2(rotmat[2, 1], rotmat[1, 1])
  #     bbox.relative_yaw_radians = float(R.from_dcm(rotmat).as_euler('zxy')[0])
  #       # Tait–Bryan?  ... But y in Argoverse is to the left?
      
  #     camera_yaw = math.atan2(calib.R[2, 1], calib.R[1, 1])
  #     # bbox.relative_yaw_to_camera_radians = [camera_yaw, 

  #     bbox.ego_to_obj = object_label_record.translation
  #     city_to_ego_se3 = loader.get_city_to_ego(uri.timestamp)
  #     # bbox.city_to_ego = common.Transform(
  #     #                       rotation=city_to_ego_se3.rotation,
  #     #                       translation=city_to_ego_se3.translation) ~~~~~~~~~~
  #     bbox.city_to_ego = city_to_ego_se3.translation




  #     from scipy.spatial.transform import Rotation as R
  #     # bbox.ego_to_camera = common.Transform(
  #     #                         rotation=R.from_dcm(calib.R).as_quat(),
  #     #                         translation=calib.T) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #     bbox.camera_norm = get_camera_normal(calib)

  #     bbox.obj_ypr_in_ego = R.from_dcm(rotmat).as_euler('zxy')

  #     from argoverse.utils.se3 import SE3
  #     x_hat = np.array([1, 0, 0])
  #     cos_theta = bbox.camera_norm.dot(x_hat)
  #     rot_axis = np.cross(bbox.camera_norm, x_hat)
  #     ego_to_cam_device_rot = R.from_rotvec(
  #       math.acos(cos_theta) * rot_axis / np.linalg.norm(rot_axis))

  #     # Recover translation from ego to camera
  #     ego_to_cam = SE3(rotation=calib.R, translation=calib.T)
  #     cam_device_from_ego_T = ego_to_cam.inverse().translation
  #     cam_device_from_ego = SE3(
  #           rotation=ego_to_cam_device_rot.as_dcm(),
  #           translation=cam_device_from_ego_T)
  #     # ego_to_cam_device.rotation = ego_to_cam_device_rot.as_dcm()
  #     # obj_from_ego = SE3(rotation=rotmat, translation=bbox.ego_to_obj)
  #     # obj_in_cam = cam_device_from_ego.right_multiply_with_se3(obj_from_ego)

  #     camera_to_obj = bbox.ego_to_obj - cam_device_from_ego_T
  #     bbox.camera_to_obj = camera_to_obj

  #     camera_to_obj_hat = camera_to_obj / np.linalg.norm(camera_to_obj)


  #     # doh_camera_norm = np.array([1, 0, 0])
  #     # cos_theta = bbox.camera_norm.dot(camera_to_obj_hat)
  #     # rot_axis = np.cross(bbox.camera_norm, camera_to_obj_hat)
  #     obj_from_ego = SE3(rotation=rotmat, translation=bbox.ego_to_obj)
  #     obj_normal = obj_from_ego.rotation.dot(x_hat)
  #     cos_theta = camera_to_obj_hat.dot(obj_normal)
  #     rot_axis = np.cross(camera_to_obj_hat, obj_normal)

  #     obj_from_ray = R.from_rotvec(
  #       math.acos(cos_theta) * rot_axis / np.linalg.norm(rot_axis))
  #     # ray_from_cam = SE3(
  #     #   rotation=ray_from_cam_normal.as_dcm(),
  #     #   translation=np.zeros(3))
  #     # obj_in_ray = obj_in_cam.right_multiply_with_se3(ray_from_cam)

  #     # obj_camera_local = R.from_dcm(obj_in_cam.rotation) * obj_from_cam.inv()
  #     # obj_camera_local = R.from_dcm(obj_in_ray.rotation)
  #     bbox.obj_ypr_camera_local = [NumpyArray(obj_from_ray.as_euler('zxy'))]

      


  #     # # Compute object pose relative to camera view; this is the object's
  #     # # pose relative to a ray cast from camera center to object centroid.
  #     # # We can use this pose as a label for predicting 'local pose'
  #     # # as described in Drago et al. https://arxiv.org/pdf/1612.00496.pdf .
  #     # # BARF: camera extrinsics can vary widely:
  #     # # * Log f9fa3960 has pitch of 0.51 rad for front center camera
  #     # # * Log 53037376 has pitch of 0.008 rad for front center camera
  #     # # And the above two logs have yaws / rolls that are off by pi.
  #     # # However, extrinsic translation has less variance, as one might expect.
  #     # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
  #     # # from argoverse.utils.calibration import get_camera_extrinsic_matrix
  #     # # calib_raw = get_camera_extrinsic_matrix(calib.calib_data)
  #     # # vehicle_SE3_sensor = calib.calib_data["value"]['vehicle_SE3_camera_']
  #     # # egovehicle_t_camera = np.array(vehicle_SE3_sensor["translation"])
  #     # # egovehicle_q_camera = vehicle_SE3_sensor["rotation"]["coefficients"]
  #     # # egovehicle_R_camera = quat2rotmat(egovehicle_q_camera)

  #     # # cam_h, cam_w = get_image_width_height(uri.camera)
  #     # # P = calib.K[:,:3].dot(calib.extrinsic[:3,:4])#[:,:-1]
  #     # # P_plus = np.linalg.pinv(P)#P.T * np.linalg.inv(P * P.T)
  #     # # pt3 = np.dot(P_plus,[.5 * cam_w, .5 * cam_h, 10.])

  #     # # P = K * | R |T|
  #     # #         |000 1|
  #     # P = calib.K.dot(calib.extrinsic)

  #     # # Zisserman pg 161
  #     # # http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
  #     # # The principal axis vector.  A ray that points along the principal 
  #     # # axis.
  #     # # P = [M | p4]; M = |..|
  #     # #                   |m3|
  #     # # pv = det(M) * m3
  #     # pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
  #     # pv_hat = pv / np.linalg.norm(pv)

      
  #     # # P_plus * [.5 * cam_w, .5 * cam_h, 10]
  #     # # ptcam = np.array([ ptcam ])
  #     # # pt3 = calib.project_image_to_ego(ptcam)
      
  #     # from argoverse.utils.se3 import SE3
  #     # ego_to_cam = SE3(rotation=calib.R, translation=calib.T).inverse()
  #     # bbox.calib_ypr = ego_to_cam.translation
  #     # bbox.relative_yaw_to_camera_radians = [
  #     #   [float(v) for v in pv_hat]
  #     # ]
      
  #     # # [[float(v) for v in 
  #     # #   R.from_dcm(egovehicle_R_camera).as_euler('zxy').tolist()] , [
  #     # #     float(v) for v in pt3.tolist()]]
  #     # obj_from_cam = (
  #     #   object_label_record.translation - ego_to_cam.translation)
  #     # obj_t_x, obj_t_y, obj_t_z = obj_from_cam 
      
  #     # yaw = R.from_euler('z', math.atan2(-obj_t_y, obj_t_x))
  #     # pitch = R.from_euler('y', math.atan2(obj_t_z, obj_t_x))
  #     # roll = R.from_euler('x', -R.from_dcm(rotmat).as_euler('zxy')[2])
  #     #   # Use camera roll; don't roll the camera when "pointing it" at obj
  #     #   # Also adjust camera roll for pi / 2 frame change that's embedded
  #     #   # into calibration JSON
      
  #     # obj_from_ray_R = (yaw * pitch * roll)

  #     # obj_in_cam_ray = R.from_dcm(rotmat) * obj_from_ray_R.inv()
  #     # bbox.obj_in_crop = obj_in_cam_ray.as_euler('zyx')
  #     #                           # yaw, pitch, roll

  #     # bbox.obj_in_crop_debug = [
  #     #   math.atan2(-obj_t_y, obj_t_x), # yaw
  #     #   math.atan2(obj_t_z, obj_t_x), # pitch
  #     #   float(R.from_dcm(ego_to_cam.rotation).as_euler('zxy')[2] - math.pi / 2) # roll
  #     # ]
  #     # bbox.obj_in_crop_xyz = [float(obj_t_x), float(obj_t_y), float(obj_t_z)]

  #   def fill_bbox_core(bbox):
  #     bbox.category_name = object_label_record.label_class

  #     bbox.im_width, bbox.im_height = get_image_width_height(uri.camera)
  #     uv = calib.project_ego_to_image(bbox.cuboid_pts)
      
  #     bbox.cuboid_pts_image = np.array([uv[:, 0] , uv[:, 1]]).T

  #     x1, x2 = np.min(uv[:, 0]), np.max(uv[:, 0])
  #     y1, y2 = np.min(uv[:, 1]), np.max(uv[:, 1])
  #     z = float(np.max(uv[:, 2]))

  #     bbox.set_x1_y1_x2_y2(x1, y1, x2, y2)

  #     num_onscreen = bbox.get_num_onscreen_corners()
  #     bbox.has_offscreen = ((z <= 0) or (num_onscreen < 4))
  #     bbox.is_visible = (
  #       z > 0 and
  #       num_onscreen > 0 and
  #       object_label_record.occlusion < 100)

  #     bbox.clamp_to_screen()
  #     bbox.z = float(z)

  #   bbox = BBox()
  #   fill_cuboid_pts(bbox)
  #   fill_bbox_core(bbox)
  #   fill_extra(bbox)
  #   return bbox



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
      self.viewport = self.uri.get_viewport()
    
  @property
  def loader(self):
    if not self._loader:
      self._loader = self.FIXTURES.get_loader(self.uri)
    return self._loader # type: AUTrackingLoader
  
  @staticmethod
  @klepto.lru_cache(maxsize=100)
  def __load_image(path):
    return imageio.imread(path)

  @property
  def image(self):
    if not util.np_truthy(self._image):
      path = self.loader.get_nearest_image_path(
                      self.uri.camera, self.uri.timestamp)
      self._image = AVFrame.__load_image(path)
      if not self.viewport.is_full_image():
        c, r, w, h = (
          self.viewport.x, self.viewport.y,
          self.viewport.width, self.viewport.height)
        self._image = self._image[r:r+h, c:c+w, :]
    return self._image
  
  @property
  def cloud(self):
    if not util.np_truthy(self._cloud):
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
    uv = uv.T

    # Correct for image origin if this frame is a crop
    uv -= np.array([self.viewport.x, self.viewport.y, 0])
    return uv

  @property
  def image_bboxes(self):
    if not self._image_bboxes:
      bboxes = self.loader.get_nearest_label_bboxes(self.uri)

      # Ingore invisible things
      self._image_bboxes = [
        bbox for bbox in bboxes
        if bbox.is_visible and self.viewport.overlaps_with(bbox)
      ]

      # Correct for image origin if this frame is a crop
      for bbox in self._image_bboxes:
        bbox.translate(-np.array(self.viewport.get_x1_y1()))
        bbox.im_width = self.viewport.width
        bbox.im_height = self.viewport.height

    return self._image_bboxes

  def get_target_bbox(self):
    if self.uri.track_id:
      for bbox in self.image_bboxes:
          if bbox.track_id == self.uri.track_id:
            return bbox
    return None

  def get_debug_image(self):
    img = np.copy(self.image)
    
    from au import plotting as aupl
    xyd = self.get_cloud_in_image()
    aupl.draw_xy_depth_in_image(img, xyd)

    target_bbox = self.get_target_bbox()
    if target_bbox:
      # Draw a highlight box first; then the draw() calls below will draw over
      # the box.
      # WHITE = (225, 225, 255)
      # target_bbox.draw_in_image(img, color=WHITE, thickness=20)

    # for bbox in self.image_bboxes:
      bbox = target_bbox
      bbox.draw_cuboid_in_image(img)
      # bbox.draw_in_image(img)
    
    return img

  def get_cropped(self, bbox):
    """Create and return a new AVFrame instance that contains the data in this
    frame cropped down to the viewport of just `bbox`."""

    uri = copy.deepcopy(self.uri)
    uri.set_crop(bbox)
    if hasattr(bbox, 'track_id') and bbox.track_id:
      uri.track_id = bbox.track_id

    frame = self.FIXTURES.get_frame(uri)
    return frame

class HardNegativeMiner(object):
  SEED = 1337
  MAX_FRACTION_ANNOTATED = 0.2
  WIDTH_PIXELS_MU_STD = (121, 50)
  HEIGHT_PIXELS_MU_STD = (121, 50)

  def __init__(self, viewport, pos_boxes):
    self._viewport = viewport
    
    # Build a binary mask where a pixel has an indicator value of 1 only if
    # one or more annotation bboxes covers that pixel
    mask = np.zeros((viewport.im_height, viewport.im_width))
    for bbox in pos_boxes:
      r1, c1, r2, c2 = bbox.get_r1_c1_r2_r2()
      mask[r1:r2+1, c1:c2+1] = 1

    # We'll use the integral image trick to make rejection sampling efficient
    class IntegralImage(object):
      def __init__(self, img):
        self.__ii = img.cumsum(axis=0).cumsum(axis=1)
      
      def get_sum(self, r1, c1, r2, c2):
        r2 -= 1 # Boundary conditions: the integral image is exclusive
        c2 -= 1 # on the distance point, but BBox is inclusive.
        return (
          self.__ii[r2, c2]
          - self.__ii[r1, c2] - self.__ii[r2, c1]
          + self.__ii[r1, c1])

    self._ii = IntegralImage(mask)
    self._random = random.Random(self.SEED + util.stable_hash(pos_boxes))

  def next_sample(self, max_attempts=1000):
    if not self._ii:
      return None

    rand = self._random
    for _ in range(max_attempts):
      v = self._viewport
      
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
      if sample.get_area() <= 0:
        continue
      
      num_anno_pixels = self._ii.get_sum(*sample.get_r1_c1_r2_r2())
      
      # Do we have enough non-annotated pixels to accept?
      if num_anno_pixels / sample.get_area() <= self.MAX_FRACTION_ANNOTATED:
        return sample
    util.log.warn(
      "Tried %s times and could not sample an unannotated box" % max_attempts)
    return None
  
  @staticmethod
  def create_miner(camera, *miner_args):

    class RingCameraMiner(HardNegativeMiner):
      # We could perhaps choose these using camera intrinsics, but instead
      # we use the empircal distributon from existing annotations
      # See cache/data/argoverse/index/image_annos/Size_stats_by_Camera.html
      WIDTH_PIXELS_MU_STD = (111.894619, 128.690585)
      HEIGHT_PIXELS_MU_STD = (92.435195, 119.881747)

    class StereoCameraMiner(HardNegativeMiner):
      # We could perhaps choose these using camera intrinsics, but instead
      # we use the empircal distributon from existing annotations
      # See cache/data/argoverse/index/image_annos/Size_stats_by_Camera.html
      WIDTH_PIXELS_MU_STD = (204.346622, 192.078106)
      HEIGHT_PIXELS_MU_STD = (204.070778, 212.397835)

    from argoverse.utils import camera_stats
    if camera in camera_stats.RING_CAMERA_LIST:
      return RingCameraMiner(*miner_args)
    elif camera in camera_stats.STEREO_CAMERA_LIST:
      return StereoCameraMiner(*miner_args)
    else:
      raise ValueError("Unknown camera: %s" % camera)




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

  def __init__(self, root_dir, log_name, FIXTURES=None):
    """Create a new loader.
    
    Args:
      root_dir: string, path to a directory containing log directories,
        e.g. /media/data/path/to/argoverse/argoverse-tracking/train1
      log_name: string, the name of the log to load,
        e.g. 5ab2697b-6e3e-3454-a36a-aba2c6f27818
      FIXTURES: `Fixtures` class that specifies paths to source data fixtures.
    """

    self.FIXTURES = FIXTURES or Fixtures

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

  @property
  def timestamp_to_pose_path(self):
    if not hasattr(self, '_timestamp_to_pose'):
      pose_paths = util.all_files_recursive(
        os.path.join(self.root_dir, self.current_log, 'poses'))
      timestamp_to_pose = {}
      for path in pose_paths:
        timestamp_to_pose[get_nanostamp_from_json_path(path)] = path
      self._timestamp_to_pose = timestamp_to_pose
    return self._timestamp_to_pose

  def get_up_lidar_extrinsic_matrix(self):
    if not hasattr(self, '_lidar_to_extrinsic'):
      import json
      config = json.load(open(self.calib_filename))
      self._lidar_to_extrinsic = {}
      for lidar in ('vehicle_SE3_up_lidar_', 'vehicle_SE3_down_lidar_'):
        self._lidar_to_extrinsic[lidar] = \
          get_lidar_extrinsic_matrix(config, lidar)
    
    # For now we just support getting the up lidar ...
    return self._lidar_to_extrinsic['vehicle_SE3_up_lidar_']

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
    return path, timestamp
  
  def get_nearest_lidar_sweep_id(self, timestamp):
    """Return the index of the lidar sweep and its timestamp in this log that
    either matches exactly or is closest to `timestamp`."""
    diff, idx = min(
              (abs(timestamp - t), idx)
              for idx, t in enumerate(self.lidar_timestamp_list))
    assert diff < 1e9, \
      "Could not find a cloud within 1 sec of %s, diff %s" % (timestamp, diff)
    return idx, self.lidar_timestamp_list[idx]

  def get_nearest_lidar_sweep(self, timestamp):
    idx, lidar_t = self.get_nearest_lidar_sweep_id(timestamp)
    @klepto.lru_cache(maxsize=10)
    def get_lidar(idx):
      return self.get_lidar(idx)
    cloud = get_lidar(idx)
    return cloud, lidar_t

  def get_nearest_label_objects(self, timestamp):
    """Load and return the `ObjectLabelRecord`s nearest to `timestamp`;
    provide either an exact match or choose the closest available."""

    idx, _ = self.get_nearest_lidar_sweep_id(timestamp)
    if idx >= len(self.label_list):
      # This most often happens on test split examples where
      # Argoverse does not yet include labels
      util.log.debug(
        "Log %s has %s labels but %s lidar sweeps; idx %s out of range" % (
          self.current_log, len(self.label_list),
          len(self.lidar_timestamp_list), idx))
      return []

    import argoverse.data_loading.object_label_record as object_label
    objs = object_label.read_label(self.label_list[idx])
      # NB: the above actually reads a *list* of label objects :P
    
    # Some of the labels are complete junk.  Argoverse filters these
    # interally in scattered places.  Let's do that in one place here.
    objs = [
      obj for obj in objs
      if not (
        np.isnan(obj.quaternion).any() or 
        np.isnan(obj.translation).any())
    ]
    
    # We hide the object timestamp in the label; I guess the Argoverse
    # authors didn't think the timestamp was important :P
    label_t = self.lidar_timestamp_list[idx]
    for obj in objs:
      obj.timestamp = label_t
    
    return objs

  def get_maybe_motion_corrected_cloud(self, timestamp):
    """Similar to `get_lidar()` but motion-corrects the entire cloud
    to (likely camera-time) `timestamp`.  Return also True if
    motion corrected."""
    cloud, lidar_t = self.get_nearest_lidar_sweep(timestamp)
    try:
      return self.get_motion_corrected_pts(cloud, lidar_t, timestamp), True
    except MissingPose:
      return cloud, False

  def get_maybe_motion_corrected_cloud_and_time(self, timestamp):
    """Similar to `get_lidar()` but motion-corrects the entire cloud
    to (likely camera-time) `timestamp`.  Return also the lidar timestamp."""
    cloud, lidar_t = self.get_nearest_lidar_sweep(timestamp)
    try:
      return self.get_motion_corrected_pts(cloud, lidar_t, timestamp), True
    except MissingPose:
      return cloud, lidar_t

  def get_cloud_in_image(
        self,
        camera,
        timestamp,
        motion_corrected=True,
        viewport=None):
    if motion_corrected:
      cloud, motion_corrected = \
        self.get_maybe_motion_corrected_cloud(timestamp)
    else:
      cloud, lidar_t = self.get_nearest_lidar_sweep(timestamp)
      motion_corrected = False

    calib = self.get_calibration(camera)

    if not viewport:
      viewport = common.BBox.of_size(*get_image_width_height(camera))

    # Limits of the cloud to crop
    x, y, w, h = (
      viewport.x, viewport.y,
      viewport.width, viewport.height)

    # Per the argoverse recommendation, this should be safe:
    # https://github.com/argoai/argoverse-api/blob/master/demo_usage/argoverse_tracking_tutorial.ipynb
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
    uv = uv.T

    # Correct for image origin if this frame is a crop
    uv -= np.array([viewport.x, viewport.y, 0])
    return uv, motion_corrected

  def get_city_to_ego(self, timestamp):
    from argoverse.data_loading.pose_loader import \
      get_city_SE3_egovehicle_at_sensor_t
    
    city_to_ego = get_city_SE3_egovehicle_at_sensor_t(
      timestamp, self.root_dir, self.current_log)
    if city_to_ego is None:
      raise MissingPose
    
    return city_to_ego

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
      raise MissingPose

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when the LiDAR sweep was recorded.
    city_SE3_ego_pts_t = get_city_SE3_egovehicle_at_sensor_t(
                              pts_timestamp,
                              self.root_dir,
                              self.current_log)
    if city_SE3_ego_pts_t is None:
      raise MissingPose
    
    # Argoverse SE3 does not want homogenous coords
    pts = np.copy(pts)
    if pts.shape[-1] == 4:
      pts = pts.T[:, :3]

    ego_dest_t_SE3_ego_pts_t = \
      city_SE3_ego_dest_t.inverse().right_multiply_with_se3(city_SE3_ego_pts_t)
    pts = ego_dest_t_SE3_ego_pts_t.transform_point_cloud(pts)
    
    from argoverse.utils.calibration import point_cloud_to_homogeneous
    return pts
  
  def print_sensor_sample_rates(self):
    """Print a report to stdout describing the sample rates of all sensors,
    labels, and localization objects."""

    def to_sec(timestamp):
      # Timestamps are in nanoseconds GPS time (but the epoch looks wrong--
      # drives look recorded in 1990).
      return timestamp * 1e-9

    def get_ts_from_json_path(path):
      nanostamp = int(path.split('_')[-1].rstrip('.json'))
      return to_sec(nanostamp)

    name_to_ts = {}

    # Labels
    name_to_ts['labels'] = np.array(
      sorted(get_ts_from_json_path(p) for p in self.label_list))
    
    # Poses
    pose_paths = util.all_files_recursive(
      os.path.join(self.root_dir, self.current_log, 'poses'))
    name_to_ts['ego_pose'] = np.array(
      sorted(get_ts_from_json_path(p) for p in pose_paths))
    
    # Lidar
    name_to_ts['lidar_fused'] = np.array(
      sorted(to_sec(t) for t in self.lidar_timestamp_list))
    
    # Cameras
    for camera in self.image_timestamp_list.keys():
      name_to_ts[camera] = np.array(
        sorted(to_sec(t) for t in self.image_timestamp_list[camera]))

    # Aggregate
    import itertools
    all_ts = sorted(itertools.chain.from_iterable(name_to_ts.values()))
    from datetime import datetime
    dt = datetime.utcfromtimestamp(all_ts[0])
    start = dt.strftime('%Y-%m-%d %H:%M:%S')
    duration = (all_ts[-1] - all_ts[0])

    # Print a report
    print('---')
    print('---')

    print('Log %s' % (self.current_log))
    print('Start %s \tDuration %s sec' % (start, duration))
    import pandas as pd
    from collections import OrderedDict
    rows = []
    for name in sorted(name_to_ts.keys()):
      def get_series(name):
        return np.array(sorted(name_to_ts[name]))
      
      series = get_series(name)
      freqs = series[1:] - series[:-1]

      lidar_series = get_series('lidar_fused')
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

    # ---
    # ---
    # Log 02cf0ce1-699a-373b-86c0-eb6fd5f4697a
    # Start 1980-01-06 01:01:34       Duration 15.95070230960846 sec
    #                 Series     Freq Hz  Diff Lidar (msec)   Duration  Support
    # 0             ego_pose  229.456981          27.981463  15.950702     3661
    # 1               labels   10.000169           0.000000  15.599737      157
    # 2          lidar_fused   10.000169           0.000000  15.599737      157
    # 3    ring_front_center   30.030025          27.547860  15.817503      476
    # 4      ring_front_left   30.030027          27.547829  15.817501      476
    # 5     ring_front_right   30.030026          27.547906  15.817502      476
    # 6       ring_rear_left   30.030031          27.547705  15.817500      476
    # 7      ring_rear_right   30.030029          27.547684  15.817500      476
    # 8       ring_side_left   30.030031          27.547732  15.817499      476
    # 9      ring_side_right   30.030036          27.547605  15.817497      476
    # 10   stereo_front_left    5.005005          15.940089  15.784199       80
    # 11  stereo_front_right    5.005004          15.940806  15.784202       80
    # ---
    # ---
    # Log 3138907e-1f8a-362f-8f3d-773f795a0d01
    # Start 1980-01-06 00:58:31       Duration 15.950732171535492 sec
    #                 Series     Freq Hz  Diff Lidar (msec)   Duration  Support
    # 0             ego_pose  229.865559          26.294311  15.922351     3661
    # 1               labels    9.999661           0.000000  15.600529      157
    # 2          lidar_fused    9.999661           0.000000  15.600529      157
    # 3    ring_front_center   30.030021          23.791747  15.584405      469
    # 4      ring_front_left   30.029949          23.787438  15.584442      469
    # 5     ring_front_right   30.030032          23.791692  15.584399      469
    # 6       ring_rear_left   30.030028          23.791658  15.584401      469
    # 7      ring_rear_right   30.030034          23.791675  15.584398      469
    # 8       ring_side_left   30.030032          23.791355  15.584399      469
    # 9      ring_side_right   30.030036          23.791633  15.584397      469
    # 10   stereo_front_left    5.005004          36.802685  15.584402       79
    # 11  stereo_front_right    5.005006          36.803061  15.584398       79
    # ---
    # ---
    # Log 70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c
    # Start 1980-01-06 03:12:41       Duration 15.950706362724304 sec
    #                 Series     Freq Hz  Diff Lidar (msec)   Duration  Support
    # 0             ego_pose  228.516526          28.184812  15.950706     3646
    # 1               labels    9.999913           0.000000  15.600136      157
    # 2          lidar_fused    9.999913           0.000000  15.600136      157
    # 3    ring_front_center   30.030032          26.389656  15.617699      470
    # 4      ring_front_left   30.030032          26.389745  15.617699      470
    # 5     ring_front_right   30.030029          26.389670  15.617700      470
    # 6       ring_rear_left   30.030024          26.389670  15.617703      470
    # 7      ring_rear_right   30.030033          26.389734  15.617698      470
    # 8       ring_side_left   30.030030          26.389771  15.617700      470
    # 9      ring_side_right   30.030031          26.389730  15.617699      470
    # 10   stereo_front_left    5.005004          15.282329  15.384604       78
    # 11  stereo_front_right    5.005005          15.282405  15.384600       78



###
### Data
###

class Fixtures(object):

  # All Argoverse tarballs served from here
  BASE_TARBALL_URL = "https://s3.amazonaws.com/argoai-argoverse"

  # If you happen to have a local copy of the tarballs, use this instead:
  #BASE_TARBALL_URL = "file:///tmp/argotars"

  ###
  ### NB: we omit the forecasting tarballs because they appear to exclude
  ### sensor data (and are therefore not useful for research).
  ###

  TRACKING_SAMPLE = "tracking_sample.tar.gz"

  SAMPLE_TARBALLS = (
    TRACKING_SAMPLE,
    # "forecasting_sample.tar.gz", Ignore forecasting for now
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
  # Ignore forecasting for now
  # (
  #   "forecasting_train.tar.gz",
  #   "forecasting_val.tar.gz",
  #   "forecasting_test.tar.gz",
  # )

  MAP_TARBALLS = (
    "hd_maps.tar.gz",
  )

  SPLITS = ('train', 'test', 'val', 'sample')

  TRAIN_TEST_SPLITS = ('train', 'val')
    # 'test' has no labels and 'sample' duplicates part of 'train'

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'argoverse')



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
      uri = av.URI.from_str(uri)
    tarball_name, log_id = uri.segment_id.split('|')
    return cls._get_loader(tarball_name, log_id)
  
  @classmethod
  @klepto.inf_cache(ignore=(0,))
  def _get_loader(cls, tarball_name, log_id):
    """Argoverse log loaders are paifully expensive because they
    scrape the filesystem in the ctor.  Here we try to cache them
    as much as possible."""
    loader = None # Build this
    # Need to find the dir corresponding to log_id
    base_path = cls.tarball_dir(tarball_name)
    for log_dir in cls.get_log_dirs(base_path):
      cur_log_id = os.path.split(log_dir)[-1]
      if cur_log_id == log_id:
        loader = AUTrackingLoader(
          os.path.dirname(log_dir), log_id, FIXTURES=cls)
        break
    
    assert loader, "Could not find log %s in %s" % (log_id, base_path)
    return loader

  # @classmethod
  # def get_frame(cls, uri):
  #   """Factory function for constructing an AVFrame that uses this Fixtures
  #   instance as fixtures."""
  #   return AVFrame(uri=uri, FIXTURES=cls)


    # # These stupid Loader objects are painfully, painfully expensive because
    # # they do massive filesystem stat()s.  Thus we try to cache using a file
    # # cache that can be shared *across Spark workers.*  The user may need to
    # # bust this cache in the event of code change.  TODO: build a spark thing so this can be done once per run ~~~~~~~
    
    
    #   key = tarball_name + '.' + log_id
    #   if key not in cls._loader_cache_map:
    #     loader = None # Build this
    #     # Need to find the dir corresponding to log_id
    #     base_path = cls.tarball_dir(tarball_name)
    #     for log_dir in cls.get_log_dirs(base_path):
    #       cur_log_id = os.path.split(log_dir)[-1]
    #       if cur_log_id == log_id:
    #         loader = AUTrackingLoader(os.path.dirname(log_dir), log_id)
    #         break
        
    #     assert loader, "Could not find log %s in %s" % (log_id, base_path)

    #     cls._loader_cache_map[key] = loader
    #     cls._loader_cache_map.sync()

    #   loader = cls._loader_cache_map[key]
    # return loader

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
        yield av.URI(
          split=split,
          dataset='argoverse',
          segment_id=tarball + '|' + log_id)

  @classmethod
  def get_image_frame_uris(cls, log_frame_uri):
    loader = cls.get_loader(log_frame_uri)
    base_uri = str(log_frame_uri)
    for camera, ts_to_path in loader.timestamp_image_dict.items():
      for ts in ts_to_path.keys():
        uri = av.URI.from_str(base_uri, camera=camera, timestamp=ts)
        yield uri

  

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



###
### StampedDatumTable Impl
###

class StampedDatumTable(av.StampedDatumTableBase):

  FIXTURES = Fixtures

  MERGE_AND_REPLACE_BIKES = True
  MAX_RIDDEN_BIKE_DISTANCE_METERS = 5
    # For each frame of labels, try to associate bikes with riders, and replace
    # source cuboids with their associated ones.

  SYNC_LIDAR_TO_CAMERA = False
  MOTION_CORRECTED_CUBOIDS = False # TODO deleteme?  use nusc-style interp? ~~~~~~~~~~~~

  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/argoverse_datum_table'

  @classmethod
  def _create_datum_rdds(cls, spark):

    segment_uris = cls.get_segment_uris()
    from collections import defaultdict
    split_to_count = defaultdict(int)
    for uri in segment_uris:
      split_to_count[uri.split] += 1
    util.log.info(
      "Found %s segments, splits: %s" % (
          len(segment_uris), dict(split_to_count)))


    PARTITIONS_PER_SEGMENT = 4 * os.cpu_count()
    PARTITIONS_PER_TASK = os.cpu_count()

    datum_rdds = []
    for segment_uri in segment_uris:
      partition_chunks = util.ichunked(
        range(PARTITIONS_PER_SEGMENT), PARTITIONS_PER_TASK)
      for partitions in partition_chunks:
        task_rdd = spark.sparkContext.parallelize(
          [(segment_uri, partition) for partition in partitions])

        def gen_partition_datums(task):
          segment_uri, partition = task
          for i, uri in enumerate(cls.iter_uris_for_segment(segment_uri)):
            if (i % PARTITIONS_PER_SEGMENT) == partition:
              yield cls.create_stamped_datum(uri)
        
        datum_rdd = task_rdd.flatMap(gen_partition_datums)
        datum_rdds.append(datum_rdd)
    return datum_rdds


  ## Public API

  @classmethod
  def get_segment_uris(cls):
    splits = cls.FIXTURES.TRAIN_TEST_SPLITS
    return list(itertools.chain.from_iterable(
      cls.FIXTURES.get_log_uris(split)
      for split in splits))

  @classmethod
  def iter_uris_for_segment(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)

    ## Poses
    for timestamp, path in loader.timestamp_to_pose_path.items():
      uri = copy.deepcopy(uri)
      uri.topic = 'ego_pose'
      uri.timestamp = timestamp
      yield uri
    
    ## Lidar
    # TODO SYNC_LIDAR_TO_CAMERA
    for timestamp in loader.lidar_timestamp_list:
      uri = copy.deepcopy(uri)
      uri.topic = 'lidar|fused'
      uri.timestamp = timestamp
      yield uri
    
    ## Cameras
    for camera in loader.timestamp_image_dict.keys():
      for timestamp in loader.timestamp_image_dict[camera].keys():
        uri = copy.deepcopy(uri)
        uri.topic = 'camera|' + camera
        uri.timestamp = timestamp
        yield uri

    ## Labels
    # TODO MOTION_CORRECTED_CUBOIDS
    label_paths = loader.label_list
    for path in label_paths:
      uri = copy.deepcopy(uri)
      uri.topic = 'labels|cuboids'
      uri.timestamp = get_nanostamp_from_json_path(path)
      yield uri

  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic.startswith('camera'):
      return cls.__create_camera_image(uri)
    elif uri.topic.startswith('lidar'):
      return cls.__create_point_cloud(uri)
    elif uri.topic == 'ego_pose':
      return cls.__create_ego_pose(uri)
    elif uri.topic == 'labels|cuboids':
      return cls.__create_cuboids_in_ego(uri)
    else:
      raise ValueError(uri)
  

  ## Support

  @classmethod
  def __get_ego_pose(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)
    
    try:
      city_to_ego = loader.get_city_to_ego(uri.timestamp)
    except MissingPose:
      diff, best_ts = min(
        (abs(pose_t - uri.timestamp), pose_t)
        for pose_t in loader.timestamp_to_pose_path.keys())
      city_to_ego = loader.get_city_to_ego(best_ts)
      util.log.warn(
        "Using approx pose (stale by %s sec) for %s" % (diff * 1e-9, str(uri)))

    ego_pose = av.Transform(
                  rotation=city_to_ego.rotation,
                  translation=city_to_ego.translation,
                  src_frame='city',
                  dest_frame='ego')
    return ego_pose

  @classmethod
  def __create_camera_image(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)

    camera = uri.topic[len('camera|'):]
    path = loader.timestamp_image_dict[camera][uri.timestamp]
    calib = loader.get_calibration(camera)
    
    cam_from_ego = av.Transform(
                        rotation=calib.R,
                        translation=calib.T,
                        src_frame='ego',
                        dest_frame=camera)
    ego_pose = cls.__get_ego_pose(uri)

    viewport = uri.get_viewport()
    w, h = get_image_width_height(camera)
    if not viewport:
      viewport = common.BBox.of_size(w, h)

    K = calib.K[:3, :3]
    ci = av.CameraImage(
        camera_name=camera,
        image_jpeg=bytearray(open(path, 'rb').read()),
        height=h,
        width=w,
        viewport=viewport,
        timestamp=uri.timestamp,
        ego_pose=ego_pose,
        cam_from_ego=cam_from_ego,
        K=K,
        principal_axis_in_ego=get_camera_normal(calib),
    )

    return av.StampedDatum.from_uri(uri, camera_image=ci)
  
  @classmethod
  def __create_point_cloud(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)

    if cls.SYNC_LIDAR_TO_CAMERA:
      cloud, lidar_timestamp = \
        loader.get_maybe_motion_corrected_cloud(uri.timestamp)
    else:
      cloud, lidar_timestamp = loader.get_nearest_lidar_sweep(uri.timestamp)
      motion_corrected = False
      assert lidar_timestamp == uri.timestamp
    
    extrinsic = loader.get_up_lidar_extrinsic_matrix()
    ego_to_sensor = av.Transform(
                        rotation=extrinsic[0:3, 0:3],
                        translation=extrinsic[0:3, 3],
                        src_frame='ego',
                        dest_frame='lidar_fused')
    ego_pose = cls.__get_ego_pose(uri)
    pc = av.PointCloud(
        sensor_name='lidar_fused',
        timestamp=lidar_timestamp, # NB: use *real* lidar timestamp if given
        cloud=cloud,
        motion_corrected=motion_corrected,
        ego_to_sensor=ego_to_sensor,
        ego_pose=ego_pose,
    )
    return av.StampedDatum.from_uri(uri, point_cloud=pc)
  
  @classmethod
  def __create_ego_pose(cls, uri):
    ego_pose = cls.__get_ego_pose(uri)
    return av.StampedDatum.from_uri(uri, transform=ego_pose)
  
  @classmethod
  def __create_cuboids_in_ego(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)

    olrs = loader.get_nearest_label_objects(uri.timestamp)
    cuboids = []
    for olr in olrs:
      cuboid = av.Cuboid()

      ## Core
      from au.fixtures.datasets.av import ARGOVERSE_CATEGORY_TO_AU_AV_CATEGORY
      cuboid.track_id = olr.track_id
      cuboid.category_name = olr.label_class
      cuboid.au_category = ARGOVERSE_CATEGORY_TO_AU_AV_CATEGORY.get(
                                              olr.label_class, 'background')
      cuboid.timestamp = olr.timestamp
        # NB: this timestamp is embedded in the label itself, perhaps might
        # not be equal to uri.timestamp
      cuboid.extra = {
        'argoverse_occlusion': str(olr.occlusion),
          # In practice, the value in this field is not meaningful
      }

      ## Box
      cuboid.box3d = olr.as_3d_bbox()
      cuboid.motion_corrected = False
      if cls.MOTION_CORRECTED_CUBOIDS:
        try:
          cuboid.box3d = loader.get_motion_corrected_pts(
                                    cuboid.box3d,
                                    olr.timestamp,
                                    uri.timestamp)
          cuboid.motion_corrected = True
        except MissingPose:
          # Garbage! Ignore.
          pass
      
      cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))

      ## Pose Etc
      cuboid.length_meters = float(olr.length)
      cuboid.width_meters = float(olr.width)
      cuboid.height_meters = float(olr.height)

      from argoverse.utils.transform import quat2rotmat
      rotmat = quat2rotmat(olr.quaternion)
        # NB: must use quat2rotmat due to Argo-specific quaternion encoding
      
      from scipy.spatial.transform import Rotation as R
      cuboid.obj_from_ego = av.Transform(
                                rotation=rotmat,
                                translation=olr.translation,
                                src_frame='ego',
                                dest_frame='obj')
      cuboid.ego_pose = cls.__get_ego_pose(uri)

      cuboids.append(cuboid)
      
    if cls.MERGE_AND_REPLACE_BIKES:
      cuboids = cls.__get_bikes_merged(cuboids)

    return av.StampedDatum.from_uri(uri, cuboids=cuboids)

  @classmethod
  def __get_bikes_merged(cls, cuboids):
    bikes = [c for c in cuboids if c.category_name in BIKE]
    riders = [c for c in cuboids if c.category_name in RIDER]

    if not bikes:
      return cuboids
    
    cuboids_out = [c for c in cuboids if c.category_name not in (BIKE + RIDER)]

    # The best pair has smallest euclidean distance between centroids
    def l2_dist(c1, c2):
      return np.linalg.norm(
        c1.obj_from_ego.translation - c2.obj_from_ego.translation)
    
    # Each rider gets assigned the nearest bike.  Note that not all bikes may
    # have riders.
    tracks_kept = set(c.track_id for c in cuboids_out)
    for rider in riders:
      distance, best_bike = min(
                              (l2_dist(rider, bike), bike)
                              for bike in bikes)

      if distance <= cls.MAX_RIDDEN_BIKE_DISTANCE_METERS:
        # Merge!
        merged_cuboid = av.Cuboid.get_merged(rider, best_bike)
        
        if rider.category_name == 'BICYCLIST':
          merged_cuboid.au_category = 'bike_with_rider'
        else: # Motorcycle, maybe moped?
          merged_cuboid.au_category = 'motorcycle_with_rider'
        
        cuboids_out.append(merged_cuboid)
        tracks_kept.add(rider.track_id)
        tracks_kept.add(best_bike.track_id)

    # Add back in any *unmerged* bikes & riders
    for c in bikes + riders:
      if c.track_id in tracks_kept:
        continue

      if c.category_name in ("BICYCLE",):
        c.au_category = 'bike_no_rider'
      elif c.category_name in ("MOPED", "MOTORCYCLE"):
        c.au_category = 'motorcycle_no_rider'
      elif c.category_name in RIDER:
        c.au_category = 'ped'
          # Don't drop unassociated riders entirely
      
      cuboids_out.append(c)
    return cuboids_out

    



###
### FrameTable Impl
###

class FrameTable(av.FrameTableBase):
  """TODO"""

  FIXTURES = Fixtures

  PROJECT_CLOUDS_TO_CAM = True
  PROJECT_CUBOIDS_TO_CAM = True
  IGNORE_INVISIBLE_CUBOIDS = True
  MOTION_CORRECTED_POINTS = True
  FILTER_MISSING_POSE = True

  MERGE_AND_REPLACE_BIKES = True
  MAX_RIDDEN_BIKE_DISTANCE_METERS = 5

  SETUP_URIS_PER_CHUNK = 1000

  ## Subclass API

  # @classmethod
  # def table_root(cls):
  #   return '/outer_root/media/seagates-ext4/au_datas/frame_table'

  @classmethod
  def _create_frame_rdds(cls, spark):
    uri_rdd = cls._get_uri_rdd(spark)

    # Try to group frames from segments together to make partitioning easier
    # and result in fewer files
    uri_rdd = uri_rdd.sortBy(lambda uri: uri.segment_id)
    
    frame_rdds = []
    uris = uri_rdd.toLocalIterator()
    for uri_chunk in util.ichunked(uris, cls.SETUP_URIS_PER_CHUNK):
      chunk_uri_rdd = spark.sparkContext.parallelize(uri_chunk)
      # create_frame = util.ThruputObserver.wrap_func(
      #                       cls.create_frame,
      #                       name='create_frame',
      #                       log_on_del=True)

      frame_rdd = chunk_uri_rdd.map(cls.create_frame)

      if cls.FILTER_MISSING_POSE:
        def has_valid_ego_pose(frame):
          return not frame.world_to_ego.is_identity()
        frame_rdd = frame_rdd.filter(has_valid_ego_pose)

      frame_rdds.append(frame_rdd)
    return frame_rdds

  ## Support
  
  @classmethod
  def _get_uri_rdd(cls, spark, splits=None):
    if not splits:
      splits = cls.FIXTURES.TRAIN_TEST_SPLITS

    util.log.info("Building frame table for splits %s" % (splits,))

    # Be careful to hint to Spark how to parallelize reads. Instantiating
    # log readers is expensive, so we have Spark do that in parallel.
    log_uris = list(
              itertools.chain.from_iterable(
                    cls.FIXTURES.get_log_uris(split)
                    for split in splits))
    util.log.info("... reading from %s logs ..." % len(log_uris))
    log_uri_rdd = spark.sparkContext.parallelize(
                            log_uris, numSlices=len(log_uris))
    uri_rdd = log_uri_rdd.flatMap(cls.FIXTURES.get_image_frame_uris).cache()

    util.log.info("... computed %s URIs." % uri_rdd.count())
    return uri_rdd

  @classmethod
  def create_frame(cls, uri):
    f = av.Frame(uri=uri)
    uri = f.uri
    f.world_to_ego = cls._get_ego_pose(uri)
    f.camera_images = cls._get_camera_images(uri)
    # f.clouds = cls._get_clouds(uri)
    # f.cuboids = cls._get_cuboids(uri)
    return f

  @classmethod
  def _get_ego_pose(cls, uri):
    loader = cls.FIXTURES.get_loader(uri)
    try:
      city_to_ego_se3 = loader.get_city_to_ego(uri.timestamp)
      return av.Transform(
              rotation=city_to_ego_se3.rotation,
              translation=city_to_ego_se3.translation)
    except MissingPose:
      return av.Transform()

  @classmethod
  def _get_camera_images(cls, uri):
    """Fetch image(s) for the camera(s) specified in `uri`; if no cameras are
    specified then fetch images for *all* cameras.  Include a projection of
    lidar points into the image; motion-correct the cloud by default."""
    loader = cls.FIXTURES.get_loader(uri)
    cameras = []
    if uri.camera:
      cameras = [uri.camera]
    else:
      from argoverse.utils import camera_stats
      cameras = CAMERA_LIST

    cis = []
    for camera in cameras:
      path, timestamp = loader.get_nearest_image_path(camera, uri.timestamp)
      calib = loader.get_calibration(camera)
      cam_from_ego = av.Transform(
                        rotation=calib.R, translation=calib.T.reshape((3, 1)))

      viewport = uri.get_viewport()
      w, h = get_image_width_height(camera)
      if not viewport:
        viewport = common.BBox.of_size(w, h)

      K = calib.K[:3, :3]
      ci = av.CameraImage(
        camera_name=camera,
        image_jpeg=bytearray(open(path, 'rb').read()),
        height=h,
        width=w,
        viewport=viewport,
        timestamp=timestamp,
        cam_from_ego=cam_from_ego,
        K=K,
        principal_axis_in_ego=get_camera_normal(calib),
      )

      if cls.PROJECT_CLOUDS_TO_CAM:
        cls.__fill_cloud_in_cam(loader, timestamp, ci)
      
      if cls.PROJECT_CUBOIDS_TO_CAM:
        cls.__fill_cuboids_in_cam(uri, timestamp, ci)

      cis.append(ci)
    return cis

  @classmethod
  def __fill_cloud_in_cam(cls, loader, timestamp, camera_image):
    if cls.MOTION_CORRECTED_POINTS:
      cloud, motion_corrected = \
        loader.get_maybe_motion_corrected_cloud(timestamp)
    else:
      cloud, timestamp = loader.get_nearest_lidar_sweep(timestamp)
      motion_corrected = False
    
    # Put the cloud in the camera frame
    cloud = camera_image.project_ego_to_image(cloud, omit_offscreen=True)
    camera_image.cloud = av.PointCloud(
        sensor_name='lidar_in_' + camera_image.camera_name,
        timestamp=timestamp, # NB: use *real* lidar timestamp if given
        cloud=cloud,
        motion_corrected=motion_corrected,
        ego_to_sensor=copy.deepcopy(camera_image.cam_from_ego),
          # Points are now in camera frame.  NB: for Argoverse, that
          # means there might also include an axis change versus the
          # ego frame
      )
  
  @classmethod
  def __fill_cuboids_in_cam(cls, uri, timestamp, camera_image):
    cuboids = cls._get_cuboids(uri, timestamp=timestamp)
    for cuboid in cuboids:      
      bbox = camera_image.project_cuboid_to_bbox(cuboid)

      if cls.IGNORE_INVISIBLE_CUBOIDS and not bbox.is_visible:
        continue

      camera_image.bboxes.append(bbox)
    
  @classmethod
  def _get_cuboids(cls, uri, timestamp=None):
    loader = cls.FIXTURES.get_loader(uri)
    if not timestamp:
      timestamp = uri.timestamp
    olrs = loader.get_nearest_label_objects(timestamp)
    cuboids = []
    for olr in olrs:
      cuboid = av.Cuboid()
      cls.__fill_core(cuboid, olr)
      cls.__fill_pts(loader, timestamp, cuboid, olr)
      cls.__fill_pose(cuboid, olr)
      cuboids.append(cuboid)
    
    if cls.MERGE_AND_REPLACE_BIKES:
      cuboids = cls.__get_bikes_merged(cuboids)
    
    return cuboids



  # def _get_clouds(cls, uri):
  #   loader = cls.FIXTURES.get_loader(uri)
  #   timestamp = uri.timestamp
  #   if cls.MOTION_CORRECTED_POINTS:
  #     cloud, motion_corrected = \
  #       loader.get_maybe_motion_corrected_cloud(timestamp)
  #   else:
  #     cloud, timestamp = loader.get_nearest_lidar_sweep(timestamp)
  #     motion_corrected = False
    
  #   # TODO: split top and bottom lidars
  #   return [
  #     av.PointCloud(
  #       sensor_name='lidar',
  #       timestamp=timestamp,
  #       cloud=cloud,
  #       motion_corrected=motion_corrected,
  #       # Leave ego_to_sensor as $I$; points are in ego frame
  #     )]

  # @classmethod
  # def _get_cuboids(cls, uri):
  #   """Construct and return a list of `av.Cuboid` instances from the given
  #   Argoverse `ObjectLabelRecord` instance.  Labels are in lidar space-time
  #   and *not* camera space-time; therefore, transforming labels into
  #   the camera domain requires (to be most precise) correction for the
  #   egomotion of the robot.  This correction can be substantial (~20cm)
  #   at high robot speed.  Apply this correction only if `motion_corrected`.
  #   """
  #   loader = cls.FIXTURES.get_loader(uri)
  #   calib = loader.get_calibration(uri.camera)
  #   olrs = loader.get_nearest_label_objects(uri.timestamp)
  #   cuboids = []
  #   for olr in olrs:
  #     cuboid = av.Cuboid()
  #     cls.__fill_core(cuboid, olr)
  #     cls.__fill_pts(loader, uri, cuboid, olr)
  #     cls.__fill_pose(calib, cuboid, olr)
  #     cuboids.append(cuboid)
  #   return cuboids

  ### Cuboid Utils

  @classmethod
  def __get_bikes_merged(cls, cuboids):
    bikes = [c for c in cuboids if c.category_name in BIKE]
    riders = [c for c in cuboids if c.category_name in RIDER]

    if not bikes:
      return cuboids
    
    cuboids_out = [c for c in cuboids if c.category_name not in (BIKE + RIDER)]

    # The best pair has smallest euclidean distance between centroids
    def l2_dist(c1, c2):
      return np.linalg.norm(
        c1.obj_from_ego.translation - c2.obj_from_ego.translation)
    
    # Each rider gets assigned the nearest bike.  Note that not all bikes may
    # not have riders.
    tracks_kept = set(c.track_id for c in cuboids_out)
    for rider in riders:
      distance, best_bike = min(
                              (l2_dist(rider, bike), bike)
                              for bike in bikes)

      if distance <= cls.MAX_RIDDEN_BIKE_DISTANCE_METERS:
        # Merge!
        merged_cuboid = av.Cuboid.get_merged(rider, best_bike)
        
        if rider.category_name == 'BICYCLIST':
          merged_cuboid.au_category = 'bike_with_rider'
        else: # Motorcycle, maybe moped?
          merged_cuboid.au_category = 'motorcycle_with_rider'
        
        cuboids_out.append(merged_cuboid)
        tracks_kept.add(rider.track_id)
        tracks_kept.add(best_bike.track_id)
    
    # Add back in any unmerged bikes & riders
    for c in bikes + riders:
      if c.track_id in tracks_kept:
        continue

      if c.category_name in ("BICYCLE",):
        c.au_category = 'bike_no_rider'
      elif c.category_name in ("MOPED", "MOTORCYCLE"):
        c.au_category = 'motorcycle_no_rider'
      elif c.category_name in RIDER:
        c.au_category = 'ped'
      
      cuboids_out.append(c)
    return cuboids_out

  @classmethod
  def __fill_core(cls, cuboid, olr):
    from au.fixtures.datasets.av import ARGOVERSE_CATEGORY_TO_AU_AV_CATEGORY
    cuboid.track_id = olr.track_id
    cuboid.category_name = olr.label_class
    cuboid.au_category = ARGOVERSE_CATEGORY_TO_AU_AV_CATEGORY.get(
                                      olr.label_class, 'background')
    cuboid.timestamp = olr.timestamp
    cuboid.extra = {
      'argoverse_occlusion': str(olr.occlusion),
        # In practice, the value in this field is not meaningful
    }

  @classmethod
  def __fill_pts(cls, loader, target_timestamp, cuboid, olr):
    cuboid.box3d = olr.as_3d_bbox()
    cuboid.motion_corrected = False
    if cls.MOTION_CORRECTED_POINTS:
      try:
        cuboid.box3d = loader.get_motion_corrected_pts(
                                  cuboid.box3d,
                                  olr.timestamp,
                                  target_timestamp)
        cuboid.motion_corrected = True
      except MissingPose:
        # Garbage! Ignore.
        pass
    
    cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
  
  @classmethod
  def __fill_pose(cls, cuboid, olr):
    cuboid.length_meters = float(olr.length)
    cuboid.width_meters = float(olr.width)
    cuboid.height_meters = float(olr.height)

    from argoverse.utils.transform import quat2rotmat
    rotmat = quat2rotmat(olr.quaternion)
      # NB: must use quat2rotmat due to Argo-specific quaternion encoding
    
    from scipy.spatial.transform import Rotation as R
    cuboid.obj_from_ego = av.Transform(
      rotation=rotmat, translation=olr.translation.reshape((3, 1)))
  

class ImageAnnoTable(object):
  """A table of argoverse annotations projected into image space."""

  FIXTURES = Fixtures

  ## Public API

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
      if not bike_rows:
        return
      
      # The best pair has smallest euclidean distance between centroids
      def l2_dist(r1, r2):
        r1_ego_to_obj = r1.ego_to_obj.arr
        r2_ego_to_obj = r2.ego_to_obj.arr
        return float(np.linalg.norm(r2_ego_to_obj - r1_ego_to_obj))

      # Each rider gets assigned the nearest bike.  Note that bikes may not
      # have riders.
      for rider in rider_rows:
        try:
          distance, best_bike = min(
                          (l2_dist(rider, bike), bike)
                          for bike in bike_rows)
        except Exception as e:
          import sys
          print(str(sys.exc_info()))
          # TODO: getting spurious "numpy cast to bool" exceptions here?
          util.log.error("Bike rider assoc wat? %s" % e)
          continue
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
  def create_frame_uri_rdd(cls, spark, splits=None):
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
    uri_rdd = cls.create_frame_uri_rdd(spark, splits=splits)

    def iter_anno_rows(uri):
      # from collections import namedtuple ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # pt = namedtuple('pt', 'x y z')

      frame = cls.FIXTURES.get_frame(uri)
      for bbox in frame.image_bboxes:
        row = {}

        # Obj
        row.update(bbox.to_row_dict())
        # # TODO make spark accept numpy and numpy float64 things
        # row = box.to_dict()
        # IGNORE = ('cuboid_pts', 'cuboid_pts_image', 'ego_to_obj')
        # for attr in IGNORE:
        #   v = row.pop(attr)
        #   if attr == 'cuboid_pts_image':
        #     continue
        #   if hasattr(v, 'shape'):
        #     if len(v.shape) == 1:
        #       row[attr] = pt(*v.tolist())
        #     else:
        #       row[attr] = [pt(*v[r, :3].tolist()) for r in range(v.shape[0])]
        
        # Anno Context
        obj_uri = copy.deepcopy(frame.uri)
        obj_uri.track_id = bbox.track_id
        row.update(
          frame_uri=str(uri),
          uri=str(obj_uri),
          **obj_uri.to_dict())
        row.update(
          city=cls.FIXTURES.get_loader(uri).city_name,
          coarse_category=AV_OBJ_CLASS_TO_COARSE.get(bbox.category_name, ''))
        
        from pyspark.sql import Row
        yield Row(**row)
    
    row_rdd = uri_rdd.flatMap(iter_anno_rows)
    df = spark.createDataFrame(row_rdd)
    df = cls._impute_rider_for_bikes(spark, df)
    return df



class AnnoReports(object):
  """HTML reports mined from the argoverse dataset"""

  ANNO_TABLE = ImageAnnoTable

  ### For histogram reports
  NUM_BINS = 20

  # Show only this many examples for each bucket.  More
  # examples -> more images -> larger plot files.
  EXAMPLES_PER_BUCKET = 10

  # For each of these metrics in ImageAnnoTable, generate a distinct plot
  # for each sub-pivot column
  SPLIT_AND_CITY = ['split', 'city']
  CATEGORY_AND_CAMERA = ['category_name', 'coarse_category', 'camera']
  ALL_SUB_PIVOTS = SPLIT_AND_CITY + CATEGORY_AND_CAMERA
  METRIC_AND_SUB_PIVOTS = (
    ('distance_meters',                 ALL_SUB_PIVOTS),
    ('length_meters',                   SPLIT_AND_CITY + ['category_name']),
    ('relative_yaw_radians',            SPLIT_AND_CITY),
    ('relative_yaw_to_camera_radians',  ALL_SUB_PIVOTS),
    ('occlusion',                       SPLIT_AND_CITY),

    # Special handling! See below
    ('ridden_bike_distance',            ALL_SUB_PIVOTS),
  )

  @classmethod
  def create_reports(cls, spark=None, dest_dir=None):
    with Spark.sess(spark) as spark:
      dest_dir = dest_dir or cls.ANNO_TABLE.FIXTURES.image_annos_reports_root()
      util.mkdir(dest_dir)
      util.log.info("Creating annotation reports in %s ..." % dest_dir)

      ## First do overall stats reports
      for title, pdf in cls.get_overall_stats_dfs(spark):
        fname = title.replace(' ', '_') + '.html'
        with open(os.path.join(dest_dir, fname), 'w') as f:
          f.write(pdf.to_html())
        util.log.info("Saved simple report: \n%s\n%s\n\n" % (title, pdf))
      
      ## Now do histogram reports
      cls.save_histogram_reports(spark, dest_dir)      

  @classmethod
  def get_overall_stats_dfs(cls, spark):
    """Generate a sequence of title/pandas.Dataframe pairs detailing overall
    dataset statistics."""
    
    ## Register anno table
    anno_df = cls.ANNO_TABLE.as_df(spark)
    anno_df.createOrReplaceTempView("annos")

    ## Register raw anno table for special queries
    # We use a Dataframe that is similar to `anno_df` but (1) includes some
    # extra information and (2) is not serializable due to null columns.
    frame_uri_rdd = cls.ANNO_TABLE.create_frame_uri_rdd(spark).cache()
    def iter_all_bbox_rows(uri):
      uri_cells = FrameURI.from_str(uri).to_dict()
      loader = cls.ANNO_TABLE.FIXTURES.get_loader(uri)
      bboxes = loader.get_nearest_label_bboxes(uri)
      yield dict(frame_uri=uri, num_bboxes=len(bboxes), **uri_cells)
      
      for bbox in bboxes:
        row = dict(bbox.to_row_dict())
        row.update(**uri_cells)
        yield row
    
    raw_anno_row_rdd = frame_uri_rdd.flatMap(iter_all_bbox_rows)
    raw_anno_df = spark.createDataFrame(raw_anno_row_rdd, samplingRatio=0.5)
    raw_anno_df.createOrReplaceTempView("raw_annos")

    title_queries = (
      ("Size Stats by Split", """
          SELECT
            split,
            AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
            AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
            COUNT(*) AS num_annos,
            COUNT(DISTINCT frame_uri) AS num_frames
          FROM annos
          GROUP BY split
          ORDER BY split"""
      ),
      ("Size Stats by City", """
          SELECT
            city,
            AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
            AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
            COUNT(*) AS num_annos,
            COUNT(DISTINCT frame_uri) AS num_frames
          FROM annos
          GROUP BY city
          ORDER BY city"""
      ),
      ("Size Stats by Category", """
          SELECT
            category_name,
            AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
            AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
            COUNT(*) AS num_annos,
            COUNT(DISTINCT frame_uri) AS num_frames
          FROM annos
          GROUP BY category_name
          ORDER BY category_name"""
      ),
      ("Size Stats by Camera", """
          SELECT
            camera,
            AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
            AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
            COUNT(*) AS num_annos,
            COUNT(DISTINCT frame_uri) AS num_frames
          FROM annos
          GROUP BY camera
          ORDER BY camera"""
      ),
      ("Frames Per Track by Camera", """
          SELECT
            camera,
            AVG(num_frames) AS num_frames_mu,
            STD(num_frames) AS num_frames_std,
            SUM(num_frames) AS num_frames_total,
            COUNT(*) AS num_tracks_total
          FROM (
            SELECT
              camera, track_id, COUNT(DISTINCT frame_uri) AS num_frames
            FROM annos
            GROUP BY camera, track_id )
          GROUP BY camera
          ORDER BY camera"""
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
            FROM annos
            GROUP BY camera, frame_uri )
          GROUP BY camera
          ORDER BY camera"""
      ),
      ("Pedestrian Stats by Distance", """
          SELECT
            camera,
            10 * MOD(ROUND(distance_meters), 10) AS distance_m_bucket,
            AVG(width) AS w_pixels_mu, STD(width) AS w_pixels_std,
            AVG(height) AS h_pixels_mu, STD(height) AS h_pixels_std,
            COUNT(*) AS num_annos,
            COUNT(DISTINCT frame_uri) AS num_frames
          FROM annos
          WHERE
            category_name = 'PEDESTRIAN' AND 
            camera in ('ring_front_center', 'stereo_front_left')
          GROUP BY camera, distance_m_bucket
          ORDER BY camera ASC, distance_m_bucket ASC"""
      ),
      ("Associated & Unassociated Bike Riders by Split", """
          SELECT
            split,
            SUM(IF(ridden_bike_track_id != '', 1, 0)) AS associated_rider,
            SUM(IF(ridden_bike_track_id != '', 0, 1)) AS free_rider
          FROM annos
          WHERE
            category_name in ("BICYCLIST", "MOTORCYCLIST")
          GROUP BY split
          ORDER BY split"""
      ),
      ("Motion-Corrected Annos by Split", """
          SELECT
            split,
            SUM(IF(motion_corrected, 1, 0)) AS motion_corrected,
            SUM(IF(motion_corrected, 0, 1)) AS not_motion_corrected
          FROM annos
          GROUP BY split
          ORDER BY split"""
      ),

      ## Special queries on raw annos
      ("Tracks That Never Appear On Camera", """
          SELECT *
          FROM (
            SELECT
              FIRST(split) AS split,
              FIRST(log_id) AS log_id,
              track_id,
              SUM(IF(is_visible, 1, 0)) AS num_invisible,
              SUM(IF(is_visible, 0, 1)) AS num_visible
            FROM raw_annos
            GROUP BY track_id )
          WHERE num_visible == 0
          ORDER BY split, log_id, track_id"""
      ),
      ("Frames With No Annotations", """
          SELECT
            COUNT(DISTINCT frame_uri)
          FROM raw_annos
          WHERE num_bboxes == 0"""
      ),
    )
    for title, query in title_queries:
      util.log.info("Running %s ..." % title)
      yield title, spark.sql(query).toPandas()

  @classmethod
  def save_histogram_reports(cls, spark, dest_dir):
    ## Histogram reports
    
    num_plots = sum(len(spvs) for metric, spvs in cls.METRIC_AND_SUB_PIVOTS)
    util.log.info("Going to generate %s plots ..." % num_plots)
    t = util.ThruputObserver(name='plotting', n_total=num_plots)
    
    # Generate plots!
    anno_df = cls.ANNO_TABLE.as_df(spark)
    for metric, sub_pivots in cls.METRIC_AND_SUB_PIVOTS:
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
          NUM_BINS = cls.NUM_BINS
          SUB_PIVOT_COL = sub_pivot
          WIDTH = 1400
          TITLE = plot_name

          EXAMPLES_PER_BUCKET = cls.EXAMPLES_PER_BUCKET

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
              TEMPLATE = """
                <a href="{href}">{title}</a><br />
                {img_tag}<br />
                <pre>{uri}</pre>
                <br />
              """
              BASE = "/view?"
              href = BASE + parse.urlencode({'uri': row.uri})

              frame = cls.ANNO_TABLE.FIXTURES.get_frame(row.uri)
              debug_img = frame.get_debug_image()
              
              if row.ridden_bike_track_id:
                # Highlight the rider's bike if possible. Rather than draw a
                # new box in the image, it's easiest to just fetch a debug
                # image for the rider and blend using OpenCV
                best_bike_uri = FrameURI.from_str(row.uri)
                best_bike_uri.track_id = row.ridden_bike_track_id
                dframe = cls.ANNO_TABLE.FIXTURES.get_frame(best_bike_uri)
                debug_img_bike = dframe.get_debug_image()
                import cv2
                debug_img[:] = cv2.addWeighted(
                  debug_img, 0.5, debug_img_bike, 0.5, 0)

              img_tag = aupl.img_to_img_tag(
                          debug_img,
                          jpeg_quality=60,
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

# from collections import namedtuple
# cropattrs = namedtuple('cropattrs', 'anno cloud_npz viewport_annos')

class CroppedObjectImageTable(dataset.ImageTable):

  # Center the object in a viewport of this size (pixels)
  VIEWPORT_WH = (270, 270)#(170, 170)
  
  # Pad the object by this many pixels against the viewport edges
  PADDING_PIXELS = 20

  # Jitter the center of any crop vs the target bbox using this gaussian
  CENTER_JITTER_MU_STD = (0, 10)

  # The front camera has on average 18.53 annotations per image
  # cache/data/argoverse/index/image_annos/Anno_Counts_by_Camera.html
  NEGATIVE_SAMPLES_PER_FRAME = 20

  # Try to break each split into this many shards
  N_SHARDS_PER_SPLIT = 100

  ANNOS = ImageAnnoTable

  TABLE_NAME = (
    'argoverse_cropped_object_%s_%s' % (VIEWPORT_WH[0], VIEWPORT_WH[1]))

  DEBUG_CROP_HW = (500, 500)

  @classmethod
  def setup(cls, spark=None):
    if not util.missing_or_empty(cls.table_root()):
      return

    with Spark.sess(spark) as spark:
      pos_df = cls._create_positives(spark)
      neg_df = cls._create_negatives(spark)
      df = Spark.union_dfs(pos_df, neg_df)
        # Need to write a DF with uniform schema because pyarrow doesn't
        # like non-uniform schemas :(
      
      cls.__save_df(df)
      util.log.info(
        "Created %s total crops." % cls.as_imagerow_df(spark).count())
      
      
      
      # df2 = cls.ANNOS.as_df(spark)
      # df2.createOrReplaceTempView('d2')
      # import pprint
      # pprint.pprint(spark.sql('select log_id, camera, first(relative_yaw_to_camera_radians) from d2 where relative_yaw_to_camera_radians is not null group by log_id, camera order by camera').collect())
      # import pdb; pdb.set_trace()
      
      
      util.log.info("Creating HTML sample report ...")
      html = cls.to_html(spark)
      
      fname = cls.TABLE_NAME + '_sample.html'
      dest = os.path.join(cls.ANNOS.FIXTURES.index_root(), fname)
      util.mkdir(cls.ANNOS.FIXTURES.index_root())
      with open(dest, 'w') as f:
        f.write(html)
      util.log.info("... saved HTML sample report to %s" % dest)      
  
  @classmethod
  def as_imagerow_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    
    # Alias column to aid ImageRow Construction
    df = df.withColumn('image_bytes', df['jpeg_bytes'])
    df = df.withColumn('label', df['category_name'])

    return df

  @classmethod
  def to_html(cls, spark, limit=500, random_sample_seed=1337):
    df = cls.as_imagerow_df(spark)
    if random_sample_seed is not None:
      fraction_approx = np.clip(float(limit) / df.count(), 0.1, 1.0)
      df = df.where(df.category_name != 'background')
      df = df.sample(
                withReplacement=False,
                fraction=fraction_approx,
                seed=random_sample_seed)
    if limit and limit >= 1:
      df = df.limit(limit)
    pdf = df.toPandas()

    def maybe_to_img(v):
      from au import plotting as aupl
      if isinstance(v, bytearray):
        try:
          from io import BytesIO
          img = imageio.imread(BytesIO(v))
          return aupl.img_to_img_tag(img)
        except Exception as e:
          util.log.error("Failed to encode image bytes %s" % (e,))
      return v

    # The HTML page will be very wide; move these cols to the left (no scroll)
    # COLS = [
    #   'category_name',
    #   'jpeg_bytes',
    #   'debug_jpeg_bytes',
    #   'camera',
    # ]
    COLS = [
      'jpeg_bytes',
      'debug_jpeg_bytes',
      'camera',

      'obj_ypr_camera_local',
      'camera_norm',        
      'camera_to_obj',
      'obj_ypr_in_ego',
      'log_id',

      # 'relative_yaw_radians',
      # 'relative_yaw_to_camera_radians',
      # 'obj_in_crop',
      # 'obj_in_crop_debug',
      # 'obj_in_crop_xyz',
      # 'calib_ypr',
      
    ]
    other_cols = set(pdf.columns) - set(COLS)
    pdf = pdf.reindex(columns=COLS + list(sorted(other_cols)))

    import pandas as pd
    with pd.option_context('display.max_colwidth', -1):
      html = pdf.to_html(
            escape=False,
            formatters=dict(
              (col, maybe_to_img)
              for col in pdf.columns
              if 'bytes' in col))
    return html

  ### Utils

  @classmethod
  def get_crop_viewport(cls, bbox):
    """Compute and return a viewport with the aspect ratio defined in
    `VIEWPORT_WH` (and respecting `PADDING_PIXELS`) from which we can
    create a crop of the object annotated in `bbox`.  Note that the
    returned bbox may extend beyond the image edges."""
    
    # Center of crop
    box_center = np.array(
      [bbox.x + .5 * bbox.width, bbox.y + .5 * bbox.height])

    rand = random.Random(util.stable_hash(str(bbox)))
    offset = np.array([
        rand.gauss(*cls.CENTER_JITTER_MU_STD),
        rand.gauss(*cls.CENTER_JITTER_MU_STD)
      ])
    offset = np.clip(offset, -cls.PADDING_PIXELS, cls.PADDING_PIXELS)
    box_center += offset
      # Jitter the center, but not so much that it exceeds the padding; then
      # the center will have moved so far that the target extends outside
      # the crop viewport.
    
    # Size of crop
    radius = .5 * max(bbox.width, bbox.height)
    padding_relative = 2 * cls.PADDING_PIXELS / np.array(cls.VIEWPORT_WH)
    radius_xy = radius / (1 - padding_relative)
      # If paddig is 10% relative, we want the crop to cover
      # 90% of the viewport
    # padding_in_target_res = radius / (
    #   1 - ())
    # radius_xy = radius + padding_in_target_res

    # Compute the crop
    x1, y1 = (box_center - radius_xy).tolist()
    x2, y2 = (box_center + radius_xy).tolist()
    cropped = copy.deepcopy(bbox)
    cropped.set_x1_y1_x2_y2(x1, y1, x2, y2)
    cropped.update(im_width=bbox.im_width, im_height=bbox.im_height)
    cropped.quantize()
    return cropped

  @classmethod
  def _to_cropped_image_anno_df(cls, spark, anno_df, and_debug=True):
    """Adjusts annos to cropped viewpoint and adds columns:
     * jpeg_bytes (cropped image encoded as max quality jpeg)
     * debug_jpeg_bytes (debug image as medium-quality jpeg)
    """

    def anno_rows_to_crop_rows(part, rows):
      t = util.ThruputObserver(name='row_crop_%s' % part)
      for row in rows:
        t.start_block()

        from au import plotting as aupl
        import cv2

        bbox = BBox.from_row_dict(row.asDict())
        cropbox = cls.get_crop_viewport(bbox)
        has_offscreen = (cropbox.get_num_onscreen_corners() != 4)
        # Skip anything that is too near (or off) the edge of the image
        if has_offscreen:
          continue

        frame = cls.ANNOS.FIXTURES.get_frame(row.uri)
        cropped_frame = frame.get_cropped(cropbox)
        cropped_anno = frame.get_target_bbox()
      
        row_out = row.asDict()
        if cropped_anno:
          row_out.update(cropped_anno.to_row_dict())
        else:
          # For negatives, there is no real anno, but do record the
          # crop bounds
          row_out.update(dict(
            (k, v) for k, v in cropbox.to_dict().items()
            if v is not None))
              # TODO create false annos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        row_out.update(cropped_frame.uri.to_dict())

        # Generate unformly-sized crop
        crop = cropped_frame.image
        crop = cv2.resize(crop, cls.VIEWPORT_WH, interpolation=cv2.INTER_NEAREST)
        row_out['jpeg_bytes'] = bytearray(util.to_jpeg_bytes(crop, quality=100))

        if and_debug:
          # Rescale debug image to save space
          crop_debug = cropped_frame.get_debug_image()
          th, tw = aupl.get_hw_in_viewport(
                      crop_debug.shape[:2], cls.DEBUG_CROP_HW)
          crop_debug = cv2.resize(
            crop_debug, (tw, th), interpolation=cv2.INTER_NEAREST)
          row_out['debug_jpeg_bytes'] = bytearray(
            util.to_jpeg_bytes(crop_debug, quality=60))
      
        yield Row(**row_out)

        t.stop_block(n=1, num_bytes=sys.getsizeof(row_out))
        t.maybe_log_progress()
    
    crop_anno_rdd = anno_df.rdd.mapPartitionsWithIndex(anno_rows_to_crop_rows)
    crop_anno_rdd = crop_anno_rdd.filter(lambda x: x is not None)
    crop_anno_df = spark.createDataFrame(crop_anno_rdd)
    return crop_anno_df

  @classmethod
  def _get_anno_df(cls, spark):
    # Pre-shard the annos for easier writes and better parallelism later
    anno_df = cls.ANNOS.as_df(spark)

    # Skip sample set, and the test set has no labels
    anno_df = anno_df.filter("split not in ('sample', 'test')")

    from pyspark.sql import functions as F
    anno_df = anno_df.withColumn(
                'shard',
                F.abs(F.hash(anno_df['frame_uri'])) % cls.N_SHARDS_PER_SPLIT)
    anno_df = anno_df.repartition(10, 'split', 'shard').persist()
    return anno_df

  @classmethod
  def _create_croppable_anno_df(cls, spark):
    anno_df = cls._get_anno_df(spark)

    # Filter
    CONDS = (
      # Motion correction almost always succeeds
      anno_df.motion_corrected == True,

      # Ignore things that extend off the side of the image
      anno_df.has_offscreen == False,

      # Ignore very small things
      anno_df.height >= 50, # pixels
      anno_df.width >= 50,  # pixels
    )
    for cond in CONDS:
      anno_df = anno_df.filter(cond)
    
    # anno_df = anno_df.filter(anno_df.frame_uri.isin(anno_df.select('frame_uri').distinct().limit(5000).rdd.flatMap(lambda x: x).collect())) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #anno_df = anno_df.repartition(1000)

    util.log.info("Found %s eligible annos." % anno_df.count())
    return anno_df
  
  @classmethod
  def __save_df(cls, crop_df):
    crop_df.write.parquet(
      cls.table_root(),
      partitionBy=['split', 'shard'],
      compression='snappy') # TODO pyarrow / lz4
    util.log.info("Wrote to %s" % cls.table_root())

  @classmethod
  def _create_positives(cls, spark):
    util.log.info("Creating positives ....")
    anno_df = cls._create_croppable_anno_df(spark)
    cropped_anno_df = cls._to_cropped_image_anno_df(spark, anno_df)
    return cropped_anno_df
  
  @classmethod
  def _create_negatives(cls, spark):
    util.log.info("Creating negatives ...")
    # We'll mine negatives from all images, TODO even those w/out annotations ? ~~~~~~~~~~~~~~~
    anno_df = cls._get_anno_df(spark)
    # anno_df = anno_df.filter(anno_df.frame_uri.isin(anno_df.select('frame_uri').distinct().limit(100).rdd.flatMap(lambda x: x).collect())) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Since `anno_df` already has all the bbox info we need, we use those
    # bboxes since reading the raw annos again from Argoverse is expensive
    bbox_df = anno_df.select(
                # We'll sample negatives from each frame.  URI includes 
                # viewport info.
                'frame_uri', 'shard',

                # Get all positives (bboxes)
                'uri', 'x', 'y', 'width', 'height')
    
    def iter_samples(key_rows):
      # Unpack
      (frame_uri, shard), rows = key_rows
      pos_boxes = [BBox.from_row_dict(row.asDict()) for row in rows]

      # Construct a negative miner
      frame_uri = FrameURI.from_str(frame_uri)
      viewport = frame_uri.get_viewport()
      miner = HardNegativeMiner.create_miner(
                        frame_uri.camera, viewport, pos_boxes)

      # Mine!
      for n in range(cls.NEGATIVE_SAMPLES_PER_FRAME):
        cropbox = miner.next_sample()
        if not cropbox:
          continue
        row = cropbox.to_dict()
        row.update(
          category_name='background',
          uri=str(frame_uri),
          frame_uri=str(frame_uri),
          split=frame_uri.split,
          shard=shard)
        yield Row(**row)

    key_rows_rdd = bbox_df.rdd.groupBy(lambda r: (r.frame_uri, r.shard))
    util.log.info(
      "... generating negatives from %s frames ..." % key_rows_rdd.count())
    negative_annos = key_rows_rdd.flatMap(iter_samples)
    anno_df = spark.createDataFrame(negative_annos)
    
    cropped_anno_df = cls._to_cropped_image_anno_df(spark, anno_df)
    return cropped_anno_df


    # def anno_row_to_imagerow(crop_spec_row):
    #   import cv2
    #   frame = AVFrame(uri=anno.uri, FIXTURES=cls.FIXTURES)
    #   cropbox = BBox(**crop_spec_row.asDict())
    #   cropped = frame.get_crop(cropbox)

    #   # # TODO clean up dataset.ImageRow so we can use it here with jpeg
    #   # def to_jpg_bytes(arr):
    #   #   import imageio
    #   #   import io
    #   #   buf = io.BytesIO()
    #   #   imageio.imwrite(buf, arr, 'jpeg', quality=100)
    #   #   return bytearray(buf.getvalue())

    #   def to_npz_bytes(arr):
    #     import io
    #     buf = io.BytesIO()
    #     np.savez_compressed(buf, arr)
    #     return bytearray(buf.getvalue())

    #   crop_img = cv2.resize(cropped.image, cls.VIEWPORT_WH)
    #   cloud = cropped.get_cloud_in_image()
    #   # TODO: expose other labels as cols, add a shard key ~~~~~~~~~~~~~~~~~~~~~~~~
    #   row = Row(
    #     uri=cropped.uri,
    #     split=cropped.split,
    #     dataset='argoverse',
    #     img_byte_jpeg=to_jpg_bytes(crop_img),
    #     label=cropbox.category_name,
    #     attrs=cropattrs(
    #       anno=crop_spec_row,
    #       cloud_npz=to_npz_bytes(cloud),
    #       viewport_annos=''))# TODO ~~~~~~also add other labels~~~~~~~~~~~~~~~~~~~~~~cropped.image_bboxes))
    #   return row
    
    # crop_spec_df = cls._create_crop_spec_df(spark)
    # imagerow_rdd = crop_spec_df.rdd.map(anno_row_to_imagerow)
    # imagerow_df = spark.createDataFrame(imagerow_rdd)
    
  
  


              
    
"""
TODO
 
 *** find images that have no annotations!
 * a ped can stand on OTHER_MOVER

 * report on number of invisible things (e.g. top of file comment)
    * report on number of motion-corrected frames
    * report on bikes no riders
 * do kl divergence or some sort of tests in 'split' plots
 * log-scale option for? plots
 * debug draw lider pts
 
 * try to measure occlusion / clutter by:
   * overlapping / subsuming cuboids .. how many are completely subsumed?
   * fraction of lidar points that overlap with cuboid in image frame but
       do NOT overlap with cuboid in 3D-- think person occluded by car
       or car thru window.
   * prolly an exception for bikes and riders
   * maybe also build occlusion graph:
      * col with edges using the above measures
      * ** then we can also mine for person-car occluders, etc
      .... can deep net detect occlusion?  
               ** can deep net discrim car w/ and w/out occluded thing behind?
"""

