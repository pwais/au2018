import os
import copy
import itertools

import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av



## Utils

def get_jpeg_size(jpeg_bytes):
  """Get the size of a JPEG image without reading and decompressing the entire
  file.  Based upon:  
   * https://github.com/shibukawa/imagesize_py/blob/master/imagesize.py#L87
  """
  import struct
  from io import BytesIO
  buf = BytesIO(jpeg_bytes)
  head = buf.read(24)
  if not head.startswith(b'\377\330'):
    raise ValueError("Invalid JPEG header")
  buf.seek(0)
  size = 2
  ftype = 0
  while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
    buf.seek(size, 1)
    byte = buf.read(1)
    while ord(byte) == 0xff:
      byte = buf.read(1)
    ftype = ord(byte)
    size = struct.unpack('>H', buf.read(2))[0] - 2
  # Now we're at a SOFn block
  buf.seek(1, 1)  # Skip `precision' byte.
  height, width = struct.unpack('>HH', buf.read(4))
  return width, height

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy array `pts` to Homogeneous coordinates of target `dim`
  if necessary"""
  if len(pts.shape) != dim + 1:
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts

def transform_from_pb(waymo_transform):
  RT = np.reshape(waymo_transform.transform, [4, 4])
  return av.Transform(rotation=RT[:3, :3], translation=RT[:3, 3])

import klepto

def mymap(cls, waymo_frame):
  # _, waymo_frame = cls_waymo_frame
  return str(waymo_frame.context.name) + str(waymo_frame.timestamp_micros)

class GetFusedCloudInEgo(object):

  @staticmethod
  def _get_all_points(wf_str):
    import tensorflow as tf
    assert tf.executing_eagerly()
    
    from waymo_open_dataset import dataset_pb2
    wf = dataset_pb2.Frame()
    wf.ParseFromString(wf_str.numpy())
    
    from waymo_open_dataset.utils import frame_utils
    range_images, camera_projections, range_image_top_pose = (
      frame_utils.parse_range_image_and_camera_projection(wf))

    # Waymo provides two returns for every lidar beam; we have to
    # fuse them manually
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        wf,
        range_images,
        camera_projections,
        range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        wf,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    all_points = np.concatenate((points_all, points_all_ri2), axis=0)
    return all_points
  
  @classmethod
  @klepto.lru_cache(maxsize=10, keymap=mymap)
  def get_points(cls, waymo_frame):
    # waymo_open_dataset requires eager mode :( so we need to scope its use
    # into a tensorflow py_func
    if not hasattr(cls, '_thruput'):
      cls._thruput = util.ThruputObserver(name='GetFusedCloudInEgo')

    with cls._thruput.observe(n=1):
      os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # TF Eager is a load of trash and ignores our session config
      import tensorflow as tf
      import tensorflow.contrib.eager as tfe

      if not hasattr(cls, '_sess'):
        cls._sess = util.tf_cpu_session()

      wf = tf.placeholder(dtype=tf.string)
      pf = tfe.py_func(GetFusedCloudInEgo._get_all_points, [wf], tf.float32)
      all_points = cls._sess.run(
        pf, feed_dict={wf: waymo_frame.SerializeToString()})
    
    cls._thruput.update_tallies(num_bytes=util.get_size_of_deep(all_points))
    cls._thruput.maybe_log_progress(every_n=10)
    return all_points

def get_category_name(class_id):
  from waymo_open_dataset import label_pb2
  class_id_to_name = dict(
    zip(label_pb2.Label.Type.values(), label_pb2.Label.Type.keys()))
  return class_id_to_name[class_id]

def get_camera_name(camera_id):
  from waymo_open_dataset import dataset_pb2
  camera_id_to_name = dict(
    zip(
      dataset_pb2.CameraName.Name.values(),
      dataset_pb2.CameraName.Name.keys()))
  return camera_id_to_name[camera_id]

def get_lidar_name(lidar_id):
  from waymo_open_dataset import dataset_pb2
  lidar_id_to_name = dict(
    zip(
      dataset_pb2.LaserName.Name.values(),
      dataset_pb2.LaserName.Name.keys()))
  return lidar_id_to_name[lidar_id]

## Data

class Fixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'waymo-od')

  TARBALLS = (
    # Sadly, the Train / Val splits are *only* indicated in the filenames

    # Train
    'training_0000.tar',
    'training_0001.tar',
    'training_0002.tar',
    'training_0003.tar',
    'training_0004.tar',
    'training_0005.tar',
    'training_0006.tar',
    'training_0007.tar',
    'training_0008.tar',
    'training_0009.tar',
    'training_0010.tar',
    'training_0011.tar',
    'training_0012.tar',
    'training_0013.tar',
    'training_0014.tar',
    'training_0015.tar',
    'training_0016.tar',
    'training_0017.tar',
    'training_0018.tar',
    'training_0019.tar',
    'training_0020.tar',
    'training_0021.tar',
    'training_0022.tar',
    'training_0023.tar',
    'training_0024.tar',
    'training_0025.tar',
    'training_0026.tar',
    'training_0027.tar',
    'training_0028.tar',
    'training_0029.tar',
    'training_0030.tar',
    'training_0031.tar',

    # Val
    'validation_0000.tar',
    'validation_0001.tar',
    'validation_0002.tar',
    'validation_0003.tar',
    'validation_0004.tar',
    'validation_0005.tar',
    'validation_0006.tar',
    'validation_0007.tar',
  )

  SPLITS = ('train', 'val')

  ## Source Data

  @classmethod
  def tarballs_dir(cls):
    # return os.path.join(cls.ROOT, 'tarballs')
    return '/outer_root/media/seagates-ext4/au_datas/waymo_open/'

  @classmethod
  def tarball_path(cls, fname):
    return os.path.join(cls.tarballs_dir(), fname)
  
  @classmethod
  def get_split(cls, fname):
    return 'train' if 'training' in fname else 'val'

class FrameTable(av.FrameTableBase):

  FIXTURES = Fixtures

  PROJECT_CLOUDS_TO_CAM = True
  PROJECT_CUBOIDS_TO_CAM = True
  IGNORE_INVISIBLE_CUBOIDS = True

  # SETUP_SEGMENTS_PER_CHUNK = os.cpu_count()
  # SETUP_URIS_PER_CHUNK = 10

  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/waymo_open_frame_table'

  @classmethod
  def create_frame(cls, uri, waymo_frame=None, record_idx=None):
    uri = av.URI.from_str(uri)
    f = av.Frame(uri=uri)
    if waymo_frame is None:
      waymo_frame = cls._get_waymo_frame(uri, record_idx=record_idx)
        # NB: this operation is expensive! See comments embedded in function
    cls._fill_ego_pose(f, waymo_frame)
    cls._fill_camera_images(f, waymo_frame)
    cls._fill_extra(f, waymo_frame)
    return f

  @classmethod
  def _create_frame_rdds(cls, spark):

    # def gen_frame_tasks(uri):
    #   segment_id = uri.segment_id
    #   record = cls._get_segment_id_to_record()[segment_id]
    #   tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
    #   tasks = [(uri, i) for i in range(len(tf_str_list))]
    #   print('%s tasks for %s' % (len(tasks), segment_id))
    #   return tasks
    
    # seg_rdd = spark.sparkContext.parallelize(cls.iter_all_segment_uris())
    # task_rdd = seg_rdd.flatMap(gen_frame_tasks)
    # print(task_rdd.count())

    # import itertools
    # iter_tasks = itertools.chain.from_iterable(
    #   gen_frame_tasks(uri) for uri in cls.iter_all_segment_uris()
    # )
    # import pdb; pdb.set_trace()


    PARTITIONS_PER_SEGMENT = 2

    frame_rdds = []
    segment_uris = list(cls.iter_all_segment_uris())
    t = util.ThruputObserver(
      name='create_frame_rdds', n_total=PARTITIONS_PER_SEGMENT * len(segment_uris))
    for segment_uri in cls.iter_all_segment_uris():
      for partition in range(PARTITIONS_PER_SEGMENT):
        with t.observe(n=1):
          segment_uri_rdd = spark.sparkContext.parallelize([segment_uri])

          def gen_tasks(uri):
            segment_id = uri.segment_id
            record = cls._get_segment_id_to_record()[segment_id]
            tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
            tasks = [
              (uri, i) for i in range(len(tf_str_list))
              if (i % PARTITIONS_PER_SEGMENT) == partition
            ]
            print('%s tasks for %s part %s' % (len(tasks), segment_id, partition))
            return tasks

          task_rdd = segment_uri_rdd.flatMap(gen_tasks)

          def load_frame(task):
            uri, record_idx = task
            f = cls.create_frame(uri, record_idx=record_idx)
            # print(f.uri, ' ', util.get_size_of_deep(f) * 1e-6, 'MB')
            return f
          
          frame_rdd = task_rdd.repartition(2 * os.cpu_count()).map(load_frame)
          frame_rdds.append(frame_rdd)
        t.maybe_log_progress(every_n=500)



    # isegment_uris = util.ichunked(
    #   cls.iter_all_segment_uris(), 1)#cls.SETUP_SEGMENTS_PER_CHUNK)
    # for segment_uris in isegment_uris:
      
    #   def gen_frame_tasks(uri):
    #     segment_id = uri.segment_id
    #     record = cls._get_segment_id_to_record()[segment_id]
    #     tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
    #     tasks = [(uri, i) for i in range(len(tf_str_list))]
    #     print('%s tasks for %s' % (len(tasks), segment_id))
    #     return tasks
      
    #   def load_frame(task):
    #     uri, record_idx = task
    #     create_frame = util.ThruputObserver.wrap_func(cls.create_frame, log_on_del=True)
    #     f = create_frame(uri, record_idx=record_idx)
    #     # print(f.uri, ' ', util.get_size_of_deep(f) * 1e-6, 'MB')
    #     return f

    #   # def iter_segment_frames(uri):
    #   #   # Breadcrumbs: to get any frame information at all, we must read and
    #   #   # decode protobufs from the whole TFRecord file.  So don't compute
    #   #   # Frame URIs, as we do for other datasets, but just generate Frames.
    #   #   t = util.ThruputObserver(name=uri.segment_id)
    #   #   for wf in cls._iter_waymo_frames(uri.segment_id):
    #   #     frame_uri = copy.deepcopy(uri)
    #   #     frame_uri.timestamp = int(wf.timestamp_micros * 1e3)
    #   #     yield cls.create_frame(uri, waymo_frame=wf)
    #   #     t.update_tallies(n=1)
    #   #     t.maybe_log_progress(every_n=10)
      
    #   segment_uri_rdd = spark.sparkContext.parallelize(segment_uris)
    #   # frame_rdd = segment_uri_rdd.flatMap(iter_segment_frames)
    #   task_rdd = segment_uri_rdd.flatMap(gen_frame_tasks)
    #   frame_rdd = task_rdd.repartition(20 * os.cpu_count() ).partitionBy(20 * os.cpu_count()).map(load_frame)
    #   frame_rdds.append(frame_rdd)
    #   # break
    
    return frame_rdds



  ## Public API

  @classmethod
  def iter_all_segment_uris(cls):
    for segment_id, record in cls._get_segment_id_to_record().items():
      yield av.URI(
              dataset='waymo-od',
              split=record.split,
              segment_id=segment_id)      
  
  @classmethod
  def iter_all_uris(cls):
    # Warning: Slow!
    for base_uri in cls.iter_all_segment_uris():
      for wf in cls._iter_waymo_frames(base_uri.segment_id):
        uri = copy.deepcopy(base_uri)
        uri.timestamp = int(wf.timestamp_micros * 1e3)
        yield uri

  ## Support
  # Portions based upon https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/master/examples/example.py

  class _SegmentRecord(object):
    __slots__ = ('fw', 'split')
    def get_reader(self):
      return self.fw.data_reader

  @classmethod
  def _get_segment_id_to_record(cls):
    if not hasattr(cls, '_segment_id_to_record'):
      util.log.info("Scanning all tarballs ...")
      
      segment_id_to_record = {}
      for fname in cls.FIXTURES.TARBALLS:
        path = cls.FIXTURES.tarball_path(fname)
        fws = util.ArchiveFileFlyweight.fws_from(path)
        for fw in fws:
          if fw.name.endswith('tfrecord'):
            record = FrameTable._SegmentRecord()
            record.fw = fw
            record.split = cls.FIXTURES.get_split(fname)
            segment_id_to_record[fw.name] = record

      from collections import defaultdict
      split_to_count = defaultdict(int)
      for record in segment_id_to_record.values():
        split_to_count[record.split] += 1
      util.log.info(
        "... found %s segments, splits: %s" % (
          len(segment_id_to_record), dict(split_to_count)))
      
      cls._segment_id_to_record = segment_id_to_record
    return cls._segment_id_to_record
  
  @classmethod
  def _iter_waymo_frames(cls, segment_id):
    from waymo_open_dataset import dataset_pb2
    record = cls._get_segment_id_to_record()[segment_id]
    tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
    for s in tf_str_list:
      wf = dataset_pb2.Frame()
      wf.ParseFromString(bytearray(s))
      yield wf
  
  @classmethod
  def _get_waymo_frame(cls, uri, record_idx=None):
    # Breakcrumbs: the frame context and timestamp is embedded in the
    # serialized protobuf message, so even if we wanted to index the Waymo
    # TFRecord files, we'd have to read and decode all of them.  Thus for now
    # we just provide an expensive linear search to look up individual frames
    # and optimize just the linear read / ETL use case.
    if record_idx is not None:
      from waymo_open_dataset import dataset_pb2
      record = cls._get_segment_id_to_record()[uri.segment_id]
      tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
      s = tf_str_list[record_idx]
      print(uri, ' protobuf ', len(s) * 1e-6, 'MB')
      wf = dataset_pb2.Frame()
      wf.ParseFromString(bytearray(s))
      return wf

    util.log.info("Doing expensive linear find ...")
    for wf in cls._iter_waymo_frames(uri.segment_id):
      timestamp = int(wf.timestamp_micros * 1e3)
      if timestamp == uri.timestamp:
        util.log.info("... found!")
        return wf
    raise ValueError("Frame not found for %s" % uri)
  
  # Filling Frames

  @classmethod
  def _fill_ego_pose(cls, f, waymo_frame):
    f.world_to_ego = transform_from_pb(waymo_frame.pose)

  @classmethod
  def _fill_extra(cls, f, waymo_frame):
    f.extra.update({
      'waymo.time_of_day':  waymo_frame.context.stats.time_of_day,
      'waymo.location':     waymo_frame.context.stats.location,
      'waymo.weather':      waymo_frame.context.stats.weather,
    })

  @classmethod
  def _fill_camera_images(cls, f, waymo_frame):
    for image in waymo_frame.images:
      ci = cls._get_camera_image(waymo_frame, image.name, viewport=f.uri.get_viewport())
      f.camera_images.append(ci)

  @classmethod
  def _get_camera_image(cls, waymo_frame, camera_idx, viewport=None):
    def get_for_camera(lst, idx):
      for el in lst:
        if el.name == idx:
          return el
      raise ValueError("Element with name %s not found" % idx)
    wf_camera_image = get_for_camera(waymo_frame.images, camera_idx)
    w, h = get_jpeg_size(wf_camera_image.image)
    if not viewport:
      from au.fixtures.datasets import common
      viewport = common.BBox.of_size(w, h)
    ci_timestamp = int(wf_camera_image.pose_timestamp * 1e9)

    # NB: Waymo protobuf defs had an error; the protobufs contain
    # the inverse of the expected transform.
    camera_calibration = get_for_camera(
                            waymo_frame.context.camera_calibrations,
                            camera_idx)
    extrinsic = np.reshape(camera_calibration.extrinsic.transform, [4, 4])
    vehicle_to_sensor = np.linalg.inv(extrinsic) # Need inverse! Waymo lies!

    # Waymo camera extrinsics keep the nominal ego frame axes: +x forward,
    # +z up, etc.  We need to add a rotation to convert to the more
    # canonical +z depth, +x right, etc.
    axes_transformation = np.array([
                            [0, -1,  0,  0],
                            [0,  0, -1,  0],
                            [1,  0,  0,  0],
                            [0,  0,  0,  1]])
    cam_from_ego_RT = axes_transformation.dot(vehicle_to_sensor)
    cam_from_ego = av.Transform(
                      rotation=cam_from_ego_RT[:3, :3],
                      translation=np.reshape(cam_from_ego_RT[:3, 3], (3, 1)))
      
    # Waymo encodes intrinsics using a very custom layout
    f_u = camera_calibration.intrinsic[0]
    f_v = camera_calibration.intrinsic[1]
    c_u = camera_calibration.intrinsic[2]
    c_v = camera_calibration.intrinsic[3]
    K = np.array([
        [f_u, 0,   c_u],
        [0,   f_v, c_v],
        [0,   0,   1  ],
    ])

    # To get the principal axis, we can use `vehicle_to_sensor` (which
    # neglects the rotation into the image plane) and simply rotate X_HAT
    # into the camera frame.
    X_HAT = np.array([1, 0, 0])
    principal_axis_in_ego = vehicle_to_sensor[:3, :3].dot(X_HAT)

    ci = av.CameraImage(
        camera_name=get_camera_name(wf_camera_image.name),
        image_jpeg=bytearray(wf_camera_image.image),
        height=h,
        width=w,
        viewport=viewport,
        timestamp=ci_timestamp,
        cam_from_ego=cam_from_ego,
        K=K,
        principal_axis_in_ego=principal_axis_in_ego,
      )
      
    if cls.PROJECT_CLOUDS_TO_CAM:
      # TODO: motion correction / rolling shutter per-camera, non-fused cloud, ...
      pc = av.PointCloud(
        sensor_name='lidar_fused',
        timestamp=int(waymo_frame.timestamp_micros * 1e3),
        cloud=GetFusedCloudInEgo.get_points(waymo_frame),
        ego_to_sensor=av.Transform(), # points are in ego frame...
        motion_corrected=False,
      )
      pc.cloud = ci.project_ego_to_image(pc.cloud, omit_offscreen=True)
      pc.sensor_name = pc.sensor_name + '_in_cam'
      ci.clouds.append(pc)

    if cls.PROJECT_CUBOIDS_TO_CAM:
      cuboids = cls._get_cuboids_in_ego(waymo_frame)
      for cuboid in cuboids:
        bbox = ci.project_cuboid_to_bbox(cuboid)
        if cls.IGNORE_INVISIBLE_CUBOIDS and not bbox.is_visible:
          continue
        ci.bboxes.append(bbox)
    
    return ci

  @classmethod
  def _get_cuboids_in_ego(cls, waymo_frame):
    cuboids = []
    for label in waymo_frame.laser_labels:
      box = label.box
      
      # Coords in ego frame
      T = np.array([
            [box.center_x],
            [box.center_y],
            [box.center_z]])
      
      # Heading is yaw in ego frame
      from scipy.spatial.transform import Rotation as R
      rotation = R.from_euler('z', box.heading).as_dcm()
      
      obj_from_ego = av.Transform(
                          rotation=rotation,
                          translation=T)

      # cosyaw = math.cos(box.heading)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
      # sinyaw = math.sin(box.heading)
      # obj_from_ego_RT = np.array([
      #                     [ cosyaw, -sinyaw, 0, tx],
      #                     [ sinyaw,  cosyaw, 0, ty],
      #                     [      0,       0, 1, tz],
      #                     [      0,       0, 0,  1]])

      l, w, h, = box.length, box.width, box.height
      box3d = obj_from_ego.apply(
          # Send the corners of the box into the ego frame
          .5 * np.array([
                # Front face
                [ l,  w,  h],
                [ l, -w,  h],
                [ l, -w, -h],
                [ l,  w, -h],
                
                # Rear face
                [-l,  w,  h],
                [-l, -w,  h],
                [-l, -w, -h],
                [-l,  w, -h],
          ])).T

      category_name = get_category_name(label.type)
      au_category = av.WAYMO_OD_CATEGORY_TO_AU_AV_CATEGORY[category_name]

      extra = {
        'waymo.detection_difficulty_level':
          str(label.detection_difficulty_level),
        'waymo.tracking_difficulty_level':
          str(label.tracking_difficulty_level),
      }
      if label.metadata:
        extra.update({
          'waymo.speed_x': str(label.metadata.speed_x),
          'waymo.speed_y': str(label.metadata.speed_y),
          'waymo.accel_x': str(label.metadata.accel_x),
          'waymo.accel_y': str(label.metadata.accel_y),
        })

      cuboid = av.Cuboid(
        track_id=label.id,
        category_name=category_name,
        au_category=au_category,
        timestamp=int(waymo_frame.timestamp_micros * 1e3),
        box3d=box3d,
        motion_corrected=False, # TODO
        length_meters=l,
        width_meters=w,
        height_meters=h,
        distance_meters=np.linalg.norm(T),
        obj_from_ego=obj_from_ego,
        extra=extra,
      )
      cuboids.append(cuboid)
    return cuboids
      
    

# print(points_all.shape)
# print(points_all_ri2.shape)

# # camera projection corresponding to each point.
# cp_points_all = np.concatenate(cp_points, axis=0)
# cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)



# n = 0
# camera_calibration = frame.context.camera_calibrations[n]
# extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
# vehicle_to_sensor = tf.matrix_inverse(extrinsic).numpy()

# f_u = camera_calibration.intrinsic[0]
# f_v = camera_calibration.intrinsic[1]
# c_u = camera_calibration.intrinsic[2]
# c_v = camera_calibration.intrinsic[3]

# K = np.array([
#     [f_u, 0,   c_u],
#     [0,   f_v, c_v],
#     [0,   0,   1  ],
# ])
# print('K', K, (f_u, f_v, c_u, c_v))



# p = maybe_make_homogeneous(points_all, dim=4)
# print('p', p)
# p_cam = vehicle_to_sensor.dot(p.T)
# print('p_cam', p_cam)

# plt.figure(figsize=(20, 12))
# plt.scatter(p_cam[0, :], p_cam[1, :])

# # plt.figure(figsize=(20, 12))
# # plt.scatter(p_cam[0, :], p_cam[2, :])

# # plt.figure(figsize=(20, 12))
# # plt.scatter(p_cam[1, :], p_cam[2, :])


# # p_cam[:2, :] = -p_cam[:2, :] / p_cam[2, :]
# p_cam = p_cam[(2, 1, 0), :]
# p_cam = p_cam[(1, 0, 2), :]
# # p_cam[2, :] *= -1
# p_cam[1, :] *= -1
# p_cam[0, :] *= -1

# uv = K.dot(p_cam)
# uv[:2, :] /= uv[2, :]

# # uv *= 1e-4

# print('uv', uv.shape, ((uv[0, :].min(), uv[0, :].max(), uv[0, :].mean())))


# plt.figure(figsize=(20, 12))
# plt.scatter(uv[0, :], uv[1, :])

# # images = sorted(frame.images, key=lambda i:i.name)
# image = tf.image.decode_jpeg(frame.images[n].image)
# h, w, c = image.numpy().shape
# print('hwc', (h, w, c))


# idx_ = np.where(
#         np.logical_and.reduce((
#           # Filter offscreen points
#           0 <= uv[0, :], uv[0, :] < w - 1.0,
#           0 <= uv[1, :], uv[1, :] < h - 1.0,
#           # Filter behind-screen points
#           uv[2, :] > 0)))
# idx_ = idx_[0]
# print('idx', len(idx_))
# uv = uv[:, idx_]
# uvd = uv.T

# plot_points_on_image(uvd, images[0], rgba)

  
