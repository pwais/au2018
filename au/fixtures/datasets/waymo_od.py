import os
import copy
import itertools

import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av



## Utils

def to_nanostamp(timestamp_micros):
  return int(timestamp_micros) * 1000

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


class GetLidarCloudInEgo(object):

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

    # Waymo provides two returns for every lidar beam; user must
    # fuse them manually.
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

    # # 3d points in vehicle frame.
    # points_all = np.concatenate(points, axis=0)
    # points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # all_points = np.concatenate((points_all, points_all_ri2), axis=0)

    # Sadly this is how we must pack things for returning in TF Eager.
    # These should be in the same order as `lidar_names` in the caller.
    return [p for p in points] + [p for p in points_ri2]
  
  @classmethod
  def get_points(cls, waymo_frame):
    # waymo_open_dataset requires eager mode :( so we need to scope its use
    # into a tensorflow py_func
    if not hasattr(cls, '_thruput'):
      cls._thruput = util.ThruputObserver(name='GetLidarCloudInEgo')

    with cls._thruput.observe(n=1):
      os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # TF Eager is a load of trash and ignores our session config
      import tensorflow as tf
      import tensorflow.contrib.eager as tfe

      if not hasattr(cls, '_sess'):
        cls._sess = util.tf_cpu_session()

      # Ordered to match how waymo_open orders points within each tensor
      # https://github.com/waymo-research/waymo-open-dataset/blob/b38845a57aa2031dd717147aa438f2e3f2166a3b/waymo_open_dataset/utils/frame_utils.py#L103
      lidar_ids = sorted([
        c.name for c in waymo_frame.context.laser_calibrations])
      lidar_names = [get_lidar_name(lid) for lid in lidar_ids]

      wf = tf.placeholder(dtype=tf.string)
      n_lasers = len(waymo_frame.lasers)
      rets = tuple(tf.float32 for _ in lidar_names + lidar_names)
      pf = tfe.py_func(GetLidarCloudInEgo._get_all_points, [wf], rets)
      result = cls._sess.run(
        pf, feed_dict={wf: waymo_frame.SerializeToString()})

      # Unpack Tensorflow Eager trash BS
      returns = lidar_names + [name + '_return_2' for name in lidar_names]
      assert len(result) == len(returns)
      name_to_points = dict(zip(returns, result))
    
    cls._thruput.update_tallies(num_bytes=util.get_size_of_deep(name_to_points))
    cls._thruput.maybe_log_progress(every_n=10)
    
    return name_to_points



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
    # if not f.uri.timestamp:
    # If given a `record_idx`, we might not know the frame's timestamp
    # until we've fetched and decoded the `waymo_frame`.
    f.uri.timestamp = to_nanostamp(waymo_frame.timestamp_micros)
    cls._fill_ego_pose(f, waymo_frame)
    cls._fill_camera_images(f, waymo_frame)
    cls._fill_extra(f, waymo_frame)
    print(uri.segment_id, f.uri.timestamp, waymo_frame.timestamp_micros, type(waymo_frame.timestamp_micros), record_idx)
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
      name='create_frame_rdd_tasks',
      n_total=PARTITIONS_PER_SEGMENT * len(segment_uris))
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
        uri.timestamp = to_nanostamp(wf.timestamp_micros)
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
      wf = dataset_pb2.Frame()
      wf.ParseFromString(bytearray(s))
      return wf

    util.log.info("Doing expensive linear find ...")
    for wf in cls._iter_waymo_frames(uri.segment_id):
      timestamp = to_nanostamp(wf.timestamp_micros)
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
    ci_timestamp = to_nanostamp(wf_camera_image.pose_timestamp)

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
        timestamp=to_nanostamp(waymo_frame.timestamp_micros),
        cloud=GetLidarCloudInEgo.get_points(waymo_frame),
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
        timestamp=to_nanostamp(waymo_frame.timestamp_micros),
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
      












class StampedDatumTable(av.StampedDatumTableBase):

  FIXTURES = Fixtures

  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/waymo_open_datum_table'

  @classmethod
  def _create_datum_rdds(cls, spark):
    
    PARTITIONS_PER_SEGMENT = 25
      # Most segments have 10Hz * 20sec = 200 frames
    TASKS_PER_DATUM_RDD = os.cpu_count()

    datum_rdds = []
    segment_ids = cls.get_all_segment_ids()
    # t = util.ThruputObserver(
    #   name='create_datum_rdd_tasks',
    #   n_total=PARTITIONS_PER_SEGMENT * len(segment_ids))
    for segment in segment_ids:
      ipartition_ids = util.ichunked(
                          range(PARTITIONS_PER_SEGMENT), TASKS_PER_DATUM_RDD)
      for partition_ids in ipartition_ids:
        # with t.observe(n=len(partition_ids)):
          tasks = [(segment, partition) for partition in partition_ids]
          segment_uri_rdd = spark.sparkContext.parallelize(tasks)

          def iter_datums(task):
            print('working on %s' % (task,))
            segment_id, partition = task
            record = cls._get_segment_id_to_record()[segment_id]
            base_uri = av.URI(
                          dataset='waymo-od',
                          split=record.split,
                          segment_id=segment_id)

            tf_str_list = \
              util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
            for record_idx in range(len(tf_str_list)):
              if (record_idx % PARTITIONS_PER_SEGMENT) == partition:
                base_uri.extra = {'waymo_frame_record_idx': record_idx}
                for sd in cls.iter_stamped_datums_for_frame(base_uri):
                  yield sd
          
          datum_rdd = segment_uri_rdd.flatMap(iter_datums)
          datum_rdds.append(datum_rdd)
        # t.maybe_log_progress(every_n=500)
    
    return datum_rdds


  ## Public API

  @classmethod
  def get_all_segment_ids(cls):
    return list(cls._get_segment_id_to_record().keys())

  @classmethod
  def get_waymo_frame(cls, uri):
    segment_id = uri.segment_id
    record_idx = None
    if 'waymo_frame_record_idx' in uri.extra:
      record_idx = int(uri.extra['waymo_frame_record_idx'])

    # Breakcrumbs: the frame context and timestamp is embedded in the
    # serialized protobuf message, so even if we wanted to index the Waymo
    # TFRecord files, we'd have to read and decode all of them.  Thus for now
    # we just provide an expensive linear search to look up individual frames
    # and optimize just the linear read / ETL use case.
    if record_idx is not None:
      from waymo_open_dataset import dataset_pb2
      record = cls._get_segment_id_to_record()[segment_id]
      tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
      s = tf_str_list[record_idx]
      wf = dataset_pb2.Frame()
      wf.ParseFromString(bytearray(s))
      return wf

    assert uri.timestamp is not None, "Need either record_idx or timestamp"
    util.log.info("Doing expensive linear find ...")
    for wf in cls._iter_waymo_frames(segment_id):
      wf_timestamp = to_nanostamp(wf.timestamp_micros)
      if wf_timestamp == uri.timestamp:
        util.log.info("... found!")
        return wf
    raise ValueError("Frame not found for %s" % uri)

  @classmethod
  def iter_stamped_datums_for_frame(cls, uri, waymo_frame=None):
    if not waymo_frame:
      waymo_frame = cls.get_waymo_frame(uri)
    
    gens = itertools.chain.from_iterable((
      cls._gen_lidar_datums(uri, waymo_frame),
      cls._gen_camera_datums(uri, waymo_frame),
      cls._gen_cuboid_datums(uri, waymo_frame),
    ))
    for sd in gens:
      yield sd


  ## Support
  # Portions based upon https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/master/examples/example.py

  @classmethod
  def _gen_lidar_datums(cls, uri, waymo_frame):
    uri = copy.deepcopy(uri)
    
    ## Ego Pose
    # It seems ego pose for lidar is only available at the frame level;
    # cameras and everything else are synchronized to lidar.
    pose_matrix = np.reshape(waymo_frame.pose.transform, [4, 4])
    ego_pose = av.Transform(
                    rotation=pose_matrix[:3, :3],
                    translation=pose_matrix[:3, 3],
                    src_frame='city',
                    dest_frame='ego')
    
    uri.topic = 'ego_pose'
    uri.timestamp = to_nanostamp(waymo_frame.timestamp_micros)
    yield av.StampedDatum.from_uri(uri, transform=ego_pose)

    # TODO: motion correction / rolling shutter per-camera, ...
    name_to_points = GetLidarCloudInEgo.get_points(waymo_frame)
    for laser in waymo_frame.lasers:
      lidar_name = get_lidar_name(laser.name)
      
      ## Calibration
      laser_calibration = None
      for lc in waymo_frame.context.laser_calibrations:
        if lc.name == laser.name:
          laser_calibration = lc
      assert lc

      # "Lidar frame to vehicle frame" at time of writing
      ego_from_lidar = np.reshape(
        laser_calibration.extrinsic.transform, [4, 4])
      ego_to_lidar = np.linalg.inv(ego_from_lidar)
      ego_to_sensor = av.Transform(
                        rotation=ego_to_lidar[:3, :3],
                        translation=ego_to_lidar[:3, 3],
                        src_frame='ego',
                        dest_frame=lidar_name)
      
      ## First Return
      laser_points = name_to_points[lidar_name]
      pc = av.PointCloud(
        sensor_name=lidar_name,
        timestamp=to_nanostamp(waymo_frame.timestamp_micros),
        cloud=laser_points,
        motion_corrected=False,
        ego_to_sensor=ego_to_sensor, 
        ego_pose=ego_pose,
        # TODO maybe beam calibration as extra
      )
      uri.topic = 'lidar|' + pc.sensor_name
      uri.timestamp = pc.timestamp
      yield av.StampedDatum.from_uri(uri, point_cloud=pc)
      
      ## Second Return
      lidar_name_ri2 = lidar_name + '_return_2'
      laser_points_ri2 = name_to_points[lidar_name_ri2]
      pc = av.PointCloud(
        sensor_name=lidar_name_ri2,
        timestamp=to_nanostamp(waymo_frame.timestamp_micros),
        cloud=laser_points_ri2,
        motion_corrected=False,
        ego_to_sensor=ego_to_sensor, 
        ego_pose=ego_pose,
        # TODO maybe beam calibration as extra
      )
      uri.topic = 'lidar|' + pc.sensor_name
      uri.timestamp = pc.timestamp
      yield av.StampedDatum.from_uri(uri, point_cloud=pc)

  @classmethod
  def _gen_camera_datums(cls, uri, waymo_frame):

    uri = copy.deepcopy(uri)
    for wf_camera_image in waymo_frame.images:
      ci_timestamp = int(wf_camera_image.pose_timestamp * 1e9)
        # NB: no idea why waymo obfuscates camera pose stamp this way...

      ## Ego Pose
      pose_matrix = np.reshape(wf_camera_image.pose.transform, [4, 4])
      ego_pose = av.Transform(
                      rotation=pose_matrix[:3, :3],
                      translation=pose_matrix[:3, 3],
                      src_frame='city',
                      dest_frame='ego')
      
      uri.topic = 'ego_pose'
      uri.timestamp = ci_timestamp
      yield av.StampedDatum.from_uri(uri, transform=ego_pose)

      ## Camera Image
      camera_name = get_camera_name(wf_camera_image.name)

      w, h = get_jpeg_size(wf_camera_image.image)
      viewport = uri.get_viewport()
      if not viewport:
        from au.fixtures.datasets import common
        viewport = common.BBox.of_size(w, h)
      
      camera_calibration = None
      for cc in waymo_frame.context.camera_calibrations:
        if cc.name == wf_camera_image.name:
          camera_calibration = cc
      assert camera_calibration

      # NB: Waymo protobuf defs had an error; the protobufs contain
      # the inverse of the expected transform.
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
          camera_name=camera_name,
          image_jpeg=bytearray(wf_camera_image.image),
          height=h,
          width=w,
          viewport=viewport,
          timestamp=ci_timestamp,
          ego_pose=ego_pose,
          cam_from_ego=cam_from_ego,
          K=K,
          principal_axis_in_ego=principal_axis_in_ego,
        )
      uri.topic = 'camera|' + ci.camera_name
      uri.timestamp = ci_timestamp
      yield av.StampedDatum.from_uri(uri, camera_image=ci)
      
  @classmethod
  def _gen_cuboid_datums(cls, uri, waymo_frame):
    
    # Cuboid ego pose is lidar ego pose
    pose_matrix = np.reshape(waymo_frame.pose.transform, [4, 4])
    ego_pose = av.Transform(
                    rotation=pose_matrix[:3, :3],
                    translation=pose_matrix[:3, 3],
                    src_frame='city',
                    dest_frame='ego')
    timestamp = to_nanostamp(waymo_frame.timestamp_micros)

    cuboids = []
    for label in waymo_frame.laser_labels:
      box = label.box
      
      # Coords in ego frame
      T = np.array([
            [box.center_x],
            [box.center_y],
            [box.center_z]])
      
      # Heading is yaw in ego frame
      from scipy.spatial.transform import Rotation
      R = Rotation.from_euler('z', box.heading).as_dcm()
      
      obj_from_ego = av.Transform(
        rotation=R, translation=T, src_frame='ego', dest_frame='obj')

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
        
        # Frame-level context
        'waymo.time_of_day':  waymo_frame.context.stats.time_of_day,
        'waymo.location':     waymo_frame.context.stats.location,
        'waymo.weather':      waymo_frame.context.stats.weather,
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
        timestamp=timestamp,
        box3d=box3d,
        motion_corrected=False, # TODO
        length_meters=l,
        width_meters=w,
        height_meters=h,
        distance_meters=np.linalg.norm(T),
        obj_from_ego=obj_from_ego,
        ego_pose=ego_pose,
        extra=extra,
      )
      cuboids.append(cuboid)
    
    uri.topic = 'labels|cuboids'
    uri.timestamp = timestamp
    yield av.StampedDatum.from_uri(uri, cuboids=cuboids)


  ## Support: Segment Access

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