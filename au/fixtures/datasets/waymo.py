import os
import copy
import itertools

import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av



## Utils

def transform_from_pb(waymo_transform):
  RT = np.reshape(waymo_transform.transform, [4, 4])
  return av.Transform(rotation=RT[:3, :3], translation=RT[:3, 3])



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
    return os.path.join(cls.ROOT, 'tarballs')

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

  SETUP_SEGMENTS_PER_CHUNK = os.cpu_count()

  ## Subclass API

  @classmethod
  def create_frame(cls, uri, waymo_frame=None):
    f = av.Frame(uri=uri)
    if waymo_frame is None:
      waymo_frame = cls._get_waymo_frame(uri)
        # NB: this operation is expensive! See comments embedded in function
    cls._fill_ego_pose(f, waymo_frame)
    cls._fill_camera_images(f, waymo_frame)
    return f

  @classmethod
  def _create_frame_rdds(cls, spark):
    frame_rdds = []
    for segment_uris in util.ichunked(uris, cls.SETUP_SEGMENTS_PER_CHUNK):
      
      def iter_segment_frames(uri):
        # Breadcrumbs: to get any frame information at all, we must read and
        # decode protobufs from the whole TFRecord file.  So don't compute
        # Frame URIs, as we do for other datasets, but just generate Frames.
        for wf in cls._iter_waymo_frames(uri.segment_id):
          frame_uri = copy.deepcopy(uri)
          frame_uri.timestamp = int(wf.teimstamp_micros * 1e3)
          yield cls.create_frame(uri, waymo_frame=wf)
      
      segment_uri_rdd = spark.sparkContext.parallelize(segment_uris)
      frame_rdd = segment_uri_rdd.flatMap(iter_segment_frames)
      frame_rdds.append(frame_rdd)
    
    return frame_rdds



  ## Public API

  @classmethod
  def iter_all_segment_uris(cls):
    for segment_id, record in self._segment_id_to_record.items():
      yield av.URI(
              dataset='waymo-od',
              split=record.split,
              segment_id=segment_id)      
  
  ## Support

  class _SegmentRecord(object):
    __slots__ = ('fw', 'split')
    def get_reader(self):
      return self.fw.data_reader

  @classmethod
  def _segment_id_to_record(cls):
    if not hasattr(cls, '__segment_id_to_record'):
      util.log.info("Scanning all tarballs ...")
      
      segment_id_to_record = {}
      for fname in cls.FIXTURES.TARBALLS:
        path = cls.FIXTURES.tarball_path(fname)
        fws = util.ArchiveFileFlyweight.fws_from(path)
        for fw in fws:
          if fw.name.endswith('tfrecord'):
            record = _SegmentRecord()
            record.fw = fw
            record.split = cls.FIXTURES.get_split(fname)
            segment_id_to_record[fw.name] = record

      util.log.info("... found %s segments." % len(segment_id_to_record))
      cls.__segment_id_to_record = segment_id_to_record
    return cls.__segment_id_to_record
  
  @classmethod
  def _iter_waymo_frames(cls, segment_id):
    from waymo_open_dataset import dataset_pb2 as open_dataset
    record = cls._segment_id_to_record[segment_id]
    tf_str_list = util.TFRecordsFileAsListOfStrings(record.fw.data_reader)
    for s in tf_str_list:
      wf = open_dataset.Frame()
      wf.ParseFromString(bytearray(s))
      yield wf
  
  @classmethod
  def _get_waymo_frame(cls, uri):
    # Breakcrumbs: the frame context and timestamp is embedded in the
    # serialized protobuf message, so even if we wanted to index the Waymo
    # TFRecord files, we'd have to read and decode all of them.  Thus for now
    # we just provide an expensive linear search to look up individual frames
    # and optimize just the linear read / ETL use case.
    for wf in cls._iter_waymo_frames(uri.segment_id):
      timestamp = int(wf.timestamp_micros * 1e3)
      if timestamp == uri.timestamp:
        return wf
    raise ValueError("Frame not found for %s" % uri)
  
  # Filling Frames

  @classmethod
  def _fill_ego_pose(cls, f, waymo_frame):
    f.world_to_ego = transform_from_pb(waymo_frame.pose)
  
  @classmethod
  def _fill_camera_images(cls, f, waymo_frame):
    for wf_camera_image in waymo_frame.images:
      frame_uri.camera = wf_camera_image.name


points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=1)



# 3d points in vehicle frame.
points_all = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)

print(points_all.shape)
print(points_all_ri2.shape)

# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)



n = 0
camera_calibration = frame.context.camera_calibrations[n]
extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
vehicle_to_sensor = tf.matrix_inverse(extrinsic).numpy()

f_u = camera_calibration.intrinsic[0]
f_v = camera_calibration.intrinsic[1]
c_u = camera_calibration.intrinsic[2]
c_v = camera_calibration.intrinsic[3]

K = np.array([
    [f_u, 0,   c_u],
    [0,   f_v, c_v],
    [0,   0,   1  ],
])
print('K', K, (f_u, f_v, c_u, c_v))

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy array `pts` to Homogeneous coordinates of target `dim`
  if necessary"""
  if len(pts.shape) != dim + 1:
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts

p = maybe_make_homogeneous(points_all, dim=4)
print('p', p)
p_cam = vehicle_to_sensor.dot(p.T)
print('p_cam', p_cam)

plt.figure(figsize=(20, 12))
plt.scatter(p_cam[0, :], p_cam[1, :])

# plt.figure(figsize=(20, 12))
# plt.scatter(p_cam[0, :], p_cam[2, :])

# plt.figure(figsize=(20, 12))
# plt.scatter(p_cam[1, :], p_cam[2, :])


# p_cam[:2, :] = -p_cam[:2, :] / p_cam[2, :]
p_cam = p_cam[(2, 1, 0), :]
p_cam = p_cam[(1, 0, 2), :]
# p_cam[2, :] *= -1
p_cam[1, :] *= -1
p_cam[0, :] *= -1

uv = K.dot(p_cam)
uv[:2, :] /= uv[2, :]

# uv *= 1e-4

print('uv', uv.shape, ((uv[0, :].min(), uv[0, :].max(), uv[0, :].mean())))


plt.figure(figsize=(20, 12))
plt.scatter(uv[0, :], uv[1, :])

# images = sorted(frame.images, key=lambda i:i.name)
image = tf.image.decode_jpeg(frame.images[n].image)
h, w, c = image.numpy().shape
print('hwc', (h, w, c))


idx_ = np.where(
        np.logical_and.reduce((
          # Filter offscreen points
          0 <= uv[0, :], uv[0, :] < w - 1.0,
          0 <= uv[1, :], uv[1, :] < h - 1.0,
          # Filter behind-screen points
          uv[2, :] > 0)))
idx_ = idx_[0]
print('idx', len(idx_))
uv = uv[:, idx_]
uvd = uv.T

plot_points_on_image(uvd, images[0], rgba)

  
