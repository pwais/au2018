import itertools
import os

import klepto
import numpy as np
import pandas as pd

from au import util
from au.fixtures.datasets import av

def read_xyz_from_ply(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  # Header is 7 lines
  lines = lines[7:]
  def to_v(l):
    x, y, z = l.split()
    return float(x), float(y), float(z)
  xyz = np.array([to_v(l) for l in lines])

  # TODO keep me? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  from math import pi
  from scipy.spatial.transform import Rotation as R
  axes_transformation = R.from_euler('zyx', [-pi/2, -pi/2, 0]).as_dcm()
  xyz = axes_transformation.dot(xyz.T).T

  return xyz

def read_K_from_path(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  itoks = itertools.chain.from_iterable(l.split() for l in lines)
  K = np.array([float(t) for t in itoks]).reshape((3, 3))
  return K

def read_RT_from_path(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  # lines 0 and 4 are newlines
  Rr1 = lines[1]
  Rr2 = lines[2]
  Rr3 = lines[3]
  R = np.array([
    [float(v) for v in Rr1.split()],
    [float(v) for v in Rr2.split()],
    [float(v) for v in Rr3.split()],
  ]).reshape((3, 3))
  T = np.array([float(v) for v in lines[5].split()]).reshape((3, 1))


  # TODO keep me? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  from math import pi
  from scipy.spatial.transform import Rotation
  axes_transformation = Rotation.from_euler('zyx', [-pi/2, -pi/2, 0]).as_dcm()
  R = axes_transformation * R

  return R, T

### Data

class Fixtures(object):

  TRAIN_ROOT = '/opt/au/au/fixtures/datasets/av_spheres/tast'


### StampedDatumTable

class StampedDatumTable(av.StampedDatumTableBase):

  FIXTURES = Fixtures

  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/av_spheres_datum'
  
  @classmethod
  def _create_datum_rdds(cls, spark):
    URIS_PER_TASK = 100
    datum_rdds = []
    for uri_chunk in util.ichunked(cls.iter_all_uris(), URIS_PER_TASK):
      uri_rdd = spark.sparkContext.parallelize(uri_chunk)
      datum_rdd = uri_rdd.map(cls.create_stamped_datum)
      datum_rdds.append(datum_rdd)
    return datum_rdds


  ## Public API

  @classmethod
  def iter_all_uris(cls):
    TOPICS = ('camera_', 'lidar_', 'ego_pose', 'cuboids')

    df = cls.artifact_df()
    ts_t_df = df[['timestamp', 'topic']].drop_duplicates()
    for row in ts_t_df.to_dict(orient='recrods').items():
      if any(t in row['topic'] for t in TOPICS):
        yield av.URI(
          dataset='av_spheres',
          split='train', # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          segment_id='TODO',
          topic=row['topic'],
          timestamp=row['timestamp')

  @classmethod
  def create_stamped_datum(cls, uri):
    # get df rows for (topic, timestamp)
    if uri.topic.startswith('lidar_'):
      return cls._create_point_cloud(uri)
    elif uri.topic.startswith('camera_'):
      return cls._create_camera_image(uri)
    elif uri.topic == 'cuboids':
      return cls._create_cuboids(uri)
    elif uri.topic == 'ego_pose':
      return cls._create_ego_pose(uri)
    else:
      raise ValueError("Don't know what to do with %s" % uri)


  ## Support

  @classmethod
  def artifact_df(cls):
    if not hasattr(cls, '_artifact_df'):
      def to_row(path):
        fname = os.path.basename(path)
        try:
          timestamp, topic, prop, ext = fname.split('.')
          return {
            'timestamp': int(timestamp),
            'topic': topic,
            'prop': prop,
            'path': path,
          }
        except Exception:
          return None
      dirpath = cls.FIXTURES.TRAIN_ROOT # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
      rows = [to_row(p) for p in util.all_files_recursive(dirpath)]
      cls._artifact_df = pd.DataFrame([r for r in rows if r is not None])
    return cls._artifact_df


  ## Creating Stamped Datums

  @classmethod
  @klepto.lru_cache(ignore=(0,), maxsize=10000)
  def _get_transform(cls, uri):
    df = cls.artifact_df()
    if uri.topic == 'ego_pose':
      query = "topic == 'ego_pose' and timestamp == %s" % uri.timestamp
      src_frame = 'city'
      dest_frame = 'ego'
    else:
      query = "topic == 'extrinsic' and prop == '%s'" % uri.topic
      src_frame = 'ego'
      dest_frame = uri.topic
    rows = df.query(query).to_dict(orient='records')
    assert rows and len(rows) == 1
    path = rows[0]['path']
    R, T = read_RT_from_path(path)

    # TODO: do axis shift? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return av.Transform(
                rotation=R,
                translation=T,
                src_frame=src_frame,
                dest_frame=dest_frame)

  @classmethod
  @klepto.lru_cache(ignore=(0,), maxsize=1000)
  def _get_path(cls, uri, prop):
    df = cls.artifact_df()
    query = "topic =='%s' and timestamp == %s and prop == '%s'" % (
      uri.topic, uri.timestamp, prop)
    rows = df.query(query).to_dict(orient='records')
    assert rows and len(rows) == 1
    return rows[0]['path']


  @classmethod
  def _create_point_cloud(cls, uri):
    ego_to_sensor = cls._get_transform(uri)
    ego_pose = cls._get_transform(
      av.URI(timestamp=uri.timestamp, topic='ego_pose'))

    pc_path = cls._get_path(uri, 'points')
    cloud = read_xyz_from_ply(path)

    pc = av.PointCloud(
        sensor_name=uri.topic,
        timestamp=uri.timestamp,
        cloud=cloud,
        motion_corrected=False,
        ego_to_sensor=ego_to_sensor,
        ego_pose=ego_pose,
    )
    return av.StampedDatum.from_uri(uri, point_cloud=pc)
  
  @classmethod
  def _create_camera_image(cls, uri):
    ego_to_sensor = cls._get_transform(uri)
    ego_pose = cls._get_transform(
      av.URI(timestamp=uri.timestamp, topic='ego_pose'))
    
    K_path = cls._get_path(av.URI(topic='intrinsic', timestamp=0), uri.topic)
    K = read_K_from_path(K_path)

    visible_path = cls._get_path(uri, 'visible')
    image_jpeg = bytearray(open(visible_path, 'rb').read())
    w, h = util.get_jpeg_size(image_jpeg)
    viewport = uri.get_viewport()
    if not viewport:
      from au.fixtures.datasets import common
      viewport = common.BBox.of_size(w, h)
    
    # TODO fixme? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To get the principal axis, we can use `ego_to_sensor` (which
    # neglects the rotation into the image plane) and simply rotate X_HAT
    # into the camera frame.
    X_HAT = np.array([1, 0, 0])
    principal_axis_in_ego = ego_to_sensor.rotation.dot(X_HAT)

    ci = av.CameraImage(
        camera_name=uri.topic,
        image_jpeg=bytearray(open(visible_path, 'rb').read()),
        height=h,
        width=w,
        viewport=viewport,
        timestamp=uri.timestamp,
        ego_pose=ego_pose,
        cam_from_ego=ego_to_sensor.get_inverse(),
        K=K,
        principal_axis_in_ego=principal_axis_in_ego,
    )

    return av.StampedDatum.from_uri(uri, camera_image=ci)

  @classmethod
  def _create_cuboids(cls, uri):
    ego_pose = cls._get_transform(
      av.URI(timestamp=uri.timestamp, topic='ego_pose'))

    df = cls.artifact_df()
    query = "topic =='cuboids' and timestamp == %s" % uri.timestamp
    rows = df.query(query).to_dict(orient='records')
    cuboids = []
    for row in rows:
      xyz = read_xyz_from_ply(row['path'])
      track_id = row['prop']

      # TODO which frame?? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      l = xyz[:,0].max() - xyz[:,0].min()
      w = xyz[:,1].max() - xyz[:,1].min()
      h = xyz[:,2].max() - xyz[:,2].min()

      front = xyz[(0,1,2,3),:]
      back =  xyz[(4,5,6,7),:]
      obj_normal = front.mean() - back.mean()
      obj_normal = obj_hat / np.linalg.norm(obj_hat)

      import math
      from scipy.spatial.transform import Rotation
      X_HAT = np.array([1, 0, 0])
      cos_theta = obj_normal.dot(X_HAT)
      rot_axis = np.cross(X_HAT, obj_normal)
      R = Rotation.from_rotvec(math.acos(cos_theta) * rot_axis).as_dcm()
      T = xyz.mean()
      obj_from_ego = av.Transform(
        rotation=R, translation=T, src_frame='ego', dest_frame='obj')
      
      category_name = 'car'
      au_category = 'car'

      cuboid = av.Cuboid(
        track_id=track_id,
        category_name=category_name,
        au_category=au_category,
        timestamp=uri.timestamp,
        box3d=xyz,
        motion_corrected=False,
        length_meters=l,
        width_meters=w,
        height_meters=h,
        distance_meters=np.linalg.norm(T),
        obj_from_ego=obj_from_ego,
        ego_pose=ego_pose,
      )
      cuboids.append(cuboid)
    
    uri.topic = 'labels|cuboids'
    yield av.StampedDatum.from_uri(uri, cuboids=cuboids)


  @classmethod
  def _create_ego_pose(cls, uri):
    ego_pose = cls._get_transform(
      av.URI(timestamp=uri.timestamp, topic='ego_pose'))
    return av.StampedDatum.from_uri(uri, transform=ego_pose)

