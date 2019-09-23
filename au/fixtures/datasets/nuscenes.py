
import os

import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av


## Utils
import shelve
from nuscenes.nuscenes import NuScenes
class MemoryEfficientNuScenes(NuScenes):
  def __init__(self, **kwargs):
    
    cache_dir = '/opt/au/cache/nuscenes_tables/' + kwargs['version'] + '/'
    if not os.path.exists(cache_dir):
      util.mkdir(cache_dir)
      util.log.info("Creating shelve caches.  Reading source JSON ...")
      nusc = NuScenes(**kwargs)
      util.log.info("... done loading JSON data ...")
      for table_name in nusc.table_names:
        util.log.info("Building shelve cache file %s ..." % table_name)
        cache_path = os.path.join(cache_dir, table_name)

        import pickle
        d = shelve.open(cache_path, protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name)
        d.update((r['token'], r) for r in rows)
        d.close()
      util.log.info("... done.")
    
    super(MemoryEfficientNuScenes, self).__init__(**kwargs)

  def _get_table_map(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):  
      cache_dir = '/opt/au/cache/nuscenes_tables/' + self.version + '/'
      cache_path = os.path.join(cache_dir, table_name)
      util.log.info("Using shelve cache %s" % cache_path)
      import pickle
      d = shelve.open(cache_path, protocol=pickle.HIGHEST_PROTOCOL)
      setattr(self, attr, d)
    return getattr(self, attr)

  def __load_table__(self, table_name):
    return self._get_table_map(table_name).values()
  
  def __make_reverse_index__(self, verbose):
    # Shelve data files have reverse indicies built-in
    return
  
  def get(self, table_name, token):
    assert table_name in self.table_names, \
      "Table {} not found".format(table_name)
    return self._get_table_map(table_name)[token]
  
  def getind(self, table_name, token):
    raise ValueError("Unsupported in this hack")


def transform_from_record(rec):
  from pyquaternion import Quaternion
  return av.Transform(
          rotation=Quaternion(rec['rotation']).rotation_matrix,
          translation=np.array(rec['translation']).reshape((3, 1)))

def get_camera_normal(K, extrinsic):
    """FMI see au.fixtures.datasets.auargoverse.get_camera_normal()"""

    # Build P
    # P = |K 0| * | R |T|
    #             |000 1|
    K_h = np.zeros((3, 4))
    K_h[:3, :3] = K
    P = K_h.dot(extrinsic)

    # Zisserman pg 161 The principal axis vector.
    # P = [M | p4]; M = |..|
    #                   |m3|
    # pv = det(M) * m3
    pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
    pv_hat = pv / np.linalg.norm(pv)
    return pv_hat


## Data

class Fixtures(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'nuscenes')

  TARBALLS = (
    'v1.0-mini.tar',
    
    'v1.0-test_meta.tar',
    'v1.0-test_blobs.tar',

    'v1.0-trainval01_blobs.tar',
    'v1.0-trainval02_blobs.tar',
    'v1.0-trainval03_blobs.tar',
    'v1.0-trainval04_blobs.tar',
    'v1.0-trainval05_blobs.tar',
    'v1.0-trainval06_blobs.tar',
    'v1.0-trainval07_blobs.tar',
    'v1.0-trainval08_blobs.tar',
    'v1.0-trainval09_blobs.tar',
    'v1.0-trainval10_blobs.tar',

    'nuScenes-map-expansion.zip',
  )

  MINI_TARBALL = 'v1.0-mini.tar'

  SPLITS = ('train', 'val', 'test', 'mini')
  
  TRAIN_TEST_SPLITS = ('train', 'val')

  ## Source Data

  @classmethod
  def tarballs_dir(cls):
    return os.path.join(cls.ROOT, 'tarballs')

  @classmethod
  def tarball_path(cls, fname):
    return os.path.join(cls.tarballs_dir(), fname)

  # @classmethod
  # def tarball_dir(cls, fname):
  #   """Get the directory for an uncompressed tarball with `fname`"""
  #   dirname = fname.replace('.tar.gz', '')
  #   return cls.tarball_path(dirname)

  # @classmethod
  # def all_tarballs(cls):
  #   return list(
  #     itertools.chain.from_iterable(
  #       getattr(cls, attr, [])
  #       for attr in dir(cls)
  #       if attr.endswith('_TARBALLS')))


  ## Derived Data
  
  @classmethod
  def dataroot(cls):
    return '/outer_root/media/seagates-ext4/au_datas/nuscenes_root'

  # @classmethod
  # def dataroot(cls):
  #   return os.path.join(cls.ROOT, 'nuscenes_dataroot')

  @classmethod
  def index_root(cls):
    return os.path.join(cls.ROOT, 'index')
  

  ## Setup

  @classmethod
  def run_import(cls, only_mini=False):
    pass

  ## Public API

  @classmethod
  def get_loader(cls, version='v1.0-trainval'):
    """Return a `nuscenes.nuscenes.NuScenes` object for the
    dataset with `version`."""
    # from nuscenes.nuscenes import NuScenes
    # nusc = NuScenes(version=version, dataroot=cls.dataroot(), verbose=True)
    # import pdb; pdb.set_trace()
    nusc = MemoryEfficientNuScenes(version=version, dataroot=cls.dataroot(), verbose=True)
    return nusc
  
  @classmethod
  def get_split_for_scene(cls, scene):
    if not hasattr(cls, '_scene_to_split'):
      from nuscenes.utils.splits import create_splits_scenes
      split_to_scenes = create_splits_scenes()

      scene_to_split = {}
      for split, scenes in split_to_scenes.items():
        # Ignore mini splits because they duplicate train/val
        if 'mini' not in split:
          for s in scenes:
            scene_to_split[s] = split
      cls._scene_to_split = scene_to_split
    return cls._scene_to_split[scene]
        


class FrameTable(av.FrameTableBase):

  FIXTURES = Fixtures

  NUSC_VERSION = 'v1.0-trainval' # E.g. v1.0-mini, v1.0-trainval, v1.0-test

  PROJECT_CLOUDS_TO_CAM = True
  PROJECT_CUBOIDS_TO_CAM = True
  IGNORE_INVISIBLE_CUBOIDS = True
  
  SETUP_URIS_PER_CHUNK = 100

  ## Subclass API

  @classmethod
  def _create_frame_rdds(cls, spark):
    uris = cls._get_camera_uris()
    print('len(uris)', len(uris))

    # TODO fixmes
    uri_rdd = spark.sparkContext.parallelize(uris)

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

      frame_rdds.append(frame_rdd)
    return frame_rdds
  
  ## Public API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/nusc_frame_table'

  @classmethod
  def create_frame(cls, uri):
    nusc = cls.get_nusc() 
    scene_to_ts_to_sample_token = cls._scene_to_ts_to_sample_token()
    sample_token = scene_to_ts_to_sample_token[uri.segment_id][uri.timestamp]
    sample = nusc.get('sample', sample_token)
    return cls._create_frame_from_sample(uri, sample)

  @classmethod
  def get_nusc(cls):
    if not hasattr(cls, '_nusc'):
      cls._nusc = cls.FIXTURES.get_loader(version=cls.NUSC_VERSION)
    return cls._nusc


  ## Support

  @classmethod
  def _get_camera_uris(cls, splits=None):
    nusc = cls.get_nusc()

    if not splits:
      splits = cls.FIXTURES.TRAIN_TEST_SPLITS

    uris = []
    for sample in nusc.sample:
      scene_record = nusc.get('scene', sample['scene_token'])
      scene_split = cls.FIXTURES.get_split_for_scene(scene_record['name'])
      if scene_split not in splits:
        continue

      for sensor, token in sample['data'].items():
        sample_data = nusc.get('sample_data', token)
        if sample_data['sensor_modality'] == 'camera':
          uri = av.URI(
                  dataset='nuscenes',
                  split=scene_split,
                  timestamp=sample['timestamp'],
                  segment_id=scene_record['name'],
                  camera=sensor)
          uris.append(uri)

    return uris
  
  @classmethod
  def _scene_to_ts_to_sample_token(cls):
    if not hasattr(cls, '__scene_to_ts_to_sample_token'):
      nusc = cls.get_nusc()
      scene_name_to_token = dict(
        (scene['name'], scene['token']) for scene in nusc.scene)
    
      from collections import defaultdict
      scene_to_ts_to_sample_token = defaultdict(dict)
      for sample in nusc.sample:
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        timestamp = sample['timestamp']
        token = sample['token']
        scene_to_ts_to_sample_token[scene_name][timestamp] = token
      
      cls.__scene_to_ts_to_sample_token = scene_to_ts_to_sample_token
    return cls.__scene_to_ts_to_sample_token

  @classmethod
  def _create_frame_from_sample(cls, uri, sample):
    f = av.Frame(uri=uri)
    cls._fill_ego_pose(uri, sample, f)
    cls._fill_camera_images(uri, sample, f)
    return f
  
  @classmethod
  def _fill_ego_pose(cls, uri, sample, f):
    nusc = cls.get_nusc()

    # For now, always set ego pose using the *lidar* timestamp, as is done
    # in nuscenes.  (They probably localize mostly from lidar anyways).
    token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    f.world_to_ego = transform_from_record(pose_record)

  @classmethod
  def _fill_camera_images(cls, uri, sample, f):
    nusc = cls.get_nusc()
    if uri.camera:
      camera_token = sample['data'][uri.camera]
      ci = cls._get_camera_image(uri, camera_token, f)
      f.camera_images.append(ci)
    else:
      raise ValueError("Grouping multiple cameras etc into frames TODO")
  
  @classmethod
  def _get_camera_image(cls, uri, camera_token, f):
    nusc = cls.get_nusc()
    sd_record = nusc.get('sample_data', camera_token)
    cs_record = nusc.get(
      'calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path, _, cam_intrinsic = nusc.get_sample_data(camera_token)
      # Ignore box_list, we'll get boxes in ego frame later
    
    viewport = uri.get_viewport()
    w, h = sd_record['width'], sd_record['height']
    if not viewport:
      from au.fixtures.datasets import common
      viewport = common.BBox.of_size(w, h)

    timestamp = sd_record['timestamp']

    ego_from_cam = transform_from_record(cs_record)
    cam_from_ego = ego_from_cam.get_inverse()
    RT_h = cam_from_ego.get_transformation_matrix(homogeneous=True)
    principal_axis_in_ego = get_camera_normal(cam_intrinsic, RT_h)
    
    ci = av.CameraImage(
          camera_name=uri.camera,
          image_jpeg=bytearray(open(data_path, 'rb').read()),
          height=h,
          width=w,
          viewport=viewport,
          timestamp=timestamp,
          cam_from_ego=cam_from_ego,
          K=cam_intrinsic,
          principal_axis_in_ego=principal_axis_in_ego,
        )
    
    if cls.PROJECT_CLOUDS_TO_CAM:
      for sensor in ('LIDAR_TOP',):
        sample = nusc.get('sample', sd_record['sample_token'])
        pc = cls._get_point_cloud_in_ego(sample, sensor=sensor)

        # Project to image
        pc.cloud = ci.project_ego_to_image(pc.cloud, omit_offscreen=True)
        pc.sensor_name = pc.sensor_name + '_in_cam'
        ci.cloud = pc
      
    if cls.PROJECT_CUBOIDS_TO_CAM:
      sample_data_token = sd_record['token']
      cuboids = cls._get_cuboids_in_ego(sample_data_token)
      for cuboid in cuboids:
        bbox = ci.project_cuboid_to_bbox(cuboid)
        if cls.IGNORE_INVISIBLE_CUBOIDS and not bbox.is_visible:
          continue
        ci.bboxes.append(bbox)
    
    return ci

  @classmethod
  def _get_point_cloud_in_ego(cls, sample, sensor='LIDAR_TOP'):
    # Based upon nuscenes.py#map_pointcloud_to_image()
    import os.path as osp
    
    from pyquaternion import Quaternion

    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.data_classes import RadarPointCloud
    
    nusc = cls.get_nusc()
    pointsensor_token = sample['data'][sensor]
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
      pc = LidarPointCloud.from_file(pcl_path)
    else:
      pc = RadarPointCloud.from_file(pcl_path)

    # Points live in the point sensor frame, so transform to ego frame
    cs_record = nusc.get(
      'calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    n_xyz = pc.points[:3, :].T
      # Throw out intensity (lidar) and ... other stuff (radar)
    return av.PointCloud(
      sensor_name=sensor,
      timestamp=pointsensor['timestamp'],
      cloud=n_xyz,
      ego_to_sensor=transform_from_record(cs_record),
      motion_corrected=False, # TODO interpolation for cam ~~~~~~~~~~~~~~~~~~~~~~~
    )
    
  @classmethod
  def _get_cuboids_in_ego(cls, sample_data_token):
    nusc = cls.get_nusc()
    boxes = nusc.get_boxes(sample_data_token)
  
    # Boxes are in world frame.  Move all to ego frame.
    from pyquaternion import Quaternion
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for box in boxes:
      # Move box to ego vehicle coord system
      box.translate(-np.array(pose_record['translation']))
      box.rotate(Quaternion(pose_record['rotation']).inverse)

    from au.fixtures.datasets.av import NUSCENES_CATEGORY_TO_AU_AV_CATEGORY
    cuboids = []
    for box in boxes:
      cuboid = av.Cuboid()

      # Core
      sample_anno = nusc.get('sample_annotation', box.token)
      cuboid.track_id = \
        'nuscenes_instance_token:' + sample_anno['instance_token']
      cuboid.category_name = box.name
      cuboid.timestamp = sd_record['timestamp']
      
      attribs = [
        nusc.get('attribute', attrib_token)['name']
        for attrib_token in sample_anno['attribute_tokens']
      ]
      cuboid.au_category = NUSCENES_CATEGORY_TO_AU_AV_CATEGORY[box.name]
      if 'cycle.with_rider' in attribs:
        if box.name == 'vehicle.bicycle':
          cuboid.au_category = 'bike_with_rider'
        else: # Probably vehicle.motorcycle 
          cuboid.au_category = 'motorcycle_with_rider'

      cuboid.extra = {
        'nuscenes_token': box.token,
        'nuscenes_attribs': '|'.join(attribs),
      }

      # Points
      cuboid.box3d = box.corners().T
      cuboid.motion_corrected = False # TODO interpolation ? ~~~~~~~~~~~~~~~~~~~~
      cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
      
      # Pose
      cuboid.width_meters = float(box.wlh[0])
      cuboid.length_meters = float(box.wlh[1])
      cuboid.height_meters = float(box.wlh[2])

      cuboid.obj_from_ego = av.Transform(
          rotation=box.orientation.rotation_matrix,
          translation=box.center.reshape((3, 1)))
      cuboids.append(cuboid)
    return cuboids