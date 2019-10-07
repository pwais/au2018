
import itertools
import os

import pandas as pd
import numpy as np

from au import conf
from au import util
from au.fixtures.datasets import av



## Utils

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

import shelve
from nuscenes.nuscenes import NuScenes
class AUNuScenes(NuScenes):

  ### Memory Efficiency
  # The base NuScenes object uses 8GB resident RAM (each instance) due to
  # the "tables" of JSON data that it loads.  Below we replace these "tables"
  # with disk-based `shelve`s in order to dramatically reduce memory usage.
  # This change is needed in order to support instantiating multiple
  # NuScenes readers per machine (e.g. for Spark)

  CACHE_ROOT = os.path.join(conf.AU_DATA_CACHE, 'nuscenes_table_cache')

  SAMPLE_DATA_TS_CACHE_NAME = 'sample_data_ts_df.parquet'

  ALL_CAMERAS = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
  )

  def _get_cache_path(self, cache_name):
    return os.path.join(self.CACHE_ROOT, self.version, cache_name)

  def __init__(self, **kwargs):
    self.version = kwargs['version']
      # Base ctor does this, but we'll do it here for convenience
    
    if util.missing_or_empty(self._get_cache_path('')):
      util.log.info("Creating shelve caches.  Reading source JSON ...")
      nusc = NuScenes(**kwargs)
        # NB: The above ctor call not only loads all JSON but also runs
        # 'reverse indexing', which edits the data loaded into memory.  We'll
        # then write the edited data below using `shelve` so that we don't have
        # to try to make `AUNuScenes` support reverse indexing itself.
      util.log.info("... done loading JSON data ...")
      
      for table_name in nusc.table_names:
        util.log.info("Building shelve cache file %s ..." % table_name)
        
        cache_path = self._get_cache_path(table_name)
        util.mkdir(os.path.dirname(cache_path))
        
        import pickle
        d = shelve.open(cache_path, protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name)
        d.update((r['token'], r) for r in rows)
        d.close()
      util.log.info("... done.")
      del nusc # Free several GB memory
    
    super(AUNuScenes, self).__init__(**kwargs)

  def _get_table(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):
      cache_path = self._get_cache_path(table_name)
      util.log.info("Using shelve cache %s" % cache_path)
      d = shelve.open(cache_path)
      setattr(self, attr, d)
    return getattr(self, attr)

  def __load_table__(self, table_name):
    return self._get_table(table_name).values()
      # Despite the type annotation, the parent class actually returns a list
      # of dicts.  This return type is a Values View (a generator-like thing)
      # and does not break any core NuScenes functionality.
  
  def __make_reverse_index__(self, verbose):
    # NB: Shelve data files have, built-in, the reverse indicies that the
    # base `NuScenes` creates.  See above.
    
    # Build a timestamp index over `sample_data`s to support efficient
    # interpolation.
    cache_path = self._get_cache_path(self.SAMPLE_DATA_TS_CACHE_NAME)
    if not os.path.exists(cache_path):
      util.log.info("Building sample_data timestamp cache ...")

      sample_to_scene = {}
      for sample in self.sample:
        scene = self.get('scene', sample['scene_token'])
        sample_to_scene[sample['token']] = scene['token']
    
      def to_ts_row(sample_data):
        row = dict(sample_data)
        row['scene_token'] = sample_to_scene[row['sample_token']]
        row['scene_name'] = self.get('scene', row['scene_token'])['name']
        return row
    
      df = pd.DataFrame(to_ts_row(r) for r in self.sample_data)
      df.to_parquet(cache_path)
      del df # Free several GB of memory

      util.log.info("... done.")

    return
  
  def get(self, table_name, token):
    assert table_name in self.table_names, \
      "Table {} not found".format(table_name)
    return self._get_table(table_name)[token]
  
  def getind(self, table_name, token):
    raise ValueError("Unsupported / unnecessary; provided by shelve")



  ### AU-added Utils

  def get_nearest_sample_data(self, scene_name, timestamp, channel=None):
    if not hasattr(self, '_sample_data_ts_df'):
      cache_path = self._get_cache_path(self.SAMPLE_DATA_TS_CACHE_NAME)
      util.log.info("Using sample_data timestamp cache at %s" % cache_path)
      self._sample_data_ts_df = pd.read_parquet(cache_path)
    
    df = self._sample_data_ts_df
    # First narrow down to the relevant scene / car and (maybe) sensor
    df = df[df['scene_name'] == scene_name]
    if channel:
      df = df[df['channel'] == channel]
    
    nearest = df.iloc[  (df['timestamp'] - timestamp).abs().argsort()[:1]  ]
    if len(nearest) > 0:
      row = nearest.to_dict(orient='records')[0]
      return row, row['timestamp'] - timestamp
    else:
      return None, 0

  def get_all_sensors(self):
    return set(itertools.chain.from_iterable(
      s['data'].keys() for s in self.sample))
    # NuScenes:
    # (TODO)
    # Lyft Level 5:
    # 'CAM_FRONT_ZOOMED', 'CAM_BACK', 'LIDAR_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    # 'CAM_BACK_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'LIDAR_TOP',
    # 'LIDAR_FRONT_LEFT', 'CAM_BACK_RIGHT'
  
  def get_all_classes(self):
    return set(anno['category_name'] for anno in self.sample_annotation)
    # NuScenes:
    # (TODO)
    # Lyft Level 5:
    # 'other_vehicle', 'bus', 'truck', 'car', 'bicycle', 'pedestrian', 
    # 'animal', 'emergency_vehicle', 'motorcycle'

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
    # return '/outer_root/media/seagates-ext4/au_datas/nuscenes_root'
    return '/outer_root/media/seagates-ext4/au_datas/lyft_level_5_root/train'

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
    """Create and return a `NuScenes` object for the given `version`."""
    nusc = AUNuScenes(version=version, dataroot=cls.dataroot(), verbose=True)
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
    # return cls._scene_to_split[scene]
    return 'train' # for lyft level 5, we assume all train for now
        


class FrameTable(av.FrameTableBase):

  FIXTURES = Fixtures

  NUSC_VERSION = 'v1.0-trainval' # E.g. v1.0-mini, v1.0-trainval, v1.0-test

  PROJECT_CLOUDS_TO_CAM = True
  PROJECT_CUBOIDS_TO_CAM = True
  IGNORE_INVISIBLE_CUBOIDS = True

  KEYFRAMES_ONLY = True#False
    # When disabled, will use motion-corrected points
  
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
    f = av.Frame(uri=uri)
    cls._fill_ego_pose(f)
    cls._fill_camera_images(f)
    return f

    # nusc = cls.get_nusc()

    # best_sample_data, diff = nusc.get_nearest_sample_data(
    #                             uri.segment_id, uri.timestamp)
    # assert diff == 0, "Can't interpolate all sensors for %s" % uri
    # assert False, best_sample_data




    # scene_to_ts_to_sample_token = cls._scene_to_ts_to_sample_token()
    # sample_token = scene_to_ts_to_sample_token[uri.segment_id][uri.timestamp]
    # sample = nusc.get('sample', sample_token)
    # return cls._create_frame_from_sample(uri, sample)

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
    if cls.KEYFRAMES_ONLY:
      import itertools
      sample_datas = itertools.chain.from_iterable(
        (nusc.get('sample_data', token)
          for sensor, token in sample['data'].items())
        for sample in nusc.sample)
    else:
      sample_datas = iter(nusc.sample_data)
    
    uris = []
    for sample_data in sample_datas:
      if sample_data['sensor_modality'] != 'camera':
        continue
        
      sample = nusc.get('sample', sample_data['sample_token'])
      scene_record = nusc.get('scene', sample['scene_token'])
      scene_split = cls.FIXTURES.get_split_for_scene(scene_record['name'])
      if scene_split not in splits:
        continue

      uris.append(av.URI(
                    dataset='nuscenes',
                    split=scene_split,
                    timestamp=sample_data['timestamp'],
                    segment_id=scene_record['name'],
                    camera=sample_data['channel']))

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

  # @classmethod
  # def _create_frame_from_sample(cls, uri, sample):
  #   f = av.Frame(uri=uri)
  #   cls._fill_ego_pose(f)
  #   cls._fill_camera_images(f)
  #   return f
  
  @classmethod
  def _fill_ego_pose(cls, f):
    nusc = cls.get_nusc()
    uri = f.uri

    # Every sample has a pose, so we should get an exact match
    best_sd, diff = nusc.get_nearest_sample_data(
                                uri.segment_id,
                                uri.timestamp)
    assert best_sd and diff == 0, "Can't interpolate pose"

    token = best_sd['ego_pose_token']
    pose_record = nusc.get('ego_pose', best_sd['ego_pose_token'])
    f.world_to_ego = transform_from_record(pose_record)

    # # For now, always set ego pose using the *lidar* timestamp, as is done
    # # in nuscenes.  (They probably localize mostly from lidar anyways).
    # token = sample['data']['LIDAR_TOP']
    # sd_record = nusc.get('sample_data', token)
    

  @classmethod
  def _fill_camera_images(cls, f):
    nusc = cls.get_nusc()
    uri = f.uri
    
    cameras = list(nusc.ALL_CAMERAS)
    if uri.camera:
      cameras = [uri.camera]
    
    for camera in cameras:
      best_sd, diff = nusc.get_nearest_sample_data(
                                uri.segment_id,
                                uri.timestamp,
                                channel=camera)
      assert best_sd

      ci = cls._get_camera_image(uri, best_sd)
      f.camera_images.append(ci)
  
  @classmethod
  def _get_camera_image(cls, uri, sd_record):
    nusc = cls.get_nusc()
    # sd_record = nusc.get('sample_data', camera_token)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    camera_token = sd_record['token']
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
      # TODO fuse
      # pc = None
      # for sensor in ('LIDAR_TOP',):
      for sensor in ('LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT'): # lyft
        # sample = nusc.get('sample', sd_record['sample_token']) ~~~~~~~~~~~~~~~~
        target_pose_token = sd_record['ego_pose_token']
        pc = cls._get_point_cloud_in_ego(uri, sensor, target_pose_token)
        if not pc:
          continue
        # if pts:
        #   pts = np.concatenate((pts, pc))
        # else:
        #   pts = pc

        # Project to image
        pc.cloud = ci.project_ego_to_image(pc.cloud, omit_offscreen=True)
        pc.sensor_name = pc.sensor_name + '_in_cam'
        ci.clouds.append(pc)
      
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
  def _get_point_cloud_in_ego(cls, uri, sensor, target_pose_token):
    # Based upon nuscenes.py#map_pointcloud_to_image()
    
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.data_classes import RadarPointCloud

    nusc = cls.get_nusc()
    
    

    # Get the cloud closest to the uri time
    pointsensor, diff = nusc.get_nearest_sample_data(
                                uri.segment_id,
                                uri.timestamp,
                                channel=sensor)
    if not pointsensor:
      # Perhaps this scene does not have `sensor`
      return None
    
    #pointsensor_token = sample['data'][sensor]
    #pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
      pc = LidarPointCloud.from_file(pcl_path)
    else:
      pc = RadarPointCloud.from_file(pcl_path)

    # Step 1: Points live in the point sensor frame.  First transform to
    # world frame:
    # 1a transform to ego
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get(
                  'calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # 1b transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Step 2: Send points into the ego frame at the target timestamp
    poserecord = nusc.get('ego_pose', target_pose_token)
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    
    # # before applying ego adjustment / interpolation.
    # #  so transform to ego frame
    # cs_record = nusc.get(
    #   'calibrated_sensor', pointsensor['calibrated_sensor_token'])
    # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    # pc.translate(np.array(cs_record['translation']))

    n_xyz = pc.points[:3, :].T
      # Throw out intensity (lidar) and ... other stuff (radar)
    return av.PointCloud(
      sensor_name=sensor,
      timestamp=pointsensor['timestamp'],
      cloud=n_xyz,
      ego_to_sensor=transform_from_record(cs_record),
      motion_corrected=(pointsensor['ego_pose_token'] != target_pose_token),
    )
    
  @classmethod
  def _get_cuboids_in_ego(cls, sample_data_token):
    nusc = cls.get_nusc()

    # NB: This helper always does motion correction (interpolation) unless
    # `sample_data_token` refers to a keyframe.
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
      
      cuboid.au_category = NUSCENES_CATEGORY_TO_AU_AV_CATEGORY[box.name]
      
      # Try to give bikes riders
      # NB: In Lyft Level 5, they appear to *not* label bikes without riders
      attribs = [
        nusc.get('attribute', attrib_token)['name']
        for attrib_token in sample_anno['attribute_tokens']
      ]
      if 'cycle.with_rider' in attribs:
        if cuboid.au_category == 'bike_no_rider':
          cuboid.au_category = 'bike_with_rider'
        elif cuboid.au_category == 'motorcycle_no_rider':
          cuboid.au_category = 'motorcycle_with_rider'
        else:
          raise ValueError(
            "Don't know how to give a rider to %s %s" % (cuboid, attribs))

      cuboid.extra = {
        'nuscenes_token': box.token,
        'nuscenes_attribs': '|'.join(attribs),
      }

      # Points
      cuboid.box3d = box.corners().T
      cuboid.motion_corrected = (not sd_record['is_key_frame'])
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