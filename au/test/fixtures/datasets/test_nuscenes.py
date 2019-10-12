TEST_URIS = (
  # Bike in front cam
  # 'avframe://dataset=nuscenes&split=val&segment_id=scene-0553&timestamp=1535489296047917&camera=CAM_FRONT',

  # # Front cam with person in truck and with cone way off to side
  # 'avframe://dataset=nuscenes&split=train&segment_id=scene-0061&timestamp=1532402928147847&camera=CAM_FRONT',
  'avframe://dataset=nuscenes&split=train&segment_id=scene-0061&timestamp=1532402928762460&camera=CAM_FRONT',
  'avframe://dataset=nuscenes&split=train&segment_id=scene-0655&timestamp=1535385107162404&camera=CAM_FRONT',
)

def test_nuscenes():

  from au.fixtures.datasets.nuscenes import FrameTable
  
  nusc = FrameTable.get_nusc()
  nusc.print_sensor_sample_rates()
  import pdb; pdb.set_trace()
  FrameTable.setup()

  return
  

  # FrameTable.NUSC_VERSION = 'v1.0-mini'
  uris = FrameTable._get_camera_uris()

  LYFT_SCENES_MULTI_LIDAR = (
    'host-a101-lidar0-1240877587199107226-1240877612099413030',
    'host-a102-lidar0-1242662270298972894-1242662295198395706',
    'host-a101-lidar0-1241472407298206026-1241472432198409706',
    'host-a101-lidar0-1242493624298705334-1242493649198973302',
    'host-a102-lidar0-1242749461398477906-1242749486298996742',
    'host-a101-lidar0-1241561147998866622-1241561172899320654',
    'host-a101-lidar0-1241462203298815998-1241462228198805706',
  )

  uris = [u for u in uris if u.segment_id in LYFT_SCENES_MULTI_LIDAR]

  # nusc = FrameTable.get_nusc()

  # scenes = (
  #   'scene-0061',
  #   'scene-0103', 
  #   'scene-0655', 
  #   'scene-0553', 
  #   'scene-0757', 
  #   'scene-0796', 
  #   'scene-0916', 
  #   'scene-1077', 
  #   'scene-1094', 
  #   'scene-1100', 
  # )

  # import itertools
  # tasks = itertools.chain.from_iterable(
  #   ((s, c) for c in nusc.ALL_CAMERAS) for s in scenes)
  # from au.spark import Spark
  # spark = Spark.getOrCreate()
  # task_rdd = spark.sparkContext.parallelize(tasks)

  # def render_vid(s_c):
  #   scene, cam = s_c

  #   vid_uris = [
  #     u for u in uris if (u.segment_id == scene and u.camera == cam)]
  #   vid_uris = sorted(vid_uris, key=lambda u: u.timestamp)

  #   def gen_debug_frames():
  #     from au import util
  #     t = util.ThruputObserver(name='%s_%s' % (scene, cam), n_total=len(vid_uris))
  #     for uri in vid_uris:
  #       with t.observe(n=1):
  #         FrameTable.NUSC_VERSION = 'v1.0-mini'
  #         f = FrameTable.create_frame(uri)
  #         ci = f.camera_images[0]
  #         import numpy as np
  #         debug_img = np.copy(ci.image)
  #         from au import plotting as aupl
  #         aupl.draw_xy_depth_in_image(debug_img, ci.cloud.cloud, alpha=0.7)
  #         for bbox in ci.bboxes:
  #           bbox.draw_in_image(debug_img)
  #       yield debug_img
  #       t.maybe_log_progress(every_n=1)
  
  #   import imageio
  #   imageio.mimwrite(
  #       '/tmp/%s_%s.mov' % (scene, cam),
  #       gen_debug_frames(),
  #       format='mov', 
  #       fps=10)
  # render_vid(('scene-0655', 'CAM_FRONT'))
  # task_rdd.foreach(render_vid)
  # return

  #asdgasgas
  # x = [a for a in nusc.sample_annotation if 'bicycle' in a['category_name']]
  # scen_toks = set([nusc.get('sample', xx['sample_token'])['scene_token'] for xx in x])

  # scene_name_to_token = dict(
  #       (scene['name'], scene['token']) for scene in nusc.scene)

  # scen_uris = [u for u in uris if scene_name_to_token[u.segment_id] in scen_toks]
  # scen_uris = [u for u in uris 
  #   if u.segment_id == 'scene-0553' and u.timestamp == 1535489297047675]# and abs(u.timestamp - 1535489297047675) <= 1]
  # import random
  # random.shuffle(scen_uris)
  #asdgasgas




  
  print(len(uris))
  import random
  random.shuffle(uris)
  # uris = TEST_URIS
  # 1535385107162404
  # uris = [u for u in uris if u.segment_id == 'scene-0655' and u.camera == 'CAM_FRONT']
  # uris.sort(key=lambda u: u.timestamp)
  for uri in uris[:50]:
    frame = FrameTable.create_frame(uri)
    fname = '|'.join((frame.uri.segment_id , str(frame.uri.timestamp) , frame.uri.camera))
    with open('/tmp/' + fname + '.html', 'w') as f:
      f.write(frame.to_html())
      print(str(frame.uri), fname)

  return
  import pdb; pdb.set_trace()




  from au.fixtures.datasets import av
  from nuscenes.nuscenes import NuScenes

  nusc = NuScenes(
          version='v1.0-mini',
          dataroot='/tmp/nuscenes_tast', verbose=True)
  
  

  # from nuscenes.utils.splits import create_splits_scenes
  # split_to_scenes = create_splits_scenes()

  # scene_to_split = {}
  # for split, scenes in split_to_scenes.items():
  #   if 'mini' not in split:
  #     for scene in scenes:
  #       scene_to_split[scene] = split
  


  # uris = []
  # for sample in nusc.sample:
  #   scene = nusc.get('scene', sample['scene_token'])
  #   for sensor, token in sample['data'].items():
  #     sample_data = nusc.get('sample_data', token)
  #     if sample_data['sensor_modality'] == 'camera':
  #       uri = av.URI(
  #               dataset='nuscenes',
  #               split=scene_to_split[scene['name']],
  #               timestamp=sample['timestamp'],
  #               segment_id=scene['name'],
  #               camera=sensor)
  #       uris.append(uri)
  
  # def get_point_cloud_in_ego(sample, sensor='LIDAR_TOP'):
  #   # Based upon nuscenes.py#map_pointcloud_to_image()
  #   import os.path as osp
    
  #   from pyquaternion import Quaternion

  #   from nuscenes.utils.data_classes import LidarPointCloud
  #   from nuscenes.utils.data_classes import RadarPointCloud
    
  #   pointsensor_token = sample['data'][sensor]
  #   pointsensor = nusc.get('sample_data', pointsensor_token)
  #   pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
  #   if pointsensor['sensor_modality'] == 'lidar':
  #     pc = LidarPointCloud.from_file(pcl_path)
  #   else:
  #     pc = RadarPointCloud.from_file(pcl_path)

  #   # Points live in the point sensor frame, so transform to ego frame
  #   cs_record = nusc.get(
  #     'calibrated_sensor', pointsensor['calibrated_sensor_token'])
  #   pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
  #   pc.translate(np.array(cs_record['translation']))
  #   return av.PointCloud(
  #     sensor_name=sensor,
  #     timestamp=pointsensor['timestamp'],
  #     cloud=pc.points,
  #     ego_to_sensor=transform_from_record(cs_record),
  #     motion_corrected=False, # TODO interpolation for cam ~~~~~~~~~~~~~~~~~~~~~~~
  #   )


  def get_frame_from_sample(uri, sample):
    # token = None
    # if uri.camera:
    #   token = sample['data'][uri.camera]

    # sd_record = nusc.get('sample_data', token)
    # cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    # sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    # pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # # Boxes are moved into *ego* frame
    # data_path, box_list, cam_intrinsic = nusc.get_sample_data(token)
    
    # f = av.Frame(uri=uri)

    # f.world_to_ego = transform_from_record(pose_record)
    
    # viewport = uri.get_viewport()
    # w, h = sd_record['width'], sd_record['height']
    # if not viewport:
    #   viewport = common.BBox.of_size(w, h)

    # timestamp = sd_record['timestamp']

    # cam_from_ego = transform_from_record(cs_record)

    # def get_camera_normal(K, extrinstic):
    #   """TODO see notes from auargoverse

    #   """
    #   # Build P
    #   # P = K * | R |T|
    #   #         |000 1|
    #   P = K.dot(extrinsic)

    #   # Zisserman pg 161 The principal axis vector.
    #   # P = [M | p4]; M = |..|
    #   #                   |m3|
    #   # pv = det(M) * m3
    #   pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
    #   pv_hat = pv / np.linalg.norm(pv)
    #   return pv_hat

    # principal_axis_in_ego = get_camera_normal(
    #                           cam_intrinsic,
    #                           cam_from_ego.get_transformation_matrix())

    # ci = av.CameraImage(
    #       camera_name=uri.camera,
    #       image_jpeg=bytearray(open(data_path, 'rb').read()),
    #       height=h,
    #       width=w,
    #       viewport=viewport,
    #       timestamp=timestamp,
    #       cam_from_ego=cam_from_ego,
    #       K=cam_intrinsic,
    #       principal_axis_in_ego=principal_axis_in_ego,
    #     )
    
    # if True#cls.PROJECT_CLOUDS_TO_CAM:
    #   for sensor in ('LIDAR_TOP',):
    #     pc = get_point_cloud_in_ego(sample, sensor=sensor)
        
    #     # Project to image
    #     pc.cloud = ci.project_ego_to_image(pc.cloud, omit_offscreen=True)
    #     ci.cloud = pc
      
    # if True#cls.PROJECT_CUBOIDS_TO_CAM:
           
    #   sample_data_token = sd_record['token']
      
    #   boxes = nusc.get_boxes(sample_data_token)
      
    #   # Boxes are in world frame.  Move to Ego frame.
    #   from pyquaternion import Quaternion
    #   sd_record = self.get('sample_data', sample_data_token)
    #   pose_record = self.get('ego_pose', sd_record['ego_pose_token'])
    #   for box in boxes:
    #     # Move box to ego vehicle coord system
    #     box.translate(-np.array(pose_record['translation']))
    #     box.rotate(Quaternion(pose_record['rotation']).inverse)

    #   for box in boxes:
    #     cuboid = av.Cuboid()

    #     # Core
    #     sample_anno = nusc.get('sample_annotation', box.token)
    #     cuboid.track_id = sample_anno['instance_token']
    #     cuboid.category_name = box.name
    #     cuboid.timestamp = sd_record['timestamp']
        
    #     attribs = [
    #       nusc.get('attribute', attrib_token)
    #       for attrib_token in sample_anno['attribute_tokens']
    #     ]
    #     cuboid.extra = {
    #       'nuscenes_token': box.token,
    #       'nuscenes_attribs': '|'.join(attrib['name'] for attrib in attribs),
    #     }

    #     # Points
    #     cuboid.box3d = box.corners().T
    #     cuboid.motion_corrected = False # TODO interpolation ~~~~~~~~~~~~~~~~~~~~
    #     cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
        
    #     # Pose
    #     cuboid.width_meters = float(box.wlh[0])
    #     cuboid.length_meters = float(box.wlh[1])
    #     cuboid.height_meters = float(box.wlh[3])

    #     cuboid.obj_from_ego = av.Transform(
    #         rotation=box.orientation.rotation_matrix,
    #         translation=box.center.reshape((3, 1)))
        

    #     bbox = ci.project_cuboid_to_bbox(cuboid)
    #     cls_IGNORE_INVISIBLE_CUBOIDS = True
    #     if cls_IGNORE_INVISIBLE_CUBOIDS and not bbox.is_visible:
    #       continue
      
    #     ci.bboxes.append(bbox)




    
    import pdb; pdb.set_trace()
    print('asgd')

  # def get_frame(uri):
  #   scene_name_to_token = dict(
  #     (scene['name'], scene['token']) for scene in nusc.scene)
    
  #   from collections import defaultdict
  #   scene_to_ts_to_sample_token = defaultdict(dict)
  #   for sample in nusc.sample:
  #     scene_name = nusc.get('scene', sample['scene_token'])['name']
  #     timestamp = sample['timestamp']
  #     token = sample['token']
  #     scene_to_ts_to_sample_token[scene_name][timestamp] = token
    
  #   sample_token = scene_to_ts_to_sample_token[uri.segment_id][uri.timestamp]
  #   sample = nusc.get('sample', sample_token)
  #   return get_frame_from_sample(uri, sample)

  get_frame(uris[0])



  camera_keyframes = [
    s for s in nusc.sample_data
    if s['is_key_frame'] and 'camera' in s['sensor_modality']
  ]

  import pdb; pdb.set_trace()

  lidar_keyframes = [
    s for s in nusc.sample_data
    if s['is_key_frame'] and 'lidar' in s['sensor_modality']
  ]
  