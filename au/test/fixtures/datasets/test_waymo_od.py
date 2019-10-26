import os

from au import conf

URIS = (
  'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380086457446912',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10724020115992582208_7660_400_7680_400_with_camera_labels.tfrecord&timestamp=1507342888193009920',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord&timestamp=1511659072142650112',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord&timestamp=1507265181881123072',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10500357041547037089_1474_800_1494_800_with_camera_labels.tfrecord&timestamp=1509298376482341888',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380080957681920',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord&timestamp=1511659071642657024',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord&timestamp=1507265184981669888',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380085557635072',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord&timestamp=1507265186784071936',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10724020115992582208_7660_400_7680_400_with_camera_labels.tfrecord&timestamp=1507342888292936960',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380084457432064',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380078957580032',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10724020115992582208_7660_400_7680_400_with_camera_labels.tfrecord&timestamp=1507342901192209920',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord&timestamp=1511659081042469120',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380083857431040',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord&timestamp=1507265175696123904',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord&timestamp=1507265190486380032',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord&timestamp=1511380074358799104',
'avframe://dataset=waymo-od&split=train&segment_id=segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord&timestamp=1511659065620167936'
)

def test_waymo_yay():
  from au.fixtures.datasets.waymo_od import StampedDatumTable
  StampedDatumTable.setup()
  return

  # segs = set(uri.segment_id for uri in URIS)
  # for seg in segs:
  #   cam_to_vid = 

  for uri in URIS:
    frame = FrameTable.create_frame(uri)
    fname = '|'.join((frame.uri.segment_id , str(frame.uri.timestamp) ))
    with open('/tmp/' + fname + '.html', 'w') as f:
      f.write(frame.to_html())
      print(str(frame.uri), fname)

  # import itertools
  # l = list(itertools.islice(FrameTable.iter_all_uris(), 1000))
  # import pdb; pdb.set_trace()
  # i = 0
  # for uri in FrameTable.iter_all_uris():
  #   print(uri)
  #   i += 1
  #   if i == 100:
  #     return
  print('yay')