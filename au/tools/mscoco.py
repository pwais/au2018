#!/usr/bin/env python3

from au.fixtures.datasets import mscoco

if __name__ == '__main__':
  from au.fixtures.datasets import auargoverse as av
  # av.ImageAnnoTable.setup()
  # av.CroppedObjectImageTable.setup()
  av.FrameTable.setup()

  # from au.fixtures.datasets.av import frame_table_to_object_detection_tfrecords

  # from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
  # AV_OBJ_CLASS_NAME_TO_ID = dict(
  #   (cname, i + 1)
  #   for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
  # AV_OBJ_CLASS_NAME_TO_ID['background'] = 0

  # from au.spark import Spark
  # spark = Spark.getOrCreate()
  # frame_table_to_object_detection_tfrecords(
  #   spark,
  #   av.FrameTable,
  #   '/outer_root/media/seagates-ext4/au_datas/frame_table_tfrecords/',
  #   AV_OBJ_CLASS_NAME_TO_ID)

  # av.AnnoReports.create_reports()

  # # NB: We can't embed this into the mscoco module due to a bug in
  # # Cloudpickle: https://github.com/cloudpipe/cloudpickle/issues/225
  # mscoco.Fixtures.run_import()
