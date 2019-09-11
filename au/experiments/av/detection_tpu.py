import os
import sys

import tensorflow as tf

sys.path = [
  '/opt/au/external/tensorflow_tpu/models',
  '/opt/au/external/tensorflow_tpu/models/official/detection'] + sys.path

class TFExampleDSInputFn(object):
  """Based upon TPU object detection InputFn:
  https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/dataloader/input_reader.py#L23
  """

  def __init__(self, tf_example_ds, params, mode):
    self._tf_example_ds = tf_example_ds
    self._mode = mode
    self._is_training = (mode == 'train')

    from dataloader import factory
    self._parser_fn = factory.parser_generator(params, mode)
    self._transpose_input = hasattr(params, 'train') and hasattr(
        params.train, 'transpose_input') and params.train.transpose_input

  def __call__(self, params):
    batch_size = params['batch_size']
    # dataset = self._tf_example_ds
    dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self._tf_example_ds))

    if self._is_training:
      dataset = dataset.repeat()

    # dataset = dataset.apply(
    #     tf.data.experimental.parallel_interleave(
    #         lambda file_name: self._dataset_fn(file_name).prefetch(1),
    #         cycle_length=32,
    #         sloppy=self._is_training))

    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parses the fetched records to input tensors for model function.
    dataset = dataset.map(self._parser_fn, num_parallel_calls=64)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
    # TPU is expensive.
    if self._transpose_input and self._is_training:

      def _transpose_images(images, labels):
        return tf.transpose(images, [1, 2, 3, 0]), labels

      dataset = dataset.map(_transpose_images, num_parallel_calls=64)

    return dataset

# https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/configs/yaml/retinanet_nasfpn.yaml
CONF = """
# ---------- RetianNet + NAS-FPN ----------
# Expected accuracy with using NAS-FPN l3-l7 and image size 640x640: 39.5
train:
  train_batch_size: 1
  total_steps: 90000
  learning_rate:
    init_learning_rate: 0.08
    learning_rate_levels: [0.008, 0.0008]
    learning_rate_steps: [60000, 80000]

architecture:
  multilevel_features: 'nasfpn'

nasfpn:
  fpn_feat_dims: 256
  min_level: 3
  max_level: 7
  num_repeats: 5
  use_separable_conv: False

retinanet_parser:
  aug_scale_min: 0.8
  aug_scale_max: 1.2
"""

def main():

  from au.fixtures.datasets import auargoverse as avse
  uris = (
    'avframe://segment_id=tracking_train2.tar.gz|5c251c22-11b2-3278-835c-0cf3cdee3f44&split=train&camera=ring_front_center&timestamp=315967787401035936',
    'avframe://segment_id=tracking_train1.tar.gz|f9fa3960-537f-3151-a1a3-37a9c0d6d7f7&split=train&camera=ring_rear_right&timestamp=315968463537902224',
    'avframe://segment_id=tracking_train1.tar.gz|1d676737-4110-3f7e-bec0-0c90f74c248f&split=train&camera=ring_front_left&timestamp=315984810796685856',
    'avframe://segment_id=tracking_train2.tar.gz|3138907e-1f8a-362f-8f3d-773f795a0d01&split=train&camera=stereo_front_left&timestamp=315968320382681104',
    'avframe://segment_id=tracking_train1.tar.gz|70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c&split=train&camera=stereo_front_left&timestamp=315976372531658128',

    # dupes
    'avframe://segment_id=tracking_train2.tar.gz|5c251c22-11b2-3278-835c-0cf3cdee3f44&split=train&camera=ring_front_center&timestamp=315967787401035936',
    'avframe://segment_id=tracking_train1.tar.gz|f9fa3960-537f-3151-a1a3-37a9c0d6d7f7&split=train&camera=ring_rear_right&timestamp=315968463537902224',
    'avframe://segment_id=tracking_train1.tar.gz|1d676737-4110-3f7e-bec0-0c90f74c248f&split=train&camera=ring_front_left&timestamp=315984810796685856',
    'avframe://segment_id=tracking_train2.tar.gz|3138907e-1f8a-362f-8f3d-773f795a0d01&split=train&camera=stereo_front_left&timestamp=315968320382681104',
    'avframe://segment_id=tracking_train1.tar.gz|70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c&split=train&camera=stereo_front_left&timestamp=315976372531658128',
    'avframe://segment_id=tracking_train2.tar.gz|5c251c22-11b2-3278-835c-0cf3cdee3f44&split=train&camera=ring_front_center&timestamp=315967787401035936',
    'avframe://segment_id=tracking_train1.tar.gz|f9fa3960-537f-3151-a1a3-37a9c0d6d7f7&split=train&camera=ring_rear_right&timestamp=315968463537902224',
    'avframe://segment_id=tracking_train1.tar.gz|1d676737-4110-3f7e-bec0-0c90f74c248f&split=train&camera=ring_front_left&timestamp=315984810796685856',
    'avframe://segment_id=tracking_train2.tar.gz|3138907e-1f8a-362f-8f3d-773f795a0d01&split=train&camera=stereo_front_left&timestamp=315968320382681104',
    'avframe://segment_id=tracking_train1.tar.gz|70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c&split=train&camera=stereo_front_left&timestamp=315976372531658128',
  )
  # uris = tuple()
  frames = [avse.FrameTable.create_frame(uri) for uri in uris]
  print('loaded %s frames' % len(frames))
  
  from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
  AV_OBJ_CLASS_NAME_TO_ID = dict(
    (cname, i + 1)
    for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
  AV_OBJ_CLASS_NAME_TO_ID['background'] = 0

  from au.fixtures.datasets import av
  tf_examples = [
    av.camera_image_to_tf_example(
      frame.uri,
      frame.camera_images[0],
      AV_OBJ_CLASS_NAME_TO_ID).SerializeToString()
    for frame in frames
  ]

  ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tf_examples))
  ds = tf_examples
  


  # From tensorflow/tpu
  from configs import retinanet_config
  from dataloader import input_reader
  from dataloader import mode_keys as ModeKeys
  from executor import tpu_executor
  from modeling import model_builder
  # sys.path.insert(0, 'tpu/models')
  from hyperparameters import params_dict

  def save_config(params, model_dir):
    if model_dir:
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      params_dict.save_params_dict_to_yaml(
          params, os.path.join(model_dir, 'params.yaml'))

  params = params_dict.ParamsDict(
      retinanet_config.RETINANET_CFG, retinanet_config.RETINANET_RESTRICTIONS)
  
  params = params_dict.override_params_dict(
        params, CONF, is_strict=True)

  params.override({
    'platform': {
      'eval_master': '',
      'tpu': '',
      'tpu_zone': '',
      'gcp_project': '',
    },
    'tpu_job_name': '',
    'use_tpu': False,
    'model_dir': '/tmp/tpu_tast_detection',
    'train': {
      'num_shards': 1#8,
    },
  }, is_strict=False)
  # params.override({
  #     'platform': {
  #         'eval_master': FLAGS.eval_master,
  #         'tpu': FLAGS.tpu,
  #         'tpu_zone': FLAGS.tpu_zone,
  #         'gcp_project': FLAGS.gcp_project,
  #     },
  #     'use_tpu': FLAGS.use_tpu,
  #     'model_dir': FLAGS.model_dir,
  #     'train': {
  #         'num_shards': FLAGS.num_cores,
  #     },
  # }, is_strict=False)
  # Only run spatial partitioning in training mode.

  params.validate()
  params.lock()
  import pprint
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  print('Model Parameters: {}'.format(params_str))

  # Builds detection model on TPUs.
  model_fn = model_builder.ModelFn(params)
  executor = tpu_executor.TpuExecutor(model_fn, params)

  # Prepares input functions for train and eval.
  train_input_fn = TFExampleDSInputFn(
      ds, params, mode=ModeKeys.TRAIN)
  eval_input_fn = TFExampleDSInputFn(
      ds, params, mode=ModeKeys.PREDICT_WITH_GT)

  # Runs the model.
  save_config(params, params.model_dir)
  # executor.prepare_evaluation()
  num_cycles = int(params.train.total_steps / params.eval.num_steps_per_eval)
  for cycle in range(num_cycles):
    tf.logging.info('Start training cycle %d.' % cycle)
    current_cycle_last_train_step = ((cycle + 1)
                                      * params.eval.num_steps_per_eval)
    executor.train(train_input_fn, current_cycle_last_train_step)
    executor.evaluate(
        eval_input_fn,
        params.eval.eval_samples // params.predict.predict_batch_size)

if __name__ == '__main__':
  main()