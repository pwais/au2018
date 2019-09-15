"""

ctpu up --zone us-central1-a --tpu-size v3-8 --machine-type n1-standard-4
ctpu --zone us-central1-a --project academic-actor-248218 list

indefinite hang:

2019-09-15 11:38:32.466701: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-09-15 11:38:32.475523: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2019-09-15 11:38:32.476009: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4ece150 executing computations on platform Host. Devices:
2019-09-15 11:38:32.476045: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-09-15 11:38:32.478664: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:250] Initialize GrpcChannelCache for job local -> {0 -> localhost:34375}
2019-09-15 11:38:32.480175: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:365] Started server with target: grpc://localhost:34375
From /opt/au/external/tensorflow_tpu/models/official/detection/executor/tpu_executor.py:55: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

^C^CTerminated





30 minutes for an OOM!

[Stage 267:(56 + 4) / 356][Stage 270:>  (0 + 0) / 1][Stage 272:>  (0 + 0) / 1]TPUPollingThread found TPU b'au2018-3' in state READY, and health HEALTHY.
[Stage 267:(190 + 4) / 356][Stage 270:>  (0 + 0) / 1][Stage 272:>  (0 + 0) / 1]TPUPollingThread found TPU b'au2018-3' in state READY, and health HEALTHY.
[Stage 267:(332 + 4) / 356][Stage 270:>  (0 + 0) / 1][Stage 272:>  (0 + 0) / 1]TPUPollingThread found TPU b'au2018-3' in state READY, and health HEALTHY.
Error recorded from infeed: Step was cancelled by an explicit call to `Session::Close()`.
Error recorded from training_loop: From /job:worker/replica:0/task:0:
Compilation failure: Ran out of memory in memory space hbm. Used 17.60G of 16.00G hbm. Exceeded hbm capacity by 1.60G.

Total hbm usage >= 17.60G:
    reserved        528.00M 
    program          17.08G 
    arguments       unknown size 

Output size unknown.

Program hbm requirement 17.08G:
    reserved          12.0K
    global           980.0K
    scoped            5.41M
    HLO temp         17.08G (92.8% utilization, 0.3% fragmentation (53.61M))

  Largest program allocations in hbm:

  1. Size: 512.00M
     Operator: op_type="OneHot" op_name="one_hot"
     Shape: f32[4,128,128,9,18]{4,3,0,2,1:T(8,128)}
     Unpadded size: 40.50M
     Extra memory due to padding: 471.50M (12.6x expansion)
     XLA label: %fusion.1668 = f32[4,128,128,9,18]{4,3,0,2,1:T(8,128)} fusion(s32[4,128,128,9]{3,0,2,1:T(4,128)} %copy.130), kind=kLoop, calls=%fused_computation.1304, metadata={op_type="OneHot" op_name="one_hot"}
     Allocation type: HLO temp
     ==========================

  2. Size: 256.00M
     Shape: bf16[4,512,512,64]{3,0,2,1:T(4,128)(2,1)}
     Unpadded size: 128.00M
     Extra memory due to padding: 128.00M (2.0x expansion)
     XLA label: %fusion.20231 = (bf16[4,512,512,64]{3,0,2,1:T(4,128)(2,1)}) fusion(f32[64]{0:T(256)} %get-tuple-element.73199, f32[64]{0:T(256)} %get-tuple-element.73198, f32[64]{0:T(256)} %get-tuple-element.63973, f32[64]{0:T(256)} %get-tuple-element.67355, bf16[4,512,51...
     Allocation type: HLO temp
     ==========================

  3. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_11"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.989 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63433, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2010, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  4. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_8"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.979 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63434, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.3330, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  5. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.981 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63418, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.1906, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  6. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_3"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.983 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63417, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2890, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  7. Size: 256.00M
     Operator: op_type="FusedBatchNormV2" op_name="resnet152/batch_normalization/FusedBatchNormV2"
     Shape: bf16[4,512,512,64]{3,0,2,1:T(4,128)(2,1)}
     Unpadded size: 128.00M
     Extra memory due to padding: 128.00M (2.0x expansion)
     XLA label: %fusion.1235 = (f32[64]{0:T(256)}, f32[64]{0:T(256)}, bf16[4,512,512,64]{3,0,2,1:T(4,128)(2,1)}) fusion(bf16[1024,1024,3,4]{3,2,1,0:T(4,128)(2,1)} %reshape.4895, bf16[7,7,3,64]{3,2,1,0:T(4,128)(2,1)} %get-tuple-element.73977), kind=kOutput, calls=%fused_co...
     Allocation type: HLO temp
     ==========================

  8. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_7"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.986 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63422, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.3242, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  9. Size: 256.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_4"
     Shape: f32[4,256,256,256]{3,0,2,1:T(4,128)}
     Unpadded size: 256.00M
     XLA label: %reduce-window.980 = f32[4,256,256,256]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,256]{3,0,2,1:T(4,128)} %get-tuple-element.63421, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2978, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  10. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_20"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.976 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63464, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2810, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  11. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_6"
     Shape: f32[4,256,256,64]{3,0,2,1:T(4,128)}
     Unpadded size: 64.00M
     Extra memory due to padding: 64.00M (2.0x expansion)
     XLA label: %reduce-window.987 = f32[4,256,256,64]{3,0,2,1:T(4,128)} reduce-window(f32[4,256,256,64]{3,0,2,1:T(4,128)} %get-tuple-element.63431, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.3154, metadata={op_type="MaxP...
     Allocation type: HLO temp
     ==========================

  12. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_27"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.1001 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63463, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2866, metadata={op_type="M...
     Allocation type: HLO temp
     ==========================

  13. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_24"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.975 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63465, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2842, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  14. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_39"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.1010 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63466, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2970, metadata={op_type="M...
     Allocation type: HLO temp
     ==========================

  15. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_12"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.978 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63499, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2098, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  16. Size: 128.00M
     Operator: op_type="Relu" op_name="resnet152/Relu_6"
     Shape: bf16[4,256,256,256]{3,0,2,1:T(4,128)(2,1)}
     Unpadded size: 128.00M
     XLA label: %fusion.1504 = bf16[4,256,256,256]{3,0,2,1:T(4,128)(2,1)} fusion(bf16[4,256,256,256]{3,0,2,1:T(4,128)(2,1)} %fusion.1503, bf16[]{:T(512)(2,1)} %multiply.40609, bf16[]{:T(512)(2,1)} %multiply.40649, f32[256]{0:T(256)} %get-tuple-element.73687, f32[256]{0:T(...
     Allocation type: HLO temp
     ==========================

  17. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_15"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.992 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63501, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2362, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  18. Size: 128.00M
     Operator: op_type="Relu" op_name="resnet152/Relu_3"
     Shape: bf16[4,256,256,256]{3,0,2,1:T(4,128)(2,1)}
     Unpadded size: 128.00M
     XLA label: %fusion.1503 = bf16[4,256,256,256]{3,0,2,1:T(4,128)(2,1)} fusion(bf16[]{:T(512)(2,1)} %multiply.40610, bf16[]{:T(512)(2,1)} %multiply.40624, f32[256]{0:T(256)} %get-tuple-element.73203, f32[256]{0:T(256)} %get-tuple-element.73202, f32[256]{0:T(256)} %rsqrt...
     Allocation type: HLO temp
     ==========================

  19. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_23"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.998 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63493, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2834, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================

  20. Size: 128.00M
     Operator: op_type="MaxPool" op_name="resnet152/MaxPool_36"
     Shape: f32[4,128,128,512]{3,0,2,1:T(4,128)}
     Unpadded size: 128.00M
     XLA label: %reduce-window.972 = f32[4,128,128,512]{3,0,2,1:T(4,128)} reduce-window(f32[4,128,128,512]{3,0,2,1:T(4,128)} %get-tuple-element.63475, f32[]{:T(256)} %constant.41841), window={size=1x7x7x1 pad=0_0x3_3x3_3x0_0}, to_apply=%max_F32.2946, metadata={op_type="Ma...
     Allocation type: HLO temp
     ==========================


        TPU compilation failed
         [[node TPUReplicateMetadata (defined at /opt/au/external/tensorflow_tpu/models/official/detection/executor/tpu_executor.py:105) ]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.





"""

import os
import sys

import tensorflow as tf

sys.path = [
  '/opt/au/external/tensorflow_tpu/models',
  '/opt/au/external/tensorflow_tpu/models/official/detection'] + sys.path

from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
AV_OBJ_CLASS_NAME_TO_ID = dict(
  (cname, i + 1)
  for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
AV_OBJ_CLASS_NAME_TO_ID['background'] = 0

class TFExampleDSInputFn(object):
  """Based upon TPU object detection InputFn:
  https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/dataloader/input_reader.py#L23
  """

  def __init__(self, frame_df, params, mode):
    self._frame_df = frame_df
    self._mode = mode
    self._is_training = (mode == 'train')

    from dataloader import factory
    self._parser_fn = factory.parser_generator(params, mode)
    self._transpose_input = hasattr(params, 'train') and hasattr(
        params.train, 'transpose_input') and params.train.transpose_input

  def __call__(self, params):
    batch_size = params['batch_size']
    # dataset = self._tf_example_ds
    # dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self._tf_example_ds))
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tf.device('/job:coordinator/task:0'):

      from au.fixtures.datasets import av
      from au.spark import Spark
      spark = Spark.getOrCreate()
      # df = spark.read.parquet('/opt/au/frame_table')
      df = spark.read.parquet(self._frame_df)
      dataset = av.frame_df_to_tf_example_ds(
                    df, AV_OBJ_CLASS_NAME_TO_ID)

      # if self._is_training:
      #   dataset = dataset.repeat()

      if self._is_training:
        dataset = dataset.shuffle(64)

      # # Parses the fetched records to input tensors for model function.
      # dataset = dataset.map(self._parser_fn, num_parallel_calls=64)
      dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
      # dataset = dataset.batch(batch_size, drop_remainder=True)

      # # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
      # # TPU is expensive.
      # if self._transpose_input and self._is_training:

      #   def _transpose_images(images, labels):
      #     return tf.transpose(images, [1, 2, 3, 0]), labels

      #   dataset = dataset.map(_transpose_images, num_parallel_calls=64)


      source_dataset = dataset
      source_iterator = dataset.make_one_shot_iterator()
      source_handle = source_iterator.string_handle()


    # https://github.com/tensorflow/tensorflow/blob/76f23b3b751150a2a81012925100bb08e406ab47/tensorflow/python/tpu/datasets.py#L51
    from tensorflow.python.data.experimental.ops import batching
    from tensorflow.python.data.experimental.ops import interleave_ops
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.data.ops import iterator_ops
    from tensorflow.python.data.ops import readers
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import function
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import functional_ops
    @function.Defun(dtypes.string)
    def LoadingFunc(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, dataset_ops.get_legacy_output_types(source_dataset),
          dataset_ops.get_legacy_output_shapes(source_dataset))
      return remote_iterator.get_next()

    def MapFn(unused_input):
      source_dataset_output_types = dataset_ops.get_legacy_output_types(
          source_dataset)
      if isinstance(source_dataset_output_types, dtypes.DType):
        output_types = [source_dataset_output_types]
      elif isinstance(source_dataset_output_types, (list, tuple)):
        output_types = source_dataset_output_types
      else:
        raise ValueError('source dataset has invalid output types')
      remote_calls = functional_ops.remote_call(
          args=[source_handle],
          Tout=output_types,
          f=LoadingFunc,
          target='/job:%s/replica:0/task:0/cpu:0' % 'coordinator')
      if len(remote_calls) == 1:
        return remote_calls[0]
      else:
        return remote_calls

    with ops.device('/job:%s' % 'worker'):
      output_dataset = dataset_ops.Dataset.range(2).repeat().map(
          MapFn, num_parallel_calls=4)
      output_dataset = output_dataset.prefetch(1)

      
      # Undo the batching used during the transfer.
      # output_dataset = output_dataset.apply(batching.unbatch()).prefetch(1)
      
      # output_dataset = set_shapes(output_dataset)


      # Parses the fetched records to input tensors for model function.
      output_dataset = output_dataset.map(self._parser_fn, num_parallel_calls=64)
      output_dataset = output_dataset.prefetch(tf.contrib.data.AUTOTUNE)
      output_dataset = output_dataset.batch(batch_size, drop_remainder=True)

      # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
      # TPU is expensive.
      if self._transpose_input and self._is_training:

        def _transpose_images(images, labels):
          return tf.transpose(images, [1, 2, 3, 0]), labels

        output_dataset = output_dataset.map(_transpose_images, num_parallel_calls=64)


    return output_dataset



# https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/configs/yaml/retinanet_nasfpn.yaml
CONF = """
# ---------- RetianNet + NAS-FPN ----------
# Expected accuracy with using NAS-FPN l3-l7 and image size 640x640: 39.5
# train batch size 64

enable_summary: True

train:
  train_batch_size: 16
  total_steps: 90000
  gradient_clip_norm: 10
  learning_rate:
    init_learning_rate: 0.0008
    learning_rate_levels: [0.08, 0.008, 0.0008]
    learning_rate_steps: [1000, 60000, 80000]

architecture:
  multilevel_features: 'nasfpn'
  use_bfloat16: True

anchor:
  min_level: 3
  max_level: 7
  num_scales: 3
  aspect_ratios: [0.5, 1.0, 2.0]
  anchor_size: 4.0

resnet:
  resnet_depth: 152
  dropblock:
    dropblock_keep_prob: 0.9
    dropblock_size: 7

retinanet_head:
  num_classes: {num_classes}
retinanet_loss:
  num_classes: {num_classes}
postprocess:
  num_classes: {num_classes}

nasfpn:
  fpn_feat_dims: 256
  min_level: 3
  max_level: 7
  num_repeats: 5
  use_separable_conv: False

retinanet_parser:
  aug_scale_min: 0.8
  aug_scale_max: 1.2
  output_size: [960, 960]
  aug_rand_hflip: True
  use_bfloat16: True

""".format(num_classes=len(AV_OBJ_CLASS_NAME_TO_ID))

DEFAULT_PARAMS = {
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
      'num_shards': 1,
    },
  }

import time
TPU_PARAMS = {
    'platform': {
      'eval_master': '',
      'tpu': 'au2018-3',
      'tpu_zone': 'us-central1-a',
      'gcp_project': 'academic-actor-248218',
      'coordinator_name': 'coordinator',
    },
    'tpu_job_name': 'worker',
    'use_tpu': True,
    'iterations_per_loop': 50,
    'model_dir': 'gs://au2018-3/tpu_test_detection/test_%s' % time.time(),
      #'/tmp/tpu_tast_detection',
    'train': {
      'num_shards': 8,
    },
  }

def main():

  # from au.spark import Spark
  # from au.spark import spark_df_to_tf_dataset
  # spark = Spark.getOrCreate()
  # # df = spark.read.parquet('/opt/au/frame_table')
  # df = spark.read.parquet('/opt/au/frame_table_broken_shard/')
  df = '/opt/au/frame_table_broken_shard/'


  # frames = [avse.FrameTable.create_frame(uri) for uri in uris]
  # print('loaded %s frames' % len(frames))

  # from au.fixtures.datasets import av
  # tf_examples = [
  #   av.camera_image_to_tf_example(
  #     frame.uri,
  #     frame.camera_images[0],
  #     AV_OBJ_CLASS_NAME_TO_ID).SerializeToString()
  #   for frame in frames
  # ]

  # ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tf_examples))
  # ds = tf_examples
  

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

  tf.logging.set_verbosity(tf.logging.DEBUG)

  params = params_dict.ParamsDict(
      retinanet_config.RETINANET_CFG, retinanet_config.RETINANET_RESTRICTIONS)
  
  params = params_dict.override_params_dict(
        params, CONF, is_strict=True)

  params.override(TPU_PARAMS, is_strict=False)
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
      df, params, mode=ModeKeys.TRAIN)
  eval_input_fn = TFExampleDSInputFn(
      df, params, mode=ModeKeys.PREDICT_WITH_GT)

  # Runs the model.
  save_config(params, params.model_dir)
  # executor.prepare_evaluation()
  num_cycles = int(params.train.total_steps / params.eval.num_steps_per_eval)
  for cycle in range(num_cycles):
    tf.logging.info('Start training cycle %d.' % cycle)
    current_cycle_last_train_step = ((cycle + 1)
                                      * params.eval.num_steps_per_eval)
    executor.train(train_input_fn, current_cycle_last_train_step)
    print('want to eval but eval segfaults')
    # executor.evaluate(
    #     eval_input_fn,
    #     params.eval.eval_samples // params.predict.predict_batch_size)

if __name__ == '__main__':
  main()



#   """
#   TPU MODE
#   """

#   import os
# import sys

# import tensorflow as tf

# sys.path = [
#   '/opt/au/external/tensorflow_tpu/models',
#   '/opt/au/external/tensorflow_tpu/models/official/detection'] + sys.path

# from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
# AV_OBJ_CLASS_NAME_TO_ID = dict(
#   (cname, i + 1)
#   for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
# AV_OBJ_CLASS_NAME_TO_ID['background'] = 0

# class TFExampleDSInputFn(object):
#   """Based upon TPU object detection InputFn:
#   https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/dataloader/input_reader.py#L23
#   """

#   def __init__(self, frame_df, params, mode):
#     self._frame_df = frame_df
#     self._mode = mode
#     self._is_training = (mode == 'train')

#     from dataloader import factory
#     self._parser_fn = factory.parser_generator(params, mode)
#     self._transpose_input = hasattr(params, 'train') and hasattr(
#         params.train, 'transpose_input') and params.train.transpose_input

#   def __call__(self, params):
#     batch_size = params['batch_size']
#     # dataset = self._tf_example_ds
#     # dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self._tf_example_ds))
    
#     from au.fixtures.datasets import av
#     dataset = av.frame_df_to_tf_example_ds(
#                   self._frame_df, AV_OBJ_CLASS_NAME_TO_ID)

#     if self._is_training:
#       dataset = dataset.repeat()

#     # dataset = dataset.apply(
#     #     tf.data.experimental.parallel_interleave(
#     #         lambda file_name: self._dataset_fn(file_name).prefetch(1),
#     #         cycle_length=32,
#     #         sloppy=self._is_training))

#     if self._is_training:
#       dataset = dataset.shuffle(64)

#     # Parses the fetched records to input tensors for model function.
#     dataset = dataset.map(self._parser_fn, num_parallel_calls=64)
#     dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#     dataset = dataset.batch(batch_size, drop_remainder=True)

#     # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
#     # TPU is expensive.
#     if self._transpose_input and self._is_training:

#       def _transpose_images(images, labels):
#         return tf.transpose(images, [1, 2, 3, 0]), labels

#       dataset = dataset.map(_transpose_images, num_parallel_calls=64)

#     return dataset

# # https://github.com/tensorflow/tpu/blob/246a41ef611b3129b6468e7ab778e6837dbfc785/models/official/detection/configs/yaml/retinanet_nasfpn.yaml
# CONF = """
# # ---------- RetianNet + NAS-FPN ----------
# # Expected accuracy with using NAS-FPN l3-l7 and image size 640x640: 39.5
# train:
#   train_batch_size: 64
#   total_steps: 90000
#   learning_rate:
#     init_learning_rate: 0.08
#     learning_rate_levels: [0.008, 0.0008]
#     learning_rate_steps: [60000, 80000]

# architecture:
#   multilevel_features: 'nasfpn'

# nasfpn:
#   fpn_feat_dims: 256
#   min_level: 3
#   max_level: 7
#   num_repeats: 5
#   use_separable_conv: False

# retinanet_parser:
#   aug_scale_min: 0.8
#   aug_scale_max: 1.2
# """

# def main():

#   from au.spark import Spark
#   from au.spark import spark_df_to_tf_dataset
#   spark = Spark.getOrCreate()
#   df = spark.read.parquet('/opt/au/frame_table_small')


#   # frames = [avse.FrameTable.create_frame(uri) for uri in uris]
#   # print('loaded %s frames' % len(frames))

#   # from au.fixtures.datasets import av
#   # tf_examples = [
#   #   av.camera_image_to_tf_example(
#   #     frame.uri,
#   #     frame.camera_images[0],
#   #     AV_OBJ_CLASS_NAME_TO_ID).SerializeToString()
#   #   for frame in frames
#   # ]

#   # ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tf_examples))
#   # ds = tf_examples
  


#   # From tensorflow/tpu
#   from configs import retinanet_config
#   from dataloader import input_reader
#   from dataloader import mode_keys as ModeKeys
#   from executor import tpu_executor
#   from modeling import model_builder
#   # sys.path.insert(0, 'tpu/models')
#   from hyperparameters import params_dict

#   def save_config(params, model_dir):
#     if model_dir:
#       if not tf.gfile.Exists(model_dir):
#         tf.gfile.MakeDirs(model_dir)
#       params_dict.save_params_dict_to_yaml(
#           params, os.path.join(model_dir, 'params.yaml'))

#   params = params_dict.ParamsDict(
#       retinanet_config.RETINANET_CFG, retinanet_config.RETINANET_RESTRICTIONS)
  
#   params = params_dict.override_params_dict(
#         params, CONF, is_strict=True)

#   params.override({
#     'platform': {
#       'eval_master': '',
#       'tpu': 'au2018',
#       'tpu_zone': 'us-central1-a',
#       'gcp_project': 'academic-actor-248218',
#     },
#     'tpu_job_name': 'worker',
#     'use_tpu': True,
#     'model_dir': 'gs://au2018-3/tpu_test_detection/',#'/tmp/tpu_tast_detection',
#     'train': {
#       'num_shards': 8,
#     },
#   }, is_strict=False)
#   # params.override({
#   #     'platform': {
#   #         'eval_master': FLAGS.eval_master,
#   #         'tpu': FLAGS.tpu,
#   #         'tpu_zone': FLAGS.tpu_zone,
#   #         'gcp_project': FLAGS.gcp_project,
#   #     },
#   #     'use_tpu': FLAGS.use_tpu,
#   #     'model_dir': FLAGS.model_dir,
#   #     'train': {
#   #         'num_shards': FLAGS.num_cores,
#   #     },
#   # }, is_strict=False)
#   # Only run spatial partitioning in training mode.

#   params.validate()
#   params.lock()
#   import pprint
#   pp = pprint.PrettyPrinter()
#   params_str = pp.pformat(params.as_dict())
#   print('Model Parameters: {}'.format(params_str))

#   # Builds detection model on TPUs.
#   model_fn = model_builder.ModelFn(params)
#   executor = tpu_executor.TpuExecutor(model_fn, params)

#   # Prepares input functions for train and eval.
#   train_input_fn = TFExampleDSInputFn(
#       df, params, mode=ModeKeys.TRAIN)
#   eval_input_fn = TFExampleDSInputFn(
#       df, params, mode=ModeKeys.PREDICT_WITH_GT)

#   # Runs the model.
#   save_config(params, params.model_dir)
#   # executor.prepare_evaluation()
#   num_cycles = int(params.train.total_steps / params.eval.num_steps_per_eval)
#   for cycle in range(num_cycles):
#     tf.logging.info('Start training cycle %d.' % cycle)
#     current_cycle_last_train_step = ((cycle + 1)
#                                       * params.eval.num_steps_per_eval)
#     executor.train(train_input_fn, current_cycle_last_train_step)
#     executor.evaluate(
#         eval_input_fn,
#         params.eval.eval_samples // params.predict.predict_batch_size)

# if __name__ == '__main__':
#   main()