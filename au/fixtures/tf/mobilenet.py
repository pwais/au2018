"""
Based upon tensorflow/models slim example
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
"""

import os

import tensorflow as tf

from au import conf
from au import util
from au.fixtures import nnmodel

class Mobilenet(nnmodel.INNModel):
  
  class Params(nnmodel.INNModel.ParamsBase):
    CHECKPOINT_TARBALL_URI = ''
    IMG_SIZE = -1
    def __init__(self):
      mobilenet_version = self.CHECKPOINT_TARBALL_URI.split('/')[-1][:-4]
      model_name = self.MODEL_NAME = 'MobileNet.' + mobilenet_version
      super(Mobilenet.Params, self).__init__(model_name=model_name)
      
      self.CHECKPOINT = mobilenet_version
      self.INPUT_TENSOR_SHAPE = [None, self.IMG_SIZE, self.IMG_SIZE, 3]

  class Small(Params):
    CHECKPOINT_TARBALL_URI = \
      'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_96.tgz'
    IMG_SIZE = 96
    DEPTH_MULTIPLIER = 0.35
    FINE = True
  
  class Medium(Params):
    CHECKPOINT_TARBALL_URI = \
      'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 0.5
    FINE = True
    
  class Large(Params):
    CHECKPOINT_TARBALL_URI = \
      'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 1.0
    FINE = False
  
  class XLarge(Params):
    CHECKPOINT_TARBALL_URI = \
      'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 1.4
    FINE = False
  
  ALL_PARAMS_CLSS = (Small, Medium, Large, XLarge)

  class GraphFactory(nnmodel.TFInferenceGraphFactory):
    def create_inference_graph(self, input_image, base_graph):
      util.download(self.params.CHECKPOINT_TARBALL_URI, self.params.MODEL_BASEDIR)
      
      self.graph = base_graph
      with self.graph.as_default():
        

        # tf_sess = util.tf_create_session()
        # saver = tf.train.Saver()#vs)
        # saver.restore(
        #   tf_sess,
        #   os.path.join(
        #     self.params.MODEL_BASEDIR,
        #     self.params.CHECKPOINT + '.ckpt'))
        
        input_image = tf.cast(input_image, tf.float32) / 128. - 1
        input_image.set_shape(self.params.INPUT_TENSOR_SHAPE)
        # input_image = tf.reshape(input_image, tuple(self.params.INPUT_TENSOR_SHAPE))

        from nets.mobilenet import mobilenet_v2
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
          # See also e.g. mobilenet_v2_035
          self.logits, self.endpoints = mobilenet_v2.mobilenet(
                                input_image,
                                is_training=False,
                                depth_multiplier=self.params.DEPTH_MULTIPLIER,
                                finegrain_classification_mode=self.params.FINE)

        # Per authors: Restore using exponential moving average since it produces
        # (1.5-2%) higher accuracy
        ema = tf.train.ExponentialMovingAverage(0.999)
        vs = ema.variables_to_restore()
        
      saver = tf.train.Saver(vs)
      checkpoint = os.path.join(
        self.params.MODEL_BASEDIR,
        self.params.CHECKPOINT + '.ckpt')
      nodes = list(self.output_names) + [input_image]
      self.graph = util.give_me_frozen_graph(
                              checkpoint,
                              nodes=self.output_names,
                              base_graph=self.graph,
                              saver=saver)

      return self.graph
    
    @property
    def output_names(self):
      return (
        'MobilenetV2/Logits/output:0',
        'MobilenetV2/embedding:0',
        'MobilenetV2/expanded_conv_16/output:0',
        'MobilenetV2/Predictions/Reshape_1:0',
      )

  # def __init__(self):
  #   self.sess = None
  #   self.predictor = None
  
  @classmethod
  def load_or_train(cls, params=None):
    model = cls()
    model.params = params or model.params
    model.igraph = Mobilenet.GraphFactory(params)
    return model
     
  def get_inference_graph(self):
    return self.igraph

# class MobilenetActivationsTable(ActivationsTable):
#   TABLE_NAME = ''

#     tf.reset_default_graph()
#     images = tf.placeholder(
#                   tf.float32,
#                   # Channels last format!
#                   (None, model.params.IMG_SIZE, model.params.IMG_SIZE, 3),
#                   name='images')
    
#     images = tf.cast(images, tf.float32) / 128. - 1
#     from nets.mobilenet import mobilenet_v2
# #     with tf.Graph().as_default():
# #       with tf.variable_scope('MobilenetV2', reuse=True) as scope:
#     with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
      
    
#     # Restore using exponential moving average since it produces (1.5-2%)
#     # higher accuracy
#     #ema = tf.train.ExponentialMovingAverage(0.999)
#     #vs = ema.variables_to_restore()
    
#     tf_sess = util.tf_create_session()
#     saver = tf.train.Saver()#vs)
#     saver.restore(
#         tf_sess,
#         os.path.join(model.params.MODEL_BASEDIR, model.params.CHECKPOINT + '.ckpt'))
#     return model
    
#   def iter_activations(self):
#     yield 1

