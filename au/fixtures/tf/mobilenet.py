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
    def create_frozen_graph_def(self):
      util.download(
        self.params.CHECKPOINT_TARBALL_URI,
        self.params.MODEL_BASEDIR)
      
      g = tf.Graph()
      with g.as_default():
        input_image = tf.placeholder(
          tf.uint8,
          self.params.INPUT_TENSOR_SHAPE,
          name=self.params.INPUT_TENSOR_NAME)
        uris = tf.placeholder(
          tf.string,
          [None],
          name=self.params.INPUT_URIS_NAME)

        input_image_norm = tf.cast(input_image, tf.float32) / 128. - 1
        input_image_norm.set_shape(self.params.INPUT_TENSOR_SHAPE)
        
        from nets.mobilenet import mobilenet_v2
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
          # See also e.g. mobilenet_v2_035
          logits, endpoints = mobilenet_v2.mobilenet(
                                input_image_norm,
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
      nodes = list(self.output_names) + [input_image, uris]
      return util.give_me_frozen_graph(
                              checkpoint,
                              nodes=nodes,
                              base_graph=g,
                              saver=saver)





    #     sess = util.tf_cpu_session()
    #     with sess.as_default():
    #       tf_model = create_model()

    #       input_image = tf.placeholder(
    #         tf.uint8,
    #         self.params.INPUT_TENSOR_SHAPE,
    #         name=self.params.INPUT_TENSOR_NAME)
    #       input_image_f = tf.cast(input_image, tf.float32)
    #       uris = tf.placeholder(
    #         tf.string,
    #         [None],
    #         name=self.params.INPUT_URIS_NAME)
          
    #       pred = tf_model(input_image_f, training=False)
    #       checkpoint = tf.train.latest_checkpoint(self.params.MODEL_BASEDIR)
    #       saver = tf.train.import_meta_graph(
    #                             checkpoint + '.meta',
    #                             clear_devices=True)
    #       return util.give_me_frozen_graph(
    #                           checkpoint,
    #                           nodes=list(self.output_names) + [input_image, uris],
    #                           saver=saver,
    #                           base_graph=g,
    #                           sess=sess)


    # def create_inference_graph(self, input_image, base_graph):
    #   util.download(self.params.CHECKPOINT_TARBALL_URI, self.params.MODEL_BASEDIR)
      
    #   # self.graph = base_graph
    #   g = tf.Graph()
    #   with g.as_default():
    #     input_image = tf.cast(input_image, tf.float32) / 128. - 1
    #     input_image.set_shape(self.params.INPUT_TENSOR_SHAPE)

    #     from nets.mobilenet import mobilenet_v2
    #     with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    #       # See also e.g. mobilenet_v2_035
    #       self.logits, self.endpoints = mobilenet_v2.mobilenet(
    #                             input_image,
    #                             is_training=False,
    #                             depth_multiplier=self.params.DEPTH_MULTIPLIER,
    #                             finegrain_classification_mode=self.params.FINE)

    #     # Per authors: Restore using exponential moving average since it produces
    #     # (1.5-2%) higher accuracy
    #     ema = tf.train.ExponentialMovingAverage(0.999)
    #     vs = ema.variables_to_restore()
        
    #   saver = tf.train.Saver(vs)
    #   checkpoint = os.path.join(
    #     self.params.MODEL_BASEDIR,
    #     self.params.CHECKPOINT + '.ckpt')
    #   # nodes = list(self.output_names) + [input_image]
    #   self.graph = util.give_me_frozen_graph(
    #                           checkpoint,
    #                           nodes=self.output_names,
    #                           base_graph=base_graph,
    #                           saver=saver)
    #   return self.graph
    
    @property
    def output_names(self):
      return (
        'MobilenetV2/Logits/output:0',
        'MobilenetV2/embedding:0',
        'MobilenetV2/expanded_conv_16/output:0',
        'MobilenetV2/Predictions/Reshape_1:0',
      )
  
  @classmethod
  def load_or_train(cls, params=None):
    model = cls()
    model.params = params or model.params
    model.igraph = Mobilenet.GraphFactory(params)
    return model
     
  def get_inference_graph(self):
    return self.igraph
