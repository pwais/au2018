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

  class Small(Params):
    CHECKPOINT_TARBALL_URI = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_96.tgz'
    IMG_SIZE = 96
    DEPTH_MULTIPLIER = 0.35
    FINE = True
  
  class Medium(Params):
    CHECKPOINT_TARBALL_URI = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 0.5
    FINE = True
    
  class Large(Params):
    CHECKPOINT_TARBALL_URI = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 1.0
    FINE = False
  
  class XLarge(Params):
    CHECKPOINT_TARBALL_URI = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz'
    IMG_SIZE = 224
    DEPTH_MULTIPLIER = 1.4
    FINE = False
    
  def __init__(self):
    self.sess = None
    self.predictor = None
  
  @classmethod
  def load_or_train(cls, params=None):
    model = cls()
    model.params = params or model.params
    
    util.download(params.CHECKPOINT_TARBALL_URI, model.params.MODEL_BASEDIR) 
    
    tf.reset_default_graph()
    images = tf.placeholder(
                  tf.float32,
                  # Channels last format!
                  (None, model.params.IMG_SIZE, model.params.IMG_SIZE, 3),
                  name='images')
    
    images = tf.cast(images, tf.float32) / 128. - 1
    from nets.mobilenet import mobilenet_v2
#     with tf.Graph().as_default():
#       with tf.variable_scope('MobilenetV2', reuse=True) as scope:
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
      # See also e.g. mobilenet_v2_035
      model.logits, model.endpoints = mobilenet_v2.mobilenet(
                                        images,
                                        is_training=False,
                                        depth_multiplier=model.params.DEPTH_MULTIPLIER,
                                        finegrain_classification_mode=model.params.FINE)
    
    # Restore using exponential moving average since it produces (1.5-2%)
    # higher accuracy
    #ema = tf.train.ExponentialMovingAverage(0.999)
    #vs = ema.variables_to_restore()
    
    if not model.params.tf_sess:
      model.params.tf_sess = util.tf_create_session()
    saver = tf.train.Saver()#vs)
    saver.restore(
        model.params.tf_sess,
        os.path.join(model.params.MODEL_BASEDIR, model.params.CHECKPOINT + '.ckpt'))
    return model
    
  def iter_activations(self):
    import pdb; pdb.set_trace()

