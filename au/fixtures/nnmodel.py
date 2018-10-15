import os

from au import conf
from au import util

class INNModel(object):
  
  class ParamsBase(object):
    def __init__(self, model_name='INNModel'):
      # Canonical places to save files
      self.MODEL_NAME = model_name
      self.MODEL_BASEDIR = os.path.join(
                                  conf.AU_MODEL_CACHE,
                                  self.MODEL_NAME)
      self.DATA_BASEDIR = os.path.join(
                              conf.AU_DATA_CACHE,
                              self.MODEL_NAME)
      self.TENSORBOARD_BASEDIR = os.path.join(
                                    conf.AU_TENSORBOARD_DIR,
                                    self.MODEL_NAME)
      
      # Tensorflow session, if available
      self.tf_sess = None
  
  def __init__(self):
    self.params = INNModel.ParamsBase()
  
  @classmethod
  def load_or_train(self, cls, params=None):
    raise NotImplementedError
  
  def iter_activations(self):
    raise NotImplementedError
