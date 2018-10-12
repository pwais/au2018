import os

from au import conf
from au import util



class INNModel(object):
  
  class ParamsBase(object):
    
    # Canonical places to save files
    MODEL_NAME = 'INNModel'
    MODEL_BASEDIR = os.path.join(conf.AU_MODEL_CACHE, MODEL_NAME)
    DATA_BASEDIR = os.path.join(conf.AU_DATA_CACHE, MODEL_NAME)
    TENSORBOARD_BASEDIR = os.path.join(conf.AU_TENSORBOARD_DIR, MODEL_NAME)
  
  def __init__(self):
    self.params = ParamsBase()
  
  @classmethod
  def load_or_train(self, cls, params=None):
    raise NotImplementedError
  
  def iter_activations(self):
    raise NotImplementedError