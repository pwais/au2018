

"""
make a dataset for 1-channel mnist things

make a dataset for our handful of images

try to coerce dataset from mscoco

make one for bbd100k

record activations for mnist
then for mobilenet on bdd100k / mscoco
take note of deeplab inference: https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=edGukUHXyymr
and we'll wanna add maskrcnn mebbe ?

SPARK_LOCAL_IP=127.0.0.1 $SPARK_HOME/bin/pyspark --packages databricks:tensorframes:0.5.0-s_2.11 --packages databricks:spark-deep-learning:1.2.0-spark2.3-s_2.11


"""


class DatasetFactoryBase(object):
  
  class ParamsBase(object):
    def __init__(self):
      self.BASE_DIR = ''
  
  @classmethod
  def create_dataset(cls):
    pass
  
  @classmethod
  def get_ctx_for_entry(cls, entry_id):
    pass