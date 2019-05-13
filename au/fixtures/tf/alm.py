# """ALM-- Activation Latent Model.  An auto-encoder, GAN, or other model
# that attempts to model the latent space of the activations of a given
# nnmodel"""

# import numpy as np

# import tensorflow as tf

# from au import util
# from au.fixtures import nnmodel
# from au.spark import Spark

# class ActivationDataset(nnmodel.ActivationsTable):

#   DATASET = ''
#   SPLIT = ''

#   LIMIT = -1 # Use all data, else ablate to this much data
  
#   TENSOR_NAMES = tuple()

#   @classmethod
#   def iter_image_rows(cls, spark):

#     # Get a master list of URIs / examples to read; this will help us
#     # cap memory usage when reading data.
#     df = spark.read.parquet(cls.table_root())
#     if cls.SPLIT:
#       df = df.filter(df.split == cls.SPLIT)
#     if cls.DATASET:
#       df = df.filter(df.dataset == cls.DATASET)
#     uris = activations_df.select('uri').distinct()

#     if cls.LIMIT >= 0:
#       uris = uris[:cls.LIMIT]
    




#     class Example(object):
#       __slots__ = ('uri', 'tensor_name_to_value')

#       def __init__(self, **kwargs):
#         for k in self.__slots__:
#           setattr(self, k, kwargs.get(k))
      
#       def as_input_vector(self):



#   def _create_df(cls, spark=None):
#     df = None
#     if spark:
#       df = spark.read.parquet(cls.table_root())
#     else:
#       import pandas as pd
#       import pyarrow.parquet as pq
#       pa_table = pq.read_table(cls.table_root())
#       df = pa_table.to_pandas()
    
#     if cls.SPLIT:
#       df = df.filter(df['split'] == cls.SPLIT)
#     return df

#   @classmethod
#   def _iter_rows(cls, spark=None):
#     df = cls._create_df(spark=spark)

#     if spark:
#       for row in df.rdd.toLocalIterator():
#         yield row.asDict()
#     else:
#       for row in df.to_dict(orient='records'):
#         yield row

#   @classmethod
#   def create_tf_dataset(cls, cycle=True, spark=None):
#     def row_to_tuple(row):
      

#     if spark:
#       df = spark.read.parquet(cls.table_root())
#       if cls.SPLIT:
#         df = df.


# def train(params, tf_config=None):
#   if tf_config is None:
#     tf_config = util.tf_create_session_config()

#   tf.logging.set_verbosity(tf.logging.DEBUG)

#   util.mkdir(params.MODEL_BASEDIR)

#   estimator = tf.estimator.Estimator(
#     model_fn=model_fn(params),
#     params=None, # Ignore TF HParams thingy
#     config=tf.estimator.RunConfig(
#       model_dir=params.MODEL_BASEDIR,
#       save_summary_steps=10,
#       save_checkpoints_secs=10,
#       session_config=tf_config,
#       log_step_count_steps=10))

#   ## Data
#   def train_input_fn():
#     # Load the dataset
#     train_ds = params.TRAIN_TABLE.to_mnist_tf_dataset(spark=spark)

#     # Flow doesn't need uri
#     train_ds = train_ds.map(lambda arr, label, uri: (arr, label))

#     if params.LIMIT >= 0:
#       train_ds = train_ds.take(params.LIMIT)
#     # train_ds = train_ds.shuffle(60000).batch(params.BATCH_SIZE)
#     # train_ds = train_ds.prefetch(10 * params.BATCH_SIZE)
#     train_ds = train_ds.batch(params.BATCH_SIZE)
#     train_ds = train_ds.cache(os.path.join(params.MODEL_BASEDIR, 'train_cache'))
#     return train_ds
  
#   def eval_input_fn():
#     test_ds = params.TEST_TABLE.to_mnist_tf_dataset(spark=spark)

#     # Flow doesn't need uri
#     test_ds = test_ds.map(lambda arr, label, uri: (arr, label))

#     if params.LIMIT >= 0:
#       test_ds = test_ds.take(params.LIMIT)
#     test_ds = test_ds.batch(params.EVAL_BATCH_SIZE)
#     test_ds = test_ds.cache(os.path.join(params.MODEL_BASEDIR, 'test_cache'))
    
#     # No idea why we return an interator thingy instead of a dataset ...
#     return test_ds.make_one_shot_iterator().get_next()