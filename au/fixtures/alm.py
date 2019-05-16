"""ALM-- Activation Latent Model.  An auto-encoder, GAN, or other model
that attempts to model the latent space of the activations of a given
nnmodel"""

import numpy as np

from au import util
from au.fixtures import nnmodel
from au.spark import Spark

class Example(object):
  __slots__ = ('uri', 'x', 'y')
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))

class ImageRowToExampleXForm(object):

  ALL_TENSORS = tuple()
  DEFAULT_MODEL = '<take first model>'
  DEFAULTS = {
    'activation_model': DEFAULT_MODEL,
    'x_activation_tensors': ALL_TENSORS, # Or None to turn off
    'y_is_visible': True,
    'x_dtype': np.float32,
    'y_dtype': np.float32,
  }

  def __init__(self, **kwargs):
    for k, v in self.DEFAULTS.iteritems():
      setattr(self, k, kwargs.get(k, v))

  def __call__(self, row):
    acts = row.attrs['activations']

    model = self.activation_model
    if model == self.DEFAULT_MODEL:
      model = acts.get_default_model()
    else:
      assert model in acts.get_models()

    x = None
    if self.x_activation_tensors is not None:
      
      tensor_names = self.x_activation_tensors
      if tensor_names == self.ALL_TENSORS:
        tensor_names = sorted(acts.get_tensor_names(model))
      
      assert tensor_names

      ts = np.concatenate(tuple(
        acts.get_tensor(model, tn).flatten()
        for tn in tensor_names))
      x = ts.flatten()
    
    y = None
    if self.y_is_visible:
      y = row.as_numpy()
    y = y.flatten()

    # MNIST FIXME
    y = y / 255.
    
    assert x is not None and y is not None
    x = x.astype(self.x_dtype)
    y = y.astype(self.y_dtype)

    return Example(uri=row.uri, x=x, y=y)


class ActivationsDataset(object):

  ACTIVATIONS_TABLE = None
  ROW_XFORM = ImageRowToExampleXForm()
  
  @classmethod
  def as_tf_dataset(cls, spark=None):
    import tensorflow as tf

    import cloudpickle

    # TODO fixme
    if not hasattr(cls, '_cache'):
      import os
      cls._cache = os.path.join('/tmp', cls.__name__ + '_cache')
      if not os.path.exists(cls._cache):
        from au.spark import Spark
        from pyspark.sql import Row
        with Spark.sess(spark) as spark:
          imagerow_rdd = cls.ACTIVATIONS_TABLE.as_imagerow_rdd(spark=spark)
          ex_rdd = imagerow_rdd.map(cls.ROW_XFORM)
          ex_rdd = ex_rdd.map(lambda ex: Row(data=bytearray(cloudpickle.dumps(ex))))
          ex_rdd = ex_rdd.cache()
          for _ in range(10):
            spark.createDataFrame(ex_rdd).write.parquet(
              cls._cache,
              mode='append',
              compression='snappy')
            print 'append'

    def make_iter_ex_tuples(s):
      def f():
        import time

        from au.spark import Spark
        from au.fixtures import dataset
        print 'spark tf session'
        start = time.time()
        with Spark.sess(s) as spark:
          imagerow_rdd, joined = cls.ACTIVATIONS_TABLE.as_imagerow_rdd(spark=spark)
          from pyspark import StorageLevel
          joined = joined.persist(StorageLevel.DISK_ONLY)
          ex_rdd = imagerow_rdd.map(cls.ROW_XFORM)
          uris = joined.select('uri').collect()
          print 'num uris', len(uris)
          def to_imagerow(row):
            irow = dataset.ImageRow(**row.asDict())
            acts = nnmodel.Activations.from_rows(row.m_t2vs)
            irow.attrs = irow.attrs or {}
            irow.attrs['activations'] = acts
            return irow
          def data_gen():
            for uri_chunk in util.ichunked(uris, 1000):
              rows = joined.filter(joined.uri.isin([u.uri for u in uri_chunk])).rdd.map(to_imagerow).map(cls.ROW_XFORM).collect()
              for r in rows:
                yield r.x, r.y, r.uri
          
          ds = tf.data.Dataset.from_generator(
            generator=data_gen,
            output_types=(tf.float32, tf.float32, tf.string))
          ds = ds.prefetch(20 * 1000)
          ds = ds.map(lambda x, y, z: (x, y, z), num_parallel_calls=8)
          n = 0
          with util.tf_data_session(ds) as (sess, ddd):
            for k in ddd():
              n += 1
              if n % 1000 == 0:
                print n, time.time() - start
          print n
          print time.time() - start
          import pdb; pdb.set_trace()


        # import pyarrow as pa
        # import pyarrow.parquet as pq
        # start = time.time()
        # print 'pyarrow'
        # n = 0
        # for path in util.all_files_recursive(cls._cache, '*.parquet'):
        #   pa_table = pq.read_table(path)
        #   for i, row in enumerate(pa_table.to_pandas().to_dict(orient='records')):
        #     n += 1
        #     if n % 10000 == 0:
        #       print i, n, time.time() - start
        # end = time.time()
        # print end - start
        # # 190 sec

        # print 'petastorm'
        # from petastorm import make_batch_reader
        # start = time.time()
        # n = 0
        # with make_batch_reader('file://' + cls._cache) as reader:
        #   for i, row in enumerate(reader):
        #     n += len(row.data)
        #     if i % 100 == 0:
        #       print n, time.time() - start
        # end = time.time()
        # print end - start
        # # 46 sec

        start = time.time()
        df = s.read.parquet(cls._cache)
        import pdb; pdb.set_trace()
        # df = df.cache()
        # print df.count()
        for i, rr in enumerate(df.rdd.toLocalIterator()):
          if i % 10000 == 0:
            print i, time.time() - start
        end = time.time()
        print end - start
          
        import pdb; pdb.set_trace()

        from au.spark import Spark
        with Spark.sess(s) as spark:
          df = spark.read.parquet(cls._cache)
          df = df.cache()
          for rr in df.rdd.toLocalIterator():
            r = cloudpickle.loads(rr.data)
            yield r.x, r.y, r.uri
      return f
    
    # TODO: make sizes predictable
    x, y, uri = make_iter_ex_tuples(spark)().next()
    cls._x_shape = x.shape
    cls._y_shape = y.shape

    ds = tf.data.Dataset.from_generator(
          generator=make_iter_ex_tuples(spark),
          output_types=(tf.float32, tf.float32, tf.string),
          output_shapes=(cls._x_shape, cls._y_shape, []))
    return ds




# class ActivationDataset(object):


#   DATASET = ''
#   SPLIT = ''

#   LIMIT = -1 # Use all data, else ablate to this much data
  
#   X_TENSORS = tuple()
#   Y_TENSORS = tuple()

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