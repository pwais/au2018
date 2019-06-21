"""ALM-- Activation Latent Model.  An auto-encoder, GAN, or other model
that attempts to model the latent space of the activations of a given
nnmodel"""

import os

import numpy as np

from au import conf
from au import util
from au.fixtures import nnmodel
from au.spark import Spark

class Example(object):
  __slots__ = ('uri', 'x', 'y')
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def astuple(self):
    return self.x, self.y, self.uri
  
  # # TODO: work around this boilerplate 
  # def to_dict(self):
  #   return {
  #     'x': self.x.tolist(), #bytearray(pickle.dumps(self.x)),
  #     'y': self.y.tolist(), #bytearray(pickle.dumps(self.y)),
  #     'uri': self.uri,
  #   }
  # @staticmethod
  # def from_row(**kwargs):
  #   return Example(
  #     x=np.array((kwargs.get('x', []))),
  #     y=np.array((kwargs.get('y', []))),
  #     uri=kwargs.get('uri'),
  #   )


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
    for k, v in self.DEFAULTS.items():
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

    # MNIST FIXME FIXMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    y = y / 255.
    
    assert x is not None and y is not None
    x = x.astype(self.x_dtype)
    y = y.astype(self.y_dtype)

    return Example(uri=row.uri, x=x, y=y)


class ActivationsDataset(object):

  ACTIVATIONS_TABLE = None
  ROW_XFORM = ImageRowToExampleXForm()

  @classmethod
  def ds_cache_root(cls):
    dirname = 'activations_dataset_%s_%s' % (
          cls.__name__,
          cls.ACTIVATIONS_TABLE.__name__,
        )
    return os.path.join(conf.AU_TABLE_CACHE, dirname)

  @classmethod
  def as_tf_dataset(cls, spark=None):
    import cloudpickle
    import tensorflow as tf
    meta_path = os.path.join(cls.ds_cache_root(), 'au_spark_tf_meta.pkl')
    if not os.path.exists(meta_path):
      util.log.info("Caching TF dataset to %s ..." % cls.ds_cache_root())

      util.mkdir(cls.ds_cache_root())

      # TODO TODO hash the dataset for auto cache-busting

      with Spark.sess(spark) as spark:
        imagerow_rdd = cls.ACTIVATIONS_TABLE.as_imagerow_rdd(spark=spark)

        def part_to_tf_cache(pid, iter_image_rows):          
          try:
            row = next(iter_image_rows)
          except StopIteration as e:
            util.log.info("Empty partition, skipping %s" % pid)
            return []
          
          ext = cls.ROW_XFORM(row).astuple()
          def get_dtype(v):
            if hasattr(v, 'dtype'):
              return tf.dtypes.as_dtype(v.dtype)
            elif isinstance(v, str):
              return tf.string
            else:
              return tf.dtypes.as_dtype(v)
          output_shapes = tuple(
            v.shape[0] if hasattr(v, 'shape') else None
            for v in ext
          )
          output_types = tuple(get_dtype(v) for v in ext)
          def gen_examples():
            yield ext
            for row in iter_image_rows:
              ex = cls.ROW_XFORM(row)
              yield ex.astuple()
          ds = tf.data.Dataset.from_generator(
                gen_examples,
                output_types=output_types,
                output_shapes=output_shapes)

          # def serialize_example(ex):
          #   example_proto = tf.train.Example(
          #     features=tf.train.Features(feature={
          #       'x': 
          #     }


          cache_path = os.path.join(cls.ds_cache_root(), 'cache_%s' % pid)
          ds = ds.cache(cache_path)
          with util.tf_data_session(ds) as (sess, iter_ds):
            n = 0
            for _ in iter_ds():
              n += 1
            util.log.info("Saved %s ex to %s" % (n,cache_path))
          return [
            (cache_path, 
             cloudpickle.dumps(output_shapes),
             cloudpickle.dumps(output_types),
            )
          ]

        meta_rdd = imagerow_rdd.mapPartitionsWithIndex(part_to_tf_cache)
        meta_rows = meta_rdd.collect()
        
        with open(meta_path, 'wb') as f:
          cloudpickle.dump(meta_rows, f)
        util.log.info("... done caching; saved meta on %s files to %s" % (
          len(meta_rows), meta_path))
    
    with open(meta_path, 'rb') as f:
      meta_rows = cloudpickle.load(f)
    
    util.log.info("Creating TF dataset from %s cache files in %s ..." % (
      len(meta_rows), cls.ds_cache_root()))
    ds = None
    for meta_row in meta_rows:
      def gen():
        assert False, "I'm a dummy generator"
        yield
      cache_path = meta_row[0]
      output_shapes = cloudpickle.loads(meta_row[1])
      output_types = cloudpickle.loads(meta_row[2])
      cds = tf.data.Dataset.from_generator(
                gen,
                output_shapes=output_shapes,
                output_types=output_types)
      cds = cds.cache(cache_path)
      if ds is None:
        ds = cds
      else:
        ds = ds.concatenate(cds)
    util.log.info("Created TF Dataset.")
    return ds


      

            


    #       output_types = [
    #             tf.dtypes.as_dtype(cls.ROW_XFORM.x_dtype),
    #             tf.dtypes.as_dtype(cls.ROW_XFORM.y_dtype),
    #             tf.string,
    #       ]

    #     if cls.N_CACHE_SHARDS >= 1:
    #       imagerow_rdd = imagerow_rdd.coalesce(cls.N_CACHE_SHARDS)

    #     # TODO: 
    #     # * use attr.s package with Example! :)
    #     # * try to not have to pickle the array data
    #     # * figure out how to simply mappartitions to Tensorflow Cache
    #     from pyspark.sql import Row
    #     def encode(row):
    #       ex = cls.ROW_XFORM(row)
    #       return Row(**ex.to_dict())
        
    #     df = spark.createDataFrame(imagerow_rdd.map(encode))
    #     df.write.parquet(
    #       cls.table_root(),
    #       mode='overwrite',
    #       compression='lz4')
    #     util.log.info("... done caching dataset to %s ..." % cls.table_root())
      
    # with Spark.sess(spark) as spark:
    #   elem_df = spark.read.parquet(cls.table_root())
    #   # import pdb; pdb.set_trace()
    #   #   elem_df = cls._elem_df_cache
    #   #   print 'cached elem df to ', path
    #   # # elem_df = elem_df.persist(StorageLevel.DISK_ONLY)

    #   def df_row_to_tf_element(row):
    #     ex = Example.from_row(**row.asDict())
    #     return ex.astuple()
      
    #   import tensorflow as tf
    #   from au.spark import spark_df_to_tf_dataset
    #   ds = spark_df_to_tf_dataset(
    #           elem_df,
    #           df_row_to_tf_element,
    #           [
    #             tf.dtypes.as_dtype(cls.ROW_XFORM.x_dtype),
    #             tf.dtypes.as_dtype(cls.ROW_XFORM.y_dtype),
    #             tf.string,
    #           ])
      
    #   # Tensorflow is dumb and can't deduce the tensor shapes at the graph
    #   # building stage.  So we have to provide a hint this way ...
    #   maybe_row = elem_df.take(1)
    #   if maybe_row:
    #     dataset_element = df_row_to_tf_element(maybe_row[0])
    #     x, y, uri = dataset_element
    #     shapes = ( # NB: must be a tuple or TF does stupid stuff
    #       tf.TensorShape(x.shape),
    #       tf.TensorShape(y.shape),
    #       None, # You might think tf.TensorShape([None]) but nope nope nope!
    #     )
    #     ds = ds.apply(tf.contrib.data.assert_element_shape(shapes))
    #   return ds




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