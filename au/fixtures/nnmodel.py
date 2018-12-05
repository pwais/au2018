import os

from au import conf
from au import util
from au.fixtures import dataset

class INNModel(object):
  """A fascade for (neural network) models. Probably needs a new name."""
  
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

      # For tensorflow models
      self.INPUT_TENSOR_SHAPE = [None, None, None, None]
      
      # For batching inference
      self.INFERENCE_BATCH_SIZE = 10


  def __init__(self):
    self.params = INNModel.ParamsBase()
  
  @classmethod
  def load_or_train(cls, params=None):
    """Create and return an instance, optionally training a model
    from scratch in the process."""
    return INNModel()
  
  def get_inference_graph(self):
    """Create and return a factory for creating inference graph(s)
    based upon this model instance."""
    return TFInferenceGraphFactory()
  


class TFInferenceGraphFactory(object):
  """Fascade for creating an inference-oriented graph
  for a Tensorflow model."""

  def __init__(self, params=None):
    self.params = params # ParamsBase
  
  def create_inference_graph(self, input_image, base_graph):
    """Create and return an inference graph based upon `base_graph`.
    Use `input_image` as a tensor of uint8 [batch, height, width, chan]
    that respects `params.INPUT_TENSOR_SHAPE`.  
    
    Subclasses can use `make_normalize_ftor()` below to specify how to
    transform `ImageRow`s to include the desired input image format
    from arbitrary images.  The inference engine will choose how to
    create those `ImageRow`s and/or the actual `input_image` data.
    """
    return base_graph
  
  def make_normalize_ftor(self):
    input_dims = self.input_tensor_shape
    target_hw = (input_dims[1], input_dims[2])
    target_nchan = input_dims[3]
    return dataset.FillNormalized(
                        target_hw=target_hw,
                        target_nchan=target_nchan)
  
#   @property
#   def input(self):
#     # must be uint8 [None, width, height, chan]
#     pass
  
  @property
  def input_tensor_shape(self):
    return self.params.INPUT_TENSOR_SHAPE
  
  @property
  def batch_size(self):
    return self.params.INFERENCE_BATCH_SIZE
  
  @property
  def output_names(self):
    return tuple()


## Utils

class FillActivationsBase(object):
  
  def __init__(self, tigraph_factory=None, model=None):
    if tigraph_factory is None:
      assert model is not None
      tigraph_factory = model.get_inference_graph()
    assert isinstance(tigraph_factory, TFInferenceGraphFactory)
    
    self.tigraph_factory = tigraph_factory
    clazzname = self.__class__.__name__
    self.overall_thruput = util.ThruputObserver(
                              name=clazzname,
                              log_on_del=True)
    self.tf_thruput = util.ThruputObserver(
                              name=clazzname + '.tf_thruput',
                              log_on_del=True)
  
  def __call__(self, iter_imagerows):
    # Identity op; fills nothing!
    for row in iter_imagerows:
      yield row

class FillActivationsTFDataset(FillActivationsBase):
  """A `FillActivationsBase` impl that leverages Tensorflow
  tf.Dataset to feed data into a tf.Graph""" 
  
  def __call__(self, iter_imagerows):

    self.overall_thruput.start_block()
    
    import Queue
    import tensorflow as tf
    
    log = util.create_log()
    
    graph = tf.Graph()
    total_rows_bytes = [0, 0]
    processed_rows = Queue.Queue()
    def iter_normalized_np_images():
      normalize = self.tigraph_factory.make_normalize_ftor()
      for row in iter_imagerows:
        row = normalize(row)
        
        processed_rows.put(row, block=True)
        total_rows_bytes[0] += 1
        total_rows_bytes[1] += row.attrs['normalized'].nbytes
        
        yield row.attrs['normalized']
    
    # We'll use the tf.Dataset.from_generator facility to read ImageRows
    # (which have the image bytes in Python memory) into the graph.  The use
    # case we have in mind is via the pyspark Dataframe / RDD API, where
    # the image bytes will need to be in Python memory anyways.  (If we want
    # to try to DMA between the Spark JVM and Tensorflow, we need to use
    # Databricks Tensorframes, which we'll consider implementing as an
    # additional subclass).  Using tf.Dataset here should at least allow
    # Tensorflow to push reading into a background thread so that Spark and/or
    # Python execution won't block inference if inference dominates. To achieve
    # multi-threaded I/O, we lean on Spark. 
    with graph.as_default():
      # Hack: let us handle batch size here ...
      input_shape = list(self.tigraph_factory.input_tensor_shape)
      input_shape = input_shape[1:]
      d = tf.data.Dataset.from_generator(
                      iter_normalized_np_images,
                      tf.uint8,
                      input_shape)
      d = d.batch(self.tigraph_factory.batch_size)
      input_image = d.make_one_shot_iterator().get_next()
    
    final_graph = self.tigraph_factory.create_inference_graph(
                                              input_image,
                                              graph)

    with final_graph.as_default():
      config = util.tf_create_session_config()
      
      # TODO: can we just use vanilla Session? & handle a tf.Dataset Stop?
      with tf.train.MonitoredTrainingSession(config=config) as sess:
#         log.info("Session devices: %s" % ','.join(
#                                   (d.name for d in sess.list_devices())))
        
        tensors_to_eval = self.tigraph_factory.output_names
        assert tensors_to_eval
        
#         import pprint
#         log.info(pprint.pformat(tf.contrib.graph_editor.get_tensors(final_graph)))
        
        while not sess.should_stop():
          with self.tf_thruput.observe():
            result = sess.run(tensors_to_eval)
              # NB: processes batch_size rows in one run()
          
          assert len(result) >= 1
          batch_size = result[0].shape[0]
          for n in range(batch_size):
            row = processed_rows.get(block=True)
              # NB: we expect worker threads to spend most of their time
              # blocking on the Tensorflow `sess.run()` call above, so
              # this Queue::get() call should in practice block very rarely.
              # Confirm using thruput stats (`overall_thruput` vs `tf_thruput`)

            activation_to_val = dict(
                        (name, result[i][n,...])
                        for i, name in enumerate(tensors_to_eval))
            row.attrs['activation_to_val'] = activation_to_val
            yield row
    
    total_rows, total_bytes = total_rows_bytes
    self.tf_thruput.update_tallies(n=total_rows, num_bytes=total_bytes)
    self.overall_thruput.stop_block(n=total_rows, num_bytes=total_bytes)



class ActivationsTable(object):

  TABLE_NAME = 'default'
  NNMODEL_CLS = None
  MODEL_PARAMS = None
  IMAGE_TABLE_CLS = None

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, cls.TABLE_NAME)
  
  @classmethod
  def setup(cls, spark=None, parallel=10):
    log = util.create_log()
    log.info("Building table %s ..." % cls.TABLE_NAME)

    spark = spark or util.Spark.getOrCreate()

    img_rdd = cls.IMAGE_TABLE_CLS.as_imagerow_rdd(spark)
    
    model = cls.NNMODEL_CLS.load_or_train(cls.MODEL_PARAMS)
    filler = FillActivationsTFDataset(model=model)

    img_rdd = img_rdd.repartition(parallel)
    activated = img_rdd.mapPartitions(filler)

    def to_activation_rows(imagerows):
      for row in imagerows:
        if row.attrs is '':
          continue

        activation_to_val = row.attrs.get('activation_to_val')
        if not activation_to_val:
          continue
        
        import pickle
        yield {
          'model_name': model.params.MODEL_NAME,
          'dataset': row.dataset,
          'split': row.split,
          'uri': row.uri,
          'activations': dict(
            (k, pickle.dumps(v))
            for k, v in activation_to_val.iteritems()),
        }
    
    activation_row_rdd = activated.mapPartitions(to_activation_rows)
    
    df = spark.createDataFrame(activation_row_rdd)
    df.show()
    writer = df.write.parquet(
                path=cls.table_root(),
                mode='overwrite',
                compression='snappy',
                partitionBy=dataset.ImageRow.DEFAULT_PQ_PARTITION_COLS)
    log.info("... wrote to %s ." % cls.table_root())

        





"""

au_image_bytes = tf.placeholder(tf.string, [], name="au_image_bytes")
image_uint8 = tf.decode_raw(image_buffer, tf.uint8, name="decode_raw")
image_float = tf.to_float(image_uint8)


make these things pluggable so we can test ... :)

a ftor that takes in image rows and manipulates them-- preprocess:
  * decode to numpy
  * resize
  * mebbe do mean sub on numpy array
  * mebbe FILL the resized numpy... see if parrow likes that?

a ftor that takes in image rows, runs inference, and fills each image
  row with activations and stuff.  use tf.Dataset.from_tensor_slices
  and feed_dict to feed in numpy.

"""