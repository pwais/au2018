import os

from au import conf
from au import util
from au.fixtures import dataset

class INNModel(object):
  """A fascade for (neural network) models. Probably needs a new name."""
  
  class ParamsBase(object):
    def __init__(self, model_name='INNModel', **overrides):
      # Canonical places to save files
      self.MODEL_NAME = model_name
      self.MODEL_BASEDIR = os.path.join(
                                  conf.AU_MODEL_CACHE,
                                  self.MODEL_NAME)
      self.DATA_BASEDIR = os.path.join(
                              conf.AU_DATA_CACHE,
                              self.MODEL_NAME)

      # For tensorflow models
      self.INPUT_TENSOR_SHAPE = [None, None, None, None]
      self.INPUT_TENSOR_NAME = 'au_inference_input'
      self.INPUT_URIS_NAME = 'au_inference_uris'
      self.NORM_FUNC = None
      
      self.TRAIN_WORKER_CLS = util.Worker

      # For batching inference
      self.INFERENCE_BATCH_SIZE = 10

      self.update(**overrides)
    
    def update(self, **overrides):
      for k, v in overrides.iteritems():
        setattr(self, k, v)

    def make_normalize_ftor(self):
      input_dims = self.INPUT_TENSOR_SHAPE
      target_hw = (input_dims[1], input_dims[2])
      target_nchan = input_dims[3]
      return dataset.FillNormalized(
                          target_hw=target_hw,
                          target_nchan=target_nchan,
                          norm_func=self.NORM_FUNC)


  def __init__(self, params=None):
    self.params = params or INNModel.ParamsBase()
  
  @classmethod
  def load_or_train(cls, params=None):
    """Create and return an instance, optionally training a model
    from scratch in the process."""
    return INNModel()
  
  def train(self):
    """Train an instance in-place (optional; some models are inference-only)."""
    pass

  def get_inference_graph(self):
    """Create and return a factory for creating inference graph(s)
    based upon this model instance."""
    return TFInferenceGraphFactory()
  


class TFInferenceGraphFactory(object):
  """Fascade for creating an inference-oriented graph
  for a Tensorflow model."""

  def __init__(self, params=None):
    self.params = params or INNModel.ParamsBase()
    self.graph = None
  
  def create_frozen_graph_def(self):
    """Create and return a string that represents a frozen GraphDef
    for the inference graph.  The graph should include input tensors
    with names `params.INPUT_TENSOR_NAME` and `params.INPUT_URIS_NAME`
    """
    return None

  def create_inference_graph(self, uris, input_image, base_graph):
    """Create and return an inference graph based upon `base_graph`.
    Use `input_image` as a tensor of uint8 [batch, height, width, chan]
    that respects `params.INPUT_TENSOR_SHAPE`.  Subclasses may simply want
    to override `create_frozen_graph_def()` above.
    
    Subclasses can use `params.make_normalize_ftor()` to specify how to
    transform `ImageRow`s to include the desired input image format
    from arbitrary images.  The inference engine will choose how to
    create those `ImageRow`s and/or the actual `input_image` data.
    """
    gdef_frozen = self.create_frozen_graph_def()
    if gdef_frozen is None:
      return base_graph

    import tensorflow as tf

    self.graph = base_graph
    with self.graph.as_default():
      tf.import_graph_def(
        gdef_frozen,
        name='',
        input_map={
          self.params.INPUT_TENSOR_NAME: input_image,
          self.params.INPUT_URIS_NAME: uris,
              # https://stackoverflow.com/a/33770771
        })

    import pprint
    util.log.info("Loaded graph:")
    util.log.info(
      '\n' +
      pprint.pformat(
        tf.contrib.graph_editor.get_tensors(self.graph)))
    return self.graph 
  
#
#
#
  # def make_normalize_ftor(self):
  #   input_dims = self.input_tensor_shape
  #   target_hw = (input_dims[1], input_dims[2])
  #   target_nchan = input_dims[3]
  #   return dataset.FillNormalized(
  #                       target_hw=target_hw,
  #                       target_nchan=target_nchan)
  
  @property
  def input_tensor_shape(self):
    return self.params.INPUT_TENSOR_SHAPE
  
  @property
  def input_uri_tensor_name(self):
    return self.params.INPUT_URIS_NAME

  @property
  def batch_size(self):
    return self.params.INFERENCE_BATCH_SIZE
  
  @property
  def output_names(self):
    return tuple()
  
  @property
  def model_name(self):
    return self.params.MODEL_NAME


## Utils

class FillActivationsBase(object):
  
  def __init__(self, tigraph_factory=None, model=None):
    if tigraph_factory is None:
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
      row['activations'] = row.get('activations', [])
      yield row

class Activations(object):
  """A pyspark-SQL-friendly wrapper for a set of activations"""

  __slots__ = ('model_name', '_tensor_to_value')

  def __init__(self, **kwargs):
    self.model_name = kwargs.get('model_name', '')
    self._tensor_to_value = {}
    if 'tensor_to_value' in kwargs:
      self.tensor_to_value = kwargs['tensor_to_value']
  
  def get_tensor_to_value(self):
    # Unpack numpy arrays
    if self._tensor_to_value is None:
      return {}
    return dict((k, v.arr) for k, v in self._tensor_to_value.iteritems())
  
  def set_tensor_to_value(self, tensor_to_value):
    # Pack numpy arrays for Spark
    from au.spark import NumpyArray
    for k, v in tensor_to_value.iteritems():
      self._tensor_to_value[k] = NumpyArray(v)
  
  tensor_to_value = property(get_tensor_to_value, set_tensor_to_value)

class FillActivationsTFDataset(FillActivationsBase):
  """A `FillActivationsBase` impl that leverages Tensorflow
  tf.Dataset to feed data into a tf.Graph""" 
  
  def __call__(self, iter_imagerows):
    self.overall_thruput.start_block()
    
    import Queue
    import tensorflow as tf

    util.log.info(
      "Filling activations for %s ..." % self.tigraph_factory.model_name)
    
    graph = tf.Graph()
    processed_rows = Queue.Queue()

    # Push normalization onto the Tensorflow tf.Dataset threadpool via the
    # generator below.  Read more after the function.
    def iter_normalized_np_images():
      normalize = self.tigraph_factory.params.make_normalize_ftor()
      for row in iter_imagerows:
        row = normalize(row)
        processed_rows.put(row)
        arr = row.attrs['normalized']
        yield row.uri, arr
        
        self.tf_thruput.update_tallies(num_bytes=arr.nbytes)
        self.overall_thruput.update_tallies(num_bytes=arr.nbytes)
    
    # We'll use the tf.Dataset.from_generator facility to read ImageRows
    # (which have the image bytes in Python memory) into the graph.  The use
    # case we have in mind is via the pyspark Dataframe / RDD API, where
    # the image bytes will need to be in Python memory anyways.  (If we want
    # to try to DMA between the Spark JVM and Tensorflow, we need to use
    # Databricks Tensorframes, which we'll consider implementing as an
    # additional subclass).  Using tf.Dataset here should at least allow
    # Tensorflow to push reading into a background thread so that Spark and/or
    # Python execution won't block inference if inference dominates. To achieve
    # multi-threaded I/O, we lean on Spark, which will typically run one
    # instance of this functor per core (thus providing some
    # cache-friendliness).
    uris_op_name = self.tigraph_factory.input_uri_tensor_name
    uris_tensor_name = uris_op_name + ':0'
    with graph.as_default():
      # Hack: let us handle batch size here ...
      input_shape = list(self.tigraph_factory.input_tensor_shape)
      input_shape = input_shape[1:]
      d = tf.data.Dataset.from_generator(
                      generator=iter_normalized_np_images,
                      output_types=(tf.string, tf.uint8),
                      output_shapes=(tf.TensorShape([]), input_shape))
      d = d.batch(self.tigraph_factory.batch_size)
      uris, input_image = d.make_one_shot_iterator().get_next()
      uris = tf.identity(uris, name=uris_op_name)
    
    util.log.info("Creating inference graph ...")
    final_graph = self.tigraph_factory.create_inference_graph(
                                              uris,
                                              input_image,
                                              graph)
    
    # with final_graph.as_default() as g:
    #   uris = tf.identity(uris, name='au_uris')
    #   assert uris.graph is g
    util.log.info("... done creating inference graph.")

    with final_graph.as_default():
      # TODO: support using single GPUs; requires running in a subprocess
      # due to Tensorflow memory madness :( 
      with util.tf_cpu_session() as sess:
          
        tensors_to_eval = list(self.tigraph_factory.output_names)
        assert tensors_to_eval
        tensors_to_eval.append(uris_tensor_name)
        
        uri_to_row_cache = {}
        while True:
          try:
            self.tf_thruput.start_block()
            output = sess.run(tensors_to_eval)
            self.tf_thruput.stop_block()
                # NB: above will processes `batch_size` rows in one run()
          except (tf.errors.OutOfRangeError, StopIteration):
            # see MonitoredTrainingSession.StepContext
            break
          
          assert len(output) >= 1
          batch_size = output[0].shape[0]
          for n in range(batch_size):
            tensor_to_value = dict(
                        (name, output[i][n,...])
                        for i, name in enumerate(tensors_to_eval))
            row_uri = str(tensor_to_value.pop(uris_tensor_name))

            # Find the row corresponding to this output
            for _ in range(100 * batch_size):
              if row_uri in uri_to_row_cache:
                break
              else:
                row = processed_rows.get()
                  # NB: we expect worker threads to spend most of their time
                  # blocking on the Tensorflow `sess.run()` call above, and
                  # sessions should process rows roughly in order (within
                  # the tolerance of this inner loop), so this Queue::get()
                  # call should be fast and correct.
                uri_to_row_cache[row.uri] = row
            row = uri_to_row_cache.pop(row_uri) # Don't let cache grow

            # Fill the row with the activation data
            if 'activations' not in row.attrs:
              row.attrs['activations'] = []  
            
            act = Activations(
                      model_name=self.tigraph_factory.model_name,
                      tensor_to_value=tensor_to_value)
            row.attrs['activations'].append(act)
            yield row
    
            self.tf_thruput.update_tallies(n=1)
            self.overall_thruput.update_tallies(n=1)
    tf.reset_default_graph()
    self.overall_thruput.stop_block()

class ActivationsTable(object):

  TABLE_NAME = 'default'
  NNMODEL_CLS = None
  MODEL_PARAMS = None
  IMAGE_TABLE_CLS = None

  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, cls.TABLE_NAME)
  
  @classmethod
  def setup(cls, spark=None):
    log = util.create_log()
    log.info("Building table %s ..." % cls.TABLE_NAME)

    spark = spark or util.Spark.getOrCreate()
    
    img_rdd = cls.IMAGE_TABLE_CLS.as_imagerow_rdd(spark)

    model = cls.NNMODEL_CLS.load_or_train(cls.MODEL_PARAMS)
    filler = FillActivationsTFDataset(model=model)

    activated = img_rdd.mapPartitions(filler)

    def to_activation_rows(imagerows):
      from pyspark.sql import Row
      for row in imagerows:
        if row.attrs is '':
          continue

        activations = row.attrs.get('activations')
        if not activations:
          continue
        
        for act in activations:
          for tensor_name, value in act._tensor_to_value.iteritems():
            yield Row(
              model_name=model.params.MODEL_NAME,
              tensor_name=tensor_name,
              tensor_value=value,
              
              dataset=row.dataset,
              split=row.split,
              uri=row.uri,
            )
    
    activation_row_rdd = activated.mapPartitions(to_activation_rows)

    df = spark.createDataFrame(activation_row_rdd)
    df.show()
    writer = df.write.parquet(
                path=cls.table_root(),
                mode='overwrite',
                compression='lz4',
                partitionBy=dataset.ImageRow.DEFAULT_PQ_PARTITION_COLS)
    log.info("... wrote to %s ." % cls.table_root())
