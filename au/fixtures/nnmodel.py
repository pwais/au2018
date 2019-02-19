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

    def to_row(self):
      from collections import OrderedDict
      row = OrderedDict()
      
      for attrname in sorted(dir(self)):
        if not attrname.startswith('_') and attrname.isupper():
          v = getattr(self, attrname)
          

      return row


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
    # # TODO fixme
    # iter_imagerows = list(iter_imagerows)
    # if not iter_imagerows:
    #   return

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
      n = 0
      for row in iter_imagerows:
        row = normalize(row)
        processed_rows.put(row)
        arr = row.attrs['normalized']
        yield row.uri, arr
        
        self.tf_thruput.update_tallies(num_bytes=arr.nbytes)
        self.overall_thruput.update_tallies(num_bytes=arr.nbytes)
        n += 1
      util.log.info("Partition had %s rows" % n)
    
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
            self.tf_thruput.maybe_log_progress(n=1000)
            self.overall_thruput.update_tallies(n=1)
            self.overall_thruput.maybe_log_progress(n=1000)
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

    from au.spark import Spark
    with Spark.sess(spark) as spark:
      # ssc, dstream = cls.IMAGE_TABLE_CLS.as_imagerow_rdd_stream(spark)
      imagerow_rdd = cls.IMAGE_TABLE_CLS.as_imagerow_rdd(spark)
      
      # Since each output row (activations) might be bigger than the input
      # row (image), explode the number of partitions to avoid OOMs in
      # the Spark Parquet-writing process at the end.
      imagerow_rdd = imagerow_rdd.repartition(
        imagerow_rdd.getNumPartitions() * 4)

      model = cls.NNMODEL_CLS.load_or_train(cls.MODEL_PARAMS)
      filler = FillActivationsTFDataset(model=model)

      # def save_activated(t, rdd):
        # if rdd.isEmpty():
        #   return
        # util.log.info("wat %s %s" % (t, rdd.getNumPartitions()))
      activated = imagerow_rdd.mapPartitions(filler)

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
      df.write.parquet(
                  path=cls.table_root(),
                  mode='overwrite',
                  compression='lz4',
                  partitionBy=dataset.ImageRow.DEFAULT_PQ_PARTITION_COLS)
        # writer.start()
        # writer.processAllAvailable()
        # writer.stop()
      log.info("... wrote to %s ." % cls.table_root())
      # dstream.foreachRDD(save_activated)
      # ssc.start()
      # ssc.stop(stopSparkContext=False, stopGraceFully=True)
      # ssc.stop()

  @classmethod
  def as_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    return df

  @classmethod
  def save_tf_embedding_projector(
              cls,
              n_vec=100, # default: ALL
              n_vec_seed=1337,
              outdir=None,
              spark=None,
              include_metadata=True,
              include_sprite=True,
              sprite_hw=(25, 25)):
    """Write Tensorboard Projector summary files to `outdir` (or the default
    model dir).  Based upon:
    https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb
    https://github.com/tensorflow/tensorflow/issues/6322
    """

    if not outdir:
      outdir = os.path.join(
                  cls.MODEL_PARAMS.MODEL_BASEDIR,
                  'tf_embedding_projector')
    util.mkdir(outdir)
    util.log.info(
      "Saving TF Projector summaries for %s to %s ..." % (
        cls.TABLE_NAME, outdir))

    rows_df = None
    uris = []
    metadata_path = ''
    sprite_path = ''
    from au.spark import Spark
    with Spark.sess(spark) as spark:
      activations_df = cls.as_df(spark)
      if n_vec >= 0:
        n = activations_df.count()
        activations_df = activations_df.sample(
              withReplacement=False,
              fraction=float(n_vec) / n,
              seed=n_vec_seed)
      rows_df = activations_df.select('uri', 'tensor_name', 'tensor_value')
      rows_df = rows_df.repartition(100 * rows_df.rdd.getNumPartitions())
      # rows_df = rows_df.sort('uri')
      uris = sorted(r.uri for r in rows_df.select('uri').distinct().collect())

      if False: #include_metadata:
        metadata_path = os.path.join(outdir, 'metadata.tsv')
        util.log.info("... writing metadata to %s ..." % metadata_path)
        
        imagerow_df = cls.IMAGE_TABLE_CLS.as_imagerow_df(spark)
        imagerow_df = imagerow_df.filter(imagerow_df.uri.isin(uris))
        imagerow_df = imagerow_df.sort('uri')

        # TODO support other labels / attributes.. use dataframe etc
        with open(metadata_path, 'wc') as f:
          f.write('Name\tClass\n') # Can this be arbitrary?
          for row in imagerow_df.select('uri', 'label').toLocalIterator():
            f.write('%s\t%s\n' % (row.uri, row.label))
        util.log.info("... wrote metadata to %s ..." % metadata_path)

      if False:#include_sprite:
        sprite_path = os.path.join(outdir, 'sprite.png')
        util.log.info("... writing sprite to %s ..." % sprite_path)

        # Get applicable images
        imagerow_df = cls.IMAGE_TABLE_CLS.as_imagerow_df(spark)
        imagerow_df = imagerow_df.filter(imagerow_df.uri.isin(uris))
        imagerow_df = imagerow_df.sort('uri')
        
        N_CHANNELS = 3
        
        # Fetch sprite-sized images
        normalize = dataset.FillNormalized(
                                target_hw=sprite_hw,
                                target_nchan=N_CHANNELS)
        def get_norm(row):
          row = dataset.ImageRow(**row.asDict())
          row = normalize(row)
          arr = row.attrs['normalized']
          return arr
        sprite_images = imagerow_df.rdd.map(get_norm).collect()
        
        # LOL horrible docs
        # https://github.com/tensorflow/tensorboard/issues/670#issuecomment-419105543
        # https://www.tensorflow.org/guide/embedding#images
        import imageio
        import math
        import numpy as np
        # Sprites must be square
        n_cols = int(math.ceil(math.sqrt(len(sprite_images))))
        n_cells = n_cols * n_cols
        
        # Pad with blank images to give us a square grid
        sprite_images.extend(
          np.zeros((sprite_hw[0], sprite_hw[1], N_CHANNELS))
          for _ in range(n_cells - len(sprite_images)))

        sprite = np.array(sprite_images)
        sprite = sprite.reshape(
                    (n_cols, n_cols, sprite_hw[0], sprite_hw[1], N_CHANNELS))
        sprite = sprite.swapaxes(1, 2).reshape((
                    n_cols * sprite_hw[0],
                    n_cols * sprite_hw[1],
                    N_CHANNELS))
        
        imageio.imwrite(sprite_path, sprite, format='png')
        util.log.info("... wrote sprite to %s ..." % sprite_path)

      # Save Projector Summaries
      import tensorflow as tf
      writer = tf.summary.FileWriter(outdir)
      with util.tf_cpu_session() as sess:
        from tensorflow.contrib.tensorboard.plugins import projector
        config = projector.ProjectorConfig()
        tensor_names = sorted(
          r.tensor_name
          for r in rows_df.select('tensor_name').distinct().collect())
        # from pyspark import StorageLevel
        # rows_df = rows_df.persist(StorageLevel.DISK_ONLY)
        for tensor_name in tensor_names:
          tensor_df = rows_df.filter(rows_df.tensor_name == tensor_name)
          tensor_df = tensor_df.select('uri', 'tensor_value')
          util.log.info(
            "... fetching %s tensors for %s ..." % (
              tensor_df.count(), tensor_name))
          # tensor_df = tensor_df.repartition(10 * tensor_df.rdd.getNumPartitions())
          # from pyspark import StorageLevel
          # tensor_df = tensor_df.persist(StorageLevel.DISK_ONLY)
          # tensor_df = tensor_df.sort(tensor_df.uri).select('tensor_value')
          # vs = [r.tensor_value for r in tensor_df.toLocalIterator()]
          
          # vs = []
          # for uris in util.ichunked(uris, 1000):
          #   vs.extend(
          #     r.tensor_value
          #     for r in tensor_df.filter(tensor_df.uri.isin(*uris)).sort('uri').collect())
          #   print 'vs', len(vs)

          vs = sorted(
                  tensor_df.toLocalIterator(),
                  key=lambda r: r.uri)
          vs = [r.tensor_value.arr for r in vs]

          util.log.info(
            "... got %s tensors for %s ..." % (len(vs), tensor_name))
          import numpy as np
          vs = np.array(vs)

          embedding_var = tf.Variable(
                              vs,
                              name=tensor_name.replace('/', '_').replace(':', '_') + '_values')
          sess.run(embedding_var.initializer)
          util.log.info("... ran session ...")

          embedding = config.embeddings.add()
          embedding.tensor_name = embedding_var.name

          if metadata_path:
            embedding.metadata_path = metadata_path

          if sprite_path:
            embedding.sprite.image_path = sprite_path
            embedding.sprite.single_image_dim.extend(
              # (width, height)
              # https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto#L22
              [sprite_hw[1], sprite_hw[0]])

        projector.visualize_embeddings(writer, config)
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(outdir, 'model2.ckpt'), 1)
      
