import os
import pickle

from au import conf
from au import util
from au.fixtures import dataset
from au.spark import Spark

class INNModel(object):
  """A fascade for discriminative (neural network) models. Probably needs a
  new name."""
  
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

class INNGenerativeModel(INNModel):
  """A fascade for generative / 'unsupervised' (neural network) models.
  Probably needs a new name."""

  def create_generator(self):
    raise ValueError("TODO")
  
  def create_transformer(self):
    raise ValueError("TODO")


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

class Activations(object):
  """A pyspark-SQL-friendly wrapper for a set of activations"""

  __slots__ = ('_model_to_tensor_to_value',)

  def __init__(self, **datum):
    self._model_to_tensor_to_value = dict(datum)

    # # Unpack numpy arrays
    # for m in self._model_to_tensor_to_value:
    #   for tn in self._model_to_tensor_to_value[m]:
    #     from au.spark import NumpyArray
    #     tv = self._model_to_tensor_to_value[m][tn]
    #     if isinstance(tv, (NumpyArray,)):
    #       self._model_to_tensor_to_value[m][tn] = tv.arr

  
  def set_tensor(self, model_name, tensor_name, tensor_value):
    self._model_to_tensor_to_value.setdefault(model_name, {})
    self._model_to_tensor_to_value[model_name][tensor_name] = tensor_value
  
  def set_tensors(self, model_name, tensor_name_to_value):
    for tn, tv in tensor_name_to_value.iteritems():
      self.set_tensor(model_name, tn, tv)

  def get_tensor(self, model_name, tensor_name):
    return self._model_to_tensor_to_value.get(model_name, {}).get(tensor_name)

  def get_models(self):
    return self._model_to_tensor_to_value.keys()
  
  def get_default_model(self):
    return sorted(self._model_to_tensor_to_value.iterkeys())[0]
  
  def get_tensor_names(self, model_name):
    return self._model_to_tensor_to_value.get(model_name, {}).keys()

  def to_rows(self):
    # pyspark sees Row as a struct / map type
    from pyspark.sql import Row
    
    # TODO refine into rowable adapter thing ...
    for model, t_to_v in self._model_to_tensor_to_value.iteritems():
      yield Row(
        model=model,
        tensor_to_value=dict(
          (tn, pickle.dumps(tv)) for tn, tv in t_to_v.iteritems()),
      )

  @staticmethod
  def from_rows(rows):
    acts = Activations()
    for row in rows:
      for tn, tvb in row.tensor_to_value.iteritems():
        acts.set_tensor(row.model, tn, pickle.loads(tvb))
    return acts
    
    # pd_df):
    # for model_name in pd_df.model_name.unique():
    #   act_df = pd_df[pdf_df.model_name == model_name]
    #   tensor_to_value = dict(
    #     (r.tensor_name, r.tensor_value)
    #     for r in act_df.iterrows())
    #   yield Activations(
    #     model_name=model_name,
    #     tensor_to_value=tensor_to_value,
    #   )

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
      row['activations'] = row.get('activations', Activations())
      yield row


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
            if not row.attrs:
              row.attrs = {}

            act = row.attrs.get('activations') or Activations()
            act.set_tensors(self.tigraph_factory.model_name, tensor_to_value)
            row.attrs['activations'] = act
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
    if os.path.exists(cls.table_root()):
      return

    cls.IMAGE_TABLE_CLS.setup(spark=spark, params=cls.MODEL_PARAMS)

    util.log.info("Building table %s ..." % cls.TABLE_NAME)
    with Spark.sess(spark) as spark:
      imagerow_rdd = cls.IMAGE_TABLE_CLS.as_imagerow_rdd(spark)
      
      # # Since each output row (activations) might be bigger than the input
      # # row (image), and since Tensorflow will need some memory, we explode
      # # the number of partitions to avoid OOMs e.g. in the Spark
      # # Parquet-writing process at the end.
      # imagerow_rdd = imagerow_rdd.repartition(
      #   imagerow_rdd.getNumPartitions() * 10)

      model = cls.NNMODEL_CLS.load_or_train(cls.MODEL_PARAMS)
      filler = FillActivationsTFDataset(model=model)
      activated = imagerow_rdd.mapPartitions(filler)

      def to_activation_rows(imagerows):
        from pyspark.sql import Row
        for row in imagerows:
          if row.attrs:
            act = row.attrs.get('activations')
            if act:
              for arow in act.to_rows():
                yield Row(
                  # Partition / indentifying keys
                  dataset=row.dataset,
                  split=row.split,
                  uri=row.uri,
                  **arow.asDict(recursive=True)
                )
      
      activation_row_rdd = activated.mapPartitions(to_activation_rows)
      df = spark.createDataFrame(activation_row_rdd)
      df.show()
      partition_cols = list(dataset.ImageRow.DEFAULT_PQ_PARTITION_COLS)
      partition_cols.append('model')
      df.write.parquet(
                  path=cls.table_root(),
                  mode='overwrite',
                  compression='lz4',
                  partitionBy=partition_cols)
      util.log.info("... wrote to %s ." % cls.table_root())

  @classmethod
  def as_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    return df

  @classmethod
  def as_imagerow_rdd(cls, spark=None):
    with Spark.sess(spark) as spark:
      
      activations_df = cls.as_df(spark)

      # Combine activations model and tensor -> value into a single column ...
      from pyspark.sql.functions import struct
      combined = activations_df.withColumn(
        'm_t2v', struct('model', 'tensor_to_value'))
      
      # ... group by URI (and others) to induce ImageRows ...
      grouped = combined.groupBy('uri', 'dataset', 'split')
      image_act_df = grouped.agg({'m_t2v': 'collect_list'})
      image_act_df = image_act_df.withColumnRenamed(
                                  'collect_list(m_t2v)', 'm_t2vs')

      imagerow_df = cls.IMAGE_TABLE_CLS.as_imagerow_df(spark)
      joined = image_act_df.join(imagerow_df, ['uri', 'dataset', 'split'])

      # ... map row things to ImageRows
      def to_imagerow(row):
        irow = dataset.ImageRow(**row.asDict())
        acts = Activations.from_rows(row.m_t2vs)
        irow.attrs = irow.attrs or {}
        irow.attrs['activations'] = acts
        return irow
      
      imagerow_rdd = joined.rdd.map(to_imagerow)
      return imagerow_rdd

  @classmethod
  def save_tf_embedding_projector(
              cls,
              n_examples=1000, # -1 for all
              outdir=None,
              spark=None,
              include_metadata=True,
              include_sprite=True,
              sprite_hw=(40, 40)):
    """Write Tensorboard Projector summary files to `outdir` (or the default
    model dir).  Based upon:
    https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb
    https://github.com/tensorflow/tensorflow/issues/6322

    NB: for large samples, you might need to cap the amount of Parquet that
    Spark can buffer, e.g.
      builder.config('spark.sql.files.maxPartitionBytes', int(64 * 1e6))
    
    TODO: filter on split, dataset, and model_name
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
      if n_examples >= 0:
        uris = activations_df.select('uri').distinct()
        n = uris.count()
        N_EXAMPLES_SEED = 1337
        sampled_uris = uris.sample(
              withReplacement=False,
              fraction=float(n_examples) / n,
              seed=N_EXAMPLES_SEED).sort('uri').limit(n_examples)
        util.log.info(
          "... restricted to %s examples ..." % sampled_uris.count())
        sampled_uris = sampled_uris.withColumnRenamed('uri', 'sampled_uri')
        activations_df = activations_df.join(
          sampled_uris,
          activations_df.uri == sampled_uris.sampled_uri)
      rows_df = activations_df.select('uri', 'tensor_name', 'tensor_value')
      # rows_df = rows_df.repartition('uri')
      # rows_df = rows_df.sort('uri')
      uris = sorted(r.uri for r in rows_df.select('uri').distinct().collect())

      if include_metadata:
        metadata_path = os.path.join(outdir, 'metadata.tsv')
        util.log.info("... writing metadata to %s ..." % metadata_path)
        
        imagerow_df = cls.IMAGE_TABLE_CLS.as_imagerow_df(spark)
        imagerow_df = imagerow_df.filter(imagerow_df.uri.isin(uris))
        imagerow_df = imagerow_df.sort('uri')

        # TODO support other labels / attributes.. use dataframe etc
        with open(metadata_path, 'wc') as f:
          f.write('Name\tClass\n') # Can this be arbitrary?
          n = 0
          for row in imagerow_df.select('uri', 'label').toLocalIterator():
            f.write('%s\t%s\n' % (row.uri, row.label))
            n += 1
            if n % 500 == 0:
              util.log.info("... wrote %s ..." % n)
        util.log.info("... wrote metadata to %s ..." % metadata_path)

      if include_sprite:
        sprite_path = os.path.join(outdir, 'sprite.png')
        util.log.info("... writing sprite to %s ..." % sprite_path)

        # Get applicable images
        imagerow_df = cls.IMAGE_TABLE_CLS.as_imagerow_df(spark)
        imagerow_df = imagerow_df.filter(imagerow_df.uri.isin(uris))
        # imagerow_df = imagerow_df.sort('uri')
        
        N_CHANNELS = 3
        
        # Fetch sprite-sized images
        normalize = dataset.FillNormalized(
                                target_hw=sprite_hw,
                                target_nchan=N_CHANNELS)
        def get_norm(row):
          row = dataset.ImageRow(**row.asDict())
          row = normalize(row)
          arr = row.attrs['normalized']
          return row.uri, arr
        uri_arrs = imagerow_df.rdd.map(get_norm).collect()
        uri_arrs.sort(key=lambda ua: ua[0])
        sprite_images = [arr for uri, arr in uri_arrs]
        
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
      
      tensor_names = sorted(
            r.tensor_name
            for r in rows_df.select('tensor_name').distinct().collect())

      import tensorflow as tf
      from tensorflow.contrib.tensorboard.plugins import projector
      config = projector.ProjectorConfig()
      # g = tf.Graph()
      # with g.as_default():
      with util.tf_cpu_session() as sess:
        sess.run(tf.global_variables_initializer())

        embedding_vars = []
        for tensor_name in tensor_names:
          name = tensor_name.replace('/', '_').replace(':', '_') + '_values'
        
          tensor_df = rows_df.filter(rows_df.tensor_name == tensor_name)
          tensor_df = tensor_df.select('uri', 'tensor_value')
          util.log.info(
            "... fetching %s tensors (in %s partitions) for %s ..." % (
              tensor_df.count(), tensor_df.rdd.getNumPartitions(), tensor_name))

          vs = sorted(
                  tensor_df.toLocalIterator(),
                  key=lambda r: r.uri)
          # TODO mebbe r.tensor_value de-pickle then flatten
          vs = [r.tensor_value.arr.flatten() for r in vs]

          util.log.info(
            "... got %s tensors for %s ..." % (len(vs), tensor_name))
          import numpy as np
          vs = np.array(vs)

          
          embedding_var = tf.Variable(vs, name=name)
          # with open(os.path.join('/tmp', name + '.npy'), 'wc') as f:
          #   np.save(f, vs)
          
          sess.run(embedding_var.initializer)
          # embedding_vars.append(embedding_var)

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

          embedding_vars.append(embedding_var)

        util.log.info("... saving checkpoint ...")
        saver = tf.train.Saver(
          embedding_vars, save_relative_paths=True)
        saver.save(sess, os.path.join(outdir, 'embedding.ckpt'), global_step=0)
        writer = tf.summary.FileWriter(outdir)
        projector.visualize_embeddings(writer, config)
          
    assert False, 'TODO testme shipme'
      


# class ILatentSpaceModel(object):
  


