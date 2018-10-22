import os

from au import conf
from au import util
from au.fixtures import dataset

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
      
#       # Tensorflow session, if available
#       self.tf_sess = None

      # For tensorflow models
      self.INPUT_TENSOR_SHAPE = [None, None, None, None]
  
  def __init__(self):
    self.params = INNModel.ParamsBase()
  
  @classmethod
  def load_or_train(self, cls, params=None):
    raise NotImplementedError
  
  def compute_activations_df(self, imagerow_df):
    raise NotImplementedError
  

class TFInferenceGraphFactory(object):
  def __init__(self, params=None):
    self.params = params # ParamsBase
  
  def create_inference_graph(self, input_image, base_graph):
    """Create and return an inference graph based upon `base_graph`.
    Use `input_image` as a tensor of uint8 [batch, width, height, chan]
    that respects `params.INPUT_TENSOR_SHAPE`.  
    
    Subclasses can use `make_normalize_ftor()` below to specify how to
    transform `ImageRow`s to include the desired input image format
    from arbitrary images.  The inference engine will choose how to
    create those `ImageRow`s and/or the actual `input_image` data.
    """
    return base_graph
  
  def make_normalize_ftor(self):
    return dataset.FillNormalized()
  
#   @property
#   def input(self):
#     # must be uint8 [None, width, height, chan]
#     pass
  
  @property
  def input_tensor_shape(self):
    return self.params.INPUT_TENSOR_SHAPE
  
  @property
  def output_names(self):
    return tuple()


class FillActivationsBase(object):
  
  def __init__(self, tigraph_factory):
    self.tigraph_factory = tigraph_factory
    clazzname = self.__class__.__name__
    self.overall_thruput = util.ThruputObserver(
                              name=clazzname,
                              log_on_del=True)
    self.tf_thruput = util.ThruputObserver(
                              name=clazzname + '.tf_thruput',
                              log_on_del=True)
  
  def __call__(self, iter_imagerows):
    for row in iter_imagerows:
      yield row
      
class FillActivationsTFDataset(object):
  
  def __call__(self, iter_imagerows):
    self.overall_thruput.start_block()
    
    import Queue
    import tensorflow as tf
    
    log = util.create_log()
    
    graph = tf.Graph()
    
    total_rows = 0
    total_bytes = 0
    processed_rows = Queue.Queue()
    def iter_normalized_np_images():
      normalize = self.tigraph_factory.make_normalize_ftor()
      for row in iter_imagerows:
        row = normalize(row)
        
        processed_rows.put(row, block=True)
        total_rows += 1
        total_bytes += row.attrs['normalized'].nbytes
        
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
      dataset = tf.Dataset.from_generator(
                     iter_normalized_np_images,
                     tf.uint8,
                     self.tigraph_factory.input_tensor_shape)
      input_image = dataset.make_one_shot_iterator().get_next()
    
    final_graph = self.tigraph_factory.create_inference_graph(
                                              input_image,
                                              graph)

    with final_graph.as_default():
      config = util.tf_create_session_config()
      
      # TODO: can we just use vanilla Session? & handle a tf.Dataset Stop?
      with tf.train.MonitoredTrainingSession(config=config) as sess:
        log.info("Session devices: %s" % ','.join(
                                  (d.name for d in sess.list_devices())))
        
        tensors_to_eval = self.tigraph_factory.output_names
        while not sess.should_stop():
          with self.tf_thruput.observe():
            result = sess.run(tensors_to_eval)
          
          assert len(result) >= 1
          batch_size = result[0].shape[0]
          for n in range(batch_size):
            row = processed_rows.get(block=True)
            name_val = zip(
                        (name, result[i][n,...])
                        for i, name in enumerate(tensors_to_eval))
            row.attrs['activation_to_val'] = dict(name_val)
            yield row
    
    self.tf_thruput.update_tallies(n=total_rows, num_bytes=total_bytes)
    self.overall_thruput.stop_block(n=total_rows, num_bytes=total_bytes)

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