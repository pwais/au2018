"""

Based upon tensorflow/models official mnist.py
https://github.com/tensorflow/models/blob/dfafba4a017c21c19dfdb60e1580f0b2ff5d361f/official/mnist/mnist.py

NB: earlier we tried mnist_eager.py, and it was less code (and much
simpler code) but a major PITA to interop with anything else.  

"""

import itertools
import os
from collections import OrderedDict

import numpy as np

import tensorflow as tf

from au import util
from au.fixtures import dataset
from au.fixtures import nnmodel

MNIST_INPUT_SIZE = (28, 28)


###
### Data
###

class MNISTDataset(dataset.ImageTable):
  TABLE_NAME = 'MNIST'
  
  SPLIT = '' # Or 'train' or 'test'

  @classmethod
  def _datasets_iter_image_rows(cls, params=None):
    params = params or MNIST.Params()
    
    log = util.create_log()
    
    def gen_dataset(ds, split):
      import imageio
      import numpy as np
    
      n = 0
      with util.tf_data_session(ds) as (sess, iter_dataset):
        for image, label in iter_dataset():
          image = np.reshape(image * 255., (28, 28, 1)).astype(np.uint8)
          label = int(label)
          row = dataset.ImageRow.from_np_img_labels(
                                      image,
                                      label,
                                      dataset=cls.TABLE_NAME,
                                      split=split,
                                      uri='mnist_%s_%s' % (split, n))
          yield row
          
          n += 1
          if params.LIMIT >= 0 and n == params.LIMIT:
            break

          if n % 100 == 0:
            log.info("Read %s records from tf.Dataset" % n)
    
    from official.mnist import dataset as mnist_dataset
    
    # Keep our dataset ops in an isolated graph
    g = tf.Graph()
    with g.as_default():
      gens = itertools.chain(
          gen_dataset(mnist_dataset.train(params.DATA_BASEDIR), 'train'),
          gen_dataset(mnist_dataset.test(params.DATA_BASEDIR), 'test'))
      for row in gens:
        yield row
  
  @classmethod
  def save_datasets_as_png(cls, params=None):
    dataset.ImageRow.write_to_pngs(
        cls._datasets_iter_image_rows(params=params))
  
  @classmethod
  def setup(cls, params=None):
    cls.save_to_image_table(cls._datasets_iter_image_rows(params=params))
  
  @classmethod
  def get_class_freq(cls, spark, and_show=True):
    df = cls.as_imagerow_df(spark)
    table = cls.__name__.lower()
    df.createOrReplaceTempView(table)

    query_base = """
      SELECT
        FIRST(split) split,
        label, 
        COUNT(*) / 
          (SELECT COUNT(*) FROM {table} WHERE split = '{split}') frac
      FROM {table}
      WHERE split = '{split}'
      GROUP BY label
      ORDER BY label, split
    """
    
    query = """
      SELECT * FROM ( ( {train_query} ) UNION ( {test_query} ) )
      ORDER BY split, label
    """.format(
          train_query=query_base.format(table=table, split='train'),
          test_query=query_base.format(table=table, split='test'))

    res = spark.sql(query)
    if and_show:
      res.show()
    return res

  @classmethod
  def to_mnist_tf_dataset(cls, spark=None):
    iter_image_rows = cls.create_iter_all_rows(spark=spark)
    def iter_mnist_tuples():
      t = util.ThruputObserver(name='iter_mnist_tuples')
      norm = MNIST.Params().make_normalize_ftor()
      for row in iter_image_rows():
        # TODO: a faster filter.  For mnist this is plenty fast.
        if cls.SPLIT is not '':
          if cls.SPLIT.lower() != row.split:
            continue

        # TODO standardize this stuff
        row = norm(row)
        arr = row.attrs['normalized']

        # # Based upon official/mnist/dataset.py
        # def normalize_image(image):
        #   # # Normalize from [0, 255] to [0.0, 1.0]
        #   # # image = tf.decode_raw(image, tf.uint8) `image` is already an array
        #   # image = tf.cast(image, tf.float32)
        #   # image = tf.reshape(image, [784])
        #   image = image.astype(float) / 255.0
        #   return np.reshape(image, (784,))

        # def decode_label(label):
        #   # NB: `label` is already an int
        #   # label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        #   label = tf.reshape(label, [])  # label is a scalar
        #   return tf.to_int32(label)

        yield arr, int(row.label), row.uri

        t.update_tallies(n=1, num_bytes=arr.nbytes)
        t.maybe_log_progress(n=1000)

    d = tf.data.Dataset.from_generator(
              generator=iter_mnist_tuples,
              output_types=(tf.float32, tf.int32, tf.string),
              output_shapes=([784], [], []))
    return d

class MNISTTrainDataset(MNISTDataset):
  SPLIT = 'train'

class MNISTTestDataset(MNISTDataset):
  SPLIT = 'test'



## From mnist.py

def create_model(data_format='channels_last'):
  """Model to recognize digits in the MNIST dataset.
  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  But uses the tf.keras API.
  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
      
      -- NB: BUT THIS IS MNIST SO THIS POINT IS LIKELY IRRELEVANT
          EVEN FOR BENCHMARKING :P

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(
            1024,
            activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator.
  
  NB: `params` is a dict but unused; Tensorflow requires this parameter
  (and for it to be named `params`).
  """
  model = create_model()#params_dict['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    LEARNING_RATE = 1e-4
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # If we are running multi-GPU, we need to wrap the optimizer.
    #if params_dict.get('multi_gpu'):
    #  optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])
    
    global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar('global_step', global_step)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, global_step))

  elif mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })

def test_dataset(params):
  from official.mnist import dataset as mnist_dataset
  test_ds = mnist_dataset.test(params.DATA_BASEDIR)
  if params.LIMIT >= 0:
    test_ds = test_ds.take(params.LIMIT)
  test_ds = test_ds.batch(params.BATCH_SIZE)
  return test_ds

# def mnist_train(params):
#   log = util.create_log()
#   tf.logging.set_verbosity(tf.logging.DEBUG)
  
#   ## Model
#   model_dir = params.MODEL_BASEDIR
#   tf.gfile.MakeDirs(params.MODEL_BASEDIR)
  
#   mnist_classifier = tf.estimator.Estimator(
#     model_fn=model_fn,
#     params=None,
#     config=tf.estimator.RunConfig(
#       model_dir=model_dir,
#       save_summary_steps=10,
#       save_checkpoints_secs=10,
#       session_config=util.tf_create_session_config(),
#       log_step_count_steps=10))
    
#   ## Data
#   def train_input_fn():
#     from official.mnist import dataset as mnist_dataset
    
#     # Load the datasets
#     train_ds = mnist_dataset.train(params.DATA_BASEDIR)
#     if params.LIMIT >= 0:
#       train_ds = train_ds.take(params.LIMIT)
#     train_ds = train_ds.shuffle(60000).batch(params.BATCH_SIZE)
#     return train_ds
  
#   def eval_input_fn():
#     test_ds = test_dataset(params)
#     # No idea why we return an interator thingy instead of a dataset ...
#     return test_ds.make_one_shot_iterator().get_next()

#   # Set up hook that outputs training logs every 100 steps.
#   from official.utils.logs import hooks_helper
#   train_hooks = hooks_helper.get_train_hooks(
#       ['ExamplesPerSecondHook',
#        'LoggingTensorHook'],
#       model_dir=model_dir,
#       batch_size=params.BATCH_SIZE)

#   # Train and evaluate model.
#   for _ in range(params.TRAIN_EPOCHS):
#     mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     log.info('\nEvaluation results:\n\t%s\n' % eval_results)

#   # Export the model
#   # TODO do we need this placeholder junk?
#   image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_image')
#   input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
#       'image': image,
#   })
#   mnist_classifier.export_savedmodel(params.MODEL_BASEDIR, input_fn)


def mnist_train(params, train_table=None, test_table=None):
  if train_table is None:
    train_table = MNISTTrainDataset
  if test_table is None:
    test_table = MNISTTestDataset

  tf.logging.set_verbosity(tf.logging.DEBUG)
  
  ## Model
  model_dir = params.MODEL_BASEDIR
  tf.gfile.MakeDirs(params.MODEL_BASEDIR)
  
  mnist_classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params=None,
    config=tf.estimator.RunConfig(
      model_dir=model_dir,
      save_summary_steps=10,
      save_checkpoints_secs=10,
      session_config=util.tf_create_session_config(),
      log_step_count_steps=10))
    
  ## Data
  def train_input_fn():    
    # Load the dataset
    train_ds = train_table.to_mnist_tf_dataset()

    # Flow doesn't need uri
    train_ds = train_ds.map(lambda arr, label, uri: (arr, label))

    if params.LIMIT >= 0:
      train_ds = train_ds.take(params.LIMIT)
    train_ds = train_ds.shuffle(60000).batch(params.BATCH_SIZE)
    return train_ds
  
  def eval_input_fn():
    test_ds = test_table.to_mnist_tf_dataset()

    # Flow doesn't need uri
    test_ds = test_ds.map(lambda arr, label, uri: (arr, label))

    if params.LIMIT >= 0:
      test_ds = test_ds.take(params.LIMIT)
    test_ds = test_ds.batch(params.BATCH_SIZE)
    
    # No idea why we return an interator thingy instead of a dataset ...
    return test_ds.make_one_shot_iterator().get_next()

  # Set up hook that outputs training logs every 100 steps.
  from official.utils.logs import hooks_helper
  train_hooks = hooks_helper.get_train_hooks(
      ['ExamplesPerSecondHook',
       'LoggingTensorHook'],
      model_dir=model_dir,
      batch_size=params.BATCH_SIZE)

  # Train and evaluate model.
  for _ in range(params.TRAIN_EPOCHS):
    mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)

  # Export the model
  # TODO do we need this placeholder junk?
  image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_image')
  input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      'image': image,
  })
  mnist_classifier.export_savedmodel(params.MODEL_BASEDIR, input_fn)


## AU Interface

class MNISTGraph(nnmodel.TFInferenceGraphFactory):
  def __init__(self, params):
    self.params = params
  
  def create_frozen_graph_def(self):
    g = tf.Graph()
    with g.as_default():
      tf_model = create_model()

      input_image = tf.placeholder(
        tf.uint8,
        self.params.INPUT_TENSOR_SHAPE,
        name=self.params.INPUT_TENSOR_NAME)
      uris = tf.placeholder(
        tf.string,
        [None],
        name=self.params.INPUT_URIS_NAME)
      
      input_image_f = tf.cast(input_image, tf.float32)
      pred = tf_model(input_image_f, training=False)
      checkpoint = tf.train.latest_checkpoint(self.params.MODEL_BASEDIR)
      saver = tf.train.import_meta_graph(
                            checkpoint + '.meta',
                            clear_devices=True)
      return util.give_me_frozen_graph(
                          checkpoint,
                          nodes=list(self.output_names) + [input_image, uris],
                          saver=saver,
                          base_graph=g)


  # def create_inference_graph(self, uris, input_image, base_graph):
  #   log = util.create_log()

  #   gdef_frozen = self.create_frozen_graph_def()

  #   self.graph = base_graph
  #   with self.graph.as_default():
  #     tf.import_graph_def(
  #       gdef_frozen,
  #       name='',
  #       input_map={
  #         'au_inf_input': tf.cast(input_image, tf.float32),
  #         'au_inf_uris': uris,
  #             # https://stackoverflow.com/a/33770771
  #       })

  #   # with base_graph.as_default():
  #   #   sess = util.tf_cpu_session()
  #   #   with sess.as_default():
  #   #     tf_model = create_model()

  #   #     # Create ops and load weights
        
  #   #     # root = tf.train.Checkpoint(model=tf_model)
  #   #     # root.restore(tf.train.latest_checkpoint(self.params.MODEL_BASEDIR))
  #   #     # log.info("Read model params from %s" % self.params.MODEL_BASEDIR)
          
  #   #     pred = tf_model(tf.cast(input_image, tf.float32), training=False)
  #   #     uris = tf.identity(uris, name='au_uris')
  #   #     checkpoint = tf.train.latest_checkpoint(self.params.MODEL_BASEDIR)
  #   #     saver = tf.train.import_meta_graph(
  #   #                           checkpoint + '.meta',
  #   #                           clear_devices=True)
  #   #     self.graph = util.give_me_frozen_graph(
  #   #                         checkpoint,
  #   #                         nodes=list(self.output_names) + [input_image, uris],
  #   #                         saver=saver,
  #   #                         base_graph=base_graph,
  #   #                         sess=sess)

  #   import pprint
  #   util.log.info("Loaded graph:")
  #   util.log.info(pprint.pformat(tf.contrib.graph_editor.get_tensors(self.graph)))
  #   return self.graph 

  @property
  def output_names(self):
    return (
      'sequential/conv2d/Relu:0',
      'sequential/conv2d_1/Relu:0',
      'sequential/dense/Relu:0',
      'sequential/dense_1/MatMul:0',
    )

# Based upon official/mnist/dataset.py
def normalize_image(image):
  # # Normalize from [0, 255] to [0.0, 1.0]
  image = image.astype(np.float32) / 255.0
  return np.reshape(image, (784,))

class MNIST(nnmodel.INNModel):

  class Params(nnmodel.INNModel.ParamsBase):
    def __init__(self):
      super(MNIST.Params, self).__init__(model_name='MNIST')
      self.BATCH_SIZE = 100
      self.LEARNING_RATE = 0.01
      self.MOMENTUM = 0.5
      self.TRAIN_EPOCHS = 2
      self.LIMIT = -1
      self.INPUT_TENSOR_SHAPE = [
                  None, MNIST_INPUT_SIZE[0], MNIST_INPUT_SIZE[1], 1]
      self.NORM_FUNC = normalize_image

  def __init__(self, params=None):
    super(MNIST, self).__init__(params=params)
    self.tf_graph = None
    self.tf_model = None
    self.predictor = None

  @classmethod
  def load_or_train(cls, params=None):    
    params = params or MNIST.Params()
    model = MNIST(params=params)

    if not os.path.exists(os.path.join(params.MODEL_BASEDIR, 'model.ckpt')):
      util.log.info("Training!")
      # subprocess allows recovery of gpu memory!  See TFSessionPool comments
      # import multiprocessing
      # p = multiprocessing.Process(target=mnist_train, args=(params,))
      # p.start()
      # p.join()
      mnist_train(params)
      util.log.info("Done training!")

    model.igraph = MNISTGraph(params)
    return model

  def get_inference_graph(self):
    return self.igraph



def setup_caches():
  MNIST.load_or_train()
  MNISTDataset.setup()



if __name__ == '__main__':
  # self-test / demo mode!
  setup_caches()
