"""

Based upon tensorflow/models official mnist.py
https://github.com/tensorflow/models/blob/dfafba4a017c21c19dfdb60e1580f0b2ff5d361f/official/mnist/mnist.py

NB: earlier we tried mnist_eager.py, and it was less code (and much
simpler code) but a major PITA to interop with anything else.  

"""

import itertools
import os
from collections import OrderedDict

import tensorflow as tf

from au import util
from au.fixtures import dataset
from au.fixtures import nnmodel

MNIST_INPUT_SIZE = (28, 28)

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

def mnist_train(params):
  log = util.create_log()
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
    from official.mnist import dataset as mnist_dataset
    
    # Load the datasets
    train_ds = mnist_dataset.train(params.DATA_BASEDIR)
    if params.LIMIT >= 0:
      train_ds = train_ds.take(params.LIMIT)
    train_ds = train_ds.shuffle(60000).batch(params.BATCH_SIZE)
    return train_ds
  
  def eval_input_fn():
    test_ds = test_dataset(params)
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
    log.info('\nEvaluation results:\n\t%s\n' % eval_results)

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
    
  def create_inference_graph(self, input_image, base_graph):
    log = util.create_log()
    
    with base_graph.as_default():  
      sess = util.tf_cpu_session()
      with sess.as_default():
        tf_model = create_model()

        # Create ops and load weights
        
        # root = tf.train.Checkpoint(model=tf_model)
        # root.restore(tf.train.latest_checkpoint(self.params.MODEL_BASEDIR))
        # log.info("Read model params from %s" % self.params.MODEL_BASEDIR)
          
        pred = tf_model(tf.cast(input_image, tf.float32), training=False)
        checkpoint = tf.train.latest_checkpoint(self.params.MODEL_BASEDIR)
        saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
        self.graph = util.give_me_frozen_graph(
                            checkpoint,
                            nodes=self.output_names,
                            saver=saver,
                            base_graph=base_graph,
                            sess=sess)

    import pprint
    log.info("Loaded graph:")
    log.info(pprint.pformat(tf.contrib.graph_editor.get_tensors(self.graph)))
    return self.graph 

  @property
  def output_names(self):
    return (
      'sequential/conv2d/Relu:0',
      'sequential/conv2d_1/Relu:0',
      'sequential/dense/Relu:0',
      'sequential/dense_1/MatMul:0',
    )

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

  def __init__(self, params=None):
    super(MNIST, self).__init__(params=params)
    self.tf_graph = None
    self.tf_model = None
    self.predictor = None

  @classmethod
  def load_or_train(cls, params=None):
    log = util.create_log()
    
    params = params or MNIST.Params()
    model = MNIST(params=params)

    if not os.path.exists(os.path.join(params.MODEL_BASEDIR, 'model.ckpt')):
      log.info("Training!")
      # subprocess allows recovery of gpu memory!  See TFSessionPool comments
      # import multiprocessing
      # p = multiprocessing.Process(target=mnist_train, args=(params,))
      # p.start()
      # p.join()
      mnist_train(params)
      log.info("Done training!")

    model.igraph = MNISTGraph(params)
    return model

  def get_inference_graph(self):
    return self.igraph


class MNISTDataset(dataset.ImageTable):
  TABLE_NAME = 'MNIST'
  
  @classmethod
  def datasets_iter_image_rows(cls, params=None):
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
          
          if params.LIMIT >= 0 and n == params.LIMIT:
            break
          n += 1
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
        cls.datasets_iter_image_rows(params=params))
  
  @classmethod
  def setup(cls, params=None):
    cls.save_to_image_table(cls.datasets_iter_image_rows(params=params))

def setup_caches():
  MNIST.load_or_train()
  MNISTDataset.init()

if __name__ == '__main__':
  # self-test / demo mode!
  setup_caches()
