"""

Based upon tensorflow/models official mnist_eager.py
https://github.com/tensorflow/models/blob/dfafba4a017c21c19dfdb60e1580f0b2ff5d361f/official/mnist/mnist_eager.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

tfe = tf.contrib.eager

from au import conf
from au import util
from au.fixtures import nnmodel

## From mnist_eager.py

def loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

def train(model, optimizer, dataset, step_counter, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  log = util.create_log()

  start = time.time()
  for (batch, (images, labels)) in enumerate(dataset):
    with tf.contrib.summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
      # Record the operations used to compute the loss given the input,
      # so that the gradient of the loss with respect to the variables
      # can be computed.
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss(logits, labels)
        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=step_counter)
      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        log.info('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
        start = time.time()


def test(model, dataset):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

  log = util.create_log()

  for (images, labels) in dataset:
    logits = model(images, training=False)
    avg_loss(loss(logits, labels))
    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  log.info('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar('loss', avg_loss.result())
    tf.contrib.summary.scalar('accuracy', accuracy.result())

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
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

## Interface

class MNistEager(nnmodel.INNModel):

  class Params(nnmodel.INNModel.ParamsBase):
    def __init__(self):
      super(MNistEager.Params, self).__init__(model_name='MNistEager')
      self.BATCH_SIZE = 100
      self.LEARNING_RATE = 0.01
      self.MOMENTUM = 0.5
      self.TRAIN_EPOCHS = 10
      self.LIMIT = -1

  def __init__(self):
    self.tf_model = None
    self.predictor = None

  @staticmethod
  def save_datasets_as_png(params=None):   
    params = params or MNistEager.Params()
    
    log = util.create_log()
    
    def save_dataset(ds, tag):
      import imageio
      import numpy as np
      
      img_dir = os.path.join(params.DATA_BASEDIR, tag, 'images')
      util.mkdir(img_dir)
      path_to_label = {}
    
      # Silly way to iterate over a tf.Dataset
      # https://stackoverflow.com/a/47917849
      iterator = ds.make_one_shot_iterator()
      next_element = iterator.get_next()
      i = 0
      with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
          image, label = sess.run(next_element)
          image = np.reshape(image * 255., (28, 28, 1)).astype(np.uint8)
          label = int(label)

          dest = os.path.abspath(os.path.join(img_dir, 'img_%s_label-%s.png' % (i, label)))
          imageio.imwrite(dest, image)
          path_to_label[dest] = label
          if ((i+1) % 100) == 0:
            log.info("Saved %s images to %s" % (i+1, img_dir))
          
          if i == params.LIMIT:
            break
          i += 1
        
      import json
      with open(os.path.join(params.DATA_BASEDIR, tag, 'path_to_label.json'), 'w') as f:
        json.dump(path_to_label, f, indent=2)
    
    from official.mnist import dataset as mnist_dataset
    save_dataset(mnist_dataset.train(params.DATA_BASEDIR), 'train')
    save_dataset(mnist_dataset.test(params.DATA_BASEDIR), 'test')

  @staticmethod
  def _train(params):
    from official.mnist import dataset as mnist_dataset
    from official.mnist import mnist

    # tf.enable_eager_execution()
    log = util.create_log()

    # Automatically determine device and data_format
    #(device, data_format) = ('/gpu:0', 'channels_first')
    #if not tf.test.is_gpu_available():
    #  (device, data_format) = ('/cpu:0', 'channels_last')
    #log.info('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    train_ds = mnist_dataset.train(params.DATA_BASEDIR).shuffle(60000)
    if params.LIMIT >= 0:
      train_ds = train_ds.take(params.LIMIT)
    train_ds = train_ds.batch(params.BATCH_SIZE)

    test_ds = mnist_dataset.test(params.DATA_BASEDIR)
    if params.LIMIT >= 0:
      test_ds = test_ds.take(params.LIMIT)
    test_ds = test_ds.batch(params.BATCH_SIZE)

    # Create the model and optimizer
    model = create_model('channels_last')
    optimizer = tf.train.MomentumOptimizer(params.LEARNING_RATE, params.MOMENTUM)

    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(params.TENSORBOARD_BASEDIR, 'train')
    test_dir = os.path.join(params.TENSORBOARD_BASEDIR, 'eval')
    tf.gfile.MakeDirs(params.TENSORBOARD_BASEDIR)

    summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
    test_summary_writer = tf.contrib.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

    # Create and restore checkpoint (if one exists on the path)
    tf.gfile.MakeDirs(params.MODEL_BASEDIR)
    checkpoint_prefix = os.path.join(params.MODEL_BASEDIR, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(
      model=model, optimizer=optimizer, step_counter=step_counter)
    checkpoint.restore(tf.train.latest_checkpoint(params.MODEL_BASEDIR))

    # Train and evaluate for a set number of epochs.
    log.info("Training!")
    # with tf.device(device):
    for _ in range(params.TRAIN_EPOCHS):
      start = time.time()
      with summary_writer.as_default():
        train(model, optimizer, train_ds, step_counter, 10)
      end = time.time()
      log.info('\nTrain time for epoch #%d (%d total steps): %f' %
            (checkpoint.save_counter.numpy() + 1,
             step_counter.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, test_ds)
      checkpoint.save(checkpoint_prefix)
    # tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
    log.info("Done training!")

  @classmethod
  def load_or_train(cls, params=None):
    model = MNistEager()
    params = params or MNistEager.Params()
    model.params = params

    tf.enable_eager_execution()

    if not os.path.exists(params.MODEL_BASEDIR):
      ## Train a model!
      MNistEager._train(params)
      tf.reset_default_graph()
    
    # Load saved model
    # 
    # sess = tf.get_default_session() or tf.Session()
    model.tf_model = create_model()
    # saver = tf.train.Saver()
    # saver.restore(sess, 
    root = tf.train.Checkpoint(model=model.tf_model)
    root.restore(tf.train.latest_checkpoint(params.MODEL_BASEDIR))

    # model.predictor = predictor.from_saved_model(params.MODEL_BASEDIR)
    print(tf.get_default_graph())
    return model

  def iter_activations(self):
    from official.mnist import dataset as mnist_dataset
    test_ds = mnist_dataset.test(self.params.DATA_BASEDIR)
    if self.params.LIMIT >= 0:
      test_ds = test_ds.take(self.params.LIMIT)
    test_ds = test_ds.batch(self.params.BATCH_SIZE)
    # sess = tf.get_default_session() or tf.Session()



    
    for (images, labels) in test_ds:
      yield self.tf_model(images, training=False)

    # iterator = test_ds.make_one_shot_iterator()
    # #sess.run(iterator.initializer)
    # while True:
    #   try:
    #     res = sess.run(self.tf_model)
    #     yield res
    #   except tf.errors.OutOfRangeError:
    #     break

def setup_caches():
  MNistEager.load_or_train()
  MNistEager.save_datasets_as_png()

if __name__ == '__main__':
  setup_caches()
