"""

Based upon tensorflow/models official mnist.py
https://github.com/tensorflow/models/blob/dfafba4a017c21c19dfdb60e1580f0b2ff5d361f/official/mnist/mnist.py

NB: earlier we tried mnist_eager.py, and it was less code (and much
simpler code) but a major PITA to interop with anything else.  

"""

import os

import tensorflow as tf

from au import util
from au.fixtures import nnmodel

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



## Interface

class MNIST(nnmodel.INNModel):

  class Params(nnmodel.INNModel.ParamsBase):
    def __init__(self):
      super(MNIST.Params, self).__init__(model_name='MNIST')
      self.BATCH_SIZE = 100
      self.LEARNING_RATE = 0.01
      self.MOMENTUM = 0.5
      self.TRAIN_EPOCHS = 2
      self.LIMIT = -1

  def __init__(self):
    self.tf_model = None
    self.predictor = None

  @staticmethod
  def save_datasets_as_png(params=None):   
    params = params or MNIST.Params()
    
    log = util.create_log()
    
    def save_dataset(ds, tag):
      import imageio
      import numpy as np
      
      img_dir = os.path.join(params.DATA_BASEDIR, tag, 'images')
      util.mkdir(img_dir)
      path_to_label = {}
    
      i = 0
      with util.tf_data_session(ds) as (sess, iter_dataset):
        for image, label in iter_dataset():
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
    
    # Keep our dataset ops in an isolated graph
    g = tf.Graph()
    with g.as_default():
      save_dataset(mnist_dataset.train(params.DATA_BASEDIR), 'train')
      save_dataset(mnist_dataset.test(params.DATA_BASEDIR), 'test')

  @staticmethod
  def _train(params):
#     from official.mnist import dataset as mnist_dataset
#     from official.mnist import mnist
# 
#     # tf.enable_eager_execution()
    log = util.create_log()
# 
#     # Automatically determine device and data_format
#     #(device, data_format) = ('/gpu:0', 'channels_first')
#     #if not tf.test.is_gpu_available():
#     #  (device, data_format) = ('/cpu:0', 'channels_last')
#     #log.info('Using device %s, and data format %s.' % (device, data_format))
# 
#     # Load the datasets
#     train_ds = mnist_dataset.train(params.DATA_BASEDIR).shuffle(60000)
#     if params.LIMIT >= 0:
#       train_ds = train_ds.take(params.LIMIT)
#     train_ds = train_ds.batch(params.BATCH_SIZE)
# 
#     test_ds = mnist_dataset.test(params.DATA_BASEDIR)
#     if params.LIMIT >= 0:
#       test_ds = test_ds.take(params.LIMIT)
#     test_ds = test_ds.batch(params.BATCH_SIZE)
# 
#     # Create the model and optimizer
#     model = create_model('channels_last')
#     optimizer = tf.train.MomentumOptimizer(params.LEARNING_RATE, params.MOMENTUM)
# 
#     # Create directories to which summaries will be written
#     # tensorboard --logdir=<output_dir>
#     # can then be used to see the recorded summaries.
#     train_dir = os.path.join(params.TENSORBOARD_BASEDIR, 'train')
#     test_dir = os.path.join(params.TENSORBOARD_BASEDIR, 'eval')
#     tf.gfile.MakeDirs(params.TENSORBOARD_BASEDIR)
# 
#     summary_writer = tf.contrib.summary.create_file_writer(
#       train_dir, flush_millis=10000)
#     test_summary_writer = tf.contrib.summary.create_file_writer(
#       test_dir, flush_millis=10000, name='test')
# 
#     # Create and restore checkpoint (if one exists on the path)
#     tf.gfile.MakeDirs(params.MODEL_BASEDIR)
#     checkpoint_prefix = os.path.join(params.MODEL_BASEDIR, 'ckpt')
#     step_counter = tf.train.get_or_create_global_step()
#     checkpoint = tf.train.Checkpoint(
#       model=model, optimizer=optimizer, step_counter=step_counter)
#     checkpoint.restore(tf.train.latest_checkpoint(params.MODEL_BASEDIR))

    # Train and evaluate for a set number of epochs.
    log.info("Training!")
    mnist_train(params)
    
#     # with tf.device(device):
#     for _ in range(params.TRAIN_EPOCHS):
#       start = time.time()
#       with summary_writer.as_default():
#         train(model, optimizer, train_ds, step_counter, 10)
#       end = time.time()
#       log.info('\nTrain time for epoch #%d (%d total steps): %f' %
#             (checkpoint.save_counter.numpy() + 1,
#              step_counter.numpy(),
#              end - start))
#       with test_summary_writer.as_default():
#         test(model, test_ds)
#       checkpoint.save(checkpoint_prefix)
    # tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
    log.info("Done training!")

  @classmethod
  def load_or_train(cls, params=None):
    log = util.create_log()
    
    model = MNIST()
    params = params or MNIST.Params()
    model.params = params

    if not os.path.exists(os.path.join(params.MODEL_BASEDIR, 'model.ckpt')):
      ## Train a model!
      MNIST._train(params)
#       tf.reset_default_graph()
    
    
#     tf.reset_default_graph()
    
    # Load saved model
    # 
    # sess = tf.get_default_session() or tf.Session()
    model.tf_model = create_model()
    # saver = tf.train.Saver()
    # saver.restore(sess, 
    root = tf.train.Checkpoint(model=model.tf_model)
    root.restore(tf.train.latest_checkpoint(params.MODEL_BASEDIR))

    # model.predictor = predictor.from_saved_model(params.MODEL_BASEDIR)
#     print(model.graph)
    import pprint
    log.info("Loaded graph:")
    log.info(pprint.pformat(tf.contrib.graph_editor.get_tensors(tf.get_default_graph())))
    return model

  def iter_activations(self):

    config = util.tf_create_session_config()
    with tf.train.MonitoredTrainingSession(config=config) as sess:
      
      # Sometimes saved model / saved graphs are finalized ...
      sess.graph._unsafe_unfinalize()
      
#       tf.keras.summary()
      
#       print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
      
      dataset = test_dataset(self.params)
      iterator = dataset.make_one_shot_iterator()
      images, labels = iterator.get_next()
      pred = self.tf_model((images, labels), training=False)
      
      TENSOR_NAMES = (
       'sequential/conv2d/Relu:0',
       'sequential/conv2d_1/Relu:0',
       'sequential/dense/Relu:0',
       'sequential/dense_1/MatMul:0',
      )
      tensors = [sess.graph.get_tensor_by_name(n) for n in TENSOR_NAMES]
      
      args = [pred] + tensors
      
      while not sess.should_stop():
        res = sess.run(args)
        name_to_val = zip(['predictions'] + list(TENSOR_NAMES), res)
        yield dict(name_to_val) 
#         yield {
#           'pred': ,#args),
# #           'conv1/Relu:0': tf.keras.activations.get('conv1/Relu:0'),
# #           'conv2/Relu:0': tf.keras.activations.get('conv2/Relu:0'),
# #           'fc1/Relu:0': tf.keras.activations.get('fc1/Relu:0'),
# #           'fc2/Relu:0': tf.keras.activations.get('fc2/Relu:0'),
#         }
#     
#     
#     with util.tf_data_session(ds) as (sess, iter_dataset):
#       for images, labels in iter_dataset():
#         yield self.tf_model(images, training=False)
    
#     
#     from official.mnist import dataset as mnist_dataset
#     test_ds = mnist_dataset.test(self.params.DATA_BASEDIR)
#     if self.params.LIMIT >= 0:
#       test_ds = test_ds.take(self.params.LIMIT)
#     test_ds = test_ds.batch(self.params.BATCH_SIZE)
#     # sess = tf.get_default_session() or tf.Session()
# 
# 
# 
#     import pdb; pdb.set_trace()
#     for (images, labels) in test_ds:
#       yield self.tf_model(images, training=False)

    # iterator = test_ds.make_one_shot_iterator()
    # #sess.run(iterator.initializer)
    # while True:
    #   try:
    #     res = sess.run(self.tf_model)
    #     yield res
    #   except tf.errors.OutOfRangeError:
    #     break

def setup_caches():
  MNIST.load_or_train()
  MNIST.save_datasets_as_png()

if __name__ == '__main__':
  setup_caches()
