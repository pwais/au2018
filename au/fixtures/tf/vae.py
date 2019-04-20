from au import util
from au.fixtures import alm
from au.fixtures import nnmodel
from au.spark import Spark

import os

import tensorflow as tf

# TODO: moveme

from au.fixtures.tf import mnist
mnist_params = mnist.MNIST.Params(
                  BATCH_SIZE=10000,
                  TRAIN_EPOCHS=30,
                  TRAIN_WORKER_CLS=util.WholeMachineWorker)
class MNISTTrainActivations(nnmodel.ActivationsTable):
  SPLIT = 'train'
  TABLE_NAME = 'mnist_train_activations'
  NNMODEL_CLS = mnist.MNIST
  MODEL_PARAMS = mnist_params
  IMAGE_TABLE_CLS = mnist_params.TRAIN_TABLE

class MNISTTestActivations(nnmodel.ActivationsTable):
  SPLIT = 'test'
  TABLE_NAME = 'mnist_test_activations'
  NNMODEL_CLS = mnist.MNIST
  MODEL_PARAMS = mnist_params
  IMAGE_TABLE_CLS = mnist_params.TEST_TABLE

class MNISTTrainActivationsDataset(alm.ActivationsDataset):
  ACTIVATIONS_TABLE = MNISTTrainActivations

class MNISTTestActivationsDataset(alm.ActivationsDataset):
  ACTIVATIONS_TABLE = MNISTTestActivations



class SimpleFCVAE(nnmodel.INNGenerativeModel):

  class Params(nnmodel.INNModel.ParamsBase):
    def __init__(self, **overrides):
      super(SimpleFCVAE.Params, self).__init__(model_name='SimpleFCVAE')
      self.BATCH_SIZE = 1000
      self.EVAL_BATCH_SIZE = 1000
      self.INFERENCE_BATCH_SIZE = 1000
      
      self.ENCODER_LAYERS = [256, 128, 64]
      self.Z_D = 2
      self.DECODER_LAYERS = [64, 128, 256]
      self.LATENT_LOSS_WEIGHT = 50.

      self.LEARNING_RATE = 1e-4
      self.TRAIN_EPOCHS = 50
      self.LIMIT = 10
      self.MULTI_GPU = True
      # self.INPUT_TENSOR_SHAPE = [
      #             None, MNIST_INPUT_SIZE[0], MNIST_INPUT_SIZE[1], 1]

      self.TRAIN_DATASET = MNISTTrainActivationsDataset
      self.TEST_DATASET = MNISTTestActivationsDataset

      self.update(**overrides)


  @classmethod
  def create_model_fn(cls, au_params):
    def model_fn(features, labels, mode, params):
      training = (mode == tf.estimator.ModeKeys.TRAIN)

      x = features

      ### Set up the model
      ## x -> z = N(z_mu, z_sigma)
      l = tf.keras.layers
      encode = tf.keras.Sequential([
        l.Dense(n_hidden, activation=tf.nn.relu)
        for n_hidden in au_params.ENCODER_LAYERS
      ])

      encoded = encode(x, training=training)
      z_mu = l.Dense(au_params.Z_D, activation=None, name='z_mu')(encoded)
      z_log_sigma_sq = l.Dense(
        au_params.Z_D, activation=None, name='z_log_sigma_sq')(encoded)
      util.tf_variable_summaries(z_mu)
      util.tf_variable_summaries(z_log_sigma_sq)

      noise = tf.keras.backend.random_normal(
                shape=tf.shape(z_log_sigma_sq),
                mean=0,
                stddev=1,
                dtype=tf.float32)
      z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * noise
      util.tf_variable_summaries(z, prefix='z')

      ## z -> y
      decode = tf.keras.Sequential([
        l.Dense(n_hidden, activation=tf.nn.relu)
        for n_hidden in au_params.DECODER_LAYERS
      ])
      decoded = decode(z, training=training)
      y_size = labels.shape[-1]
      logits = l.Dense(y_size, activation=None, name='logits')(decoded)
      y = tf.sigmoid(logits, name='y')
      util.tf_variable_summaries(logits)
      util.tf_variable_summaries(y)

      ### Set up losses
      ## Reconstruction Loss: cross-entropy of x and y
      # recon_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
      #   labels=labels, logits=logits, name='recon_loss'))
      y_ = labels
      eps = 1e-10
      recon_loss = tf.reduce_mean(
        -tf.reduce_sum(
          y * tf.log(y_ + eps) + (1 - y) * tf.log(1 - y_ + eps), axis=1))
      
      # Latent Loss: KL divergence between Z and N(0, 1)
      latent_loss = tf.reduce_mean(
          -0.5 * tf.reduce_sum(
          1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),
          axis=1))

      total_loss = recon_loss + au_params.LATENT_LOSS_WEIGHT * latent_loss
      total_loss = tf.identity(total_loss, name='total_loss')

      with tf.name_scope('loss'):
        tf.summary.scalar('recon_loss', recon_loss)
        tf.summary.scalar('latent_loss', latent_loss)
        tf.summary.scalar('total_loss', total_loss)

        mse = tf.metrics.mean_squared_error(labels, y, name='MSE')
        tf.summary.scalar('MSE', mse[1])
      
      ### Estimator Interface
      if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: no clue if this is right
        predictions = {
          'probabilities': y,
        }
        return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
      elif mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        tf.summary.scalar('global_step', global_step)

        LR = au_params.LEARNING_RATE
        tf.identity(LR, 'learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=LR)
        train_op = optimizer.minimize(total_loss, global_step)

        return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=total_loss,
            eval_metric_ops={
              'MSE': mse,
            })
    return model_fn



  @classmethod
  def load_or_train(cls, params=None):
    # TODO: load from checkpoint ...
    params = params or SimpleFCVAE.Params()

    spark = Spark.getOrCreate()
    params.TRAIN_DATASET.ACTIVATIONS_TABLE.setup(spark=spark)
    params.TEST_DATASET.ACTIVATIONS_TABLE.setup(spark=spark)

    model_dir = params.MODEL_BASEDIR
    tf.gfile.MakeDirs(params.MODEL_BASEDIR)

    config = tf.estimator.RunConfig(
                  model_dir=model_dir,
                  save_summary_steps=10,
                  save_checkpoints_secs=10,
                  log_step_count_steps=10)
    if params.MULTI_GPU:
      dist = tf.contrib.distribute.MirroredStrategy()
      config = config.replace(
        train_distribute=dist,
        eval_distribute=dist,
        session_config=util.tf_create_session_config())
    else:
      config = config.replace(
        session_config=util.tf_create_session_config(restrict_gpus=[]))

    model_fn = cls.create_model_fn(params)
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      params=None,
      config=config)

    def train_input_fn():
      train_ds = params.TRAIN_DATASET.as_tf_dataset(spark=spark)
      if params.LIMIT >= 0:
        train_ds = train_ds.take(params.LIMIT)

      # This flow doesn't need uri
      train_ds = train_ds.map(lambda arr, label, uri: (arr, label))
      train_ds = train_ds.batch(params.BATCH_SIZE)
      train_ds = train_ds.cache()#os.path.join(params.MODEL_BASEDIR, 'train_cache'))
      # train_ds = train_ds.prefetch(10)
      train_ds = train_ds.repeat(10000)
      return train_ds
    
    def eval_input_fn():
      test_ds = params.TEST_DATASET.as_tf_dataset(spark=spark)
      if params.LIMIT >= 0:
        test_ds = test_ds.take(params.LIMIT)

      # This flow doesn't need uri
      test_ds = test_ds.map(lambda arr, label, uri: (arr, label))
      test_ds = test_ds.batch(params.EVAL_BATCH_SIZE)
      # test_ds = test_ds.cache(os.path.join(params.MODEL_BASEDIR, 'test_cache'))
      return test_ds
    
    # Set up hook that outputs training logs
    # from official.utils.logs import hooks_helper
    # train_hooks = hooks_helper.get_train_hooks(
    #     ['ExamplesPerSecondHook', 'LoggingTensorHook'],
    #     model_dir=model_dir,
    #     batch_size=params.BATCH_SIZE)
    
    # Train and evaluate model.
    for t in range(params.TRAIN_EPOCHS):
      estimator.train(input_fn=train_input_fn)#, hooks=train_hooks)
      if t % 10 == 0 or t >= params.TRAIN_EPOCHS - 1:
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)

if __name__ == '__main__':
  SimpleFCVAE.load_or_train()


      