from au import util
from au.fixtures import nnmodel

import tensorflow as tf

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
      self.TRAIN_EPOCHS = 2
      self.LIMIT = -1
      # self.INPUT_TENSOR_SHAPE = [
      #             None, MNIST_INPUT_SIZE[0], MNIST_INPUT_SIZE[1], 1]

      self.TRAIN_TABLE = MNISTTrainDataset
      self.TEST_TABLE = MNISTTestDataset

      self.update(**overrides)

  def _create_model_fn(self):
    def model_fn(features, labels, mode, params):
      training = (model == tf.estimator.ModeKeys.TRAIN)

      ### Set up the model
      ## x -> z = N(z_mu, z_sigma)
      l = tf.keras.layers
      encode = tf.keras.Sequential([
        l.Dense(n_hidden, activation=tf.nn.relu)
        for n_hidden in self.params.ENCODER_LAYERS
      ])

      encoded = encode(features, training=training)
      z_mu = l.Dense(self.Z_D, activation=None, name='z_mu')(encoded)
      z_log_sigma_sq = l.Dense(
        self.Z_D, activation=None, name='z_log_sigma_sq')(encoded)
      util.tf_variable_summaries(z_mu)
      util.tf_variable_summaries(z_log_sigma_sq)

      noise = tf.keras.backend.random_normal(
                shape=tf.shape(z_log_sigma_sq),
                mean=0,
                stddev=1,
                dtype=tf.float32)
      z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * noise
      util.tf_variable_summaries(z)

      ## z -> y
      decode = tf.keras.Sequential([
        l.Dense(n_hidden, activation=tf.nn.relu)
        for n_hidden in self.params.DECODER_LAYERS
      ])
      decoded = decode(z, training=training)
      y_size = tf.shape(labels)[-1]
      logits = l.Dense(y_size, activation=None, name='logits')(decoded)
      y = tf.sigmoid(logits, name='y')
      util.tf_variable_summaries(logits)
      util.tf_variable_summaries(y)

      ### Set up losses
      ## Reconstruction Loss: cross-entropy of x and y
      recon_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='recon_loss')
      
      # Latent Loss: KL divergence between Z and N(0, 1)
      latent_loss = tf.reduce_mean(
          -0.5 * tf.reduce_sum(
          1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),
          axis=1))

      total_loss = recon_loss + self.LATENT_LOSS_WEIGHT * latent_loss
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

        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train_op = optimizer.minimize()

        return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
              'MSE': mse,
            })
    return model_fn

        
        
      