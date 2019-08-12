
from au.spark import Spark
from au.spark import spark_df_to_tf_dataset

from au import conf
from au import util

import tensorflow as tf
import numpy as np

from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
AV_OBJ_CLASS_NAME_TO_ID = dict(
  (cname, i + 1)
  for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
AV_OBJ_CLASS_NAME_TO_ID['background'] = 0






class model_fn_vae_hybrid(object):

  def __init__(self, au_params):
    self.params = au_params
  
  def __call__(self, features, labels, mode, params):

    FEATURES_SHAPE = [None, 170, 170, 3]
    features.set_shape(FEATURES_SHAPE) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels.set_shape([None,])

    obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)

    features = tf.cast(features, tf.float32) / 128. - 1

    util.tf_variable_summaries(tf.cast(labels, tf.float32), prefix='labels')
    util.tf_variable_summaries(features, prefix='features')

    ### Set up Mobilenet
    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
      # See also e.g. mobilenet_v2_035
      logits, endpoints = mobilenet_v2.mobilenet(
                            features,
                            num_classes=len(AV_OBJ_CLASS_NAME_TO_ID),
                            is_training=is_training,
                            depth_multiplier=self.params.DEPTH_MULTIPLIER,
                            finegrain_classification_mode=self.params.FINE)
      
      embedding = endpoints['layer_19']
      preds_scores = endpoints['Predictions']
    preds = tf.argmax(preds_scores, axis=1)
    with tf.name_scope('mobilenet'):
      util.tf_variable_summaries(logits)
      util.tf_variable_summaries(embedding)
      util.tf_variable_summaries(tf.cast(preds, tf.float32))
      util.tf_variable_summaries(preds_scores)
    

    ### Set up the VAE model
    ENCODER_LAYERS = [512, 256, 128, 64]
    DECODER_LAYERS = [64, 128, 256, 512]
    Z_D = 32
    LATENT_LOSS_WEIGHT = 50.
    VAE_LOSS_WEIGHT = 2.
    
    with tf.name_scope('vae'):
      x = tf.contrib.layers.flatten(embedding)
      features_flat = tf.contrib.layers.flatten(features)

      ## x -> z = N(z_mu, z_sigma)
      l = tf.keras.layers
      encode = tf.keras.Sequential([
        l.Dense(n_hidden, activation=tf.nn.relu)
        for n_hidden in ENCODER_LAYERS
      ])

      encoded = encode(x, training=is_training)
      z_mu = l.Dense(Z_D, activation=None, name='z_mu')(encoded)
      z_log_sigma_sq = l.Dense(
        Z_D, activation=None, name='z_log_sigma_sq')(encoded)
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
        for n_hidden in DECODER_LAYERS
      ])
      decoded = decode(z, training=is_training)
      y_size = features_flat.shape[-1]
      vae_logits = l.Dense(y_size, activation=None, name='logits')(decoded)
      y = tf.sigmoid(vae_logits, name='y')
      util.tf_variable_summaries(vae_logits, prefix='vae_logits')
      util.tf_variable_summaries(y, prefix='y')

      ### Set up losses
      ## Reconstruction Loss: cross-entropy of y (predicted) and y_ (target)
      y_ = features_flat
      eps = 1e-10
      recon_loss = tf.reduce_mean(
        -tf.reduce_sum(
          y * tf.log(y_ + eps) + (1. - y) * tf.log(1. - y_ + eps), axis=1))
      tf.debugging.check_numerics(y, 'y_nan')
      tf.debugging.check_numerics(y_, 'y_target_nan')
      tf.debugging.check_numerics(recon_loss, 'recon_loss_nan')
      
      # Latent Loss: KL divergence between Z and N(0, 1)
      latent_loss = tf.reduce_mean(
          -0.5 * tf.reduce_sum(
          1. + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),
          axis=1))

      total_vae_loss = recon_loss + LATENT_LOSS_WEIGHT * latent_loss

      with tf.name_scope('loss'):
        tf.summary.scalar('recon_loss', recon_loss)
        tf.summary.scalar('latent_loss', latent_loss)
        tf.summary.scalar('total_vae_loss', total_vae_loss)

        mse = tf.metrics.mean_squared_error(y_, y, name='MSE')
        tf.summary.scalar('MSE', mse[1])

    ### Set up total loss with supervised model
    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    BKG_CLASS_IDX = AV_OBJ_CLASS_NAME_TO_ID['background']
    class_weights[BKG_CLASS_IDX] *= .1
    class_weights /= class_weights.sum()

    weights = tf.gather(class_weights, labels)

    # Get supervised loss and combine
    with tf.name_scope('supervised'):
      supervised_loss = tf.losses.sparse_softmax_cross_entropy(
                            labels=labels,
                            logits=logits,
                            weights=weights)
      tf.summary.scalar('supervised_loss', supervised_loss)
    
    total_loss = supervised_loss + VAE_LOSS_WEIGHT * total_vae_loss
    tf.summary.scalar('total_loss', total_loss)

  

    ### Extra summaries
    accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
    tf.summary.scalar('accuracy', accuracy[1])
    for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
      class_labels = tf.cast(tf.equal(labels, class_id), tf.int64)
      class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

      tf.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
      tf.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
      tf.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
      tf.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

      # Monitor Supervised
      class_true = tf.boolean_mask(features, class_labels)
      class_pred = tf.boolean_mask(features, class_preds)
      tf.summary.image('supervised_labels', class_true, max_outputs=3, family=class_name)
      tf.summary.image('supervised_pred', class_pred, max_outputs=3, family=class_name)

      # Monitor VAE
      Y_SHAPE = list(FEATURES_SHAPE)
      Y_SHAPE[0] = -1
      y_unflat = tf.reshape(y, Y_SHAPE)
      y_true = tf.boolean_mask(y_unflat, class_labels)
      y_pred = tf.boolean_mask(y_unflat, class_preds)
      tf.summary.image('vae_labels', y_true, max_outputs=3, family=class_name)
      tf.summary.image('vae_pred', y_pred, max_outputs=3, family=class_name)


    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'classes': preds, #tf.argmax(logits, axis=1),
          'probabilities': tf.nn.softmax(logits),
      }
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-4
      
      loss = total_loss
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
      
      global_step = tf.train.get_or_create_global_step()
      tf.summary.scalar('global_step', global_step)
      
      train_op = optimizer.minimize(loss, global_step)
      
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }
      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(labels, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('preds_support_dist/' + class_name, class_preds)

        metric_kwargs = {
          'labels': class_labels,
          'predictions': class_preds,
        }
        name_to_ftor = {
          'accuracy': tf.metrics.accuracy,
          'auc': tf.metrics.auc,
          'precision': tf.metrics.precision,
          'recall': tf.metrics.recall,
        }
        for name, ftor in name_to_ftor.items():
          full_name = name + '/' + class_name
          eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)

      # Sadly we need to force this tensor to be computed in eval
      logging_hook = tf.train.LoggingTensorHook(
        {"obs_str_tensor" : obs_str_tensor}, every_n_iter=1)
      
      # Sadly we need to do this explicitly for eval
      summary_hook = tf.train.SummarySaverHook(
            save_secs=3,
            output_dir='/tmp/av_mobilenet_test/eval',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      hooks = [logging_hook, summary_hook]

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=hooks)


























class model_fn_simple_ff(object):

  def __init__(self, au_params):
    self.params = au_params
  
  def __call__(self, features, labels, mode, params):

    features.set_shape([None, 170, 170, 3]) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels.set_shape([None,])

    obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)

    features = tf.cast(features, tf.float32) / 128. - 1

    tf.summary.histogram('labels', labels)
    tf.summary.histogram('features', features)

    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
      # See also e.g. mobilenet_v2_035
      logits, endpoints = mobilenet_v2.mobilenet(
                            features,
                            num_classes=len(AV_OBJ_CLASS_NAME_TO_ID),
                            is_training=is_training,
                            depth_multiplier=self.params.DEPTH_MULTIPLIER,
                            finegrain_classification_mode=self.params.FINE)
    
    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    class_weights[0] *= .1
    class_weights /= class_weights.sum()

    weights = tf.gather(class_weights, labels)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
    preds_scores = endpoints['Predictions']
    preds = tf.argmax(preds_scores, axis=1)

    for k, v in endpoints.items():
      KS = ('layer_19', 'Predictions', 'Logits')
      if any(kval in k for kval in KS):
        tf.summary.histogram('mobilenet_' + k, v)

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'classes': preds, #tf.argmax(logits, axis=1),
          'probabilities': tf.nn.softmax(logits),
      }
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-4
      
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

      # If we are running multi-GPU, we need to wrap the optimizer.
      #if params_dict.get('multi_gpu'):
      # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

      
      accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)

      # Name tensors to be logged with LoggingTensorHook.
      tf.identity(LEARNING_RATE, 'learning_rate')
      tf.identity(loss, 'cross_entropy')
      tf.identity(accuracy[1], name='train_accuracy')

      # Save accuracy scalar to Tensorboard output.
      tf.summary.scalar('train_accuracy', accuracy[1])
      
      global_step = tf.train.get_or_create_global_step()
      tf.summary.scalar('global_step', global_step)

      tf.summary.histogram('train_loss', loss)
      tf.summary.histogram('logits', logits)

      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(labels, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

        class_true = tf.boolean_mask(features, class_labels)
        class_pred = tf.boolean_mask(features, class_preds)
        tf.summary.image('train_true/' + class_name, class_true, max_outputs=10)
        tf.summary.image('train_pred/' + class_name, class_pred, max_outputs=10)

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=optimizer.minimize(loss, global_step))

    elif mode == tf.estimator.ModeKeys.EVAL:
      
      # preds = tf.argmax(logits, axis=1)
      classes = labels

      tf.summary.histogram('eval_preds', preds)
      tf.summary.histogram('logits', logits)

      accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
      tf.summary.scalar('eval_accuracy', accuracy[1])
      tf.summary.scalar('eval_loss', loss)

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }
      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(classes, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('preds_support_dist/' + class_name, class_preds)

        metric_kwargs = {
          'labels': class_labels,
          'predictions': class_preds,
        }
        name_to_ftor = {
          'accuracy': tf.metrics.accuracy,
          'auc': tf.metrics.auc,
          'precision': tf.metrics.precision,
          'recall': tf.metrics.recall,
        }
        for name, ftor in name_to_ftor.items():
          full_name = name + '/' + class_name
          eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)


      logging_hook = tf.train.LoggingTensorHook(
        {"obs_str_tensor" : obs_str_tensor}, every_n_iter=1)
      summary_hook = tf.train.SummarySaverHook(
            save_secs=3,
            output_dir='/tmp/av_mobilenet_test/eval',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      hooks = [logging_hook, summary_hook]

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=hooks)

def main():

  tf_config = util.tf_create_session_config()
  # tf_config = util.tf_cpu_session_config()

  model_dir = '/tmp/av_mobilenet_test'
  util.mkdir(model_dir)

  #import os
  #import json
  #os.environ['TF_CONFIG'] = json.dumps({
  #  "cluster": {
  #    "worker": ["localhost:1122", "localhost:1123"],
#	  "chief": ["localhost:1120"],
#    },
#    'task': {'type': 'worker', 'index': 0}
#  })

  config = tf.estimator.RunConfig(
             model_dir=model_dir,
             save_summary_steps=10,
             save_checkpoints_secs=10,
             session_config=tf_config,
             log_step_count_steps=10)
  gpu_dist = tf.contrib.distribute.MirroredStrategy()
  # cpu_dist = tf.contrib.distribute.MirroredStrategy(devices=['/cpu:0'])
  
  config = config.replace(train_distribute=gpu_dist)#, eval_distribute=gpu_dist)#cpu_dist)

  from au.fixtures.tf.mobilenet import Mobilenet
  params = Mobilenet.Medium()
  av_classifier = tf.estimator.Estimator(
    # model_fn=model_fn_simple_ff(params),
    model_fn=model_fn_vae_hybrid(params),
    params=None,
    config=config)

  with Spark.getOrCreate() as spark:
    df = spark.read.parquet('/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/opt/au/cache/argoverse_cropped_object_170_170')
    print('num images', df.count())
    #df = df.cache()

    def to_example(row):
      import imageio
      from io import BytesIO
      img = imageio.imread(BytesIO(row.jpeg_bytes))
      label = AV_OBJ_CLASS_NAME_TO_ID[row.category_name]
      return img, label
   
    # BATCH_SIZE = 300
    BATCH_SIZE = 200
    tdf = df.filter(df.split == 'train')#spark.createDataFrame(df.filter(df.split == 'train').take(3000))
    def train_input_fn():
      train_ds = spark_df_to_tf_dataset(tdf, to_example, (tf.uint8, tf.int64), logging_name='train')
      #train_ds = train_ds.cache()
      #train_ds = train_ds.repeat(3)
      train_ds = train_ds.batch(BATCH_SIZE)

      # train_ds = train_ds.take(5)

      train_ds = train_ds.shuffle(400)
      # train_ds = add_stats(train_ds)
      return train_ds

    edf = df.filter(df.split == 'val')#spark.createDataFrame(df.filter(df.split == 'val').take(3000))
    def eval_input_fn():
      eval_ds = spark_df_to_tf_dataset(edf, to_example, (tf.uint8, tf.int64), logging_name='test')
      eval_ds = eval_ds.batch(BATCH_SIZE)

      eval_ds = eval_ds.take(100)

      # eval_ds = add_stats(eval_ds)
      return eval_ds

    # # Set up hook that outputs training logs every 100 steps.
    # from official.utils.logs import hooks_helper
    # train_hooks = hooks_helper.get_train_hooks(
    #    ['ExamplesPerSecondHook',
    #    'LoggingTensorHook'],
    #    model_dir=model_dir,
    #    batch_size=BATCH_SIZE)#params.BATCH_SIZE)

    TRAIN_EPOCHS = 100
    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=TRAIN_EPOCHS)
    #eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=120)
    #tf.estimator.train_and_evaluate(av_classifier, train_spec, eval_spec)


    ## Train and evaluate model.
    #TRAIN_EPOCHS = 1000

    is_training = [True]
    def run_eval():
      import time
      time.sleep(120)
      while is_training[0]:
        util.log.info("Running eval ...")
        eval_config = config.replace(session_config=util.tf_cpu_session_config())
        eval_av_classifier = tf.estimator.Estimator(
                                model_fn=model_fn(params),
                                params=None,
                                config=eval_config)
        eval_results = eval_av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
        util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    
    import threading
    eval_thread = threading.Thread(target=run_eval)
    # eval_thread.start()

    for t in range(TRAIN_EPOCHS):
      av_classifier.train(input_fn=train_input_fn)
      
      #, hooks=train_hooks)
      # if t % 10 == 0 or t >= TRAIN_EPOCHS - 1:
      # summary_hook = tf.train.SummarySaverHook(
      #       save_secs=3,
      #       output_dir='/tmp/av_mobilenet_test/eval',
      #       scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      # eval_results = av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
      # util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    is_training[0] = False
    # eval_thread.join()


if __name__ == '__main__':
  main()
