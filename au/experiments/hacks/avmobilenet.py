
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
    N_CLASSES = len(AV_OBJ_CLASS_NAME_TO_ID)

    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    BKG_CLASS_IDX = AV_OBJ_CLASS_NAME_TO_ID['background']
    # class_weights[BKG_CLASS_IDX] *= .1
    class_weights /= class_weights.sum()
    CWEIGHTS = tf.gather(class_weights, labels)

    features.set_shape(FEATURES_SHAPE) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels.set_shape([None,])

    obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)

    # Mobilenet requires normalization
    features = tf.cast(features, tf.float32) / 128. - 1

    features = tf.debugging.check_numerics(features, 'features_nan')

    with tf.name_scope('input'):
      util.tf_variable_summaries(tf.cast(labels, tf.float32), prefix='labels')
      util.tf_variable_summaries(features, prefix='features')

    ### Set up Mobilenet
    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    scope = mobilenet_v2.training_scope(is_training=is_training)
    with tf.contrib.slim.arg_scope(scope):
      # image (?, 170, 170, 3) -> embedding (?, 6, 6, 1280)
      _, endpoints = mobilenet_v2.mobilenet(
                              features,
                              num_classes=N_CLASSES,
                              base_only=True,
                              is_training=is_training,
                              depth_multiplier=self.params.DEPTH_MULTIPLIER,
                              finegrain_classification_mode=self.params.FINE)
      
      embedding = endpoints['layer_19']
    
    ### Set up the VAE model
    Z_D = 32
    ENCODER_LAYERS = [1000, 2 * Z_D]
    DECODER_LAYERS = [2 * Z_D, 1000]
    # LATENT_LOSS_WEIGHT = 50.
    # VAE_LOSS_WEIGHT = 2.
    # EPS = 1e-10

    ### VAE Input
    # Mobilenet uses tf.nn.relu6
    x = embedding / 3 - 1
    util.tf_variable_summaries(x)

    ## Encode
    ## x -> z = N(z_mu, z_sigma)
    l = tf.keras.layers
    encode = tf.keras.Sequential([
      l.Dense(n_hidden, activation=tf.nn.relu)
      for n_hidden in ENCODER_LAYERS
    ])
    x_flat = tf.contrib.layers.flatten(x)
    encoded = encode(x_flat, training=is_training)
    
    z_mu_layer = l.Dense(Z_D)
    z_mu = z_mu_layer(encoded)
    
    z_log_sigma_sq_layer = l.Dense(Z_D)
    z_log_sigma_sq = z_log_sigma_sq_layer(encoded)

    util.tf_variable_summaries(z_mu)
    util.tf_variable_summaries(z_log_sigma_sq)

    noise = tf.keras.backend.random_normal(
              shape=tf.shape(z_log_sigma_sq),
              mean=0,
              stddev=1,
              dtype=tf.float32)
    z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * noise
    util.tf_variable_summaries(z)

    ## Latent Loss: KL divergence between Z and N(0, 1)
    # latent_loss = tf.reduce_mean(
    #   -0.5 * tf.reduce_sum(
    #     1. + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),
    #     axis=1))
    latent_loss = -0.5 * tf.reduce_sum(
              1. + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq))
    tf.summary.scalar('latent_loss', latent_loss)

    ## Decode
    ## z -> y
    decode = tf.keras.Sequential([
      l.Dense(n_hidden, activation=tf.nn.relu)
      for n_hidden in DECODER_LAYERS
    ])
    y = decode(z, training=is_training)
    util.tf_variable_summaries(y)
    
    ## Class Prediction Head
    ## y -> class'; loss(class, class')
    # TODO: consider dedicating latent vars to this head as in
    # http://people.csail.mit.edu/rosman/papers/iros-2018-variational.pdf
    predict_layer = l.Dense(N_CLASSES, activation=tf.nn.softmax)
    logits = predict_layer(y)
    labels_pred = tf.sigmoid(logits)

    util.tf_variable_summaries(logits)
    multiclass_loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=labels,
                        logits=logits,
                        weights=CWEIGHTS)
    tf.summary.scalar('multiclass_loss', multiclass_loss)

    preds = tf.argmax(labels_pred, axis=-1)
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('preds', preds)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
    tf.summary.scalar('accuracy', accuracy[1])

    # ## Reconstruction Head: Embedding
    # ## y -> x'; loss(x, x')
    # x_shape = list(x.shape[1:])
    # x_size = int(np.prod(x_shape))
    # x_decoder = l.Dense(x_size, activation=tf.nn.tanh) # x is in [-1, 1]
    # x_hat_flat = x_decoder(y)
    # x_hat = tf.reshape(x_hat_flat, [-1] + x_shape)
    # util.tf_variable_summaries(x_hat)

    # #recon_embed_loss = tf.losses.absolute_difference(x, x_hat)
    # recon_embed_loss = tf.losses.mean_squared_error(x, x_hat)
    # tf.summary.scalar('recon_embed_loss', recon_embed_loss)

    #   # # TODO: keras.losses.binary_crossentropy ? TODO try L1 loss?
    #   # # http://people.csail.mit.edu/rosman/papers/iros-2018-variational.pdf
    #   # loss = tf.reduce_mean(
    #   #   -tf.reduce_sum(
    #   #     y * tf.log(y_ + EPS) + (1 - y) * tf.log(1. - y_ + EPS), axis=1))
    #   # tf.debugging.check_numerics(y, 'y_nan')
    #   # tf.debugging.check_numerics(y_, 'y_target_nan')
    #   # tf.debugging.check_numerics(loss, 'loss_nan')

    ## Reconstruction Head: Image
    ## y -> image'; loss(image, image')
    image = features
    filters = [512, 256, 128, 64]
    decode_image = tf.keras.Sequential([
      l.Convolution2DTranspose(
        filters=f, kernel_size=3, activation=None,
        strides=2, padding='same')
      for f in filters
    ])
    y_expanded_shape = [-1, 10, 10, 10]
    assert np.prod(y_expanded_shape[1:]) == int(y.shape[-1])
    y_expanded = tf.reshape(y, y_expanded_shape)
      # For GANs, it seems people just randomly reshape noise into
      # some dimensions that are plausibly deconv-able.
    decoded_image_base = decode_image(y_expanded, training=is_training)
    
    # Upsample trick for perfect fit
    # https://github.com/SimonKohl/probabilistic_unet/blob/master/model/probabilistic_unet.py#L60
    image_hw = (int(image.shape[1]), int(image.shape[2]))
    image_c = int(image.shape[-1])
    upsampled = tf.image.resize_images(
                    decoded_image_base,
                    image_hw,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True)
    image_hat_layer = tf.keras.layers.Conv2D(
                          filters=image_c, kernel_size=5,
                          activation='tanh',
                            # Need to be in [-1, 1] to match image domain
                          padding='same')
    image_hat = image_hat_layer(upsampled)
    util.tf_variable_summaries(image_hat)
    tf.summary.image(
      'reconstruct_image', image, max_outputs=10, family='image')
    tf.summary.image(
      'reconstruct_image', image_hat, max_outputs=10, family='image_hat')
    # recon_image_loss = tf.losses.mean_squared_error(image, image_hat)
    recon_image_loss = tf.losses.absolute_difference(image, image_hat)
    tf.summary.scalar('recon_image_loss', recon_image_loss)
    
    ## Total Loss
    total_loss = (
      0.01 * latent_loss +
      0.30 * N_CLASSES * N_CLASSES * multiclass_loss +
      # 0.30 * int(np.prod(embedding.shape[1:])) * recon_embed_loss +
      0.30 * int(np.prod(image.shape[1:])) * recon_image_loss)
    tf.summary.scalar('total_loss', total_loss)
    
    ### Extra summaries
    # for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
    #   class_rows = tf.equal(labels, class_id)
    #   class_labels = tf.boolean_mask(labels, class_rows)
    #   class_preds = tf.boolean_mask(preds, class_rows)

    #   class_recall = tf.metrics.accuracy(
    #                       labels=tf.equal(class_labels, class_id),
    #                       predictions=tf.equal(class_preds, class_id))
    #   tf.summary.scalar('class_recall/' + class_name, class_recall)

      # tf.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
      # tf.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
      # tf.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
      # tf.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

      # Monitor Supervised
      # class_true = tf.boolean_mask(features, class_labels)
      # class_pred = tf.boolean_mask(features, class_preds)
      # tf.summary.image('supervised_labels', class_true, max_outputs=3, family=class_name)
      # tf.summary.image('supervised_pred', class_pred, max_outputs=3, family=class_name)

    #   # Monitor VAE
    #   Y_SHAPE = list(FEATURES_SHAPE)
    #   Y_SHAPE[0] = -1
    #   y_unflat = tf.reshape(y, Y_SHAPE)
    #   y_true = tf.boolean_mask(y_unflat, class_labels)
    #   y_pred = tf.boolean_mask(y_unflat, class_preds)
    #   tf.summary.image('vae_labels', y_true, max_outputs=3, family=class_name)
    #   tf.summary.image('vae_pred', y_pred, max_outputs=3, family=class_name)


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
      LEARNING_RATE = 1e-5
      
      loss = total_loss
      optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
      # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
      
      global_step = tf.train.get_or_create_global_step()
      # tf.summary.scalar('global_step', global_step)
      
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
    
    if util.missing_or_empty('/tmp/balanced_sample'):
      df = spark.read.parquet('/opt/au/cache/argoverse_cropped_object_170_170_small')
    # df = spark.read.parquet('/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/opt/au/cache/argoverse_cropped_object_170_170')
      from au.spark import get_balanced_sample
      categories = [
        "background",
        "VEHICLE",
        "PEDESTRIAN",
      ]
      df = df.filter(df.category_name.isin(categories))
      fair_df = get_balanced_sample(df, 'category_name', n_per_category=10000)
      
      # Re-shard
      import pyspark.sql.functions as F
      fair_df = fair_df.withColumn(
                'fair_shard',
                F.abs(F.hash(fair_df['uri'])) % 10)
      fair_df = fair_df.select(*list(set(fair_df.columns) - set(['shard'])))
      fair_df = fair_df.withColumn('shard', fair_df['fair_shard'])

      fair_df.write.parquet('/tmp/balanced_sample', partitionBy=['split', 'shard'])
      print("wrote to ", '/tmp/balanced_sample')
    
    df = spark.read.parquet('/tmp/balanced_sample')
    print('num images', df.count())

    def to_example(row):
      import imageio
      from io import BytesIO
      img = imageio.imread(BytesIO(row.jpeg_bytes))
      label = AV_OBJ_CLASS_NAME_TO_ID[row.category_name]
      return img, label
   
    # BATCH_SIZE = 300
    BATCH_SIZE = 50
    tdf = df.filter(df.split == 'train')
    def train_input_fn():
      train_ds = spark_df_to_tf_dataset(tdf, to_example, (tf.uint8, tf.int64), logging_name='train')
      
      train_ds = train_ds.cache()
      train_ds = train_ds.repeat(10)
      train_ds = train_ds.shuffle(1000)
      train_ds = train_ds.batch(BATCH_SIZE)
      

      # train_ds = train_ds.take(5)

      
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
