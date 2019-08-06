
from au.spark import Spark
from au.spark import spark_df_to_tf_dataset

from au import conf
from au import util

import tensorflow as tf

from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
AV_OBJ_CLASS_NAME_TO_ID = dict(
  (cname, i + 1)
  for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
AV_OBJ_CLASS_NAME_TO_ID['background'] = 0

class model_fn(object):

  def __init__(self, au_params):
    self.params = au_params
  
  def __call__(self, features, labels, mode, params):

    features.set_shape([None, 170, 170, 3])
    labels.set_shape([None,])

    features = tf.cast(features, tf.float32) / 128. - 1

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

    if mode == tf.estimator.ModeKeys.PREDICT:
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
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-4
      
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

      # If we are running multi-GPU, we need to wrap the optimizer.
      #if params_dict.get('multi_gpu'):
      # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

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
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      
      preds = tf.argmax(logits, axis=1)
      classes = labels

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }

      # # Added in au
      # for k in range(self.au_params.TEST_TABLE.N_CLASSES):
      #   k_name = str(k)
      #   metric_kwargs = {
      #     'labels': tf.cast(tf.equal(classes, k), tf.int64),
      #     'predictions': tf.cast(tf.equal(preds, k), tf.int64),
      #   }
      #   name_to_ftor = {
      #     'accuracy': tf.metrics.accuracy,
      #     'auc': tf.metrics.auc,
      #     'precision': tf.metrics.precision,
      #     'recall': tf.metrics.recall,
      #   }
      #   for name, ftor in name_to_ftor.items():
      #     full_name = name + '_' + k_name
      #     eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops)

def main():

  tf_config = util.tf_create_session_config()

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
  #cpu_dist = tf.contrib.distribute.MirroredStrategy(devices=['/cpu:0'])
  
  config = config.replace(train_distribute=gpu_dist, eval_distribute=gpu_dist)#cpu_dist)

  from au.fixtures.tf.mobilenet import Mobilenet
  params = Mobilenet.Medium()
  av_classifier = tf.estimator.Estimator(
    model_fn=model_fn(params),
    params=None,
    config=config)

  with Spark.getOrCreate() as spark:
    df = spark.read.parquet('/outer_root/media/seagates-ext4/au_datas/gcloud_tables/gcloud_tables/argoverse_cropped_object_170_170')#'/opt/au/cache/argoverse_cropped_object_170_170')
    print('num images', df.count())
    #df = df.cache()

    def to_example(row):
      import imageio
      from io import BytesIO
      img = imageio.imread(BytesIO(row.jpeg_bytes))
      label = AV_OBJ_CLASS_NAME_TO_ID[row.category_name]
      return img, label
    
    BATCH_SIZE = 300
    def train_input_fn():
      tdf = df.filter(df.split == 'train')
      ds = spark_df_to_tf_dataset(tdf, to_example, (tf.uint8, tf.int64))
      train_ds = ds
      #train_ds = train_ds.cache()
      #train_ds = train_ds.repeat(3)
      train_ds = train_ds.batch(BATCH_SIZE)
      train_ds = train_ds.shuffle(400)
      return train_ds

    def eval_input_fn():
      edf = df.filter(df.split == 'val')
      ds = spark_df_to_tf_dataset(df, to_example, (tf.uint8, tf.int64))
      eval_ds = ds
      eval_ds = eval_ds.batch(BATCH_SIZE)
      return eval_ds

    # Set up hook that outputs training logs every 100 steps.
    #from official.utils.logs import hooks_helper
    #train_hooks = hooks_helper.get_train_hooks(
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
    for t in range(TRAIN_EPOCHS):
      av_classifier.train(input_fn=train_input_fn)#hooks=train_hooks)
      # if t % 10 == 0 or t >= TRAIN_EPOCHS - 1:
      eval_results = av_classifier.evaluate(input_fn=eval_input_fn)
      util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)

if __name__ == '__main__':
  main()
