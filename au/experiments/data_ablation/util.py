from au import conf
from au import util

import os

def experiment_basedir(exp_family_name):
  return os.path.join(
    conf.AU_EXPERIMENTS_DIR,
    'data_ablation',
    exp_family_name)

def params_to_tf_summary_df(spark, params):
  df = None
  
  reader = util.TFSummaryReader(glob_events_from_dir=params.MODEL_BASEDIR)
  rows = [r.as_row() for r in reader]
  if rows:
    df = spark.createDataFrame(rows)

  from pyspark.sql import Row
  consts = util.as_row_of_constants(params)
  
  # TODO: make array with None and empty dict Spark-compatible
  consts.pop('INPUT_TENSOR_SHAPE')
  consts.pop('TRAIN_TABLE_TARGET_DISTRIBUTION')
  params_df = spark.createDataFrame([Row(**consts)])
  if df is None:
    df = params_df
  else:
    # We can't just add literal columns using pyspark `lit` function
    # because that function does type inference that doesn't match
    # `createDataFrame` (in particular, `lit` uses Int but
    # `createDataFrame` uses Long) :S
    df = df.crossJoin(params_df)
    
    
  # else:
  #   for k, v in consts.iteritems():
  #     df = df.withColumn(k, lit(v))
  df.printSchema()
  df.show()
  return df
