from au import conf
from au import util

import os

def experiment_basedir(exp_family_name):
  return os.path.join(
    conf.AU_EXPERIMENTS_DIR,
    'data_ablation',
    exp_family_name)

def params_to_tf_summary_df(spark, all_params):
  all_params = list(all_params)

  rdd = spark.sparkContext.parallelize([
    (hash(p), p.MODEL_BASEDIR)
    for p in all_params
  ])
  def iter_rows(h_path):
    from pyspark.sql import Row
    h, path = h_path
    for r in util.TFSummaryReader(glob_events_from_dir=path):
      row = r.as_row()
      d = row.asDict()
      d['params_hash'] = h
      yield Row(**d)
  rdd = rdd.flatMap(iter_rows)
  df = spark.createDataFrame(rdd)

  # reader = util.TFSummaryReader(glob_events_from_dir=params.MODEL_BASEDIR)
  # rdd = spark.parallelize(reader)
  # rdd = rdd.flatMap(lambda r: r.as_row()
  # rows = [r.as_row() for r in reader]
  # if rows:
  # df = spark.createDataFrame([
  def params_to_row(p):
    from pyspark.sql import Row
    d = util.as_row_of_constants(p)
    d.pop('INPUT_TENSOR_SHAPE')
    d.pop('TRAIN_TABLE_TARGET_DISTRIBUTION')
    d['params_hash'] = hash(p)
    return Row(**d)

  df2 = spark.createDataFrame(params_to_row(p) for p in all_params)
  
  joined = df.join(df2, on='params_hash', how='inner')
    # This join is super fast because:
    #   (1) `df2` is quite small
    #   (2) Spark will maintain one partition per `params_hash` for `df`
  return joined
  # assert False, joined.show()

  # rdd2 = spark.sparkContext.parallelize([
  #   (hash(p), util.as_row_of_constants(p))
  #   for p in all_params
  # ])
  # rdd2 = rdd2.map()

  # joined = rdd.outerJoin(rdd2)
  # assert False, joined.take(1)

  # # from pyspark.sql import Row
  # consts = util.as_row_of_constants(params)
  
  # # TODO: make array with None and empty dict Spark-compatible
  # consts.pop('INPUT_TENSOR_SHAPE')
  # consts.pop('TRAIN_TABLE_TARGET_DISTRIBUTION')

  # from pyspark.sql.functions import lit
  # for k, v in consts.iteritems():
  #   df = df.withColumn(k, lit(v))

  # # params_df = spark.createDataFrame([Row(**consts)])
  # # if df is None:
  # #   df = params_df
  # # else:
  # #   # We can't just add literal columns using pyspark `lit` function
  # #   # because that function does type inference that doesn't match
  # #   # `createDataFrame` (e.g. Long vs Int) :S
  # #   df = df.crossJoin(params_df)
    
    
  # # else:
  # #   for k, v in consts.iteritems():
  # #     df = df.withColumn(k, lit(v))
  # # df.printSchema()
  # return df
