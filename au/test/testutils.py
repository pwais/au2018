from au import spark

class LocalSpark(spark.Spark):
  MASTER = 'local[8]'
