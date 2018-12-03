from au import util

class LocalSpark(util.Spark):
  MASTER = 'local[8]'
