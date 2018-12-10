"""A module with Spark-related utilities"""

from au import util

import pickle
from contextlib import contextmanager

import numpy as np

try:
  import findspark
  findspark.init()

  import pyspark
  from pyspark.sql import types

except Exception as e:
  msg = """
      This portion of AU requires Spark, which in turn requires Java 8 or
      higher.  Mebbe try installing using:
        $ pip install pyspark
      That will fix import errors.  To get Java, try:
        $ apt-get install -y default-jdk && \
          echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
      Original error: %s
  """ % (e,)
  raise Exception(msg)


class Spark(object):
  MASTER = None
  CONF = None
  CONF_KV = None
  HIVE = False
  
  @classmethod
  def _setup(cls):
    # TODO set up egg to ship to workers ...
    pass
  
  @classmethod
  def getOrCreate(cls):
    cls._setup()

    import pyspark
    from pyspark import sql
    builder = sql.SparkSession.builder
    if cls.MASTER is not None:
      builder = builder.master(cls.MASTER)
    if cls.CONF is not None:
      builder = builder.config(conf=cls.CONF)
    if cls.CONF_KV is not None:
      for k, v in cls.CONF_KV.iteritems():
        builder = builder.config(k, v)
    if cls.HIVE:
      # TODO fixme see mebbe https://creativedata.atlassian.net/wiki/spaces/SAP/pages/82255289/Pyspark+-+Read+Write+files+from+Hive
      # builder = builder.config("hive.metastore.warehouse.dir", '/tmp') 
      # builder = builder.config("spark.sql.warehouse.dir", '/tmp')
      builder = builder.enableHiveSupport()
    return builder.getOrCreate()
  
  @classmethod
  @contextmanager
  def sess(cls):
    spark = cls.getOrCreate()
    yield spark

  @staticmethod
  def test_pi(spark):
    util.log.info("Running PI ...")
    sc = spark.sparkContext
    num_samples = 1000000
    def inside(p):
      import random
      x, y = random.random(), random.random()
      return x*x + y*y < 1
    count = sc.parallelize(range(0, num_samples)).filter(inside).count()
    pi = 4 * float(count) / num_samples
    util.log.info("Pi estimate: %s" % pi)
    assert abs(pi - 3.14) < 0.1, "Spark program had an error?"



## Spark UDTs
# These utils are based upon Spark's DenseVector:
# https://github.com/apache/spark/blob/044b33b2ed2d423d798f2a632fab110c46f41567/python/pyspark/mllib/linalg/__init__.py#L239
# https://apache.googlesource.com/spark/+/refs/heads/master/python/pyspark/sql/tests.py#119
# Sadly they don't have a UDT for tensors... not even in Tensorframes
# https://github.com/databricks/tensorframes   o_O

class NumpyArrayUDT(types.UserDefinedType):
  """SQL User-Defined Type (UDT) for *opaque* numpy arrays.  Unlike Spark's
  DenseVector, this class preserves array shape.
  """

  @classmethod
  def sqlType(cls):
    # NB: this is actually an instance method in practice O_O !
    return types.StructType([
      types.StructField("np_bytes", types.BinaryType(), False)
    ])

  @classmethod
  def module(cls):
    return 'au.spark'

  def serialize(self, a):
    return [a.get_bytes()]

  def deserialize(self, datum):
    return NumpyArray.from_bytes(datum[0])

  def simpleString(self):
    return "numpy.arr"

class NumpyArray(object):
  __slots__ = ('arr',)
  
  __UDT__ = NumpyArrayUDT()

  def __init__(self, arr):
    self.arr = arr

  def get_bytes(self):
    return pickle.dumps(self.arr)
    # buf = io.BytesIO()
    # np.save(buf, self.arr)
    #   # NB: do NOT use savez / gzip b/c we'll let Snappy compress things.
    # return buf.getvalue()

  @staticmethod
  def from_bytes(b):
    # arr = np.load(io.BytesIO(b), encoding='bytes')
    arr = pickle.loads(b)
    return NumpyArray(arr)

  def __repr__(self):
    return "NumpyArray:" + self.arr.__repr__()
  
  def __str__(self):
    return "NumpyArray" + self.arr.__str__()

  def __eq__(self, other):
    return isinstance(other, self.__class__) and other.arr == self.arr

