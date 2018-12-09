from au import util

try:
    util.Spark._setup()
    import pyspark
except Exception as e:
    raise Exception("This module requires pyspark. %s" % (e,))

## Spark UDTs
# These utils are based upon Spark's DenseVector:
# https://github.com/apache/spark/blob/044b33b2ed2d423d798f2a632fab110c46f41567/python/pyspark/mllib/linalg/__init__.py#L239
# https://apache.googlesource.com/spark/+/refs/heads/master/python/pyspark/sql/tests.py#119
# Sadly they don't have a UDT for tensors... not even in Tensorframes
# https://github.com/databricks/tensorframes   o_O

from pyspark.sql import types

import numpy as np

import io

class NumpyArrayUDT(types.UserDefinedType):
  """SQL User-Defined Type (UDT) for *opaque* numpy arrays.  Unlike Spark's
  DenseVector, this class preserves array shape.
  """

  @classmethod
  def sqlType(cls):
    # NB: this is actually an instance method in practice O_O !
    return types.StructType([
      StructField("np_bytes", types.BinaryType(), False)
    ])

  @classmethod
  def module(cls):
    return 'au.sparkutil'

  def serialize(self, a):
    return [a.get_bytes()]

  def deserialize(self, datum):
    return NumpyArray.from_bytes(datum[0])

  def simpleString(self):
    return "numpy.arr"

class NumpyArray(object):
  __slots__ = ('arr',)
  
  __UDT__ = NumpyArrayUDT

  def __init__(self, arr):
    self.arr = arr

  def get_bytes(self):
    buf = io.BytesIO()
    np.save(buf, a)
      # NB: do NOT use savez / gzip b/c we'll let Snappy compress things.
    return buf.get_value()

  @staticmethod
  def from_bytes(b):
    buf.io.BytesIO(b)
    return NumpyArray(np.load(buf))

  def __repr__(self):
    return "NumpyArray:" + self.arr.__repr__()
  
  def __str__(self):
    return "NumpyArray" + self.arr.__str__()

  def __eq__(self, other):
    return isinstance(other, self.__class__) and other.arr == self.arr

