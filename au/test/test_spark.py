from au import util
from au.spark import NumpyArray
from au.test import testconf
from au.test import testutils

import os

import pytest

def test_spark():
  with testutils.LocalSpark.sess() as spark:
    testutils.LocalSpark.test_pi(spark)

def test_spark_ships_local_src_in_egg(monkeypatch):
  def foo(_):
    # Normally, pytest puts the local source tree on the PYTHONPATH.  That
    # setting gets inherited when Spark forks a python subprocess to run
    # this function.  Remove the source tree from the PYTHONPATH here
    # in order to force pyspark to read from the egg file / SparkFiles.
    import sys
    if '/opt/au' in sys.path:
      sys.path.remove('/opt/au')
    if '' in sys.path:
      sys.path.remove('')
    
    from au import util
    s = util.ichunked([1, 2, 3], 3)
    assert list(s) == [(1, 2, 3)]
    
    return util.get_sys_info()
  
  with testutils.LocalSpark.sess() as spark:
    sc = spark.sparkContext
    N = 10
    rdd = sc.parallelize(range(N))
    res = rdd.map(foo).collect()
    assert len(res) == N
    paths = [info['filepath'] for info in res]
    assert all('au_spark_temp-0.0.0-py2.7.egg' in p for p in paths)

# def test_spark_au_tensorflow():
#   def foo(_):
#     from au import util
#     sess = util.tf_create_session_config()

def test_spark_numpy_df():
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'spark_numpy_df')
  util.cleandir(TEST_TEMPDIR)
  

  import numpy as np
  rows = [
    {
      'id': 1,
      'a': np.array([1]), 
      'b': np.array( [ [1] ] ),
      'c': np.array([[[1]], [[2]], [[3]]]),
    },
    {
      'id': 2,
      'a': np.array([]),
      'b': None,
      'c': None,
    },
  ]

  # Test serialization numpy <-> parquet
  with testutils.LocalSpark.sess() as spark:
    from pyspark.sql import Row

    wrapped_rows = [
      Row(**dict((k, NumpyArray(v)) for k, v in row.iteritems()))
      for row in rows
    ]

    df = spark.createDataFrame(wrapped_rows)
    df.show()
    outpath = os.path.join(TEST_TEMPDIR, 'rowdata')
    df.write.parquet(outpath)

    df2 = spark.read.parquet(outpath)
    decoded_wrapped_rows = df2.collect()
    
    decoded_rows = [
      dict((k, v.arr if v else v) for k, v in row.asDict().iteritems())
      for row in decoded_wrapped_rows
    ]
    
    # We can't do assert sorted(rows) == sorted(decoded_rows)
    # because numpy syntatic sugar breaks ==
    import pprint
    def sorted_row_str(rowz):
      return pprint.pformat(sorted(rowz, key=lambda row: row['id']))
    assert sorted_row_str(rows) == sorted_row_str(decoded_rows)

