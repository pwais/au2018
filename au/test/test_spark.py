from au import util
from au.spark import NumpyArray
from au.test import testconf
from au.test import testutils

import os

import pytest

def test_spark():
  with testutils.LocalSpark.sess() as spark:
    testutils.LocalSpark.test_pi(spark)

def test_spark_numpy_df():
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'spark_numpy_df')
  util.mkdir(TEST_TEMPDIR)
  util.rm_rf(TEST_TEMPDIR)
  

  import numpy as np
  rows = [
    {
      'a': np.array([1]), 
      'b': np.array( [ [1] ] ),
      'c': np.array([[[1]], [[2]], [[3]]]),
    },
    {
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
    # df.show()
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
    import warnings
    with warnings.catch_warnings():
      # These DeprecationWarning things are obnoxious af
      warnings.filterwarnings("ignore", category=DeprecationWarning)
      assert pprint.pformat(sorted(rows)) == pprint.pformat(sorted(decoded_rows))

