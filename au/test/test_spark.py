from au import util
from au.spark import NumpyArray
from au.test import testconf
from au.test import testutils

import os

import pytest

@pytest.mark.slow
def test_spark_selftest():
  testutils.LocalSpark.selftest()

@pytest.mark.slow
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

@pytest.mark.slow
def test_spark_archive_zip():
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'test_spark_archive_zip')
  util.cleandir(TEST_TEMPDIR)
  
  # Create the fixture
  ss = ['foo', 'bar', 'baz']
  
  fixture_path = os.path.join(TEST_TEMPDIR, 'test.zip')
  
  import zipfile
  with zipfile.ZipFile(fixture_path, mode='w') as z:
    for s in ss:
      z.writestr(s, s)
  
  with testutils.LocalSpark.sess() as spark:
    rdd = testutils.LocalSpark.archive_rdd(spark, fixture_path)
    name_data = rdd.map(lambda entry: (entry.name, entry.data)).collect()
    assert sorted(name_data) == sorted((s, s) for s in ss)

# TODO test run_callables