from au import util
from au.spark import NumpyArray
from au.spark import spark_df_to_tf_dataset
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


@pytest.mark.slow
def test_spark_df_to_tf_dataset():
  with testutils.LocalSpark.sess() as spark:

    import numpy as np
    import tensorflow as tf
    from pyspark.sql import Row

    def tf_dataset_to_list(ds):
      with util.tf_data_session(ds) as (sess, iter_dataset):
        return list(iter_dataset())

    df = spark.createDataFrame([
      Row(id='r1', x=1, y=[3., 4., 5.]),
      Row(id='r2', x=2, y=[6.]),
      Row(id='r3', x=3, y=[7., 8., 9.]),
    ])

    # Test empty
    ds = spark_df_to_tf_dataset(
            df.filter('x == False'), # Empty!
            spark_row_to_tf_element=lambda r: ('test',),
            tf_element_types=(tf.string,))
    assert tf_dataset_to_list(ds) == []

    # Test simple
    ds = spark_df_to_tf_dataset(
            df,
            spark_row_to_tf_element=lambda r: (r.x,),
            tf_element_types=(tf.int64,))
    assert sorted(tf_dataset_to_list(ds)) == [(1,), (2,), (3,)]

    # Test Complex
    ds = spark_df_to_tf_dataset(
            df,
            spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
            tf_element_types=(tf.int64, tf.string, tf.float64))
    expected = [
      (1, 'r1', np.array([3., 4., 5.])),
      (2, 'r2', np.array([6.])),
      (3, 'r3', np.array([7., 8., 9.])),
    ]
    items = zip(sorted(tf_dataset_to_list(ds)), sorted(expected))
    for actual, exp in items:
      assert len(actual) == len(exp)
      for i in range(len(actual)):
        np.testing.assert_array_equal(actual[i], exp[i])

# TODO test run_callables
