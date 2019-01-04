from au import util
from au.spark import NumpyArray
from au.test import testconf
from au.test import testutils

import os

import pytest

@pytest.mark.slow
def test_spark():
  with testutils.LocalSpark.sess() as spark:
    testutils.LocalSpark.test_pi(spark)

@pytest.mark.slow
def test_spark_ships_local_src_in_egg(monkeypatch):
  EXPECTED_EGG_NAME = 'au-0.0.0-py2.7.egg'

  def foo(_):
    # Normally, pytest puts the local source tree on the PYTHONPATH.  That
    # setting gets inherited when Spark forks a python subprocess to run
    # this function.  Remove the source tree from the PYTHONPATH here
    # in order to force pyspark to read from the egg file / SparkFiles.
    # We may safely edit the PYTHONPATH here because this code is run in a
    # child python process that will soon exit.
    import sys
    if '/opt/au' in sys.path:
      sys.path.remove('/opt/au')
    if '' in sys.path:
      sys.path.remove('')

    ## Check for the egg, which Spark puts on the PYTHONPATH
    egg_path = ''
    for p in sys.path:
      if EXPECTED_EGG_NAME in p:
        egg_path = p
    assert egg_path, 'Egg not found in sys.path %s' % (sys.path,)

    ## Is the egg any good?
    import zipfile
    f = zipfile.ZipFile(egg_path)
    egg_contents = f.namelist()
    assert any('au' in fname for fname in egg_contents), egg_contents

    ## Use the egg!
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
    assert all(EXPECTED_EGG_NAME in p for p in paths)

@pytest.mark.slow
def test_spark_tensorflow():
  def foo(x):
    import tensorflow as tf

    a = tf.constant(x)
    b = tf.constant(3)

    from au import util
    sess = util.tf_create_session()
    res = sess.run(a * b)
    return res == 3 * x
  
  with testutils.LocalSpark.sess() as spark:
    sc = spark.sparkContext
    N = 10
    rdd = sc.parallelize(range(N))
    res = rdd.map(foo).collect()
    assert len(res) == N
    assert all(res)

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