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

@pytest.mark.slow
def test_spark_tf_dataset():
  with testutils.LocalSpark.sess() as spark:
    rdd = spark.sparkContext.parallelize(range(int(1e5)), numSlices=10)
    from pyspark.sql import Row
    df = spark.createDataFrame(rdd.map(lambda x: Row(x=x)))

    from pyspark.sql.functions import spark_partition_id
    df = df.withColumn('_pid', spark_partition_id())
    num_partitions = df.rdd.getNumPartitions()

    import tensorflow as tf
    import multiprocessing
    import numpy as np

    def gen_rows(pid):
      # assert False, pid
      pds = tf.data.Dataset.from_tensors(pid)

      def _gen_ds(ppid):
        # assert False, (ppid, ppid.decode())
        part_df = df.filter('_pid == %s' % ppid)
        part_df.show()
        return np.array([[r.x] for r in part_df.collect()], dtype=np.int64)
        # def iter_results():
        #   for r in part_df.collect():
        #     yield (r.x,)
        # return tf.data.Dataset.from_generator(
        #           generator=iter_results,
        #           output_types=(tf.int64,),
        #           output_shapes=(tf.TensorShape([]),))
      
      return pds.map(lambda p: tuple(tf.py_func(_gen_ds, [p], [tf.int64])))

      
    
    ds = tf.data.Dataset.from_tensor_slices(range(num_partitions))
    # import pdb; pdb.set_trace()
    ds = ds.apply(
            tf.data.experimental.parallel_interleave(
              gen_rows,
              cycle_length=multiprocessing.cpu_count(),
              sloppy=True))
    
    # ds = gen_rows(0)
    with util.tf_data_session(ds) as (sess, iter_dataset):
      n = 0
      for x in iter_dataset():
        n += 1
        print x
      print n


    

    import pdb; pdb.set_trace()
    print 'moof'


# TODO test run_callables