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


    # rdd = spark.sparkContext.parallelize(range(int(1e6)), numSlices=50)
    # from pyspark.sql import Row
    # df = spark.createDataFrame(rdd.map(lambda x: Row(x=x, y=[x*1, x*2, x*3])))
    # df.write.parquet('/tmp/tasttast', mode='overwrite', compression='none')
    # df = spark.read.parquet('/tmp/tasttast')
    # df = df.repartition(100)

    # from pyspark.sql.functions import spark_partition_id
    # df = df.withColumn('_pid', spark_partition_id())
    
    # num_partitions = df.rdd.getNumPartitions()
    # print 'num_partitions', num_partitions

    # import tensorflow as tf
    # import multiprocessing
    # import numpy as np


    # # def make_gen(ppid):
    # #   def p_gen():
    # #     part_df = df.filter('_pid == %s' % ppid)
    # #     part_df.show()
    # #     rows = part_df.collect()
    # #     for row in rows:
    # #       yield row.x, row.y
    # #   return p_gen
    # #   return tf.data.Dataset.from_generator(
    # #               p_gen,
    # #               output_types=(tf.int64, tf.int64))

    # # dss = []
    # # for ppid in range(num_partitions):
    # #   cur_ds = tf.data.Dataset.from_generator(
    # #               make_gen(ppid),
    # #               output_types=(tf.int64, tf.int64))
    # #   cur_ds = cur_ds.batch(50)
    # #   dss.append(cur_ds)
    # #   # if not ds:
    # #   #   ds = cur_ds
    # #   # else:
    # #   #   ds = ds.concatenate(cur_ds)

    # def gen_ds(ppid_tensor):
    #   pds = tf.data.Dataset.from_tensors(ppid_tensor)

    #   def _gen_ds(ppid):
    #     part_df = df.filter('_pid == %s' % ppid)
    #     print ppid
    #     rows = part_df.collect()
    #     def xform(r):
    #       return np.array([r.x]), np.array([r.y]), str(r.y)
        
    #     xformed = [xform(row) for row in rows]
    #     import itertools
    #     cwise = list(itertools.izip(*xformed))
    #     return cwise

    #     # return [
    #     #   (np.array([row.x]), np.array([row.y]))
    #     #   for row in rows
    #     # ]

    #     # def iter_results():
    #     #   for r in part_df.collect():
    #     #     yield (r.x,)
    #     # return tf.data.Dataset.from_generator(
    #     #           generator=iter_results,
    #     #           output_types=(tf.int64,),
    #     #           output_shapes=(tf.TensorShape([]),))
      
    #   return pds.map(
    #     lambda p: tuple(tf.py_func(
    #       _gen_ds, [p], [tf.int64, tf.int64, tf.string]  )))

    
    
    # # ds = tf.data.Dataset.from_tensor_slices(range(num_partitions))
    # # idss = iter(dss)
    # ds = tf.data.Dataset.range(num_partitions).apply(
    #         tf.data.experimental.parallel_interleave(
    #           gen_ds,
    #           cycle_length=multiprocessing.cpu_count(),
    #           sloppy=True))
    # ds = ds.prefetch(10)
    # ds = ds.apply(tf.contrib.data.unbatch())
    
    # # ds = gen_rows(0)
    # with util.tf_data_session(ds) as (sess, iter_dataset):
    #   n = 0
    #   for x in iter_dataset():
    #     n += 1
    #     # print x
    #     if n % 1000 == 0:
    #       print n
    #   print n


    

    # # import pdb; pdb.set_trace()
    # # print 'moof'


# TODO test run_callables