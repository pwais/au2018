from au import util
from au.spark import NumpyArray
from au.spark import RowAdapter
from au.test import testconf
from au.test import testutils

import os

import pytest

@pytest.mark.slow
def test_spark_selftest():
  testutils.LocalSpark.selftest()

class Slotted(object):
  __slots__ = ('foo', 'bar')
  
  def __init__(self, **kwargs):
    # NB: ctor for convenience; SparkTypeAdapter does not require it
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def __repr__(self):
    return "Slotted(%s)" % ([(k, getattr(self, k)) for k in self.__slots__],)

class Unslotted(object):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
  
  def __repr__(self):
    return "Unslotted" + str(sorted(self.__dict__.items()))

@pytest.mark.slow
def test_row_adapter():
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'spark_row_adapter_test')
  util.cleandir(TEST_TEMPDIR)

  import numpy as np
  rows = [
    {
      'id': 1,
      'np_number': np.float32(1.),
      'a': np.array([1]), 
      'b': {
        'foo': np.array( [ [1] ], dtype=np.uint8)
      },
      'c': [
        np.array([[[1.]], [[2.]], [[3.]]])
      ],
      'd': Slotted(foo=5, bar="abc"),
      'e': [Slotted(foo=6, bar="def")],
      'f': Unslotted(meow=4),
      'e': Unslotted() # Intentionally empty; adapter should set nothing
    },

    # Include a mostly empty row below to exercise Spark type validation.
    # Spark will ensure the row below and row above have the same schema;
    # note that `None` (or 'null') is only allowed for Struct / Row types.
    {
      'id': 2,
      'np_number': np.float32(2.),
      'a': np.array([]),
      'b': {},
      'c': [],
      'd': None,
      'e': [],
      'f': None,
      'e': None,
    },
  ]

  # Test serialization numpy <-> parquet
  with testutils.LocalSpark.sess() as spark:
    from pyspark.sql import Row

    adapted_rows = [RowAdapter.to_row(r) for r in rows]

    df = spark.createDataFrame(adapted_rows)
    df.show()
    outpath = os.path.join(TEST_TEMPDIR, 'rowdata')
    df.write.parquet(outpath)

    df2 = spark.read.parquet(outpath)
    decoded_wrapped_rows = df2.collect()
    
    decoded_rows = [
      RowAdapter.from_row(row)
      for row in decoded_wrapped_rows
    ]
    decoded_rows = [r.asDict() for r in decoded_rows]
    
    # We can't do assert sorted(rows) == sorted(decoded_rows)
    # because numpy syntatic sugar breaks ==
    import pprint
    def sorted_row_str(rowz):
      return pprint.pformat(sorted(rowz, key=lambda row: row['id']))
    assert sorted_row_str(rows) == sorted_row_str(decoded_rows)

@pytest.mark.slow
def test_spark_numpy_df():
  ###
  ### DEPRECATED
  ###
  
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
      Row(**dict((k, NumpyArray(v)) for k, v in row.items()))
      for row in rows
    ]

    df = spark.createDataFrame(wrapped_rows)
    df.show()
    outpath = os.path.join(TEST_TEMPDIR, 'rowdata')
    df.write.parquet(outpath)

    df2 = spark.read.parquet(outpath)
    decoded_wrapped_rows = df2.collect()
    
    decoded_rows = [
      dict((k, v.arr if v else v) for k, v in row.asDict().items())
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
  ss = [b'foo', b'bar', b'baz']
  
  fixture_path = os.path.join(TEST_TEMPDIR, 'test.zip')
  
  import zipfile
  with zipfile.ZipFile(fixture_path, mode='w') as z:
    for s in ss:
      z.writestr(s.decode('utf-8'), s)
  
  with testutils.LocalSpark.sess() as spark:
    rdd = testutils.LocalSpark.archive_rdd(spark, fixture_path)
    name_data = rdd.map(lambda entry: (entry.name, entry.data)).collect()
    assert sorted(name_data) == sorted((s.decode('utf-8'), s) for s in ss)

# @pytest.mark.slow
# def test_spark_df_to_tf_dataset():

#   with testutils.LocalSpark.sess() as spark:
#     df = spark.read.parquet('/tmp/au_test/test_argoverse/tables/argoverse_cropped_object_170_170')
#     import pdb; pdb.set_trace()
#     print('moof')

#   import tensorflow as tf

#   import pyarrow as pa
#   import pyarrow.parquet as pq
#   pa_table = pq.read_table(
#     '/tmp/au_test/test_argoverse/tables/argoverse_cropped_object_170_170',
#     )#columns=['uri', 'frame_uri', 'track_id', 'jpeg_bytes'])
#   df = pa_table.to_pandas()

#   print(df)

#   from tensorflow_io.arrow import ArrowDataset
#   dataset = ArrowDataset.from_pandas(df)

#   iterator = dataset.make_one_shot_iterator()
#   next_element = iterator.get_next()

#   with tf.Session() as sess:
#     for i in range(len(df)):
#       print(sess.run(next_element))


@pytest.mark.slow
def test_spark_df_to_tf_dataset():
  from au.spark import spark_df_to_tf_dataset
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

    # # Test empty
    # ds = spark_df_to_tf_dataset(
    #         df.filter('x == False'), # Empty!
    #         spark_row_to_tf_element=lambda r: ('test',),
    #         tf_element_types=(tf.string,))
    # assert tf_dataset_to_list(ds) == []

    # # Test simple
    # ds = spark_df_to_tf_dataset(
    #         df,
    #         spark_row_to_tf_element=lambda r: (r.x,),
    #         tf_element_types=(tf.int64,))
    # assert sorted(tf_dataset_to_list(ds)) == [(1,), (2,), (3,)]

    # # Test Complex
    # ds = spark_df_to_tf_dataset(
    #         df,
    #         spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
    #         tf_element_types=(tf.int64, tf.string, tf.float64))
    expected = [
      (1, b'r1', np.array([3., 4., 5.])),
      (2, b'r2', np.array([6.])),
      (3, b'r3', np.array([7., 8., 9.])),
    ]
    # items = list(zip(sorted(tf_dataset_to_list(ds)), sorted(expected)))
    # for actual, exp in items:
    #   print('actual', actual,'exp', exp)
    #   assert len(actual) == len(exp)
    #   for i in range(len(actual)):
    #     np.testing.assert_array_equal(actual[i], exp[i])
    
    # Test Large
    from sys import getsizeof
    import random
    from au.spark import Spark
    # fixture = df.collect()
    def gen_data(n):
      import numpy as np
      # for i in range(10):
      y = np.random.rand(2 ** 15).tolist()
      return Row(part=n % 100, id=str(n), x=1, y=y)
    rdd = spark.sparkContext.parallelize(range(10000))
    udf = spark.createDataFrame(rdd.map(gen_data))
    # udf.write.parquet('/tmp/yay_pq_test', partitionBy=['part'], mode='overwrite')
    print('wrote')
    # udf = spark.read.parquet('cache/tables/argoverse_image_annos')
    # udf.write.parquet('/tmp/yay_pq_test', partitionBy=['log_id'], mode='overwrite')
    udf = spark.read.parquet('/tmp/yay_pq_test')

    # udf = Spark.union_dfs(*(df for _ in range(100)))
    
    print(udf.count())
    n_expect = udf.count()
    # udf.show()
    ds = spark_df_to_tf_dataset(
            udf,
            spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
            # spark_row_to_tf_element=lambda r: (r.x, r.uri, r.length_meters),
            tf_element_types=(tf.int64, tf.string, tf.float64))
    import itertools
    iexpected = itertools.cycle(expected)
    n = 0
    t = util.ThruputObserver()
    with util.tf_data_session(ds) as (sess, iter_dataset):
      t.start_block()
      for actual in iter_dataset():
        exp = iexpected.__next__()
        # print(actual,exp)
        n += 1
        t.update_tallies(n=1)
        for i in range(len(actual)):
          t.update_tallies(num_bytes=getsizeof(actual[i]))
        t.maybe_log_progress()
        #   np.testing.assert_array_equal(actual[i], exp[i])  
      t.stop_block()
    print(t)
    print(n, n_expect)
    assert n == n_expect

    # items = list(zip(sorted(tf_dataset_to_list(ds)), sorted(expected)))
    
    # items = itertools.cycle(items)
    # for actual, exp in items:
    #   # print('actual', actual,'exp', exp)
    #   assert len(actual) == len(exp)
    #   for i in range(len(actual)):
    #     np.testing.assert_array_equal(actual[i], exp[i])

# TODO test run_callables

@pytest.mark.slow
def test_get_balanced_sample():
  from au.spark import get_balanced_sample
  from pyspark.sql import Row
  
  VAL_TO_COUNT = {
    'a': 10,
    'b': 100,
    'c': 1000,
  }
  rows = []
  for val, count in VAL_TO_COUNT.items():
    for _ in range(count):
      i = len(rows)
      rows.append(Row(id=i, val=val))
  
  def _get_category_to_count(df):
    from collections import defaultdict
    rows = df.collect()

    category_to_count = defaultdict(int)
    for row in rows:
      category_to_count[row.val] += 1
    return category_to_count

  def check_sample_in_expectation(df, n_per_category, expected):
    import numpy as np
    import pandas as pd
    N_SEEDS = 10
    rows = [
      _get_category_to_count(
        get_balanced_sample(
          df, 'val',
          n_per_category=n_per_category,
          seed=100*s))
      for s in range(N_SEEDS)
    ]
    pdf = pd.DataFrame(rows)
    pdf = pdf.fillna(0)

    ks = sorted(expected.keys())
    mu = pdf.mean()
    actual_arr = np.array([mu[k] for k in ks])
    expected_arr = np.array([expected[k] for k in ks])
    
    import numpy.testing as npt
    npt.assert_allclose(actual_arr, expected_arr, rtol=0.2)
      # NB: We can only test to about 20% accuracy with this few samples

  with testutils.LocalSpark.sess() as spark:
    df = spark.createDataFrame(rows)

    check_sample_in_expectation(
      df, n_per_category=1, expected={'a': 1, 'b': 1, 'c': 1})

    check_sample_in_expectation(
      df, n_per_category=10, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=20, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=None, expected={'a': 10, 'b': 10, 'c': 10})
