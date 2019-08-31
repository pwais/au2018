"""A module with Spark-related utilities"""

import sys

from au import conf
from au import util

import os
import pickle
from contextlib import contextmanager

import numpy as np

# Try to find Spark / Java, and produce a helpful error message otherwise
try:
  import findspark
  findspark.init()

  # In python3, we filter:
  #  "py4j-0.10.7-src.zip/py4j/java_gateway.py:2020: 
  #      DeprecationWarning: invalid escape sequence \*"
  if sys.version_info.major >= 3:
    import warnings
    warnings.filterwarnings(
      action='ignore',
      message=r'invalid escape sequence')
    
  import pyspark
  from pyspark.sql import types

except Exception as e:
  msg = """
      This portion of AU requires Spark, which in turn requires Java 8 or
      higher.  Mebbe try installing using:
        $ pip install pyspark
      That will fix import errors.  To get Java, try:
        $ apt-get install -y openjdk-8-jdk && \
          echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
      Original error: %s
  """ % (e,)
  raise Exception(msg)

def egg_py_suffix():
  py_vers = (sys.version_info.major, sys.version_info.minor)
  py_suffix = '-py{}.{}.egg'.format(*py_vers)
  return py_suffix

class Spark(object):
  # Default to local Spark master
  MASTER = None
  CONF = None
  CONF_KV = None
  HIVE = False

  # Optional
  SRC_ROOT = os.path.join(conf.AU_ROOT, 'au')
  
  @classmethod
  def _create_egg(cls, src_root=None, tmp_path=None):
    """Build a Python Egg from the current project and return a path
    to the artifact.  

    Why an Egg?  `pyspark` supports zipfiles and egg files as Python artifacts.
    One might wish to use a wheel instead of an egg.  See this excellent
    article and repo:
     * https://bytes.grubhub.com/managing-dependencies-and-artifacts-in-pyspark-7641aa89ddb7
     * https://github.com/alekseyig/spark-submit-deps
    
    The drawbacks to using a wheel include:
     * wheels often require native libraries to be installed (e.g. via
        `apt-get`), and those deps are typically best baked into the Spark
        Worker environment (versus installed every job run).
     * The `BdistSpark` example above is actually rather slow, especially
        when Tensorflow is a dependency, and `BdistSpark` must run before
        every job is submitted.
     * Spark treats wheels as zip files and unzips them on every run; this
        unzip operation can be very expensive if the zipfile contains large
        binaries (e.g. tensorflow)
    
    In comparison, an Egg provides the main benefits we want (to ship project
    code, often pre-committed code, to workers).
    """

    log = util.create_log()

    if tmp_path is None:
      import tempfile
      tempdir = tempfile.gettempdir()

      SUBDIR_NAME = 'au_eggs_%s' % os.getpid()
      tmp_path = os.path.join(tempdir, SUBDIR_NAME)
      util.cleandir(tmp_path)

    if src_root is None:
      log.info("Trying to auto-resolve path to src root ...")
      try:
        import inspect
        path = inspect.getfile(inspect.currentframe())
        src_root = os.path.dirname(os.path.abspath(path))
        i_am_in_a_module = os.path.exists(
          os.path.join(os.path.dirname(src_root), '__init__.py'))
        if i_am_in_a_module:
          src_root = os.path.abspath(os.path.join(src_root, os.pardir))
      except Exception as e:
        log.info(
          "Failed to auto-resolve src root, "
          "falling back to %s" % cls.SRC_ROOT)
        src_root = cls.SRC_ROOT
    
    if sys.version_info.major >= 3:
      # For whatever reason,
      # In py 2.7.x, setuptools wants the path of the python module
      # In py 3.x, setuptools wants the directory containing the python module
      src_root = os.path.dirname(src_root)
    log.info("Using source root %s " % src_root)

    # Below is a programmatic way to run something like:
    # $ cd /opt/au && python setup.py clearn bdist_egg
    # Based upon https://github.com/pypa/setuptools/blob/a94ccbf404a79d56f9b171024dee361de9a948da/setuptools/tests/test_bdist_egg.py#L30
    # See also: 
    # * https://github.com/pypa/setuptools/blob/f52b3b1c976e54df7a70db42bf59ca283412b461/setuptools/dist.py
    # * https://github.com/pypa/setuptools/blob/46af765c49f548523b8212f6e08e1edb12f22ab6/setuptools/tests/test_sdist.py#L123
    # * https://github.com/pypa/setuptools/blob/566f3aadfa112b8d6b9a1ecf5178552f6e0f8c6c/setuptools/__init__.py#L51
    from setuptools.dist import Distribution
    from setuptools import PackageFinder
    MODNAME = os.path.split(src_root)[-1]
    dist = Distribution(attrs=dict(
        script_name='setup.py',
        script_args=[
          'clean',
          'bdist_egg', 
            '--dist-dir', tmp_path,
            '--bdist-dir', os.path.join(tmp_path, 'workdir'),
        ],
        name=MODNAME,
        src_root=src_root,
        packages=PackageFinder.find(where=src_root),
    ))
    log.info("Generating egg to %s ..." % tmp_path)
    with util.quiet():
      dist.parse_command_line()
      dist.run_commands()

    egg_path = os.path.join(tmp_path, MODNAME + '-0.0.0' + egg_py_suffix())
    assert os.path.exists(egg_path), "Can't find {}".format(egg_path)
    log.info("... done.  Egg at %s" % egg_path)
    return egg_path

    # NB: This approach didn't work so well:
    # Typically we want to give spark the egg from:
    #  $ python setup.py bdist_egg
    # from setuptools.command import bdist_egg
    # cmd = bdist_egg.bdist_egg(bdist_dir=os.path.dirname(setup_py_path), editable=True)
    # cmd.run()

  @classmethod
  def egg_path(cls):
    if not hasattr(cls, '_cached_egg_path'):
      cls._cached_egg_path = cls._create_egg()
    return cls._cached_egg_path

  @classmethod
  def _setup(cls):
    # Warm the cache
    egg_path = cls.egg_path()

  @classmethod
  def getOrCreate(cls):
    cls._setup()

    import pyspark
    
    from pyspark import sql
    builder = sql.SparkSession.builder
    if cls.MASTER is not None:
      builder = builder.master(cls.MASTER)
    elif 'SPARK_MASTER' in os.environ:
      # spark-submit honors this env var
      builder = builder.master(os.environ['SPARK_MASTER'])
    if cls.CONF is not None:
      builder = builder.config(conf=cls.CONF)
    if cls.CONF_KV is not None:
      for k, v in cls.CONF_KV.items():
        builder = builder.config(k, v)
    builder = builder.config('spark.task.maxFailures', '10')
    builder = builder.config('spark.port.maxRetries', '96')
      # For local instances with many CPUs, let Spark use tons of ports

    # # FIXME parquet sizes need this for laptop activation tables to work
    builder = builder.config('spark.sql.files.maxPartitionBytes', int(8 * 1e6))
    # TODO want large memory thingy for local mode only
    bulder = builder.config('spark.driver.memory', '8g')
    bulder = builder.config('spark.executor.memory', '8g')
    if cls.HIVE:
      # TODO fixme see mebbe https://creativedata.atlassian.net/wiki/spaces/SAP/pages/82255289/Pyspark+-+Read+Write+files+from+Hive
      # builder = builder.config("hive.metastore.warehouse.dir", '/tmp') 
      # builder = builder.config("spark.sql.warehouse.dir", '/tmp')
      builder = builder.enableHiveSupport()
    spark = builder.getOrCreate()

    # spark.sparkContext.setLogLevel('INFO')

    spark.sparkContext.addPyFile(cls.egg_path())
    return spark
  
  @classmethod
  @contextmanager
  def sess(cls, *args):
    if args and args[0]:
      spark = args[0]
      yield spark
    else:
      spark = cls.getOrCreate()
      yield spark

  @staticmethod
  def install

  @staticmethod
  def archive_rdd(spark, path):
    fws = util.ArchiveFileFlyweight.fws_from(path)
    return spark.sparkContext.parallelize(fws)

  @staticmethod
  def thruput_accumulator(spark, **thruputKwargs):
    from pyspark.accumulators import AccumulatorParam
    class ThruputObsAccumulator(AccumulatorParam):
      def zero(self, v):
        return v or util.ThruputObserver()
      def addInPlace(self, value1, value2):
        value1 += value2
        return value1
    
    return spark.sparkContext.accumulator(
                util.ThruputObserver(**thruputKwargs),
                ThruputObsAccumulator())


  @staticmethod
  def num_executors(spark):
    # NB: Not a public API! But likely stable.
    # https://stackoverflow.com/a/42064557
    return spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()

  @staticmethod
  def run_callables(spark, callables, parallel=-1):
    import cloudpickle
      # Spark uses regular pickle for data;
      # here we need cloudpickle for code
    callable_bytess = [cloudpickle.dumps(c) for c in callables]
    if parallel <= 0:
      parallel = len(callable_bytess)

    rdd = spark.sparkContext.parallelize(callable_bytess, numSlices=parallel)
    def invoke(callable_bytes):
      import cloudpickle
      c = cloudpickle.loads(callable_bytes)
      res = c()
      return callable_bytes, cloudpickle.dumps(res)

    rdd = rdd.map(invoke)
    all_results = [
      (cloudpickle.loads(callable_bytes), cloudpickle.loads(res))
      for callable_bytes, res in rdd.collect()
    ]
    return all_results

  # @staticmethod
  # def df_to_dstream(spark, df, *colnames):
  #   assert colnames, "Need at least one column"
  #   util.log.info(
  #     "Fetching chunks of %s based on columns %s ..." % (df, colnames))
  #   cols = df.select(*colnames)
  #   chunks_rdd = cols.rdd.mapPartitions(lambda rows: [[tuple(r) for r in rows]])
  #   chunks = chunks_rdd.collect()
  #   util.log.info("... got %s chunks / partitions ..." % len(chunks))
    
  #   rdds = []
  #   for i, chunk in enumerate(chunks):
  #     chunk_df = df
  #     print 'start'
  #     import pandas as pd
  #     qqq = spark.createDataFrame(pd.DataFrame(dict((col, colvals) for col, colvals in zip(colnames, zip(*chunk)))))
  #     print 'end'
  #     print 'start2'
  #     # qqq.show()
  #     joined = df.join(qqq.hint('broadcast'), on=[getattr(df, col) == getattr(qqq, col) for col in colnames], how='inner')
  #     print 'end2'
  #     # for col, colvals in zip(colnames, zip(*chunk)):
  #     #   chunk_df = chunk_df.filter(chunk_df[col].isin(*colvals))
  #     # rdds.append(chunk_df.rdd)
  #     rdd = joined.rdd
  #     # rdd = rdd.repartition(Spark.num_executors(spark))
  #     rdds.append(rdd)
  #     util.log.info(
  #       "... prepared %s / %s chunks: %s IDs in table %s" % (
  #         i + 1, len(chunks), len(chunk), chunk_df))
    
  #   from pyspark.streaming import StreamingContext
  #   ssc = StreamingContext(spark.sparkContext, 1)
  #   dstream = ssc.queueStream(rdds)
  #   return ssc, dstream


  @staticmethod
  def union_dfs(*dfs):
    """Return the union of a sequence DataFrames and attempt to merge
    the schemas of each (i.e. union of all columns).
    Based upon https://stackoverflow.com/a/40404249
    """
    if not dfs:
      return dfs
    
    df = dfs[0]
    for df_other in dfs[1:]:
      left_types = {f.name: f.dataType for f in df.schema}
      right_types = {f.name: f.dataType for f in df_other.schema}
      left_fields = set(
        (f.name, f.dataType, f.nullable) for f in df.schema)
      right_fields = set(
        (f.name, f.dataType, f.nullable) for f in df_other.schema)

      from pyspark.sql.functions import lit

      # First go over `df`-unique fields
      for l_name, l_type, l_nullable in left_fields.difference(right_fields):
          if l_name in right_types:
              r_type = right_types[l_name]
              if l_type != r_type:
                  raise TypeError(
                    "Union failed. Type conflict on field %s. left type %s, right type %s" % (l_name, l_type, r_type))
              else:
                  raise TypeError(
                    "Union failed. Nullability conflict on field %s. left nullable %s, right nullable %s"  % (l_name, l_nullable, not(l_nullable)))
          df_other = df_other.withColumn(l_name, lit(None).cast(l_type))

      # Now go over `df_other`-unique fields
      for r_name, r_type, r_nullable in right_fields.difference(left_fields):
          if r_name in left_types:
              l_type = right_types[r_name]
              if r_type != l_type:
                  raise TypeError(
                    "Union failed. Type conflict on field %s. right type %s, left type %s" % (r_name, r_type, l_type))
              else:
                  raise TypeError(
                    "Union failed. Nullability conflict on field %s. right nullable %s, left nullable %s" % (r_name, r_nullable, not(r_nullable)))
          df = df.withColumn(r_name, lit(None).cast(r_type))
      df = df.unionByName(df_other)
    return df




  ### Test Utilities (for unit tests and more!)

  @classmethod
  def selftest(cls):
    with cls.sess() as spark:
      # spark.sparkContext.setLogLevel("INFO")
      cls.test_pi(spark)
      cls.test_egg(spark)
      cls.test_tensorflow(spark)

  @staticmethod
  def test_pi(spark):
    util.log.info("Running PI ...")
    sc = spark.sparkContext
    num_samples = 1000000
    def inside(p):
      import random
      x, y = random.random(), random.random()
      return x*x + y*y < 1
    count = sc.parallelize(list(range(0, num_samples))).filter(inside).count()
    pi = 4 * float(count) / num_samples
    util.log.info("Pi estimate: %s" % pi)
    assert abs(pi - 3.14) < 0.1, "Spark program had an error?"

  @staticmethod
  def test_egg(spark):
    EXPECTED_EGG_NAME = 'au-0.0.0' + egg_py_suffix()

    def worker_test(_):
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
      assert egg_path, "Egg not found in {}".format(sys.path)

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
  
    util.log.info("Testing egg ...")
    
    sc = spark.sparkContext
    N = max(1, Spark.num_executors(spark)) # Try to test all executors
    rdd = sc.parallelize(list(range(N)), numSlices=N)
    res = rdd.map(worker_test).collect()
    assert len(res) == N
    paths = [info['filepath'] for info in res]
    assert all(EXPECTED_EGG_NAME in p for p in paths)

    util.log.info("Test success!  Worker info:")
    def format_info(info):
      s = """
        Host: {hostname} {host}
        Egg: {filepath}
        Internet connectivity: {have_internet}
        Num CPUs: {n_cpus}
        Memory:
        {memory}

        PYTHONPATH:
        {PYTHONPATH}

        nvidia-smi:
        {nvidia_smi}

        Disk:
        {disk_free}
        """.format(**info)
      return '\n'.join(l.lstrip() for l in s.split('\n'))
    info_str = '\n\n'.join(format_info(info) for info in res)
    util.log.info('\n\n' + info_str)
  
  @staticmethod
  def test_tensorflow(spark):

    # Tensorflow Devices
    # TODO this util can crash with 'failed to allocate 2.2K' :P even with
    # a lock? wat??
    #with atomic_ignore_exceptions():
    #  from tensorflow.python.client import device_lib
    #  devs = device_lib.list_local_devices()
    #  info['tensorflow_devices'] = [str(v) for v in devs]

    def foo(x):
      import tensorflow as tf

      a = tf.constant(x)
      b = tf.constant(3)

      from au import util
      sess = util.tf_create_session()
      res = sess.run(a * b)

      assert res == 3 * x
      
      import socket
      info = {
        'hostname': socket.gethostname(),
        'gpu': tf.test.gpu_device_name(),
      }
      return info

    util.log.info("Testing Tensorflow ...")
    
    sc = spark.sparkContext
    N = max(1, Spark.num_executors(spark)) # Try to test all executors
    y = list(range(N))
    rdd = sc.parallelize(y)
    res = rdd.map(foo).collect()
    assert len(res) == N

    util.log.info("... Tensorflow success!  Info:")
    import pprint
    util.log.info('\n\n' + pprint.pformat(res) + '\n\n')



class K8SSpark(Spark):
  MASTER = conf.AU_K8S_SPARK_MASTER
  CONF_KV = {
    'spark.kubernetes.container.image':
      conf.AU_SPARK_WORKER_IMAGE,

    # In practice, we need to set this explicitly in order to get the
    # proper driver IP address to the workers.  This choice may break
    # cluster mode where the driver process will run in the cluster
    # instead of locally.  This choice may also break in certain networking
    # setups.  Spark networking is a pain :(
    'spark.driver.host':
      os.environ.get('SPARK_LOCAL_IP', util.get_non_loopback_iface()),
  }

## Spark UDTs
# These utils are based upon Spark's DenseVector:
# https://github.com/apache/spark/blob/044b33b2ed2d423d798f2a632fab110c46f41567/python/pyspark/mllib/linalg/__init__.py#L239
# https://apache.googlesource.com/spark/+/refs/heads/master/python/pyspark/sql/tests.py#119
# Sadly they don't have a UDT for tensors... not even in Tensorframes
# https://github.com/databricks/tensorframes   o_O
#
# BREADCRUMBS: so these UDTs can't be used in nested structs :( pyspark
# isn't smart enough.  

class NumpyArrayUDT(types.UserDefinedType):
  """SQL User-Defined Type (UDT) for *opaque* numpy arrays.  Unlike Spark's
  DenseVector, this class preserves array shape.  An unlike Spark's
  DenseMatrix, this class supports arbitrary shape.  See also
  pyspark.mllib.linalg.MatrixUDT.

  TODO: make an arbitrary pickleable wrapper ....
  """

  @classmethod
  def sqlType(cls):
    # NB: this is actually an instance method in practice O_O !
    return types.StructType([
      types.StructField("np_bytes", types.BinaryType(), False)

      # TODO: we'd like to make arrays more portable, but we need
      # to infer the type of the numpy array values ...
      # types.StructField("dtype_name", types.StringType(), False),
      # types.StructField("shape",
      #   types.ArrayType(types.IntegerType(), False), False),
      # types.StructField("values",
      #   types.ArrayType(  ???  , False), False),
    ])

  @classmethod
  def module(cls):
    return 'au.spark'

  def __hash__(self):
    return hash(self.simpleString())

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

  @staticmethod
  def from_bytes(b):
    arr = pickle.loads(b)
    return NumpyArray(arr)

  def __repr__(self):
    return "NumpyArray:" + self.arr.__repr__()
  
  def __str__(self):
    return "NumpyArray" + self.arr.__str__()

  def __eq__(self, other):
    return isinstance(other, self.__class__) and other.arr == self.arr



def get_balanced_sample(spark_df, col, n_per_category=None, seed=1337):
  """Given a column `col` in `spark_df`, return a *balanced* sample
  (countering class imbalances in `spark_df[col]`).  Optionally limit the
  sample to having up to `n_per_category` examples for every distinct
  categorical value of `spark_df[col]`."""
  from pyspark.sql import functions as F
  category_to_count_df = spark_df.groupBy(col).agg(F.count('*'))
  category_to_count = category_to_count_df.rdd.collectAsMap()
  assert category_to_count

  # We will only sample as many as the rarest category
  numerator = min(category_to_count.values())
  if n_per_category is not None:
    numerator = min(numerator, n_per_category)
  fractions = dict(
    (category, float(numerator) / count)
    for category, count in category_to_count.items()
  )
  return spark_df.sampleBy(col, fractions=fractions, seed=seed)
  


def spark_df_to_tf_dataset(
      spark_df,
      spark_row_to_tf_element, # E.g. lambda r: (np.array[0],),
      tf_element_types, # E.g. [tf.int64]
      non_deterministic_element_order=True,
      num_reader_threads=-1,
      logging_name='spark_tf_dataset'):
    """Create a tf.data.Dataset that reads from the Spark Dataframe
    `spark_df`.  Executes parallel reads using the Tensorflow's internal
    (native code) threadpool.  Each thread reads a single Spark partition
    at a time.

    This utility is similar to Petastorm's `make_reader()` but is far simpler
    and leverages Tensorflow's build-in threadpool (so we let Tensorflow
    do the read scheduling).

    Args:
      spark_df (pyspark.sql.DataFrame): Read from this Dataframe
      spark_row_to_tf_element (func): 
        Use this function to map each pyspark.sql.Row in `spark_df`
        to a tuple that represents a single element of the
        induced TF Dataset.
      tf_element_types (tuple):
        The types of the elements that `spark_row_to_tf_element` returns;
        e.g. (tf.float32, tf.string).
      non_deterministic_element_order (bool):
        Allow the resulting tf.data.Dataset to have elements in
        non-deterministic order for speed gains.
      num_reader_threads (int):
        Tell Tensorflow to use this many reader threads, or use -1
        to provision one reader thread per CPU core.
      logging_name (str):
        Log progress under this name.
    
    Returns:
      tf.data.Dataset: The induced TF Datset with one element per
        row in `spark_df`.
    """

    ## Somewhat slower
    #import tensorflow as tf
    #tuple_rdd = spark_df.rdd.map(spark_row_to_tf_element)
    #tuple_rdd = tuple_rdd.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    #ds = tf.data.Dataset.from_generator(
    #          tuple_rdd.toLocalIterator, tf_element_types)
    #return ds








    if num_reader_threads < 1:
      import multiprocessing
      num_reader_threads = multiprocessing.cpu_count()
    num_reader_threads = 8
    df = spark_df

    # Each Tensorflow reader thread will read a single Spark partition
    # from pyspark.sql.functions import spark_partition_id
    # df = spark_df.withColumn('_spark_partition_id', spark_partition_id())
    # df = spark_df.withColumn('part', spark_df['log_id'])
    
    
    # df = df.repartition(df['part'])
    # df = spark_df
    # pids = df.select('_spark_part_id').distinct().rdd.flatMap(lambda x: x).collect()
    print('getting shards')
    pids = df.select('shard').distinct().rdd.flatMap(lambda x: x).collect()
    print(len(pids), pids)


    import threading
    import tensorflow as tf
    pid_ds = tf.data.Dataset.from_tensor_slices(pids)
    
    class PartitionToRows(object):
      def __init__(self):
        self.overall_thruput = util.ThruputObserver(
                                    name=logging_name,
                                    log_on_del=True,
                                    n_total=df.count())#len(pids))  ~~~~~~~~~~~~~~~~~~~~~~
        self.overall_thruput.start_block()
        self.lock = threading.Lock()
      
      def __call__(self, pid):
        part_df = df.filter(df['shard'] == int(pid))
        # part_df = df.filter('part == %s' % pid)
          # Careful! This can be a linear scan :(
        rows = part_df.rdd.repartition(1000).map(spark_row_to_tf_element).toLocalIterator()#persist(pyspark.StorageLevel.MEMORY_AND_DISK).toLocalIterator()#collect()
        util.log.info("Reading partition %s " % pid)#had %s rows" % (pid, len(rows)))
        t = util.ThruputObserver(name='Partition %s' % pid, log_on_del=True)
        t.start_block()
        for row in rows:
          yield row
          t.update_tallies(n=1, num_bytes=util.get_size_of_deep(row))
        t.stop_block()
        util.log.info("Done reading partition %s, stats:\n %s" % (pid, t))
        with self.lock:
          # Since partitions are read in parallel, we need to maintain
          # independent timing stats for the main thread
          self.overall_thruput.stop_block(n=t.n, num_bytes=t.num_bytes)
          self.overall_thruput.maybe_log_progress(every_n=1)
          self.overall_thruput.start_block()

      # import pyspark.sql
      # part_df.createOrReplaceTempView('part_df_%s' % pid)
      # spark = pyspark.sql.SQLContext(part_df._sc)
      # spark.sql("""
      #   SELECT split, category_name, COUNT(*) c
      #   FROM part_df_%s
      #   GROUP BY split, category_name
      #   ORDER BY split, c DESC
      # """ % pid).show()
    
    ds = pid_ds.interleave(
       lambda pid_t: \
         tf.data.Dataset.from_generator(
           PartitionToRows(), 
           args=(pid_t,),
           output_types=tf_element_types),
       cycle_length=num_reader_threads,
       num_parallel_calls=num_reader_threads)
    #ds = pid_ds.apply(
    #  tf.compat.v2.data.experimental.parallel_interleave(
    #    lambda pid_t: 
    #      tf.data.Dataset.from_generator(
    #        get_rows, 
    #        args=(pid_t,),
    #        output_types=tf_element_types),
    #  cycle_length=num_reader_threads,
    #  sloppy=non_deterministic_element_order))
    return ds




















    # gens_dss = [
      
    #   for pid in pids
    # ]

    # ds = gens_dss[0]
    # for i in range(1, len(gens_dss)):
    #   ds.concatentate(gens_dss[i])
    # return ds

    # pid_ds.apply(
    #   tf.data.experimental.parallel_interleave(



    # def make_gen(pid_array):
    #   def gen():
    #     for row in get_rows(pid_array):
    #       print('row', row)
    #       yield row
    #   return tf.data.Dataset.from_generator(gen, tf_element_types)
    # ds = pid_ds.apply(
    #   tf.data.experimental.parallel_interleave(
    #     make_gen,
    #     cycle_length=num_reader_threads,
    #     sloppy=True))
    # return ds

    
      


      # def pid_to_element_cols(pid):
        
        
      #   if not rows:
      #     # Tensorflow expects empty numpy columns of promised dtype
      #     import numpy as np
      #     util.log.warn("No rows!")
      #     return tuple(
      #       np.empty(0, dtype=tf_dtype.as_numpy_dtype)
      #       for tf_dtype in tf_element_types
      #     )
      #   util.log.info("FetchED partition %s" % pid)
      #   xformed = [spark_row_to_tf_element(row) for row in rows]

      #   # Sadly TF py_func can't easily return a list of objects, just a
      #   # tuple of arrays.  So we re-organize the rows into columns, each
      #   # which has a known type.
      #   import itertools
      #   cwise = list(zip(*xformed))

      #   return cwise
      
      # pds = pds.map(
      #   lambda p: tf.numpy_function(
      #     pid_to_element_cols, [p], tf_element_types))
      #       # NB: why tuple()? https://github.com/tensorflow/tensorflow/issues/12396#issuecomment-323407387
      
    #   # import uuid
    #   # path = '/tmp/cache_yay_' + str(uuid.uuid4())
    #   # print 'save cache for pid to ', path
    #   # pds = pds.cache(path)
    #   # print 'warming cache'
    #   # with util.tf_data_session(pds) as (sess, iter_dataset):
    #   #   n = 0
    #   #   for _ in iter_dataset():
    #   #     n += 1
    #   #   print 'warmed', n
    #   # util.rm_rf(path + '_0.lockfile')
    #   return pds


# # FIXME
# def spark_df_to_tf_dataset(
#       spark_df,
#       spark_row_to_tf_element, # E.g. lambda r: (np.array[0],),
#       tf_element_types, # E.g. [tf.int64]
#       non_deterministic_element_order=True,
#       num_reader_threads=-1):
#     """Create a tf.data.Dataset that reads from the Spark Dataframe
#     `spark_df`.  Executes parallel reads using the Tensorflow's internal
#     (native code) threadpool.  Each thread reads a single Spark partition
#     at a time.

#     This utility is similar to Petastorm's `make_reader()` but is far simpler
#     and leverages Tensorflow's build-in threadpool (so we let Tensorflow
#     do the read scheduling).

#     Args:
#       spark_df (pyspark.sql.DataFrame): Read from this Dataframe
#       spark_row_to_tf_element (func): 
#         Use this function to map each pyspark.sql.Row in `spark_df`
#         to a tuple that represents a single element of the
#         induced TF Dataset.
#       tf_element_types (tuple):
#         The types of the elements that `spark_row_to_tf_element` returns;
#         e.g. (tf.float32, tf.string).
#       non_deterministic_element_order (bool):
#         Allow the resulting tf.data.Dataset to have elements in
#         non-deterministic order for speed gains.
#       num_reader_threads (int):
#         Tell Tensorflow to use this many reader threads, or use -1
#         to provision one reader thread per CPU core.
    
#     Returns:
#       tf.data.Dataset: The induced TF Datset with one element per
#         row in `spark_df`.
#     """

#     if num_reader_threads < 1:
#       import multiprocessing
#       num_reader_threads = multiprocessing.cpu_count()

#     # Each Tensorflow reader thread will read a single Spark partition
#     from pyspark.sql.functions import spark_partition_id
#     df = spark_df.withColumn('_spark_part_id', spark_partition_id())
    
#     import tensorflow as tf
#     def to_dataset(pid_tensor):
#       """Given a Tensor containing a single Spark partition ID,
#       return a TF Dataset that contains all elements from that partition."""
#       pds = tf.data.Dataset.from_tensors(pid_tensor)
      
#       def pid_to_element_cols(pid):
#         pid = pid[0]
#         util.log.info("Fetching partition %s" % pid)
#         part_df = df.filter('_spark_part_id == %s' % pid)
#           # TODO: this is a linear scan :(  find a more efficient way
#         rows = part_df.collect()
#         if not rows:
#           # Tensorflow expects empty numpy columns of promised dtype
#           import numpy as np
#           util.log.warn("No rows!")
#           return tuple(
#             np.empty(0, dtype=tf_dtype.as_numpy_dtype)
#             for tf_dtype in tf_element_types
#           )
#         util.log.info("FetchED partition %s" % pid)
#         xformed = [spark_row_to_tf_element(row) for row in rows]

#         # Sadly TF py_func can't easily return a list of objects, just a
#         # tuple of arrays.  So we re-organize the rows into columns, each
#         # which has a known type.
#         import itertools
#         cwise = list(zip(*xformed))

#         return cwise
      
#       pds = pds.map(
#         lambda p: tf.numpy_function(
#           pid_to_element_cols, [p], tf_element_types))
#             # NB: why tuple()? https://github.com/tensorflow/tensorflow/issues/12396#issuecomment-323407387
      
#       # import uuid
#       # path = '/tmp/cache_yay_' + str(uuid.uuid4())
#       # print 'save cache for pid to ', path
#       # pds = pds.cache(path)
#       # print 'warming cache'
#       # with util.tf_data_session(pds) as (sess, iter_dataset):
#       #   n = 0
#       #   for _ in iter_dataset():
#       #     n += 1
#       #   print 'warmed', n
#       # util.rm_rf(path + '_0.lockfile')
#       return pds



#     # maybe_row = df.take(1)
#     # assert maybe_row
#     # ex = spark_row_to_tf_element(maybe_row[0])
#     # def get_dtype(v):
#     #   if hasattr(v, 'dtype'):
#     #     return tf.dtypes.as_dtype(v.dtype)
#     #   elif isinstance(v, (basestring, unicode)):
#     #     return tf.string
#     #   else:
#     #     return tf.dtypes.as_dtype(v)
#     # output_shapes = tuple(
#     #   v.shape[0] if hasattr(v, 'shape') else None
#     #   for v in ex
#     # )
#     # output_types = tuple(get_dtype(v) for v in ex)

#     # def gen_examples(p):
#     #   util.log.info("Fetching partition %s" % p)
#     #   print 'fetch', p
#     #   part_df = df.filter('_spark_part_id == %s' % p)
#     #   for i, row in enumerate(part_df.collect()):
#     #     yield spark_row_to_tf_element(row)
#     #     print 'y', p, i
#     #   print 'fetch done', p
#     #   util.log.info("FetchED partition %s" % p)

#     # import threading
#     # l = threading.Lock()
#     # ppid = [-1]
#     # path_prefix = '/tmp/cache_yay_'
#     # def to_dataset(pid_tensor):
#     #   # print 'pid_tensor', pid_tensor
#     #   # with l:
#     #   #   ppid[0] += 1
#     #   #   pid = ppid[0]
#     #   # print 'pid', pid
#     #   pds = tf.data.Dataset.from_generator(
#     #             gen_examples,
#     #             output_types=output_types,
#     #             output_shapes=output_shapes,
#     #             args=[pid_tensor])
#     #   # pds = pds.cache(tf.strings.format(path_prefix + '{}', pid_tensor))
#     #   return pds




#     # for _ in range(10):
#     #   print 'df.rdd.getNumPartitions()', df.rdd.getNumPartitions()
#     # dss = [tf.data.Dataset.from_tensors([pid]) for pid in range(df.rdd.getNumPartitions())]


#     ds = tf.data.Dataset.from_tensor_slices(
#       [n for n in df.select('_spark_part_id').distinct().collect()]
#     )
    
#     # range(df.rdd.getNumPartitions())
#     # ds = ds.interleave(to_dataset,
#     #           cycle_length=10 * num_reader_threads)
#     ds = ds.apply(
#             # Use `parallel_interleave` to have the Tensorflow reader
#             # threadpool read in parallel from Spark
#             tf.data.experimental.parallel_interleave(
#               to_dataset,
#               cycle_length=num_reader_threads,
#               sloppy=non_deterministic_element_order))
    
#     # `ds` is now a dataset where elements are grouped by Spark partition, e.g.
#     # [x1_p1, x2_p1, ...], [x1_p2, x2_p2, ...], ...
#     # We want a dataset that's flat:
#     # [x1_p1, x2_p1, ..., x1_p2, x2_p2, ...], ...
#     # The user expects a tf.data.Dataset that fascades a flat sequence of
#     # elements.  (Because, among other things, the user wants to choose
#     # a batch size independent of Spark partition size).  Thus we
#     # use `unbatch` below.
#     ds = ds.apply(tf.data.experimental.unbatch())
#     return ds


"""
def spark_df_to_tf_dataset(
      spark_df,
      spark_row_to_tf_element, # E.g. lambda r: (np.array[0],),
      tf_element_types, # E.g. [tf.int64]
      non_deterministic_element_order=True,
      num_reader_threads=-1):
    ""Create a tf.data.Dataset that reads from the Spark Dataframe
    `spark_df`.  Executes parallel reads using the Tensorflow's internal
    (native code) threadpool.  Each thread reads a single Spark partition
    at a time.

    This utility is similar to Petastorm's `make_reader()` but is far simpler
    and leverages Tensorflow's build-in threadpool (so we let Tensorflow
    do the read scheduling).

    Args:
      spark_df (pyspark.sql.DataFrame): Read from this Dataframe
      spark_row_to_tf_element (func): 
        Use this function to map each pyspark.sql.Row in `spark_df`
        to a tuple that represents a single element of the
        induced TF Dataset.
      tf_element_types (tuple):
        The types of the elements that `spark_row_to_tf_element` returns;
        e.g. (tf.float32, tf.string).
      non_deterministic_element_order (bool):
        Allow the resulting tf.data.Dataset to have elements in
        non-deterministic order for speed gains.
      num_reader_threads (int):
        Tell Tensorflow to use this many reader threads, or use -1
        to provision one reader thread per CPU core.
    
    Returns:
      tf.data.Dataset: The induced TF Datset with one element per
        row in `spark_df`.
    ""

    if num_reader_threads < 1:
      import multiprocessing
      num_reader_threads = multiprocessing.cpu_count()

    # Each Tensorflow reader thread will read a single Spark partition
    from pyspark.sql.functions import spark_partition_id
    df = spark_df.withColumn('_spark_part_id', spark_partition_id())
    
    import tensorflow as tf
    def to_dataset(pid_tensor):
      ""Given a Tensor containing a single Spark partition ID,
      return a TF Dataset that contains all elements from that partition.""
      pds = tf.data.Dataset.from_tensors(pid_tensor)
      
      def pid_to_element_cols(pid):
        # path = '/tmp/yay_%s' % pid
        # import pickle
        if True:#not os.path.exists(path):
          util.log.info("Fetching partition %s" % pid)
          part_df = df.filter('_spark_part_id == %s' % pid)
          rows = part_df.collect()
          if not rows:
            # Tensorflow expects empty numpy columns of promised dtype
            import numpy as np
            print 'no rows'
            return tuple(
              np.empty(0, dtype=tf_dtype.as_numpy_dtype)
              for tf_dtype in tf_element_types
            )
          util.log.info("FetchED partition %s" % pid)
          xformed = [spark_row_to_tf_element(row) for row in rows]

          # Sadly TF py_func can't easily return a list of objects, just a
          # tuple of arrays.  So we re-organize the rows into columns, each
          # which has a known type.
          import itertools
          cwise = list(itertools.izip(*xformed))
        #   pickle.dump(cwise, open(path, 'wb'))
        # # else:
        # #   print 'read cached', path
        # cwise = pickle.load(open(path, 'rb'))
        return cwise
      
      def pid_to_dataset(pid):
        

      # return pds.map(
      #   lambda p: tuple(tf.py_func(
      #     pid_to_element_cols, [p], tf_element_types)))
      #       # NB: why tuple()? https://github.com/tensorflow/tensorflow/issues/12396#issuecomment-323407387
      return pds.apply(pid_to_dataset)
    
    
    for _ in range(10):
      print 'df.rdd.getNumPartitions()', df.rdd.getNumPartitions()
    ds = tf.data.Dataset.range(df.rdd.getNumPartitions())
    ds = ds.apply(
            # Use `parallel_interleave` to have the Tensorflow reader
            # threadpool read in parallel from Spark
            tf.data.experimental.parallel_interleave(
              to_dataset,
              cycle_length=num_reader_threads,
              sloppy=non_deterministic_element_order))
    
    # `ds` is now a dataset where elements are grouped by Spark partition, e.g.
    # [x1_p1, x2_p1, ...], [x1_p2, x2_p2, ...], ...
    # We want a dataset that's flat:
    # [x1_p1, x2_p1, ..., x1_p2, x2_p2, ...], ...
    # The user expects a tf.data.Dataset that fascades a flat sequence of
    # elements.  (Because, among other things, the user wants to choose
    # a batch size independent of Spark partition size).  Thus we
    # use `unbatch` below.
    ds = ds.apply(tf.contrib.data.unbatch())
    return ds
    """
