"""A module with Spark-related utilities"""

from au import conf
from au import util

import os
import pickle
from contextlib import contextmanager

import numpy as np

# Try to provide a helpful error message if we can't find Spark / Java
try:
  import findspark
  findspark.init()

  import pyspark
  from pyspark.sql import types

except Exception as e:
  msg = """
      This portion of AU requires Spark, which in turn requires Java 8 or
      higher.  Mebbe try installing using:
        $ pip install pyspark
      That will fix import errors.  To get Java, try:
        $ apt-get install -y default-jdk && \
          echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
      Original error: %s
  """ % (e,)
  raise Exception(msg)


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

      SUBDIR_NAME = 'au_eggs'
      tmp_path = os.path.join(tempdir, SUBDIR_NAME)
      util.cleandir(tmp_path)

    if src_root is None:
      log.info("Trying to auto-resolve path to src root ...")
      try:
        import inspect
        path = inspect.getfile(inspect.currentframe())
        src_root = os.path.dirname(os.path.abspath(path))
      except Exception as e:
        log.info(
          "Failed to auto-resolve src root, "
          "falling back to %s" % cls.SRC_ROOT)
        src_root = cls.SRC_ROOT
    
    src_root = '/opt/au'
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

    egg_path = os.path.join(tmp_path, MODNAME + '-0.0.0-py2.7.egg')
    assert os.path.exists(egg_path)
    log.info("... done.  Egg at %s" % egg_path)
    return egg_path

    # This didn't work so well ...
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
    if cls.CONF is not None:
      builder = builder.config(conf=cls.CONF)
    if cls.CONF_KV is not None:
      for k, v in cls.CONF_KV.iteritems():
        builder = builder.config(k, v)
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

  @staticmethod
  def df_to_dstream(spark, df, *colnames):
    assert colnames, "Need at least one column"
    util.log.info(
      "Fetching chunks of %s based on columns %s ..." % (df, colnames))
    cols = df.select(*colnames)
    chunks_rdd = cols.rdd.mapPartitions(lambda rows: [[tuple(r) for r in rows]])
    chunks = chunks_rdd.collect()
    util.log.info("... got %s chunks / partitions ..." % len(chunks))
    
    rdds = []
    for i, chunk in enumerate(chunks):
      chunk_df = df
      print 'start'
      import pandas as pd
      qqq = spark.createDataFrame(pd.DataFrame(dict((col, colvals) for col, colvals in zip(colnames, zip(*chunk)))))
      print 'end'
      print 'start2'
      # qqq.show()
      joined = df.join(qqq.hint('broadcast'), on=[getattr(df, col) == getattr(qqq, col) for col in colnames], how='inner')
      print 'end2'
      # for col, colvals in zip(colnames, zip(*chunk)):
      #   chunk_df = chunk_df.filter(chunk_df[col].isin(*colvals))
      # rdds.append(chunk_df.rdd)
      rdd = joined.rdd
      # rdd = rdd.repartition(Spark.num_executors(spark))
      rdds.append(rdd)
      util.log.info(
        "... prepared %s / %s chunks: %s IDs in table %s" % (
          i + 1, len(chunks), len(chunk), chunk_df))
    
    from pyspark.streaming import StreamingContext
    ssc = StreamingContext(spark.sparkContext, 1)
    dstream = ssc.queueStream(rdds, oneAtATime=False)
    return ssc, dstream


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
      left_fields = set((f.name, f.dataType, f.nullable) for f in df.schema)
      right_fields = set((f.name, f.dataType, f.nullable) for f in df_other.schema)

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
    count = sc.parallelize(range(0, num_samples)).filter(inside).count()
    pi = 4 * float(count) / num_samples
    util.log.info("Pi estimate: %s" % pi)
    assert abs(pi - 3.14) < 0.1, "Spark program had an error?"

  @staticmethod
  def test_egg(spark):
    EXPECTED_EGG_NAME = 'au-0.0.0-py2.7.egg'

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
  
    util.log.info("Testing egg ...")
    
    sc = spark.sparkContext
    N = max(1, Spark.num_executors(spark)) # Try to test all executors
    rdd = sc.parallelize(range(N), numSlices=N)
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
    y = range(N)
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

class NumpyArrayUDT(types.UserDefinedType):
  """SQL User-Defined Type (UDT) for *opaque* numpy arrays.  Unlike Spark's
  DenseVector, this class preserves array shape.

  TODO: make an arbitrary pickleable wrapper ....
  """

  @classmethod
  def sqlType(cls):
    # NB: this is actually an instance method in practice O_O !
    return types.StructType([
      types.StructField("np_bytes", types.BinaryType(), False)
    ])

  @classmethod
  def module(cls):
    return 'au.spark'

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
    # buf = io.BytesIO()
    # np.save(buf, self.arr)
    #   # NB: do NOT use savez / gzip b/c we'll let Snappy compress things.
    # return buf.getvalue()

  @staticmethod
  def from_bytes(b):
    # arr = np.load(io.BytesIO(b), encoding='bytes')
    arr = pickle.loads(b)
    return NumpyArray(arr)

  def __repr__(self):
    return "NumpyArray:" + self.arr.__repr__()
  
  def __str__(self):
    return "NumpyArray" + self.arr.__str__()

  def __eq__(self, other):
    return isinstance(other, self.__class__) and other.arr == self.arr

