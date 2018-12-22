"""A module with Spark-related utilities"""

from au import conf
from au import util

import os
import pickle
from contextlib import contextmanager

import numpy as np

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
    # os.environ['PYSPARK_SUBMIT_ARGS'] = '--py-files %s' % egg_path

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

    spark.sparkContext.addPyFile(cls.egg_path())
    return spark
  
  @classmethod
  @contextmanager
  def sess(cls):
    spark = cls.getOrCreate()
    yield spark

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



## Spark UDTs
# These utils are based upon Spark's DenseVector:
# https://github.com/apache/spark/blob/044b33b2ed2d423d798f2a632fab110c46f41567/python/pyspark/mllib/linalg/__init__.py#L239
# https://apache.googlesource.com/spark/+/refs/heads/master/python/pyspark/sql/tests.py#119
# Sadly they don't have a UDT for tensors... not even in Tensorframes
# https://github.com/databricks/tensorframes   o_O

class NumpyArrayUDT(types.UserDefinedType):
  """SQL User-Defined Type (UDT) for *opaque* numpy arrays.  Unlike Spark's
  DenseVector, this class preserves array shape.
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

