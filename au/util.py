import itertools
import os
import shutil
import sys
import time

from contextlib import contextmanager

### Logging
_LOGS = {}
def create_log(name='au'):
  if name not in _LOGS:
    import logging
    LOG_FORMAT = "%(asctime)s\t%(name)-4s %(process)d : %(message)s"
    log = logging.getLogger("au")
    log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    log.addHandler(console_handler)
    _LOGS[name] = log
  return _LOGS[name]
log = create_log()


### Pythonisms

def ichunked(seq, n):
  """Generate chunks of `seq` of size (at most) `n`.  More efficient
  and less junk than itertools recipes version using izip_longest...
  """
  n = max(1, n)
  seq = iter(seq)
  while True:
    chunk = tuple(itertools.islice(seq, n))
    if chunk:
      yield chunk
    else:
      break

class ThruputObserver(object):
  
  def __init__(self, name='', log_on_del=False):
    self.n = 0
    self.num_bytes = 0
    self.ts = []
    self.name = name
    self.log_on_del = log_on_del
    self._start = None
  
  @contextmanager
  def observe(self, n=0, num_bytes=0):
    start = time.time()
    yield
    end = time.time()
    
    self.n += n
    self.num_bytes += num_bytes
    self.ts.append(end - start)
  
  def start_block(self):
    self._start = time.time()
  
  def update_tallies(self, n=0, num_bytes=0):
    self.n += n
    self.num_bytes += num_bytes
  
  def stop_block(self, n=0, num_bytes=0):
    end = time.time()
    self.n += n
    self.num_bytes += num_bytes
    self.ts.append(end - self._start)
    self._start = None
  
  @staticmethod
  def union(thruputs):
    u = ThruputObserver()
    for t in thruputs:
      u.n += t.n
      u.num_bytes += t.num_bytes
      u.ts.extend(t.ts)
    return u
  
  def __str__(self):
    import numpy as np
    import tabulate

    gbytes = 1e-9 * self.num_bytes
    total_time = sum(self.ts) or float('nan')

    stats = (
      ('N thru', self.n),
      ('N chunks', len(self.ts)),
      ('total time (sec)', total_time),
      ('total GBytes', gbytes),
      ('overall GBytes/sec', gbytes / total_time if total_time else '-'),
      ('Hz', float(self.n) / total_time if total_time else '-'),
      ('Latency (per chunk)', ''),
      ('avg (sec)', np.mean(self.ts) if self.ts else '-'),
      ('p50 (sec)', np.percentile(self.ts, 50) if self.ts else '-'),
      ('p95 (sec)', np.percentile(self.ts, 95) if self.ts else '-'),
      ('p99 (sec)', np.percentile(self.ts, 99) if self.ts else '-'),
    )
    summary = tabulate.tabulate(stats)
    if self.name:
      summary = self.name + '\n' + summary
    return summary
  
  def __del__(self):
    if self.log_on_del:
      log = create_log()
      log.info('\n' + str(self) + '\n')
    

### I/O

try:
  import pathlib
except ImportError:
  import pathlib2 as pathlib
  # TODO use six?

def mkdir(path):
  pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def rm_rf(path):
  shutil.rmtree(path)

def download(uri, dest):
  """Fetch `uri`, which is a file or archive, and put in `dest`, which
  is either a destination file path or destination directory."""
  
  # Import urllib
  try:
    import urllib.error as urlliberror
    import urllib.request as urllib
    HTTPError = urlliberror.HTTPError
    URLError = urlliberror.URLError
  except ImportError:
    import urllib2 as urllib
    HTTPError = urllib.HTTPError
    URLError = urllib.URLError
  
  import tempfile
  
  import patoolib
 
  if os.path.exists(dest):
    return
  
  def show_progress(percentage):
    COLS = 70
    full = int(COLS * percentage / 100)
    bar = full * "#" + (COLS - full) * " "
    sys.stdout.write(u"\u001b[1000D[" + bar + "] " + str(percentage) + "%")
    sys.stdout.flush()
  
  fname = os.path.split(uri)[-1]
  tempdest = tempfile.NamedTemporaryFile(suffix='_' + fname)
  try:
    log.info("Fetching %s ..." % uri)
    response = urllib.urlopen(uri)
    size = int(response.info().get('Content-Length').strip())
    log.info("... downloading %s MB ..." % (float(size) * 1e-6))
    chunk = min(size, 8192)
    downloaded = 0
    while True:
      data = response.read(chunk)
      if not data:
        break
      tempdest.write(data)
      downloaded += len(data)
      show_progress(100 * downloaded / size)
    sys.stdout.write("")
    sys.stdout.flush()
    log.info("... fetched!")
  except HTTPError as e:
    raise Exception("[HTTP Error] {code}: {reason}."
                        .format(code=e.code, reason=e.reason))
  except URLError as e:
    raise Exception("[URL Error] {reason}.".format(reason=e.reason))
  
  tempdest.flush()
  
  try:
    # Is it an archive? expand!
    mkdir(dest)
    patoolib.extract_archive(tempdest.name, outdir=dest)
    log.info("Extracted archive.")
  except Exception:
    # Just move the file
    shutil.move(tempdest.name, dest)
  log.info("Downloaded to %s" % dest)


### Tensorflow

def tf_create_session_config(extra_opts=None):
  extra_opts = extra_opts or {}
  
  import tensorflow as tf
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False
  
  # Let the system pick number of threads
#   config.intra_op_parallelism_threads = 0
#   config.inter_op_parallelism_threads = 0
  
  for k, v in extra_opts.iteritems():
    setattr(config, k, v)
  return config

def tf_create_session(config=None):
  config = config or tf_create_session_config()

  import tensorflow as tf
  sess = tf.Session(config=config)
  return sess

@contextmanager
def tf_data_session(dataset, sess=None, config=None):
  import tensorflow as tf

  # Must declare these before the graph gets finalized below
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  
  # Silly way to iterate over a tf.Dataset
  # https://stackoverflow.com/a/47917849
  config = config or tf_create_session_config()
  sess = sess or tf.train.MonitoredTrainingSession(config=config)  
  with sess as sess:
    def iter_dataset():
      while not sess.should_stop():
        yield sess.run(next_element)
    yield sess, iter_dataset


### Spark

class Spark(object):
  MASTER = None
  CONF = None
  CONF_KV = None
  HIVE = False
  
  @classmethod
  def _setup(cls):
    log.info("Finding spark ...")
    import findspark
    findspark.init()
    log.info("... found!")

    # TODO set up egg to ship to workers ...
  
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
    return builder.getOrCreate()
  
  @classmethod
  @contextmanager
  def sess(cls):
    spark = cls.getOrCreate()
    yield spark

  @staticmethod
  def test_pi(spark):
    log.info("Running PI ...")
    sc = spark.sparkContext
    num_samples = 1000000
    def inside(p):
      import random
      x, y = random.random(), random.random()
      return x*x + y*y < 1
    count = sc.parallelize(range(0, num_samples)).filter(inside).count()
    pi = 4 * float(count) / num_samples
    log.info("Pi estimate: %s" % pi)
    assert abs(pi - 3.14) < 0.1, "Spark program had an error?"
