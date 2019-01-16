import itertools
import os
import shutil
import subprocess
import sys
import threading
import time

from contextlib import contextmanager

### Logging
_LOGS = {}
def create_log(name='au'):
  global _LOGS
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
  
  def __init__(self, name='', log_on_del=False, only_stats=None):
    self.n = 0
    self.num_bytes = 0
    self.ts = []
    self.name = name
    self.log_on_del = log_on_del
    self.only_stats = only_stats or []
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
      u += t
    return u
  
  def __iadd__(self, other):
    self.n += other.n
    self.num_bytes += other.num_bytes
    self.ts.extend(other.ts)
    return self

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
    if self.only_stats:
      stats = tuple(
        (name, value)
        for name, value in stats
        if name in self.only_stats
      )

    summary = tabulate.tabulate(stats)
    if self.name:
      summary = self.name + '\n' + summary
    return summary
  
  def __del__(self):
    if self.log_on_del:
      log = create_log()
      log.info('\n' + str(self) + '\n')

@contextmanager
def quiet():
  old_stdout = sys.stdout
  old_stderr = sys.stderr
  f = open(os.devnull, 'w')
  new_stdout = sys.stdout = f
  new_stderr = sys.stderr = f
  try:
    yield new_stdout, new_stderr
  finally:
    new_stdout.seek(0)
    new_stderr.seek(0)
    sys.stdout = old_stdout
    sys.stderr = old_stderr


@contextmanager
def imageio_ignore_warnings():
  # Imageio needs some fix: https://github.com/imageio/imageio/issues/376
  import imageio.core.util
  def silence_imageio_warning(*args, **kwargs):
    pass
  old = imageio.core.util._precision_warn
  imageio.core.util._precision_warn = silence_imageio_warning
  try:
    yield
  finally:
    imageio.core.util._precision_warn = old


def run_cmd(cmd, collect=False):
  log = create_log()
  
  cmd = cmd.replace('\n', '').strip()
  log.info("Running %s ..." % cmd)
  if collect:
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
  else:
    subprocess.check_call(cmd, shell=True)
    out = None
  log.info("... done with %s " % cmd)
  return out


def get_non_loopback_iface():
  # https://stackoverflow.com/a/1267524
  import socket
  non_loopbacks = [
    ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")
  ]
  if non_loopbacks:
    return non_loopbacks[0]

  # Get an iface that can connect to Google DNS ...
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  ifrace = s.getsockname()[0]
  s.close()
  return iface


_SYS_INFO_LOCK = threading.Lock()
def get_sys_info():
  global _SYS_INFO_LOCK
  log = create_log()

  log.info("Listing system info ...")

  info = {}
  info['filepath'] = os.path.abspath(__file__)
  info['PYTHONPATH'] = ':'.join(sys.path)
  
  @contextmanager
  def atomic_ignore_exceptions():
    with _SYS_INFO_LOCK:
      try:
        yield
      except Exception:
        pass

  def safe_cmd(cmd):
    with atomic_ignore_exceptions():
      return run_cmd(cmd, collect=True) or ''

  info['nvidia_smi'] = safe_cmd('nvidia-smi')
  info['cpuinfo'] = safe_cmd('cat /proc/cpuinfo')
  info['disk_free'] = safe_cmd('df -h')
  info['ifconfig'] = safe_cmd('ifconfig')

  import socket
  info['hostname'] = socket.gethostname()
  info['host'] = get_non_loopback_iface()

  import multiprocessing
  info['n_cpus'] = multiprocessing.cpu_count()
  
  log.info("... got all system info.")

  return info

### ArchiveFileFlyweight

class _IArchive(object):
    __SLOTS__ = ('archive_path', 'thread_data')
    
    def __init__(self, path):
      self.archive_path = path
      self.thread_data = threading.local()

    def _setup(self, archive_path):
      pass

    @classmethod
    def list_names(cls, archive_path):
      return []

    def _archive_get(self, name):
      raise KeyError("Interface stores no data")

    def get(self, name):
      self._setup(self.archive_path)
      return self._archive_get(name)

class _ZipArchive(_IArchive):
  
  def _setup(self, archive_path):
    if not hasattr(self.thread_data, 'zipfile'):
      import zipfile
      self.thread_data.zipfile = zipfile.ZipFile(archive_path)
  
  def _archive_get(self, name):
    return self.thread_data.zipfile.read(name)

  @classmethod
  def list_names(cls, archive_path):
    import zipfile
    return zipfile.ZipFile(archive_path).namelist()

class ArchiveFileFlyweight(object):

  __SLOTS__ = ('name', 'archive')

  def __init__(self, name='', archive=None):
    self.name = name
    self.archive = archive

  @staticmethod
  def fws_from(archive_path):
    if archive_path.endswith('zip'):
        archive = _ZipArchive(archive_path)
        names = _ZipArchive.list_names(archive_path)
        return [
          ArchiveFileFlyweight(name=name, archive=archive)
          for name in names
        ]
    else:
      raise ValueError("Don't know how to read %s" % archive_path)

  @property
  def data(self):
    return self.archive.get(self.name)



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

def all_files_recursive(root_dir):
  for path in pathlib.Path(root_dir).glob('**/*'):
    path = str(path) # pathlib uses PosixPath thingies ...
    if os.path.isfile(path):
      yield path

def cleandir(path):
  mkdir(path)
  rm_rf(path)
  mkdir(path)

def missing_or_empty(path):
  if not os.path.exists(path):
    return True
  else:
    for p in all_files_recursive(path):
      return False
    return True

def is_stupid_mac_file(path):
  fname = os.path.basename(path)
  return fname.startswith('._') or fname in ('.DS_Store',)

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

def give_me_frozen_graph(
          checkpoint,
          nodes=None,
          blacklist=None,
          base_graph=None,
          sess=None,
          saver=None):
  """
  Tensorflow has several ways to load checkpoints / graph artifacts.
  It's impossible to know if some API is stable or if tomorrow somebody
  will invent something new and break everything (e.g. TF Eager).  
  Sam Abrahams wrote a book on Tensorflow
  ( https://www.amazon.com/TensorFlow-Machine-Intelligence-hands--introduction-ebook/dp/B01IZ43JV4/ )
  and one time couldn't tell me definitively which API to use.  What's more is
  that freeze_graph.py is an optional script instead of a module in
  Tensorflow.  Chaos!!

  So, based upon spark-dl's `strip_and_freeze_until()`
  ( https://github.com/databricks/spark-deep-learning/blob/4daa1179f498df4627310afea291133539ce7001/python/sparkdl/graph/utils.py#L199 ),
  here's a utility for getting a frozen, serializable, pyspark-friendly
  graph from a checkpoint artifact metagraph thingy.
  """

  def op_name(v):
    name = v
    if hasattr(v, 'name'):
      name = v.name
    if ':' not in name:
      return name
    toks = name.split(':')
    assert len(toks) <= 2, (toks, v, name)
    return toks[0]

  import tensorflow as tf
  graph = base_graph or tf.Graph()
  if nodes:
    ops = [graph.get_operation_by_name(op_name(n)) for n in nodes]
  else:
    ops = graph.get_operations()
  # if blacklist:
  #   for n in blacklist:
  #     ops.remove(graph.get_operation_by_name(op_name(n)))

  with graph.as_default():
    with (sess or tf_create_session()) as sess:

      saver = saver or tf.Saver()
      log.info("Reading from checkpoint %s ..." % checkpoint)
      saver.restore(sess, checkpoint)
      log.info("... done.")

      gdef_frozen = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(add_shapes=True)
        [op.name for op in ops])
        # variable_names_blacklist=blacklist)
  
  g = tf.Graph()
  with g.as_default():
    tf.import_graph_def(gdef_frozen, name='')
  return g
  
