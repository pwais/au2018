import os
import shutil
import sys

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


  