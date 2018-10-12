import sys


## Utils

try:
  import pathlib
except ImportError:
  import pathlib2 as pathlib

def mkdir(path):
  pathlib.Path(path).mkdir(exist_ok=True)
  
## Logging
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
