import os

AU_ROOT = os.environ.get('AU_ROOT', '/opt/au')

AU_CACHE = os.environ.get('AU_CACHE', os.path.join(AU_ROOT, 'cache'))

AU_DATA_CACHE = os.environ.get(
                    'AU_DATA_CACHE',
                    os.path.join(AU_CACHE, 'data'))

AU_MODEL_CACHE = os.environ.get(
                    'AU_MODEL_CACHE',
                    os.path.join(AU_CACHE, 'models'))

AU_TENSORBOARD_DIR = os.environ.get(
                    'AU_TENSORBOARD_DIR',
                    os.path.join(AU_CACHE, 'tensorboard'))
