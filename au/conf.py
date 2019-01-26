import os

AU_ROOT = os.environ.get('AU_ROOT', '/opt/au')

AU_CACHE = os.environ.get('AU_CACHE', os.path.join(AU_ROOT, 'cache'))

AU_CACHE_TMP = os.environ.get('AU_CACHE_TMP', os.path.join(AU_CACHE, 'tmp'))

# AU_IMAGE_TABLE_ROOT = os.environ.get(
#                     'AU_IMAGE_TABLE_ROOT',
#                     os.path.join(AU_CACHE, 'image_table'))

AU_DATA_CACHE = os.environ.get(
                    'AU_DATA_CACHE',
                    os.path.join(AU_CACHE, 'data'))

AU_TABLE_CACHE = os.environ.get(
                    'AU_TABLE_CACHE',
                    os.path.join(AU_CACHE, 'tables'))

AU_MODEL_CACHE = os.environ.get(
                    'AU_MODEL_CACHE',
                    os.path.join(AU_CACHE, 'models'))

AU_TENSORBOARD_DIR = os.environ.get(
                    'AU_TENSORBOARD_DIR',
                    os.path.join(AU_CACHE, 'tensorboard'))

AU_IMAGES_SAMPLE = os.environ.get(
                'AU_IMAGENET_SAMPLE',
                os.path.join(AU_ROOT, 'au', 'fixtures', 'images'))

AU_IMAGENET_SAMPLE_IMGS_DIR = os.environ.get(
                'AU_IMAGENET_SAMPLE_IMGS_DIR',
                os.path.join(AU_IMAGES_SAMPLE, 'imagenet'))

AU_IMAGENET_SAMPLE_LABELS_PATH = os.environ.get(
                'AU_IMAGENET_SAMPLE_LABELS_PATH',
                os.path.join(AU_IMAGES_SAMPLE, 'imagenet_fname_to_label.json'))

AU_DY_TEST_FIXTURES = os.environ.get(
                    'AU_DY_TEST_FIXTURES',
                    os.path.join(AU_CACHE, 'test'))

# Assume kubectl proxy is active
AU_K8S_SPARK_MASTER = os.environ.get(
                        'AU_K8S_SPARK_MASTER',
                        'k8s://http://127.0.0.1:8001')
AU_SPARK_WORKER_IMAGE = os.environ.get(
                            'AU_SPARK_WORKER_IMAGE',
                            'au2018/spark')