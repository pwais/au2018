import os

from au import conf

TEST_TEMPDIR_ROOT = '/tmp/au_test'
MNIST_TEST_IMG_PATH = os.path.join(conf.AU_ROOT, 'au/test/mnist_test_img.png')

def use_tempdir(monkeypatch, test_tempdir):
  from au import util
  monkeypatch.setattr(conf, 'AU_CACHE', test_tempdir)
  monkeypatch.setattr(conf, 'AU_CACHE_TMP', os.path.join(test_tempdir, 'tmp'))
  monkeypatch.setattr(conf, 'AU_DATA_CACHE', os.path.join(test_tempdir, 'data'))
  monkeypatch.setattr(conf, 'AU_TABLE_CACHE', os.path.join(test_tempdir, 'tables'))
  monkeypatch.setattr(conf, 'AU_MODEL_CACHE', os.path.join(test_tempdir, 'models'))
  monkeypatch.setattr(conf, 'AU_TENSORBOARD_DIR', os.path.join(test_tempdir, 'tensorboard'))
  
  util.mkdir(test_tempdir)
  util.rm_rf(test_tempdir)
