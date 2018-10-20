import os

import imageio
import numpy as np

from au import conf
from au import util
from au.fixtures.dataset import ImageRow
from au.test import testconf

def test_imagerow_demo(monkeypatch):
  
  ## We can create an empty row; all members are strings
  row = ImageRow()
  assert row.dataset == ''
  assert row.split == ''
  assert row.uri == ''
  assert row.image_bytes == ''
    
  ## Invariants for a row lacking image data:
  assert not row.as_numpy().any(), "Image has no bytes"
  assert row.to_debug() is None, "No image bytes to write"
  
  ## We can use kwargs to init any desired attribute
  row = ImageRow(dataset='test1')
  assert row.dataset == 'test1'
  assert row.image_bytes == ''
  
  ## ImageRows and dicts are interchangeable (see kwargs demo above) 
  empty_row_as_dict = {
    'dataset': '',
    'split': '',
    'uri': '',
    'image_bytes': '',
    'label_type': '',
    'label_bytes': '',
  }
  assert ImageRow().to_dict() == empty_row_as_dict
  
  
  ## We can instantiate from a file on disk
  row = ImageRow.from_path(testconf.MNIST_TEST_IMG_PATH, dataset='test2')
  assert row.dataset == 'test2'
  assert len(row.image_bytes) == 250
  assert row.as_numpy().shape == (28, 28)
  
  ## We can dump a row to disk for quick inspection
  with monkeypatch.context() as m: 
    m.setattr(conf, 'AU_CACHE_TMP', testconf.TEST_TEMPDIR_ROOT)
    dest = row.to_debug(fname='ImageRowTest.png')
    assert os.path.exists(dest)
    expected = imageio.imread(testconf.MNIST_TEST_IMG_PATH)
    np.testing.assert_array_equal(row.as_numpy(), expected)
  
  
  
  ## The real warrant for ImageRow is so that we can store datasets of
  ## images using parquet and manipulate them easily using Spark and Pandas
  rows = ImageRow.rows_from_images_dir(
            conf.AU_IMAGENET_SAMPLE_IMGS_DIR,
            dataset='d')
  rows = list(rows)
  assert len(rows) > 4
  train = rows[:4]
  test = rows[4:]
  
  for r in train:
    r.split = 'train'
  for r in test:
    r.split = 'test'
  
  PQ_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'ImageRow_pq_demo')
  util.mkdir(PQ_TEMPDIR)
  util.rm_rf(PQ_TEMPDIR)
  
  ImageRow.write_to_parquet(train, PQ_TEMPDIR)
  ImageRow.write_to_parquet(test, PQ_TEMPDIR)
  
  
    
    

  