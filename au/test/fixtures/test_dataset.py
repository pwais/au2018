import os
import unittest

import imageio
import numpy as np

from au import conf
from au import util
from au.fixtures.dataset import FillNormalized
from au.fixtures.dataset import ImageRow
from au.fixtures.dataset import ImageTable
from au.test import testconf
from au.test import testutils



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
    'label': '',
    'attrs': '',
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
  assert len(rows) >= 6
  
#   # Rows can have labels of various types, too
#   rows[0].label = 'fake_label'
#   rows[1].label = 4 # fake label
#   rows[2].label = [1, 2]
#   rows[3].label = np.array([1, 2])
#   rows[4].label = {'key': 'value'}
  
  train = rows[:4]
  test = rows[4:]
  
  for r in train:
    r.split = 'train'
  for r in test:
    r.split = 'test'
  
  PQ_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'ImageRow_pq_demo')
  util.cleandir(PQ_TEMPDIR)
  
  ImageRow.write_to_parquet(train, PQ_TEMPDIR)
  ImageRow.write_to_parquet(test, PQ_TEMPDIR)
  
  # pyarrow's parquet writer should have created some nice partitioned
  # directories
  for split in ('train', 'test'):
    d = os.path.join(PQ_TEMPDIR, 'dataset=d', 'split=%s' % split) 
    assert os.path.exists(d)
  
  # Now try reading it back
  import pandas as pd
  import pyarrow as pa
  import pyarrow.parquet as pq
  
  pa_table = pq.read_table(PQ_TEMPDIR)
  df = pa_table.to_pandas()
  
  # Did we read back the correct images?
  assert set(df['uri']) == set(r.uri for r in rows)
  
  # Are the splits correct?
  expected_uri_to_split = {}
  expected_uri_to_split.update((r.uri, r.split) for r in train)
  expected_uri_to_split.update((r.uri, r.split) for r in test)
  df_rows = df.loc[:,['uri', 'split']].to_dict(orient='records')
  actual_uri_to_split = dict((d['uri'], d['split']) for d in df_rows)
  assert actual_uri_to_split == expected_uri_to_split
  
  # Check the table contents; we should see the image bytes are identical
  # to the files
  for decoded_row in ImageRow.from_pandas(df):
    assert os.path.exists(decoded_row.uri)
    expected_bytes = open(decoded_row.uri, 'rb').read()
    assert decoded_row.image_bytes == expected_bytes


  ## We can also dump sets of rows as PNGs partitioned by dataset 
  with monkeypatch.context() as m: 
    m.setattr(conf, 'AU_DATA_CACHE', testconf.TEST_TEMPDIR_ROOT)
    ImageRow.write_to_pngs(rows)

    def expect_file(relpath, uri_to_expected):
      path = os.path.join(testconf.TEST_TEMPDIR_ROOT, relpath) 
      assert os.path.exists(path)
      expected_bytes = open(uri_to_expected, 'rb').read()
      actual_bytes = open(path, 'rb').read()
      assert expected_bytes == actual_bytes

    expect_file(os.path.join('d/train', train[0].fname()), train[0].uri)
    expect_file(os.path.join('d/test', test[0].fname()), test[0].uri)

def test_imagetable_demo(monkeypatch):
  
  TABLE_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'ImageTable_pq_demo')
  util.cleandir(TABLE_TEMPDIR)
  
  with monkeypatch.context() as m: 
    m.setattr(conf, 'AU_TABLE_CACHE', TABLE_TEMPDIR)
  
    ImageTable.setup()
  
    test_img_path = os.path.join(
                        conf.AU_IMAGENET_SAMPLE_IMGS_DIR,
                        '2929331372_398d58807e.jpg')
    rows = ImageTable.get_rows_by_uris((test_img_path, 'not_in_table'))
    assert len(rows) == 1
    row = rows[0]
    
    expected_bytes = open(test_img_path, 'rb').read()
    assert row.image_bytes == expected_bytes
    assert row.label == 'coffee'
    
    assert len(list(ImageTable.iter_all_rows())) == 6

class TestFillNormalized(unittest.TestCase):
  def test_identity(self):
    f = FillNormalized()
    row = ImageRow.from_path(testconf.MNIST_TEST_IMG_PATH)
    row = f(row)
    np.testing.assert_array_equal(row.as_numpy(), row.attrs['normalized'])
  
  def test_resize(self):
    f = FillNormalized(target_hw=(10, 14))
    row = ImageRow.from_path(testconf.MNIST_TEST_IMG_PATH)
    assert row.as_numpy().shape == (28, 28)
    
    row = f(row)
    assert row.attrs['normalized'].shape == (10, 14)
  
  def test_nchan(self):
    f = FillNormalized(target_nchan=1)

    test_img_path = os.path.join(
                        conf.AU_IMAGENET_SAMPLE_IMGS_DIR,
                        '2929331372_398d58807e.jpg')

    row = ImageRow.from_path(test_img_path)
    assert row.as_numpy().shape == (375, 500, 3)
    
    row = f(row)
    assert row.attrs['normalized'].shape == (375, 500, 1)

  def test_transform(self):
    def normalize(x):
      return x - 10
    f = FillNormalized(norm_func=normalize)
    row = ImageRow.from_path(testconf.MNIST_TEST_IMG_PATH)
    row = f(row)
    expected = row.as_numpy() - 10
    np.testing.assert_array_equal(expected, row.attrs['normalized'])
  
def test_create_video():
  fps = 29.97
  n = 50
  h = 64
  w = 128
  vid_bytes = testutils.create_video(n=n, h=h, w=w, format='mov', fps=fps)
  
  VID_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'test_create_video')
  util.cleandir(VID_TEMPDIR)
  path = os.path.join(VID_TEMPDIR, 'test_video.mov')
  with open(path, 'wc') as f:
    f.write(vid_bytes)
  
  import imageio
  reader = imageio.get_reader(path)
  meta = reader.get_meta_data()
  assert meta['fps'] == fps
  assert meta['nframes'] == n
