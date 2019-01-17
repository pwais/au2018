from au import util
from au.test import testconf
from au.test import testutils

import os

def test_ichunked():
  
  def list_ichunked(*args):
    return list(util.ichunked(*args))
  
  for n in range(10):
    assert list_ichunked([], n) == []
  
  for n in range(10):
    assert list_ichunked([1], n) == [(1,)]
  
  assert list_ichunked((1, 2, 3, 4, 5), 1) == [(1,), (2,), (3,), (4,), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 2) == [(1, 2), (3, 4), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 3) == [(1, 2, 3), (4, 5)]
  assert list_ichunked((1, 2, 3, 4, 5), 4) == [(1, 2, 3, 4), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 5) == [(1, 2, 3, 4, 5)]
  assert list_ichunked((1, 2, 3, 4, 5), 6) == [(1, 2, 3, 4, 5)]
  
  assert list_ichunked('abcde', 4) == [('a', 'b', 'c', 'd'), ('e',)]

def test_thruput_observer():
  t1 = util.ThruputObserver()
  assert str(t1)
  
  t2 = util.ThruputObserver()
  
  import random
  import time
  MAX_WAIT = 0.1
  for _ in range(10):
    with t2.observe(n=1, num_bytes=1):
      time.sleep(random.random() * MAX_WAIT)
  
  assert str(t2)
  
  u = util.ThruputObserver.union((t1, t2))
  assert str(u) == str(t2)

def test_sys_info():
  info = util.get_sys_info()
  assert 'au' in info['filepath']

def test_archive_fliyweight_zip():
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'test_archive_flyweight_zip')
  util.cleandir(TEST_TEMPDIR)
  
  # Create the fixture
  ss = ['foo', 'bar', 'baz']
  
  fixture_path = os.path.join(TEST_TEMPDIR, 'test.zip')
  
  import zipfile
  with zipfile.ZipFile(fixture_path, mode='w') as z:
    for s in ss:
      z.writestr(s, s)
  
  fws = util.ArchiveFileFlyweight.fws_from(fixture_path)
  assert len(fws) == len(ss)
  datas = [fw.data for fw in fws]
  assert sorted(datas) == sorted(ss)

def test_ds_store_is_stupid():
  assert util.is_stupid_mac_file('/yay/.DS_Store')
  assert util.is_stupid_mac_file('.DS_Store')
  assert util.is_stupid_mac_file('._.DS_Store')

def test_gpu_pool(monkeypatch):
  rows = util.GPUInfo.get_infos()
  print rows