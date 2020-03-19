from au import util
from au.test import testconf
from au.test import testutils

import os
import unittest

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

def test_row_of_constants():
  as_row = util.as_row_of_constants

  # Don't row-ify junk
  assert as_row(5) == {}
  assert as_row('moof') == {}
  assert as_row(dict) == {}
  assert as_row([]) == {}

  # Row-ify class and instance constants
  class Foo(object):
    nope = 1
    _NOPE = 2
    YES = 3
    YES2 = {'bar': 'baz'}
    def __init__(self):
      self.YUP = 4
      self._nope = 5
  
  assert as_row(Foo) == {
                      'YES': 3,
                      'YES2': {'bar': 'baz'},
                    }
  assert as_row(Foo()) == {
                      'YES': 3,
                      'YES2': {'bar': 'baz'},
                      'YUP': 4,
                    }
  
  # Don't row-ify containers of stuff
  assert as_row([Foo()]) == {}

  # Do recursively row-ify
  class Bar(object):
    FOO_CLS = Foo
    FOO_INST = Foo()
    def __init__(self):
      self.MY_FOO = Foo()

  assert as_row(Bar) == {
                      'FOO_CLS': 'Foo',
                      'FOO_CLS_YES': 3,
                      'FOO_CLS_YES2': {'bar': 'baz'},

                      'FOO_INST': 'Foo',
                      'FOO_INST_YES': 3,
                      'FOO_INST_YES2': {'bar': 'baz'},
                      'FOO_INST_YUP': 4,
                    }
  assert as_row(Bar()) == {
                      'FOO_CLS': 'Foo',
                      'FOO_CLS_YES': 3,
                      'FOO_CLS_YES2': {'bar': 'baz'},

                      'FOO_INST': 'Foo',
                      'FOO_INST_YES': 3,
                      'FOO_INST_YES2': {'bar': 'baz'},
                      'FOO_INST_YUP': 4,

                      'MY_FOO': 'Foo',
                      'MY_FOO_YES': 3,
                      'MY_FOO_YES2': {'bar': 'baz'},
                      'MY_FOO_YUP': 4,
                    }



class TextProxy(unittest.TestCase):
  class Foo(object):
    def __init__(self):
      self.x = 1

  def test_raw_obj(self):
    global_foo = TextProxy.Foo()
    global_foo.x = 0

    foo = TextProxy.Foo()
    assert foo.x == 1
    del foo
    global_foo.x == 0

  def test_wrapped_with_custom_dtor(self):
    global_foo = TextProxy.Foo()
    global_foo.x = 0

    class FooProxy(util.Proxy):
      def _on_delete(self):
        global_foo.x = 2
  
    foo = FooProxy(TextProxy.Foo())
    assert foo.x == 1
    del foo
    global_foo.x == 2



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



NVIDIA_SMI_MOCK_OUTPUT = (
"""index, name, utilization.memory [%], name, memory.total [MiB], memory.free [MiB], memory.used [MiB]
0, GeForce GTX 1060 with Max-Q Design, 2, GeForce GTX 1060 with Max-Q Design, 6072, 5796, 276
""")

NVIDIA_SMI_MOCK_OUTPUT_8_K80s = (
"""index, name, utilization.memory [%], name, memory.total [MiB], memory.free [MiB], memory.used [MiB]
0, Tesla K80, 0, Tesla K80, 11441, 11441, 0
1, Tesla K80, 0, Tesla K80, 11441, 11441, 0
2, Tesla K80, 0, Tesla K80, 11441, 11441, 0
3, Tesla K80, 0, Tesla K80, 11441, 11441, 0
4, Tesla K80, 0, Tesla K80, 11441, 11441, 0
5, Tesla K80, 0, Tesla K80, 11441, 11441, 0
6, Tesla K80, 0, Tesla K80, 11441, 11441, 0
7, Tesla K80, 5, Tesla K80, 11441, 11441, 0
"""
)

def test_gpu_get_infos(monkeypatch):
  # Test Smoke
  rows = util.GPUInfo.get_infos()
  
  # Test parsing a fixture
  def mock_run_cmd(*args, **kwargs):
    return NVIDIA_SMI_MOCK_OUTPUT
  monkeypatch.setattr(util, 'run_cmd', mock_run_cmd)
  rows = util.GPUInfo.get_infos()
  
  expected = util.GPUInfo()
  expected.index = 0
  expected.name = 'GeForce GTX 1060 with Max-Q Design'
  expected.mem_util_frac = 0.02
  expected.mem_free = 5796000000
  expected.mem_used = 276000000
  expected.mem_total = 6072000000
  assert rows == [expected]

def test_gpu_pool_no_gpus(monkeypatch):
  environ = dict(os.environ)
  environ['CUDA_VISIBLE_DEVICES'] = ''
  monkeypatch.setattr(os, 'environ', environ)

  pool = util.GPUPool()

  # We should never get any handles
  for _ in range(10):
    hs = pool.get_free_gpus()
    assert len(hs) == 0

def test_gpu_pool_one_gpu(monkeypatch):
  # Pretend we have one GPU
  def mock_run_cmd(*args, **kwargs):
    return NVIDIA_SMI_MOCK_OUTPUT
  monkeypatch.setattr(util, 'run_cmd', mock_run_cmd)

  pool = util.GPUPool()
  
  # We can get one GPU
  hs = pool.get_free_gpus()
  assert len(hs) == 1
  assert hs[0].index == 0

  # Subsequent fetches will fail
  for _ in range(10):
    h2 = pool.get_free_gpus()
    assert len(h2) == 0
  
  # Free the GPU
  del hs

  # Now we can get it again
  hs3 = pool.get_free_gpus()
  assert len(hs3) == 1
  assert hs3[0].index == 0

def test_gpu_pool_eight_k80s(monkeypatch):
  # Pretend we have one GPU
  def mock_run_cmd(*args, **kwargs):
    return NVIDIA_SMI_MOCK_OUTPUT_8_K80s
  monkeypatch.setattr(util, 'run_cmd', mock_run_cmd)

  pool = util.GPUPool()
  
  # We can get one GPU
  hs = pool.get_free_gpus()
  assert len(hs) == 1
  assert hs[0].index == 0

  # Subsequent fetches will fail
  # for _ in range(10):
  #   h2 = pool.get_free_gpus()
  #   assert len(h2) == 0
  
  # Free the GPU
  del hs

  # Now we can get it again
  hs3 = pool.get_free_gpus()
  assert len(hs3) == 1
  assert hs3[0].index == 1
