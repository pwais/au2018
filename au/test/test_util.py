from au import util

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

def test_spark():
  with util.Spark.sess() as spark:
    util.Spark.test_pi(spark)
