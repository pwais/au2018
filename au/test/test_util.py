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
