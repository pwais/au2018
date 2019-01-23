"""Magic pytest file that conf(igures py)test"""

import pytest

def pytest_addoption(parser):
  parser.addoption(
    "--runslow", action="store_true", default=False, help="run slow tests")
  parser.addoption(
    "--runhighmem", action="store_true", default=False, help="run high-memory tests")


def pytest_collection_modifyitems(config, items):
  if not config.getoption("--runslow"):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
      if "slow" in item.keywords:
        item.add_marker(skip_slow)
  if not config.getoption("--runhighmem"):
    skip_highmem = pytest.mark.skip(reason="need --runslow and --runhighmem option to run")
    for item in items:
      if "highmem" in item.keywords:
        item.add_marker(skip_highmem)
