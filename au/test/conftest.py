"""Magic pytest file that conf(igures py)test"""

import pytest

def pytest_configure(config):
    config.addinivalue_line("markers",
        "slow: Tests that are slow and/or require subprocesses.")

def pytest_addoption(parser):
  parser.addoption(
    "--runslow", action="store_true", default=False, help="run slow tests")

def pytest_collection_modifyitems(config, items):
  if not config.getoption("--runslow"):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
      if "slow" in item.keywords:
        item.add_marker(skip_slow)
