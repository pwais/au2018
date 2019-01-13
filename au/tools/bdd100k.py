#!/usr/bin/env python

from au.fixtures.datasets import bdd100k

if __name__ == '__main__':
  # NB: We can't embed this into the bdd100k module due to a bug in
  # Cloudpickle: https://github.com/cloudpipe/cloudpickle/issues/225
  bdd100k.Fixtures.run_import()