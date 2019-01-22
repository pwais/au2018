#!/usr/bin/env python

from au.fixtures.datasets import mscoco

if __name__ == '__main__':
  # NB: We can't embed this into the mscoco module due to a bug in
  # Cloudpickle: https://github.com/cloudpipe/cloudpickle/issues/225
  mscoco.Fixtures.run_import()
