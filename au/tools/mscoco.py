#!/usr/bin/env python3

from au.fixtures.datasets import mscoco

if __name__ == '__main__':
  from au.fixtures.datasets import auargoverse as av
  av.ImageAnnoTable.setup()
  av.CroppedObjectImageTable.setup()
  # av.AnnoReports.create_reports()

  # # NB: We can't embed this into the mscoco module due to a bug in
  # # Cloudpickle: https://github.com/cloudpipe/cloudpickle/issues/225
  # mscoco.Fixtures.run_import()
