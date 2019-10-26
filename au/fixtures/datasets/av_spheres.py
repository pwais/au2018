import itertools
import os

import numpy as np
import pandas as pd

from au import util
from au.fixtures.datasets import av

def read_xyz_from_ply(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  # Header is 7 lines
  lines = lines[7:]
  def to_v(l):
    x, y, z = l.split()
    return float(x), float(y), float(z)
  xyz = np.array([to_v(l) for l in lines])
  return xyz

def read_K_from_path(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  itoks = itertools.chain.from_iterable(l.split() for l in lines)
  K = np.array([float(t) for t in itoks]).reshape((3, 3))
  return K

def read_RT_from_path(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  # lines 0 and 4 are newlines
  Rr1 = lines[1]
  Rr2 = lines[2]
  Rr3 = lines[3]
  R = np.array([
    [float(v) for v in Rr1.split()],
    [float(v) for v in Rr2.split()],
    [float(v) for v in Rr3.split()],
  ])
  T = np.array([float(v) for v in lines[5].split()])
  return R, T

class StampedDatumTable(av.StampedDatumTableBase):

  ## Subclass API

  @classmethod
  def table_root(cls):
    return '/outer_root/media/seagates-ext4/au_datas/av_spheres_datum'
  
  ## Public API

  @classmethod
  def get_all_uris(cls):
    ## select distinct (topic, timestamps) and return

  @classmethod
  def create_stamped_datum(cls, uri):
    # get df rows for (topic, timestamp)
    if uri.topic.startswith('lidar_'):
      # extrinsic
      # pose lookup
      # ply file
      pass
    elif uri.topic.startswith('camera_'):
      # extrinsic
      # intrinsic
      # pose lookup
      # jpeg
      pass
    elif uri.topic == 'cuboids':
      # ply file
      # pose lookup
      pass
    elif uri.topic == 'ego_pose':
      # pose lookup
      pass
    else:
      #maybe ignore for now


  ## Support

  @classmethod
  def artifact_df(cls):
    if not hasattr(cls, '_artifact_df'):
      def to_row(path):
      fname = os.path.basename(path)
      try:
        timestamp, topic, prop, ext = fname.split('.')
        return {
          'timestamp': int(timestamp),
          'topic': topic,
          'prop': prop,
          'path': path,
        }
      except Exception:
        return None

      rows = [to_row(p) for p in util.all_files_recursive(dirpath)]
      cls._artifact_df = pd.DataFrame([r for r in rows if r is not None])
    return cls._artifact_df




