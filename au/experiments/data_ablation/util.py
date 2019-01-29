from au import conf

import os

def experiment_basedir(exp_family_name):
  return os.path.join(
    conf.AU_EXPERIMENTS_DIR,
    'data_ablation',
    exp_family_name)