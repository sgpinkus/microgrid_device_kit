#!/usr/bin/env python3
''' Convenience script to just solve for outright cost minimized balanced flow - no market sim crap.
'''
from os.path import basename
import logging
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
import device_kit

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
np_printoptions = {
  'linewidth': 1e6,
  'threshold': 1e6,
  'formatter': {
    'float_kind': lambda v: '%+0.3f' % (v,),
    'int_kind': lambda v: '%+0.3f' % (v,),
  },
}
np.set_printoptions(**np_printoptions)


def main():
  parser = argparse.ArgumentParser(description='Run a power market simulation.')
  parser.add_argument('scenario', action='store',
    help='name of a python module containing scenario to run'
  )
  # parser.add_argument('data_dir', type=str,
  #   help='directory containing simulation results'
  # )
  args = parser.parse_args()

  (deviceset, meta, cb) = load_scenario(**vars(args))
  deviceset.sbounds = (0,0)
  (x, solve_meta) = device_kit.solve(deviceset, p=0) # Convenience convex solver.
  print(solve_meta.message)
  df = pd.DataFrame.from_dict(dict(deviceset.map(x)), orient='index')
  plot_bars(df, meta.get('title') if meta else None, cb)
  df.loc['total'] = df.sum()
  pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)
  print(df.sort_index())

def plot_bars(df, title, cb=None, aggregation_level=2):
  df_sums = df.groupby(lambda l: '.'.join(l.split('.')[0:aggregation_level])).sum()
  cm=plt.get_cmap('Paired', len(df_sums)+1)
  f = plt.figure(0)
  for (i, (device_label, r)) in enumerate(df_sums.iterrows()):
    plt.bar(range(0, len(df.columns)), r, label=device_label, width=1, edgecolor=cm.colors[i], fill=False, linewidth=1)
  plt.xlim(0, len(df.columns)+12)
  plt.legend()
  plt.title(title)
  plt.ylabel('Power (kWh)')
  plt.xlabel('Time (H)')
  if cb:
    cb('after-update', f)
  plt.savefig('run.png')
  plt.clf()


def load_scenario(scenario):
  def make_module_path(s):
    ''' Convert apossible filepath to a module-path. Does nothing it s is already a module-path '''
    return s.replace('.py', '').replace('/', '.').replace('..', '.').lstrip('.')
  scenario = importlib.import_module(make_module_path(scenario))
  meta = scenario.meta if hasattr(scenario, 'meta') else None
  cb = scenario.matplot_network_writer_hook if hasattr(scenario, 'matplot_network_writer_hook') else None
  return (scenario.make_deviceset(), meta, cb)


main()
