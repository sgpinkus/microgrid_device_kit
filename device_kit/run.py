#!/usr/bin/env python3
''' Convenience script to just solve for outright cost minimized balanced flow - no market sim crap.
'''
from os.path import splitext
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import device_kit
from device_kit.loaders import builder_loader, module_loader


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np_printoptions = {
  'linewidth': 1e6,
  'threshold': 1e6,
  'formatter': {
    'float_kind': lambda v: '%+0.3f' % (v,),
    'int_kind': lambda v: '%+0.3f' % (v,),
  },
}
np.set_printoptions(**np_printoptions)
pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)

def main():
  parser = argparse.ArgumentParser(
    description='Run a device_kit scenario and print result to stdout',
    prog='device_kit'
  )
  parser.add_argument('filename', action='store',
    help='name of a python module file containing scenario to run'
  )
  # parser.add_argument('data_dir', type=str,
  #   help='directory containing simulation results'
  # )
  parser.add_argument('--loader', '-l', action='store', default='module', type=str,
    help='loader to use to load scenario file')
  args = parser.parse_args()
  loader = globals()[f'{args.loader}_loader']
  (deviceset, meta, cb) = loader.load_file(args.filename)
  deviceset.sbounds = (0,0)
  print(str(deviceset))
  (x, solve_meta) = device_kit.solve(deviceset, solver_options={'ftol': 1e-6 }, cb=Cb()) # Convenience convex solver.
  print(solve_meta.message)
  df = pd.DataFrame.from_dict(dict(deviceset.map(x)), orient='index')
  plot_bars(df, meta.get('title') if meta else None, splitext(args.filename)[0] + '.png', cb)
  df.loc['total'] = df.sum()
  df['cumulative'] = df.sum(axis=1)
  print(df.sort_index())
  df.to_csv(splitext(args.filename)[0] + '.csv', float_format='%.3f')


def plot_bars(df, title, filename, cb=None, aggregation_level=2, ):
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
  plt.savefig(filename)
  plt.clf()


class Cb():
  def __init__(self):
    self.i = 0

  def __call__(self, device, x):
    logger.info('step=%d; cost=%.6f' % (self.i, device.cost(x, 0)))
    self.i += 1


main()
