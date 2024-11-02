#!/usr/bin/env python3
''' Convenience script to just solve for outright cost minimized balanced flow - no market sim crap.
'''
from os.path import splitext, basename
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import device_kit
from device_kit.loaders import builder_loader, module_loader
# from device_kit.utils import get_device_by_id
from plots import *

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
  parser.add_argument('--loader', '-l', action='store', default='module', type=str,
    help='loader to use to load scenario file')
  args = parser.parse_args()
  loader = globals()[f'{args.loader}_loader']
  (deviceset, meta, cb) = loader.load_file(args.filename)
  deviceset.sbounds = (0,0)
  print(str(deviceset))
  (x, solve_meta) = device_kit.solve(deviceset, solver_options={'ftol': 1e-6}, cb=Cb()) # Convenience convex solver.
  print(solve_meta.message)

  df = pd.DataFrame.from_dict(dict(deviceset.map(x)), orient='index')
  df.loc['total'] = df.sum()

  # x = dict(deviceset.map(x))['nw.supply']
  # d = get_device_by_id(deviceset, 'supply')
  # print(d.costv(x, 0))
  # print(d.deriv(x, 0))

  output_filename = splitext(basename(args.filename))[0]
  title = meta.get('title') if meta else None
  plot_stacked_bars(df, title, output_filename + '.png')
  df['cumulative'] = df.sum(axis=1)
  print(df.sort_index())
  df.to_csv(output_filename + '.csv', float_format='%.3f')


class Cb():
  def __init__(self):
    self.i = 0

  def __call__(self, device, x):
    logger.info('step=%d; cost=%.6f' % (self.i, device.cost(x, 0)))
    self.i += 1


if __name__ == '__main__':
  main()
