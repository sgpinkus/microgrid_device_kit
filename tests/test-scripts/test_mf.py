'''
Informal test for multi flow devices which is implemented via MFDeviceSet.

  - There are two flows, e and h.
  - There is a single producer device that is is constrained to produce e in strict proportion.
  - For all other devices we have gross subsititutes between e,h.
  - We compare this to the case where there is only one flow but everything else is identical.

Expected result:

  - The objective value should be identical and aggregate flows should be identical.

'''
import sys
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import device_kit
from device_kit import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


np.random.seed(109)
pd.set_option('display.float_format', lambda v: '%+0.4f' % (v,),)
cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
devices = OrderedDict([
    ('uncntrld', Device('uncntrld', 24, (np.maximum(0, np.random.random(24)*2-1),))),
    ('scalable', IDevice2('scalable', 24, (0., 2), (6, 24), d0=0.1)),
    ('shiftable', CDevice('shiftable', 24, (0, 2), (6, 24), a=0.5)),
    ('generator', GDevice('generator', 24, (-50,0), None, **{'cost_coeffs': cost})),
])


def make_model():
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  print(devices['scalable'])
  model = DeviceSet('site1', list(devices.values()),
    sbounds=(0,0)
  )
  return model


def make_model_mf():
  model = SubBalancedDeviceSet('site1', [
      MFDeviceSet(devices['uncntrld'], ['e', 'h']),
      MFDeviceSet(devices['scalable'], ['e', 'h']),
      MFDeviceSet(devices['shiftable'], ['e', 'h']),
      TwoRatioMFDeviceSet(devices['generator'], ['e', 'h'], [-1,-8]),
      DeviceSet('sink', [Device('h', 24, (-0,100))])
    ],
    sbounds=(0,0),
    labels=['h']
  )
  return model


def solve_model():
  model = make_model()
  (x, solve_meta) = solve(model, p=0, solver_options={'maxiter': 500, 'ftol': 1e-6})
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  df.sort_index(inplace=True)
  return (model, x, df.transpose())


def solve_model_mf():
  model = make_model_mf()
  (x, solve_meta) = solve(model, p=0, solver_options={'maxiter': 500, 'ftol': 1e-6})
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  df.loc['site1.uncntrld'] = df.filter(regex='uncntrld', axis=0).sum(axis=0)
  df.loc['site1.scalable'] = df.filter(regex='scalable', axis=0).sum(axis=0)
  df.loc['site1.shiftable'] = df.filter(regex='shiftable', axis=0).sum(axis=0)
  df.loc['site1.generator'] = df.filter(regex='generator', axis=0).sum(axis=0)
  df.sort_index(inplace=True)
  return (model, x, df.transpose())


def main():
  (model1, x1, df1) = solve_model()
  # Tabular
  print(df1, '\n', df1.sum(), '\nu:', model1.cost(x1, 0), '\nuv:', model1.costv(x1, p=0))
  print('-'*100)
  (model2, x2, df2) = solve_model_mf()
  print(df2, '\n', df2.sum(), '\nu:', model2.cost(x2, 0), '\nuv:', model2.costv(x2, p=0))
  print('-'*100)
  print(df2['site1.generator.e']/df2['site1.generator.h'])
  print('-'*100)
  # Generator show total demand.
  plt.step(np.arange(24), df1['site1.generator'], label='A.generator')
  plt.step(np.arange(24), df2['site1.generator'], label='B.generator')
  plt.legend()
  plt.show()
  # Devices should be the same.
  plt.plot(devices['generator']._cost_fn(np.ones(24)), color='y', label='cost')
  plt.plot(df1['site1.scalable'], color='r', label='site1.scalable')
  plt.plot(df1['site1.shiftable'], color='r', ls='--', label='site1.shiftable')
  plt.plot(df2['site1.scalable'], color='b', label='site2.scalable')
  plt.plot(df2['site1.shiftable'], color='b', ls='--', label='site2.shiftable')
  plt.plot(df1['site1.uncntrld'], color='k', label='uncntrld')
  plt.legend()
  plt.xlim(0, None)
  plt.show()
  return(model1, df1, model2, df2)

if __name__ == '__main__':
  main()
