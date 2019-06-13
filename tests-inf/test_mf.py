''' Informal test for multi flow devices which is implemented via MFDeviceSet mainly '''
import sys
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import device_kit
from device_kit import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)


def random_uncntrld():
  np.random.seed(109)
  return np.minimum(2, np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24))))


cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
devices = OrderedDict([
    ('uncntrld', Device('uncntrld', 24, (random_uncntrld(),))),
    ('scalable', IDevice2('scalable', 24, (0., 2), (12, 18))),
    ('shiftable', CDevice('shiftable', 24, (0, 2), (12, 24), a=0.5)),
    ('generator', GDevice('generator', 24, (-50,0), None, **{'cost': cost})),
])


def make_model():
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  model = DeviceSet('site1', list(devices.values()),
    sbounds=(0,0)
  )
  return model


def make_model_mf():
  model = SubBalancedDeviceSet('site1', [
      MFDeviceSet(devices['uncntrld'], ['e', 'h']),
      MFDeviceSet(devices['scalable'], ['e', 'h']),
      MFDeviceSet(devices['shiftable'], ['e', 'h']),
      TwoRatioMFDeviceSet(devices['generator'], ['e', 'h'], [1,8]),
      DeviceSet('sink', [Device('h', 24, (-0,100))])
    ],
    sbounds=(0,0),
    labels=['h']
  )
  return model


def solve_model():
  model = make_model()
  (x, solve_meta) = solve(model, p=0, solver_options={'maxiter': 500, 'ftol': 5e-6})
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  df.sort_index(inplace=True)
  return (model, x, df.transpose())


def solve_model_mf():
  model = make_model_mf()
  (x, solve_meta) = solve(model, p=0, solver_options={'maxiter': 1000, 'ftol': 5e-6})
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  df.loc['site1.scalable'] = df.filter(regex='scalable', axis=0).sum(axis=0)
  df.loc['site1.generator'] = df.filter(regex='generator', axis=0).sum(axis=0)
  df.sort_index(inplace=True)
  return (model, x, df.transpose())


def solve(device, p, s0=None, solver_options={}, prox=False):
  _solver_options = {'ftol': 1e-6, 'maxiter': 1000, 'disp': True}
  _solver_options.update(solver_options)
  print(_solver_options)

  args = {
    'fun': lambda s, p=p: -1*device.u(s, p),
    'x0':  s0 if s0 is not None else device.project(np.zeros(device.shape)),
    'jac': lambda s, p=p: -1*device.deriv(s, p),
    'method': 'SLSQP',
    'bounds': device.bounds,
    'constraints': device.constraints,
    'tol': 1e-6,
    'options': _solver_options,
  }
  o = minimize(**args)
  return ((o.x).reshape(device.shape), o)


def main():
  (model1, x1, df1) = solve_model()
  print(df1)
  print(df1.sum())
  print(model1.u(x1, p=0))
  print('-'*100)
  (model2, x2, df2) = solve_model_mf()
  print(df2)
  print(df2.sum())
  print(model2.u(x2, p=0))
  print('-'*100)
  dfc = pd.DataFrame()
  plt.plot(df1['site1.generator'], label='A.generator')
  plt.plot(df2['site1.generator'], label='B.generator')
  plt.legend()
  plt.show()
  dfc['cost'] = devices['generator']._cost_fn(np.ones(24))
  dfc['scalable_A'] = df1['site1.scalable']
  dfc['scalable_B'] = df2['site1.scalable']
  dfc['shiftable_A'] = df1['site1.shiftable']
  dfc['shiftable_B'] = df2['site1.shiftable']
  dfc['uncntrld'] = df1['site1.uncntrld']
  dfc.plot(legend=True)
  plt.xlim(0, None)
  plt.show()

if __name__ == '__main__':
  main()
