''' Informal test for multi flow devices which is implemented via MFDeviceSet mainly '''
import sys
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import device_kit
from device_kit import *
from device_kit.mfdeviceset import MFDeviceSet
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
    ('shiftable', CDevice('shiftable', 24, (0, 2), (12, 24))),
    ('generator', GDevice('generator', 24, (-50,0), None, **{'cost': cost})),
])


class SubBalancedMFDeviceSet(DeviceSet):
  ''' Find all devices under this device set matching ".*{label}$" and apply an additional balancing
  constraint - i.e. labelled flows sum to zero at all times. '''
  _label = []

  def __init__(self, id, devices, sbounds, labels=[]):
    super().__init__(id, devices, sbounds)
    self.labels = labels

  @property
  def constraints(self):
    constraints = super().constraints
    shape = self.shape
    flat_shape = shape[0]*shape[1]
    leaf_devices = OrderedDict(self.leaf_devices())
    for label in self.labels:
      labelled = [k for k, v in enumerate(leaf_devices.keys()) if re.match('.*\.{label}$'.format(label=label), v)]
      col_jac = np.zeros(shape[0])
      col_jac[labelled] = 1
      for i in range(0, len(self)): # for each time
        constraints += [{
          'type': 'eq',
          'fun': lambda s, i=i: s.reshape(shape)[labelled, i].sum(),
          'jac': lambda s, i=i, j=col_jac: zmm(s.reshape(shape), i, axis=1, fn=lambda r: j).reshape(flat_shape)
        }]
    return constraints


class TwoRatioMFDeviceSet(MFDeviceSet):
  ''' Add a constraint saying flow one must equal k times flow two. Only 2 flows supported because
  adding the constraint for >2 flow is more difficult and I don't need it.
  '''
  ratios = None

  def __init__(self, device:Device, flows, ratios):
    super().__init__(device, flows)
    if len(flows) != 2:
      raise ValueError('More than two flows not supported.')
    if ratios is not None and not len(ratios) == len(flows):
      raise ValueError('Flows and flow ratios must have same length')
    self.ratios = ratios

  @property
  def constraints(self):
    constraints = super().constraints
    shape = self.shape
    flat_shape = shape[0]*shape[1]
    for i in range(0, len(self)): # for each time
      constraints += [{
        'type': 'eq',
        'fun': lambda s, i=i, r=self.ratios: s.reshape(shape)[0,i]*r[0] - s.reshape(shape)[1,i]*r[1],
        'jac': lambda s, i=i, r=self.ratios: zmm(s.reshape(shape), i, axis=1, fn=lambda x: np.array([r[0], -r[1]])).reshape(flat_shape)
      }]
    return constraints


def make_model():
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  model = DeviceSet('site1', list(devices.values()),
    sbounds=(0,0)
  )
  return model


def make_model_mf():
  model = SubBalancedMFDeviceSet('site1', [
      devices['uncntrld'],
      MFDeviceSet(devices['scalable'], ['e', 'h']),
      devices['shiftable'],
      TwoRatioMFDeviceSet(devices['generator'], ['e', 'h'], [1,8]),
      DeviceSet('sink', [Device('h', 24, (0,100))])
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
