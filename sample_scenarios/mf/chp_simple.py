'''
Multi flow CHP scenario. BROKEN.
'''
import numpy as np
from device_kit import *


def make_deviceset():
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  devices = OrderedDict([
      ('uncntrld', Device('uncntrld', 24, (random_uncntrld(),))),
      ('scalable', IDevice2('scalable', 24, (0., 2), (12, 18))),
      ('shiftable', CDevice('shiftable', 24, (0, 2), (12, 24))),
      ('generator', GDevice('generator', 24, (-50,0), None, **{'cost': cost})),
  ])
  return MFDeviceSet('site1', [
      devices['uncntrld'],
      MFDeviceSet(devices['scalable'], ['e', 'h']),
      devices['shiftable'],
      TwoRatioMFDeviceSet(devices['generator'], ['e', 'h'], [1,8]),
      DeviceSet('sink', [Device('h', 24, (0,100))])
    ],
  )


def random_uncntrld():
  np.random.seed(109)
  return np.minimum(2, np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24))))


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
