'''
Simple single IDevice test scenario:
  - Day starts at 00:00. Price peaks at midday, from lows at beginning, end of day.
  - Utility split into am/pm:
    - am: quadratic
    - pm: pow 4
'''
import numpy as np
from device_kit import *


meta = {
  'title': 'A single IDevice with different am/pm curve params\nunder a varying price.'
}


def make_deviceset():
  return DeviceSet('nw', list(make_devices().values()))


def make_devices():
  basis = 24
  cost = np.stack((np.sin(np.linspace(0, np.pi, basis))*0.5+0.1, np.ones(basis)*0.001, np.zeros(basis)), axis=1)
  return {
    # 'demand1': IDevice(
    #   'demand1',
    #   basis,
    #   np.stack((np.zeros(basis), np.ones(basis)*1), axis=1),
    #   **{
    #     'a': 0.1,
    #     'b': np.concatenate((np.ones(int(basis/2))*2, np.ones(int(basis/2))*4))
    #   }
    # ),
    'demand2': IDevice2(
      'demand2',
      basis,
      np.stack((np.zeros(basis), np.ones(basis)*1), axis=1),
      **{
        'p_l': -1,
        'p_h': -0.1*np.concatenate((np.ones(int(basis/2))*2, np.ones(int(basis/2))*4))
      }
    ),
    'supply': GDevice(
      'supply',
      basis,
      np.stack((-20*np.ones(basis), np.zeros(basis)), axis=1),
      **{'cost_coeffs': cost}
    )
  }
