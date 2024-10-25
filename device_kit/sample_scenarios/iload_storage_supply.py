'''
Adds a battery to idevice_scenario.py and different particular settings. Created this to debug a
optimization error 8: 'Positive directional derivative for linesearch' when using these and similar
settings and LCLNetwork. Hmmm, if I just ignore it results is as expected.
'''
import numpy as np
from device_kit import *


def make_deviceset():
  return DeviceSet('site', [
    IDevice(
      'demand',
      24,
      np.stack((200*np.ones(24), 400*np.ones(24)), axis=1),
      None,
      **{'a': 0.1, 'b': 4, 'c': 2.5},
    ),
    SDevice(
      'battery',
      24,
      np.stack((-9500*np.ones(24), 9500*np.ones(24)), axis=1),
      None,
      **{'c1': 0.01, 'c2': 0.0, 'c3': 0.0, 'reserve': 0.5, 'capacity': 32000}
    ),
    GDevice(
      'supply',
      24,
      np.stack((-50000*np.ones(24), np.zeros(24)), axis=1),
      None,
      **{'cost_coeffs': np.array([0.05, 0.1, 0])},
    )
  ])
