'''
Single Device and Generator with quadratic cost curve 24x1H day. The Device has no preference for
energy itself, however must consume at least 12kWh at any time of the day. Max rating is 3kW, thus
the minimum time window in which to operate is 4H.

Given the quadratic cost curve ordinarily the optimal solution would be  to consume 0.5kW in every
hour. However, the device's utility function includes a factor to penalize dispersion from some
arbitrary mean. This should cause consumtion to lump together. The lumpiness is a result of a tension
between the quadratic costs and dispersion factor.
'''
import numpy as np
from device_kit import *
from device_kit.functions import *


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%d' % (v,),
  },
)
meta = {
  'title': '''Just testing'''
}
dimension = 48
demand_bounds = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085],[0.746,1.085]]).reshape((dimension, 2))
supply_bounds = -1*np.array([[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2],[0,2]]).reshape((dimension, 2))
supply_bounds = np.array([supply_bounds[:,1], supply_bounds[:,0]]).reshape((dimension, 2))
print(supply_bounds)


def make_deviceset():
  return DeviceSet('nw', [
    Device(
      'demand',
      dimension,
      demand_bounds,
    ),
    Device(
      'supply',
      dimension,
      supply_bounds,
    )
  ])