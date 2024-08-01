import numpy as np
from device_kit import Device


class PVDevice(Device):
  ''' The PVDevice (Generator) models a simple costless generator such as a simple solar panel. It
  produces power (-ve consumption), at zero cost and has a time varying maximum capacity. The client is
  expected to code the max generating capacity of the device at different times of the day via `lbounds`.
  We enforce that hbounds must be zero for this device (can't consume).

  PVDevice does not have any device specific parameters: A simple PV device can be modeled just as
  a specific configuration of the base `Device` class, plus the utility function listed in this
  class. For example for PV device one might set lbounds of `Device` to something like this:

    solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
    lbounds = -1*np.minimum(max_rate, solar_intensity*efficiency*area)

  '''

  def uv(self, s, p):
    ''' Utility is profit which = revenue since costs are zero. '''
    return -s*p

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cbounds(self):
    return self._cbounds

  @bounds.setter
  def bounds(self, bounds):
    ''' @override bounds setter to ensure hbounds <= 0. '''
    Device.bounds.fset(self, bounds)
    bounds = np.array(bounds)
    if not (self.hbounds <= 0).all():
      raise ValueError('hbounds must be <= 0')

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' @override don't allow cbounds
    @todo to allow -ve cbounds.
    '''
    if cbounds is None:
      self._cbounds = None
    else:
      raise ValueError('cbounds not allowed for PVDevice currently')
