import numpy as np
from lcl.device import Device


class CDevice(Device):
  ''' Overrides Device to provide a utility function based on the sum resource consumption.
    The particular utility curve is described by 2 params. U = a(Q) + b. Since Q is summed up a,b are scalars.
  '''

  def u(self, r, p):
    return (self.a*r.sum() + self.b) - (r*p).sum()

  def deriv(self, r, p):
    return self.a - p

  @property
  def params(self):
    return Device.params.fget(self)

  @property
  def a(self):
    return self.params[0]

  @property
  def b(self):
    return self.params[1]

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if len(params) != 2:
      raise ValueError('params must be have size two')
    (a,b) = params
    if a < 0 or b < 0:
      raise ValueError('params a and b must be >= 0')
    self._params = params
