import numpy as np
from device_kit import Device


class CDevice(Device):
  ''' Overrides Device to provide a utility function based on the sum resource consumption.
    The particular utility curve is described by 2 params. U = a(Q) + b. Since Q is summed up a,b are scalars.
  '''
  _a = 0  # Slope
  _b = 0  # Offset

  def u(self, s, p):
    return (self.a*s.sum() + self.b) - (s*p).sum()

  def deriv(self, s, p):
    return np.ones(len(self))*(self.a - p)

  def hess(self, s, p=0):
    return np.zeros((len(self), len(self)))

  @property
  def params(self):
    return {'a': self.a, 'b': self.b}

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if params is None:
      (self._a, self._b) = (0, 0)
      return
    if not isinstance(params, dict):
      raise ValueError('params incorrect type')
    (a, b) = (params['a'], params['b'])
    if a < 0 or b < 0:
      raise ValueError('params a and b must be >= 0')
    (self._a, self._b) = (a, b)
