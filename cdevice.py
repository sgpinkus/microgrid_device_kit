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
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @a.setter
  def a(self, a):
    if a < 0:
      raise ValueError('param a must be >= 0')
    self._a = a

  @b.setter
  def b(self, b):
    if b < 0:
      raise ValueError('param a must be >= 0')
    self._b = b
