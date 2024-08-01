import numpy as np
from device_kit import Device
from device_kit.functions import poly2d


class F():
  ''' Default null function for ADevice. f parameter has to look like this. '''

  def __call__(self, x):
    return 0

  def deriv(self, n=1):
    return lambda x: 0

  def hess(self):
    return lambda x: 0


class ADevice(Device):
  ''' Device that takes an arbitrary utility function and constraints. '''
  _f = F()
  _constraints = []


  def u(self, s, p):
    s = s.reshape(len(self))
    return self.f(s) - (s*p).sum()

  def deriv(self, s, p):
    s = s.reshape(len(self))
    return self.f.deriv()(s) - p

  def hess(self, s, p=0):
    s = s.reshape(len(self))
    return self.f.hess()(s)

  @property
  def constraints(self):
    return Device.constraints.fget(self) + self._constraints

  @property
  def f(self):
    return self._f

  @f.setter
  def f(self, f):
    if not hasattr(f, '__call__') and hasattr(f, 'deriv') and hasattr(f, 'hess'):
      raise ValueError('Invalid parameter type for \'f\' [%s]' % (type(f)))
    self._f = f
