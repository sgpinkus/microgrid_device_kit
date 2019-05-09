import numpy as np
from device_kit import Device, IDevice, IDevice2


class CDevice2(Device):
  ''' Same curve as IDevice2 but applied to the scalar sum of consumption. '''
  _d0 = 0
  _d1 = 1

  def u(self, s, p):
    return IDevice2._u(s.sum(), d1=self.d1, d0=self.d0, x_l=self.cbounds[0], x_h=self.cbounds[1]) - (s*p).sum()

  def deriv(self, s, p):
    return IDevice2._deriv(s.sum(), self.d1, self.d0, self.cbounds[0], self.cbounds[1])*np.ones(len(self)) - p

  def hess(self, s, p=0):
    return np.eye(len(self))*IDevice2._hess(s.sum(), self.d1, self.d0, self.cbounds[0], self.cbounds[1])

  @property
  def d0(self):
    return self._d0

  @property
  def d1(self):
    return self._d1

  @d0.setter
  def d0(self, v):
    d0 = self._validate_param(v)
    if not (self.d1 >= d0).all():
      raise ValueError('param d0 must be <= d1')
    self._d0 = d0

  @d1.setter
  def d1(self, v):
    d1 = self._validate_param(v)
    if not (self.d0 <= d1).all():
      raise ValueError('param d0 must be <= d1')
    self._d1 = d1

  def _validate_param(self, p):
    v = np.array(p)
    if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('param must be scalar or same length as device (%d)' % (len(self),))
    if not (v >= 0).all():
      raise ValueError('param must be >= 0')
    return v
