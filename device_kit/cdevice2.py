import numpy as np
from device_kit import Device, IDevice, IDevice2


class CDevice2(Device):
  ''' Same curve as IDevice2 but applied to the scalar sum of consumption. '''
  _p_h = 0
  _p_l = -1

  def cost(self, s, p):
    return IDevice2._cost(s.sum(), p_l=self.p_l, p_h=self.p_h, x_l=self.cbounds[0], x_h=self.cbounds[1]) - (s*p).sum()

  def deriv(self, s, p):
    return IDevice2._deriv(s.sum(), self.p_l, self.p_h, self.cbounds[0], self.cbounds[1])*np.ones(len(self)) - p

  def hess(self, s, p=0):
    return np.eye(len(self))*IDevice2._hess(s.sum(), self.p_l, self.p_h, self.cbounds[0], self.cbounds[1])

  @property
  def p_h(self):
    return self._p_h

  @property
  def p_l(self):
    return self._p_l

  @p_h.setter
  def p_h(self, v):
    p_h = self._validate_param(v)
    if not (self.p_l <= p_h).all():
      raise ValueError('param p_h must be >= p_l')
    self._p_h = p_h

  @p_l.setter
  def p_l(self, v):
    p_l = self._validate_param(v)
    if not (self.p_h >= p_l).all():
      raise ValueError('param p_h must be >= p_l')
    self._p_l = p_l

  def _validate_param(self, p):
    v = np.array(p)
    if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('param must be scalar or same length as device (%d)' % (len(self),))
    if not (v <= 0).all():
      raise ValueError('param must be <= 0')
    return v
