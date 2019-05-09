import numpy as np
import numdifftools as nd
from device_kit import Device


class IDevice2(Device):
  ''' Provides a utility function that is additively separable, concave, and increasing.
  The particular utility curve is described by 2 params and *also* uses on min/max consumption bounds
  setting. All params may be ndarrays of len(self), or scalars.

  For a given time slot, let r_min, r_max be the min, max consumption as specified by Device.bounds
  for the time slot. p0, p1 is the derivative of a decreasing quadratic section at r_max, r_min
  respectively.

  The utility value is indeterminate when r_max == r_min, but returns 0 in this case. Same for deriv.
  '''
  _d0 = 0
  _d1 = 1

  @staticmethod
  def _u(x, d1, d0, x_l, x_h):
    ''' The utility function on scalar. '''
    if x_l == x_h:
      return 0
    return (x_h - x_l)*np.poly1d([(d0 - d1)/2, d1, 0])(IDevice2.scale(x, x_l, x_h))

  @staticmethod
  def _deriv(x, d1, d0, x_l, x_h):
    ''' The derivative of utility function on a scalar. Returned valus is expansion of:
         np.poly1d([(d0 - d1)/2, d1, 0]).deriv()(IDevice2.scale(x, x_l, x_h))
    '''
    if x_l == x_h:
      return 0
    return (d0 - d1)*IDevice2.scale(x, x_l, x_h) + d1

  @staticmethod
  def _hess(x, d1, d0, x_l, x_h):
    ''' The 2nd derivative of utility function on a scalar. '''
    if x_l == x_h:
      return 0
    return (d0 - d1)/(x_h - x_l)

  @staticmethod
  def scale(x, x_l, x_h):
    return (x - x_l)/(x_h - x_l)

  def uv(self, s, p):
    return np.vectorize(IDevice2._u, otypes=[float])(s, self.d1, self.d0, self.lbounds, self.hbounds) - s*p

  def u(self, s, p):
    return self.uv(s, p).sum()

  def deriv(self, s, p):
    return np.vectorize(IDevice2._deriv, otypes=[float])(s.reshape(len(self)), self.d1, self.d0, self.lbounds, self.hbounds) - p

  def hess(self, s, p=0):
    return np.diag(np.vectorize(IDevice2._hess, otypes=[float])(s.reshape(len(self)), self.d1, self.d0, self.lbounds, self.hbounds))

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
