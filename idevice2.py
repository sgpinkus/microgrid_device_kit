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
  d0 = 0
  d1 = 1

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
  def params(self):
    return {'d1': self.d1, 'd0': self.d0}

  @params.setter
  def params(self, params):
    ''' Sanity check params.  '''
    if params is None:
      return
    if not isinstance(params, dict):
      raise ValueError('params to IDevice2 must be a dictionary')
    # Use preset params as defaults, ignore extras.
    p = IDevice2.params.fget(self)
    p.update({k: v for k, v in params.items() if k in IDevice2.params.fget(self).keys()})
    p = {k: np.array(v) for k, v in p.items()}
    for k, v in p.items():
      if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('params are required to have same length as device (%d)' % (len(self),))
      if not (v >= 0).all():
        raise ValueError('param %s must be >= 0' % (k,))
    if not (p['d1'] >= p['d0']).all():
      raise ValueError('param d1 must be >= d0')
    (self.d1, self.d0) = (p['d1'], p['d0'])
