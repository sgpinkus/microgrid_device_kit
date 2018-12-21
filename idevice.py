import numpy as np
import numdifftools as nd
from device import Device


class IDevice(Device):
  ''' Provides a utility function that is additively separable, concave, and increasing.
  The particular utility curve is described by 4 params and *also* uses on min/max consumption bounds
  setting. All params may be ndarrays of len(self), or scalars.

  For a given time slot, let r_min, r_max be the min, max consumption as specified by Device.bounds
  for the time slot.

    - a is a -ve offset from the peak of the utility curve at which it intersects the r_max. This
        means the gradient of the curve is always >= 0 at r_max.
    - b is the degree of the polynomial. Must be a +ve integer.
    - c is a scaling factor: the range of utility is [0,c] over [r_min, r_max], if a, d are both 0.
    - d is an offset that defines where the root of the curve is.

  The utility value is indeterminate when r_max == r_min, but returns 0 in this case. Same for deriv.
  '''
  a = 0
  b = 2
  c = 1
  d = 0

  @staticmethod
  def _u(x, a, b, c, d, x_l, x_h):
    ''' The utility function on scalar. '''
    if x_l == x_h:
      return 0
    n = lambda x: ((x-x_l)/(x_h-x_l))*(1-a)  # Normalize x to [0, 1-a]
    s = c/(1-a**b)  # Scaling factor
    return s*(-1*(1 - n(x))**b + 1) + d

  @staticmethod
  def _deriv(x, a, b, c, d, x_l, x_h):
    ''' The derivative of utility function on scalar. '''
    if x_l == x_h:
      return 0
    n = lambda x: ((x-x_l)/(x_h-x_l))*(1-a)  # Normalize x to [0, 1-a]
    s = c/(1-a**b)  # Scaling factor
    return ((1-a)/(x_h-x_l))*b*s*((1 - n(x))**(b-1))

  def uv(self, s, p):
    return np.vectorize(IDevice._u, otypes=[float])(s, self.a, self.b, self.c, self.d, self.lbounds, self.hbounds) - s*p

  def u(self, s, p):
    return self.uv(s, p).sum()

  def deriv(self, s, p):
    return np.vectorize(IDevice._deriv, otypes=[float])(s, self.a, self.b, self.c, self.d, self.lbounds, self.hbounds) - p

  def hess(self, s, p=0):
    ''' Return Hessian diagonal approximation. @todo write IDevice._hess(...). '''
    return np.diag(nd.Hessdiag(lambda x: self.u(x, 0))(s.reshape(len(self))))

  @property
  def params(self):
    return {'a': self.a, 'b': self.b, 'c': self.c, 'd': self.d}

  @params.setter
  def params(self, params):
    ''' Sanity check params.  '''
    if params is None:
      return
    if not isinstance(params, dict):
      raise ValueError('params to IDevice must be a dictionary')
    # Use preset params as defaults, ignore extras.
    p = IDevice.params.fget(self)
    p.update({k: v for k, v in params.items() if k in IDevice.params.fget(self).keys()})
    p = {k: np.array(v) for k, v in p.items()}
    for k, v in p.items():
      if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('params are required to have same length as device (%d)' % (len(self),))
      if not (v >= 0).all():
        raise ValueError('param %s must be >= 0' % (k,))
    (a, b, c, d) = (p['a'], p['b'], p['c'], p['d'])
    if not (a <= self.hbounds - self.lbounds).all():
      raise ValueError('param a (offset) cannot be larger than extent of domain')
    if not (b > 0).all():
      raise ValueError('param b must be > 0')
    (self.a, self.b, self.c, self.d) = (a, b, c, d)
