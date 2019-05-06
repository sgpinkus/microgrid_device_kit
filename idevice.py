import numpy as np
import numdifftools as nd
from device_kit import Device


class IDevice(Device):
  ''' Provides a utility function that is separable, concave, and increasing. The particular utility
  curve is described by 3 params and also uses on min/max consumption bounds setting. All params may
  be ndarrays of len(self) to specify different parameters for each interval or scalars to specify
  the same value for all intervals.

  For a given interval, x be the device flow, x_min, x_max be the min, max flow as specified by
  Device.bounds for the time slot. The three parameters are:

    - a is a -ve offset from the peak of the utility curve.
    - b is the degree of the polynomial and must be a +ve integer.
    - c is a scaling factor so that the range of utility is [0, c] over [x_min, x_max].

  The three parameters are used as follows:

    m(s(x)) = m(y) = c*k(1 - (1 - y)^b)

  where s(x) scales x between [x_min, x_max] to [0, 1-a] (by default a = 0, so by default to [0, 1]).

  We have invariants m(0) = 0 and m`(1) = 0. If y can range over the full domain [0,1] we also have m(1) = 1.
  However we actually wanted (why did we want this again??) m(y) = 1 when y = 1-a. The k term makes
  is so when k = 1/(1-a**b)

  Note, the utility value and its derivation is indeterminate when x_max == x_min, but returns 0 in
  this case.
  '''
  _a = 0
  _b = 2
  _c = 1
  _s = None

  @staticmethod
  def _u(x, a, b, c, x_l, x_h):
    ''' The utility function on scalar. '''
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*(1 - (1 - IDevice.s(x, x_l, x_h, a))**b)

  @staticmethod
  def _deriv(x, a, b, c, x_l, x_h):
    ''' The derivative of utility function on scalar. '''
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*((1-a)/(x_h-x_l))*b*((1 - IDevice.s(x, x_l, x_h, a))**(b-1))

  @staticmethod
  def _hess(x, a, b, c, x_l, x_h):
    if x_l == x_h:
      return 0
    return -(c/(1-a**b))*((1-a)/(x_h-x_l))**2*b*(b-1)*((1 - IDevice.s(x, x_l, x_h, a))**(b-2))

  @staticmethod
  def s(x, x_l, x_h, a):
    return (1-a)*((x - x_l)/(x_h - x_l))

  def uv(self, s, p):
    return np.vectorize(IDevice._u, otypes=[float])(s, self.a, self.b, self.c, self.lbounds, self.hbounds) - s*p

  def u(self, s, p):
    return self.uv(s, p).sum()

  def deriv(self, s, p):
    return np.vectorize(IDevice._deriv, otypes=[float])(s.reshape(len(self)), self.a, self.b, self.c, self.lbounds, self.hbounds) - p

  def hess(self, s, p=0):
    return np.diag(np.vectorize(IDevice._hess, otypes=[float])(s.reshape(len(self)), self.a, self.b, self.c, self.lbounds, self.hbounds))

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @property
  def c(self):
    return self._c

  @property
  def params(self):
    return {'a': self.a, 'b': self.b, 'c': self.c}

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
    (a, b, c) = (p['a'], p['b'], p['c'])
    if not (a <= 1).all():
      raise ValueError('param a must be <= 1')
    if not (b > 0).all():
      raise ValueError('param b must be > 0')
    (self._a, self._b, self._c) = (a, b, c)
