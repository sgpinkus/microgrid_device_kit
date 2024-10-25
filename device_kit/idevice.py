import numpy as np
from device_kit import Device


class IDevice(Device):
  ''' Provides a load cost function that is separable, convex, and decreasing. The particular cost
  curve is described by 3 params and also uses on min/max consumption bounds setting. All params may
  be ndarrays of len(self) to specify different parameters for each interval or scalars to specify
  the same value for all intervals.

  For a given interval, let x be the device flow, x_min, x_max be the min, max flow as specified by
  Device.bounds for the time slot. Consider a convex decreasing curve over [x_min, x_max]. We desire
  the derivative at x_min > x_min and at x_max <= 0. The three parameters fit a convex polynomial with those
  characteristics over the said range:

    - a is a -ve offset from the root of the cost curve at which x_max occurs.
    - b is the degree of the polynomial and must be a +ve integer.
    - c is a scaling factor so that the range of cost is [0, c] over [x_min, x_max].

  The three parameters are used as follows:

    m(y) = c*k*(1 - (1 - s(x))^b), where k = 1/(1-a**b), and s(x) is x scaled into domain [0,1-a].

  We have invariants m(0) = 1 and m`(1) = 0. If y can range over the full domain [0,1] we also have m(1) = 0.
  However we actually want m(y) = 1 when y = 1-a. The k term makes
  is so when

  Note, the cost value and its derivation is indeterminate when x_max == x_min, but returns 0 in
  this case.
  '''
  _a = 0
  _b = 2
  _c = 1
  _s = None

  @staticmethod
  def _cost(x, a, b, c, x_l, x_h):
    ''' The cost function on scalar. '''
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*((1 - IDevice.scale(x, x_l, x_h, a))**b)

  @staticmethod
  def _deriv(x, a, b, c, x_l, x_h):
    ''' The derivative of cost function on scalar. '''
    if x_l == x_h:
      return 0
    return -(c/(1-a**b))*((1-a)/(x_h-x_l))*b*((1 - IDevice.scale(x, x_l, x_h, a))**(b-1))

  @staticmethod
  def _hess(x, a, b, c, x_l, x_h):
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*((1-a)/(x_h-x_l))**2*b*(b-1)*((1 - IDevice.scale(x, x_l, x_h, a))**(b-2))

  @staticmethod
  def scale(x, x_l, x_h, a):
    return (1-a)*((x - x_l)/(x_h - x_l))

  def costv(self, s, p):
    return np.vectorize(IDevice._cost, otypes=[float])(s, self.a, self.b, self.c, self.lbounds, self.hbounds) - s*p

  def cost(self, s, p):
    return self.costv(s, p).sum()

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

  @a.setter
  def a(self, a):
    self._a = IDevice._validate_param(a, len(self))

  @b.setter
  def b(self, b):
    IDevice._validate_param(b, len(self))
    if not (np.array(b) > 0).all():
      raise ValueError('param b must be > 0')
    self._b = b

  @c.setter
  def c(self, c):
    self._c = IDevice._validate_param(c, len(self))

  @staticmethod
  def _validate_param(p, length):
    v = np.array(p)
    if not (v.ndim == 0 or len(v) == length):
        raise ValueError('param must be scalar or same length as device (%d)' % (length,))
    if not (v >= 0).all():
      raise ValueError('param must be >= 0')
    return p
