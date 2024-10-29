import numpy as np
from device_kit import Device
from device_kit.functions import ABCQuadraticCost


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
  _cost_fn = None

  def __init__(self, id, length, bounds, cbounds=None, **kwargs):
    super().__init__(id, length, bounds, cbounds=None, **kwargs)
    self._cost_fn = ABCQuadraticCost(self.a, self.b, self.c, self.lbounds, self.hbounds)

  def costv(self, s, p):
    return self._cost_fn(s) + s*p

  def cost(self, s, p):
    return self.costv(s, p).sum()

  def deriv(self, s, p):
    return self._cost_fn.deriv(s) + p

  def hess(self, s, p=0):
    return self._cost_fn.hess(s)

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
