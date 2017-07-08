import numpy as np
from lcl.device import Device

class IDevice(Device):
  ''' Overrides Device to provide a utility function based on instaneous resource consumptions
  (actually it's summed up). The particular utility curve is described by 4 params but also
  uses on min/max consumption setting as parameters to the utility function. All params may be 1D
  nparrays of len(self) or scalars.

  The value is indeterminate when x_h == x_l. We return 0 utility value in this case. Same for deriv().
  '''
  _a = 0 # is a -ve offset at which the curve intersects the q_max
  _b = 2 # defines the slopiness. Must be a +ve integer.
  _c = 1 # Scaling factor. The range of utility is [0,c] over [q_min, q_max]
  _d = 0 # An offset. Should be non -ve.

  @classmethod
  def _u(cls, x, a, b, c, d, x_l, x_h):
    ''' The utility function on scalar. '''
    if(x_l == x_h):
      return 0
    n = lambda x: ((x-x_l)/(x_h-x_l))*(1-a) # Normalize x to [0, 1-a]
    s = c/(1-a**b) # Scaling factor
    return s*(-1*(1 - n(x))**b + 1) + d

  @classmethod
  def _deriv(cls, x, a, b, c, d, x_l, x_h):
    ''' The derivative of utility function on scalar. '''
    if(x_l == x_h):
      return 0
    n = lambda x: ((x-x_l)/(x_h-x_l))*(1-a) # Normalize x to [0, 1-a]
    s = c/(1-a**b) # Scaling factor
    return ((1-a)/(x_h-x_l))*b*s*((1 - n(x))**(b-1))

  def uv(self, r, p):
    _uv = np.vectorize(IDevice._u, otypes=[float])
    return _uv(r, self.a, self.b, self.c, self.d, self.lbounds, self.hbounds) - r*p

  def u(self, r, p):
    return self.uv(r,p).sum()

  def deriv(self, r, p):
    _deriv = np.vectorize(IDevice._deriv, otypes=[float])
    return _deriv(r, self.a, self.b, self.c, self.d, self.lbounds, self.hbounds) - p

  @property
  def params(self):
    return {'a': self._a, 'b': self._b, 'c': self._c, 'd': self._d}

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
  def d(self):
    return self._d

  @params.setter
  def params(self, params):
    ''' Sanity check params.  '''
    if not isinstance(params, dict):
      raise ValueError('params to IDevice must be a dictionary')
    # Use preset params as defaults, ignore extras.
    p = IDevice.params.fget(self)
    p.update({k: v for k, v in params.items() if k in IDevice.params.fget(self).keys()})
    p = {k: np.array(v) for k, v in p.items()}
    for k,v in p.items():
      if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('params are required to have same length as device (%d)' % (len(self),))
      if not (v >= 0).all():
        raise ValueError('param %s must be >= 0' % (k,))
    (a,b,c,d) = (p['a'], p['b'], p['c'], p['d'])
    if not (a <= self.hbounds - self.lbounds).all():
      raise ValueError('param a (offset) cannot be larger than extent of domain')
    if not (b > 0).all():
      raise ValueError('param b must be > 0')
    (self._a, self._b, self._c, self._d) = (a,b,c,d)
