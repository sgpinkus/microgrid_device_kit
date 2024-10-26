'''
Various preference functions. Can be used with ADevice.
'''
from abc import ABC, abstractmethod
import numpy as np
from numpy import stack, hstack, zeros, ones, poly1d, polyadd
import numdifftools as nd


class Function(ABC):
  @abstractmethod
  def __call__(self, x):
    pass

  @abstractmethod
  def deriv(self, x):
    pass

  @abstractmethod
  def hess(self, x):
    pass


class NullFunction(Function):
  ''' Default null function. f parameter has to ADevice has to look like this. '''

  def __call__(self, x):
    return 0

  def deriv(self, n=1):
    return lambda x: 0

  def hess(self):
    return lambda x: 0


class SumFunction(Function):
  ''' '''
  functions = []

  def __init__(self, functions: list[Function]) -> None:
    self.functions = functions

  def __call__(self, x):
    return np.array([v(x) for v in self.functions]).sum()

  def deriv(self, n=1):
    return lambda x: np.array([v.deriv()(x) for v in self.functions]).sum()

  def hess(self):
    return lambda x: np.array([v.hess()(x) for v in self.functions]).sum()


class poly2d():
  ''' Wraps up an vector of poly1ds. Makes it slightly easier to get derivs. '''
  _c = None
  _polys = None

  def __init__(self, c):
    ''' `c` should be an array of arrays. Each array the coeffs of a poly1d. '''
    self._c = np.array(c)
    self._polys = [poly1d(c) for c in self._c]

  def __call__(self, x):
    return np.array([self._polys[k](v) for k, v in enumerate(x.reshape(len(self)))])

  def __len__(self):
    return len(self._c)

  def deriv(self, n=1):
    return poly2d([polyadd(zeros(len(c)), poly1d(c).deriv(n).coeffs) for c in self._c])

  def hess(self):
    return lambda x: np.diag(self.deriv(2)(x))

  @property
  def coeffs(self):
    return self._c


class InformationEntropy():
  ''' Information entropy preference function. Entropy is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return self.c*InformationEntropy.info_entropy(r)

  def deriv(self):
    return nd.Jacobian(lambda x: self.c*InformationEntropy.info_entropy(x))

  def hess(self):
    return nd.Hessian(lambda x: self.c*InformationEntropy.info_entropy(x))

  @staticmethod
  def info_entropy(r):
    r = np.array([i for i in np.abs(r) if i])
    r = r/r.sum()
    return (r*np.log(r)).sum()


class TemporalVariance():
  ''' Equivalent to moment of inertia if time is distance and consumption is mass. Inertia is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return self.c*TemporalVariance.inertia(r)

  def deriv(self):
    return nd.Jacobian(lambda x: self.c*TemporalVariance.inertia(x))

  def hess(self):
    return nd.Hessian(lambda x: self.c*TemporalVariance.inertia(x))

  @staticmethod
  def inertia(r):
    t = np.arange(len(r))
    return (((t - TemporalVariance.com(r))**2)*r).sum()

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)


class CobbDouglas():
  ''' Cobb Douglas *with* constant returns to scale. '''

  def __init__(self, a, c=1):
    self.a = a # Cobb Douglas coefficients.
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return self.c*CobbDouglas.cobb_douglas(r, self.a)

  def deriv(self):
    return nd.Jacobian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))

  def hess(self):
    return nd.Hessian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))

  @staticmethod
  def cobb_douglas(r, a):
    return (r**(a/a.sum())).prod()

class QuadraticCost1():
  a = 0
  b = 2
  c = 1
  x_l = None
  x_h = None

  def __init__(self, a, b, c, x_l, x_h):
    [self.a, self.a, self.c, self.x_l, self.x_h] = a, b, c, x_l, x_h
    self._cost_fn = lambda x: np.vectorize(QuadraticCost1._cost, otypes=[float])(x, self.a, self.b, self.c, self.x_l, self.x_h)
    self._deriv_fn = lambda x: np.vectorize(QuadraticCost1._deriv, otypes=[float])(np.array(x).reshape(-1), self.a, self.b, self.c, self.x_l, self.x_h)
    self._hess_fn = lambda x: np.diag(np.vectorize(QuadraticCost1._hess, otypes=[float])(np.array(x).reshape(-1), self.a, self.b, self.c, self.x_l, self.x_h))

  def __call__(self, x):
    return self._cost_fn(x).sum()

  def deriv(self):
    return self._deriv_fn

  def hess(self):
    return self._hess_fn

  @staticmethod
  def _cost(x, a, b, c, x_l, x_h):
    ''' The cost function on scalar. '''
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*((1 - QuadraticCost1.scale(x, x_l, x_h, a))**b)

  @staticmethod
  def _deriv(x, a, b, c, x_l, x_h):
    ''' The derivative of cost function on scalar. '''
    if x_l == x_h:
      return 0
    return -(c/(1-a**b))*((1-a)/(x_h-x_l))*b*((1 - QuadraticCost1.scale(x, x_l, x_h, a))**(b-1))

  @staticmethod
  def _hess(x, a, b, c, x_l, x_h):
    if x_l == x_h:
      return 0
    return (c/(1-a**b))*((1-a)/(x_h-x_l))**2*b*(b-1)*((1 - QuadraticCost1.scale(x, x_l, x_h, a))**(b-2))

  @staticmethod
  def scale(x, x_l, x_h, a):
    return (1-a)*((x - x_l)/(x_h - x_l))


class QuadraticCost2():
  p_l = -1
  p_h = 0
  x_l = x_h = None
  _cost_fn = None
  _deriv_fn = None
  _hess_fn = None

  def __init__(self, p_l, p_h, x_l, x_h):
    [self.p_l, self.p_h, self.x_l, self.x_h] = p_l, p_h, x_l, x_h
    self._cost_fn = lambda x: np.vectorize(QuadraticCost2._cost, otypes=[float])(x, self.p_l, self.p_h, self.x_l, self.x_h)
    self._deriv_fn = lambda x: np.vectorize(QuadraticCost2._deriv, otypes=[float])(np.array(x).reshape(-1), self.p_l, self.p_h, self.x_l, self.x_h)
    self._hess_fn = lambda x: np.diag(np.vectorize(QuadraticCost2._hess, otypes=[float])(np.array(x).reshape(-1), self.p_l, self.p_h, self.x_l, self.x_h))

  def __call__(self, r):
    return self._cost_fn(r).sum()

  def deriv(self):
    return self._deriv_fn

  def hess(self):
    return self._hess_fn

  @staticmethod
  def _cost(x, p_l, p_h, x_l, x_h):
    ''' The cost function on scalar. '''
    if x_l == x_h:
      return 0
    b = p_l
    a = (p_h - p_l)/2
    c = a*(-b/(2*a))**2 + b*(-b/(2*a))
    return (x_h - x_l)*np.poly1d([a, b, 0])((x - x_l)/(x_h - x_l)) - c*(x_h - x_l)

  @staticmethod
  def _deriv(x, p_l, p_h, x_l, x_h):
    ''' The derivative of cost function on a scalar. Returned valus is expansion of:
         np.poly1d([(p_h - p_l)/2, p_l, 0]).deriv()(QuadraticCost12.scale(x, x_l, x_h))
    '''
    if x_l == x_h:
      return 0
    return (p_h - p_l)*((x - x_l)/(x_h - x_l)) + p_l

  @staticmethod
  def _hess(x, p_l, p_h, x_l, x_h):
    ''' The 2nd derivative of cost function on a scalar. '''
    if x_l == x_h:
      return 0
    return (p_h - p_l)/(x_h - x_l)