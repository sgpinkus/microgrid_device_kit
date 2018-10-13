import numpy as np


class poly2d():
  ''' Wraps up an vector of poly1ds. Makes it slightly easier to get derivs. '''
  _c = None
  _f = None
  _polys = None

  def __init__(self, c):
    ''' `c` should be an array of arrays. Each array the coeffs of a poly1d. '''
    self._c = np.array(c)
    self._polys = [np.poly1d(c) for c in self._c]
    self._f = lambda x: np.array([self._polys[k](v) for k, v in enumerate(x.reshape(len(self)))])

  def __call__(self, x):
    return self._f(x)

  def __len__(self):
    return len(self._c)

  def deriv(self, n=1):
    return poly2d([np.poly1d(c).deriv(n).coeffs for c in self._c])

  def hess(self):
    return lambda x: np.diag(self.deriv(2)(x))

  @property
  def coeffs(self):
    return self._c
