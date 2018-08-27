import numpy as np


class poly2d():
  ''' Wraps up an vector of poly1ds. Makes it slightly easier to get derivs. '''
  _c = None
  _f = None

  def __init__(self, c):
    ''' `c` should be an array of arrays. Each array the coeffs of a poly1d. '''
    len(c)
    [len(v) for v in c]
    self._c = list(c)
    self._f = lambda x: np.array([np.poly1d(c)(v) for c, v in zip(self._c, x)])

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
