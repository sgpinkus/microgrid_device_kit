import numpy as np
from powermarket.device import Device


class UDevice(Device):
  ''' Device that takes an arbitrary utility function (should have been decoupled like this in the
  first place - argh). This device assumes strictly linear utility for cost (like all other devices).
  '''
  _f = 0

  def u(self, r, p):
    return self._f(r) - (r*p).sum()

  def deriv(self, r, p):
    return self._f.deriv()(r) - p

  def hess(self, r, p=0):
    return self._f.hess()(r)

  @property
  def params(self):
    return {'f': self.f}

  @property
  def f(self):
    return self._f

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if not isinstance(params, dict):
      raise ValueError('params incorrect type')
    f = params['f']
    if not hasattr(f, '__call__') and hasattr(f, 'deriv') and hasattr(f, 'hess'):
      raise ValueError('invalid parameter type')
    self._f = f
