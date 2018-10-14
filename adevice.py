import numpy as np
from powermarket.device import Device
from powermarket.device.poly2d import poly2d


class ADevice(Device):
  ''' Device that takes an arbitrary utility function and constraints. '''
  f = poly2d(np.zeros((1,24)))
  constraints = []

  def u(self, s, p):
    return self.f(s) - (s*p).sum()

  def deriv(self, s, p):
    return self.f.deriv()(s) - p

  def hess(self, s, p=0):
    return self.f.hess()(s)

  @property
  def params(self):
    return {'f': self.f, 'constraints': self.constraints}

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if not isinstance(params, dict):
      raise ValueError('params incorrect type')
    _params = self.params
    _params.update(params)
    if not hasattr(_params['f'], '__call__') and hasattr(_params['f'], 'deriv') and hasattr(_params['f'], 'hess'):
      raise ValueError('Invalid parameter type for \'f\' [%s]' % (type(f)))
    self.f = _params['f']
    self.constraints = _params['constraints']
