import numpy as np
from device_kit import Device, IDevice2


class CDevice2(Device):
  ''' Same curve as IDevice2 but applied to the scalar sum of consumption. '''
  d0 = 0
  d1 = 1

  def u(self, s, p):
    return IDevice2._u(s.sum(), d1=self.d1, d0=self.d0, x_l=self.cbounds[0], x_h=self.cbounds[1]) - (s*p).sum()

  def deriv(self, s, p):
    return IDevice2._deriv(s.sum(), self.d1, self.d0, self.cbounds[0], self.cbounds[1])*np.ones(len(self)) - p

  def hess(self, s, p=0):
    return np.eye(len(self))*IDevice2._hess(s.sum(), self.d1, self.d0, self.cbounds[0], self.cbounds[1])

  @property
  def params(self):
    return {'d1': self.d1, 'd0': self.d0}

  @params.setter
  def params(self, params):
    ''' Sanity check params.  '''
    if params is None:
      return
    if not isinstance(params, dict):
      raise ValueError('params to IDevice2 must be a dictionary')
    # Use preset params as defaults, ignore extras.
    p = IDevice2.params.fget(self)
    p.update({k: v for k, v in params.items() if k in IDevice2.params.fget(self).keys()})
    p = {k: np.array(v) for k, v in p.items()}
    for k, v in p.items():
      if not (v.ndim == 0 or len(v) == len(self)):
        raise ValueError('params are required to have same length as device (%d)' % (len(self),))
      if not (v >= 0).all():
        raise ValueError('param %s must be >= 0' % (k,))
    if not (p['d1'] >= p['d0']).all():
      raise ValueError('param d1 must be >= d0')
    (self.d1, self.d0) = (p['d1'], p['d0'])
