import numpy as np
from numpy import stack, hstack, zeros, ones
import numdifftools as nd
from powermarket.device import ADevice
from powermarket.device.functions import *


class BlobDevice(ADevice):
  ''' Overrides ADevice but constraint and f are not params. Only c is '''
  c = 1

  @property
  def params(self):
    return {'c': self.c}

  @params.setter
  def params(self, params):
    params = {} if params is None else params
    self.c = params['c'] if 'c' in params else 1
    params['f'] = TemporalVariance(self.c)
    ADevice.params.fset(self, params)


class TemporalVariance():
  ''' Callable preference function suitable for ADevice.f
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return -self.c*TemporalVariance.inertia(r)

  def deriv(self):
    return nd.Jacobian(lambda x: -self.c*TemporalVariance.inertia(x))

  def hess(self):
    return nd.Hessian(lambda x: -self.c*TemporalVariance.inertia(x))

  @staticmethod
  def inertia(r):
    t = np.arange(len(r))
    return (((t - TemporalVariance.com(r))**2)*r).sum()

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)
