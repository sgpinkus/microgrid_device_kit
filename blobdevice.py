import numpy as np
from numpy import stack, hstack, zeros, ones
import numdifftools as nd
from device_kit import ADevice
from device_kit.functions import *


class BlobDevice(ADevice):
  ''' Overrides ADevice but constraint and f are not params. Only c is '''
  c = 1
  w = None

  @property
  def params(self):
    return {'c': self.c, 'w': self.w}

  @params.setter
  def params(self, params):
    params = {} if params is None else params
    self.c = params['c'] if 'c' in params else 1
    self.w = params['w']
    params['f'] = WindowPenalty(c=self.c, w=self.w)
    ADevice.params.fset(self, params)


class WindowPenalty():
  ''' Equivalent to moment of inertia if time is distance and consumption is mass. Inertia is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, w, c=0.1):
    self.c = c
    self.w = w

  def __call__(self, r):
    return -self.c*(WindowPenalty.weights(r, self.w)*r).sum()

  def deriv(self):
    return lambda x: -self.c*WindowPenalty.weights(x, self.w)

  def hess(self):
    return lambda x: np.zeros((len(x), len(x)))

  @staticmethod
  def weights(r, w):
    return np.maximum(0, np.abs(np.arange(len(r)) - WindowPenalty.com(r)) - w/2)

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)
