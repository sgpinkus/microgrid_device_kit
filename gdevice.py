import numpy as np
from device_kit import Device
from .functions import poly2d


class GDevice(Device):
  ''' Simple generator device. Strictly produces power. Cost of power is specified by an arbitrary
  polynomial.

  GDevice takes a single parameter `cost`, which defines the coefficients of the polynomial at each
  timeslot. `cost` can be a single array of coeffs or an array of len(self) arrays of coeffs (one for
  each timeslot). Thermal generators generally have quadratic cost curves.

  Generation is always indicated by negative values. However the cost function provided should define
  *positive* costs over a *positive* range. Example [1,1,0] for cost(q) = q**2 + q

  Time variable max generation capacity is specified by setting `lbounds`. GDevice enforces that
  hbounds must be <=0 for this device (can never consume).

  Cost function for all timeslots are independent. Start up / shut down and min/max runtimes are not
  considered.

  Note for generators "utility" is interpreted as profit (which is revenue - cost)
  '''
  _cost_fn = None
  _cost_d1_fn = None
  _cost_d2_fn = None

  def uv(self, s, p):
    ''' Get utility vector for s, p. '''
    return -1*s*p - self._cost_fn(-s)

  def u(self, s, p):
    return self.uv(s, p).sum()

  def deriv(self, s, p):
    ''' Get jacobian vector of the utility at `r`, at price `p` '''
    return self._cost_d1_fn(-s.reshape(len(self))) - p

  def hess(self, s, p=0):
    ''' Return hessian. Hessdiag == Hessian for the given utility function.
    @todo actually easy to deriv explicitly ...
    '''
    return -1*np.diag(self._cost_d2_fn(-s.reshape(len(self))))

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cost(self):
    ''' Return arrays of coeffs of cost function not an actual function. '''
    return self._cost_fn.coeffs

  @property
  def params(self):
    return {
      'cost': self.cost,
    }

  @property
  def cbounds(self):
    return self._cbounds

  @bounds.setter
  def bounds(self, bounds):
    ''' @override bounds setter to ensure hbounds <= 0. '''
    Device.bounds.fset(self, bounds)
    bounds = np.array(bounds)
    if not (self.hbounds <= 0).all():
      raise ValueError('hbounds must be <= 0')

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' @override don't allow cbounds
    @todo to allow -ve cbounds.
    '''
    if cbounds is None:
      self._cbounds = None
    else:
      raise ValueError('cbounds not allowed for GDevice currently')

  @params.setter
  def params(self, params):
    ''' Sanity check params. Always called at init by contract. '''
    if not isinstance(params, dict):
      raise ValueError('params to GDevice must be a dictionary')
    self.cost = params['cost']

  @cost.setter
  def cost(self, cost):
    if np.array(cost).ndim == 1:
      self._cost_fn = np.poly1d(cost)
      self._cost_d1_fn = self._cost_fn.deriv()
      self._cost_d2_fn = self._cost_fn.deriv(2)
    elif np.array(cost).ndim == 2:
      self._cost_fn = poly2d(cost)
      self._cost_d1_fn = self._cost_fn.deriv()
      self._cost_d2_fn = self._cost_fn.deriv(2)
    else:
      raise ValueError('cost param must be array with 1 or 2 dimensions.')
