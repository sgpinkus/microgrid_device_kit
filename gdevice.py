import numpy as np
from device_kit import Device
from .functions import poly2d


class GDevice(Device):
  ''' Simple generator device. Strictly produces power. Cost of power is specified by an arbitrary
  polynomial. Thermal generators generally have quadratic cost curves but allow arbitrary cost curves.
  Device takes a single parameter `cost`, which defines the coefficients of the polynomial at each
  timeslot. `cost` can be a single array of coeffs or an array of len(self) arrays of coeffs (one for
  each timeslot).

  Generation is always indicated by negative values. However the cost function provided should define
  *positive* costs over a *positive* range. Example [1,1,0] for cost(q) = q**2 + q

  Time variable max generation capacity is specified by setting `lbounds`. GDevice enforces that
  hbounds must be <=0 for this device (can never consume).

  Cost function for all timeslots are independent. Start up / shut down and min/max runtimes are not
  considered.

  Note for generators, `utility` is interpreted as profit (which is revenue - cost)

  @todo Could just allow arbitrary bounds. Then this could be used like IDevice too.
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
    return -p + self._cost_d1_fn(-s)

  def hess(self, s, p=0):
    ''' Return hessian. Hessdiag == Hessian for the given utility function.
    @todo actually easy to deriv explicitly ...
    '''
    return -1*np.diag(self._cost_d2_fn(-s))

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
    ''' @override bounds setter to ensure hbounds = 0. '''
    if len(bounds) != len(self):
      raise ValueError('bounds has wrong length (%d)' % len(bounds))
    bounds = np.array(bounds)
    lbounds = np.array(bounds[:, 0])
    hbounds = np.array(bounds[:, 1])
    if not (hbounds <= 0).all():
      raise ValueError('hbounds must be all zeros')
    Device.bounds.fset(self, bounds)

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
