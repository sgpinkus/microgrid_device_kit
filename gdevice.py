import numpy as np
import numdifftools as nd
from powermarket.device import Device


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
  _cost = [0,]
  _cost_function = None

  def uv(self, r, p):
    ''' Get utility vector for r, p. '''
    return -1*r*p - self._cost_function(-r)

  def u(self, r, p):
    return self.uv(r, p).sum()

  def deriv(self, r, p):
    ''' Get jacobian vector of the utility at `r`, at price `p` '''
    return -p + self._deriv_function(-r)

  def hess(self, r, p=0):
    ''' @todo actually easy to deriv explicitly ... '''
    return nd.Hessian(lambda x: self.u(x,0))(r)

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cost(self):
    ''' Return arrays of coeffs of cost function not an actual function. '''
    return self._cost

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
    lbounds = np.array(bounds[:,0])
    hbounds = np.array(bounds[:,1])
    if not (hbounds <= 0).all():
      raise ValueError('hbounds must be all zeros')
    Device.bounds.fset(self, bounds)

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' @override don't allow cbounds
    @todo to allow -ve cbounds.
    '''
    if cbounds == None:
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
      self._cost_function = np.poly1d(cost)
      self._deriv_function = np.poly1d(cost).deriv()
    elif np.array(cost).ndim == 2:
      _cost = [np.poly1d(c) for c in cost]
      self._cost_function = lambda r: np.array([c(r) for c, r in zip(_cost, r)])
      self._deriv_function = lambda r:np.array([c.deriv()(r) for c, r in zip(_cost, r)])
    else:
      raise ValueError('cost param must be array with 1 or 2 dimensions.')
    self._cost = cost
