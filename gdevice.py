import numpy as np
from lcl.device import Device


class GDevice(Device):
  ''' Simple generator device. Strictly produces power. Cost of power is specified by some polynomial.
  Thermal generator generally have quadratic cost curves but polynomial allow arbitrary cost curves.
  The single cost parameter is either a single array of polynomial cooefs or and array of such coeff
  arrays (one for each timeslot). Cost is currently limited to being strictly separable in time slots.
  Note generation is always indicate by negative values and cost functions should take this into account.

  Max generation capacity can be time variable by setting `lbounds`. We enforce that hbounds must
  be <=0 for this device (can't consume). Start up / shut down and min/max runtimes are not considered.

  Note for generators, `utility` is interpreted as profit (which is revenue - cost)

  @todo Allow cost to be an array of arrays (one for each timeslot).
  '''
  _cost = [0,]

  def uv(self, r, p):
    ''' Get utility vector for r, p. '''
    return -1*p*r - self.cost(-r)

  def u(self, r, p):
    return self.uv(r, p).sum()

  def deriv(self, r, p):
    ''' Get jacobian vector of the utility at `r`, at price `p` '''
    return -1*p + self._deriv(-r)

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cbounds(self):
    return self._cbounds

  @property
  def cost(self):
    return self._cost

  @property
  def params(self):
    return {
      'cost': self.cost,
    }

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
    p = self.params
    p.update(params)
    self.cost = p['cost']

  @cost.setter
  def cost(self, cost):
    cost = np.array(cost)
    if cost.ndim == 1:
      self._cost = np.poly1d(cost)
      self._deriv = np.poly1d(cost).deriv()
    elif cost.ndim == 2:
      cost = [np.poly1d(c) for c in cost]
      self._cost = lambda r: np.array([c(r) for c, r in zip(cost, r)])
      self._deriv = lambda r: np.array([c.deriv()(r) for c, r in zip(cost, r)])
    else:
      raise ValueError('cost param must be 1 or 2 dim array.')
