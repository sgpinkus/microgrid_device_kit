import re
import numpy as np
from pprint import pformat
from lcl.projection import *

class Device:
  ''' Base class for any type of device (AKA appliance). Devices are all characterised by having:

      - A fixed length `len` which is the number of timeslots the device consumes/produces some resource.
      - A list of low/high resource consumption `bounds` of length `len`.
      - Optional cummulative bounds (`cbounds`) for resource consumption over `len`.
      - A concave differentiable utility function `u`, which represents how much value the device
          gets from consuming/producing a given resource vector of length `len` at some price.

    This class is more or less a dumb container for these settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function. Sub devices can also
    implement more complex constraints than acheivable with bounds and cbounds. Constraints should
    be convex but this is not currently enforced.

    Device should be considered immutable. The available setters are all used on init.

    Python3 @properties have been used throughout these classes. They mainly serve as very verbose,
    and slow way to protect a field, by only defining a getter. Setters are also sparingly defined.

    @todo deprecate price as a parameter to u(), deriv() (can just set price to 0 for now).
  '''
  _id = None              # The identifier of this device.
  _len = 0                # Fixed length of the following vectors / the planning window.
  _bounds = None          # Vector of 2-tuple min/max bounds on r.
  _cbounds = None         # 2-Tuple cummulative min/max bounds. Cummulative bounds are optional.
  _feasible_region = None # Convex region representing bounds and cbounds. Convenience.

  def __init__(self, id, length, bounds, cbounds=None, params=None):
    ''' Initially set resource to closest feasible point to mid point between bounds. and price to zero. '''
    if type(id) != str or not re.match('^(?i)[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be a non empty string matching ^(?i)[a-z0-9][a-z0-9_-]*$')
    if not hasattr(bounds, '__len__'):
      raise ValueError('bounds must be a sequence type')
    if len(bounds) != length:
      raise ValueError('bounds has incorrect length')
    self._len = length
    self._id = id
    self.bounds = bounds
    self.cbounds = cbounds
    self.params = params

  def __str__(self):
    ''' Print main settings. Dont print the actual min/max bounds vectors because its too verbose. '''
    return 'id=%s; len=%d; *bounds=%.3f/%.3f; cbounds=%s; params=%s' % \
      (self.id, len(self), self.lbounds.min(), self.hbounds.max(), self.cbounds, pformat(self.params))

  def __len__(self):
    return self._len

  def u(self, r, p):
    ''' Get scalar utility value for `r` consumption at price `p` '''
    return 0

  def deriv(self, r, p):
    ''' Get jacobian vector of the utility at `r`, and price `p` '''
    return np.zeros(len(self))

  @property
  def id(self):
    return self._id

  @property
  def bounds(self):
    return self._bounds

  @property
  def lbounds(self):
    return np.array(self.bounds[:,0])

  @property
  def hbounds(self):
    return np.array(self.bounds[:,1])

  @property
  def cbounds(self):
    return self._cbounds

  @property
  def minsum(self):
    return self.cbounds[0] if self.cbounds is not None else self.lbounds.sum()

  @property
  def maxsum(self):
    return self.cbounds[1] if self.cbounds is not None else self.hbounds.sum()

  @property
  def params(self):
    return self._params

  @property
  def feasible_region(self):
    return self._feasible_region

  @property
  def care_time(self):
    ''' (lbounds == hbouds) && (lbounds == 0) '''
    return np.array(((self.lbounds != 0) | (self.lbounds != self.hbounds)), dtype=float)

  @bounds.setter
  def bounds(self, bounds):
    if len(bounds) != len(self):
      raise ValueError('bounds has wrong length (%d)' % len(bounds))
    bounds = np.array(bounds)
    lbounds = np.array(bounds[:,0])
    hbounds = np.array(bounds[:,1])
    if not np.vectorize(lambda v: v is None)(bounds).all() and not (hbounds - lbounds >= 0).all():
      raise ValueError('max bound must be >= min bound for all min/max bound pairs: %s' % (str(hbounds - lbounds),))
    self._bounds = bounds
    self._build_feasible_region()

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' Set cbounds ensuring they are feasible wrt (l|h)bounds. '''
    if cbounds == None:
      self._cbounds = None
      return
    if len(cbounds) != 2:
      raise ValueError('len(cbounds) must be 2')
    if cbounds[1] - cbounds[0] < 0:
      raise ValueError('max cbound (%f) must be >= min cbound (%f)' % (cbounds[1], cbounds[0]))
    if self.lbounds.sum() > cbounds[1]:
      raise ValueError('cbounds infeasible; min possible sum (%f) is > max cbounds (%f)' % (self.lbounds.sum(), cbounds[1]))
    if self.hbounds.sum() < cbounds[0]:
      raise ValueError('cbounds infeasible; max possible sum (%f) is < min cbounds (%f)' % (self.hbounds.sum(), cbounds[0]))
    self._cbounds = cbounds
    self._build_feasible_region()

  @params.setter
  def params(self, params):
    if params != None:
      raise ValueError()
    self._params = params

  def is_feasible(self, r):
    ''' Check bounds and cbounds '''
    if not (r - self.bounds[0] >= 0).all():
      return False
    if not (r - self.bounds[1] <= 0).all():
      return False
    if self.cbounds:
      if self.cbounds[0] != None and r.sum() < self.cbounds[0]:
        return False
      if self.cbounds[1] != None and r.sum() > self.cbounds[1]:
        return False
    return True

  def project(self, r):
    ''' Project vector r onto own feasible region. I.e. make `r` feasible. '''
    return self._feasible_region.project(r)

  def to_list(self):
    ''' Serialize '''
    return (str(self.__class__), self.id, len(self), list(self.bounds), self.cbounds, self.params)

  def fromlist(self, l):
    ''' return l[0](l[1], l[2], np.array(l[3]), l[4], l[5]) '''
    raise NotImplementedError()

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Contraints at this level are just cbounds. bounds constraints are generally handled separately.
    '''
    constraints = []
    if self.cbounds:
      constraints += [{
        'type': 'ineq',
        'fun': lambda r: r.dot(np.ones(len(self))) - self.cbounds[0],
        'jac': lambda r: np.ones(len(self))
      },
      {
        'type': 'ineq',
        'fun': lambda r: self.cbounds[1] - r.dot(np.ones(len(self))),
        'jac': lambda r: -1*np.ones(len(self))
      }]
    return constraints

  def _build_feasible_region(self):
    region = HyperCube(self.bounds)
    if self.cbounds is not None:
      region = Intersection(region, Slice(np.ones(len(self)), self.cbounds[0], self.cbounds[1]))
    self._feasible_region = region
