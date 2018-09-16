import re
import numpy as np
from pprint import pformat
from powermarket.projection import *
from powermarket.device import *


class Device:
  ''' Base class for any type of device (AKA appliance). Devices are all characterised by having:

      - A fixed length `len` which is the number of timeslots the device consumes/produces some resource.
      - A list of low/high resource consumption `bounds` of length `len`.
      - Optional cummulative bounds (`cbounds`) for resource consumption over `len`.
      - A concave differentiable utility function `u`, which represents how much value the device
          gets from consuming/producing a given resource vector of length `len` at some price.

    This class is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function.

    Because the bounds and cbounds constraints are so common they are implemented in this base Device
    class for convenience. Sub devices can implement more complex constraints than acheivable with
    bounds` and cbounds. Constraints should be convex but this is not currently enforced. Note try
    to maintain:

      - Device is stateless.
      - Device should be considered immutable (the currently available setters are all used on init).
      - Device is serializable and and constructable from the serialization.

    Note Python3 @properties have been used throughout these classes. They mainly serve as very
    verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.
  '''
  _id = None              # The identifier of this device.
  _len = 0                # Fixed length of the following vectors / the planning window.
  _bounds = None          # Vector of 2-tuple min/max bounds on r.
  _cbounds = None         # 2-Tuple cummulative min/max bounds. Cummulative bounds are optional.
  _feasible_region = None # Convex region representing *only* bounds and cbounds. Convenience.
  _params = None

  def __init__(self, id, length, bounds, cbounds=None, params=None):
    ''' Initially set resource to closest feasible point to mid point between bounds. and price to
    zero. Sub classes should just override `params.setter` not `__init__`.
    @todo params should have just been **kwargs.
    '''
    if type(id) != str or not re.match('^(?i)[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be a non empty string matching ^(?i)[a-z0-9][a-z0-9_-]*$')
    self._len = length
    self._id = id
    self.bounds = bounds
    self.cbounds = cbounds
    self.params = params
    self._build_feasible_region()

  def __str__(self):
    ''' Print main settings. Dont print the actual min/max bounds vectors because its too verbose. '''
    return 'id=%s; len=%d; *bounds=%.3f/%.3f; cbounds=%s; params=%s' % \
      (self.id, len(self), self.lbounds.min(), self.hbounds.max(), self.cbounds, pformat(self.params))

  def __len__(self):
    return self._len

  def u(self, r, p):
    ''' Get scalar utility value for `r` consumption, at price (parameter) `p`. This base Device's
    utility function makes an assumption device cares linearly about costs. Generally all sub devices
    should do this too.
    '''
    return (-r*p).sum()

  def deriv(self, r, p):
    ''' Get jacobian vector of the utility at `r`, at price `p`, which is just -p. '''
    return -p

  def hess(self, r, p=0):
    ''' Get hessian vector of the utility at `r`, at price `p`. With linear utility for the numeriare
    price drops out.
    '''
    return np.zeros((len(self), len(self)))

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
    ''' Get params in same format as passed in. '''
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
    ''' Set bounds. Can be input as a pair which will be repeated len times. Conversion is one way currently. '''
    if not hasattr(bounds, '__len__'):
      raise ValueError('bounds must be a sequence type')
    if len(bounds) == 2:
      bounds = np.array([bounds for i in range(0,len(self))])
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
    if not hasattr(cbounds, '__len__') or len(cbounds) != 2:
      raise ValueError('len(cbounds) must be 2')
    if cbounds[1] - cbounds[0] < 0:
      raise ValueError('max cbound (%f) must be >= min cbound (%f)' % (cbounds[1], cbounds[0]))
    if self.lbounds.sum() > cbounds[1]:
      raise ValueError('cbounds infeasible; min possible sum (%f) is > max cbounds (%f)' % (self.lbounds.sum(), cbounds[1]))
    if self.hbounds.sum() < cbounds[0]:
      raise ValueError('cbounds infeasible; max possible sum (%f) is < min cbounds (%f)' % (self.hbounds.sum(), cbounds[0]))
    self._cbounds = tuple(cbounds)
    self._build_feasible_region()

  @params.setter
  def params(self, params):
    if params != None:
      raise ValueError()

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

  def to_dict(self):
    ''' Serialize '''
    return {
      'id': self.id,
      'length': len(self),
      'bounds': list(self.bounds),
      'cbounds': self.cbounds,
      'params': self.params
    }

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)

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
