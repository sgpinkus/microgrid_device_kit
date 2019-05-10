import re
import numpy as np
import numbers
from pprint import pformat
from device_kit import BaseDevice
from device_kit.projection import *


class Device(BaseDevice):
  ''' BaseDevice implementation for single Device. '''
  _id = None              # The identifier of this device.
  _len = 0                # Fixed length of the following vectors / the planning window.
  _bounds = None          # Vector of 2-tuple min/max bounds on r.
  _cbounds = None         # 2-Tuple cummulative min/max bounds. Cummulative bounds are optional.
  _feasible_region = None  # Convex region representing *only* bounds and cbounds. Convenience.
  _keys = None

  def __init__(self, id, length, bounds, cbounds=None, **meta):
    ''' Validate and set field. Build an incomplete feasible_region for convenience sake. Class allows
    any arbitrary field to be passed which are set on the object and add to field that will be
    serialized with an instance (fields that are set on the object after init are not serialized. This
    is by design). Sub-classes may choose to override this initializer to provide complex object
    construction. In this case they must add additional keys to _keys or also override the to_dict()
    method. Alternatively they may just define setters which will be called for all keys in **meta.
    '''
    if not isinstance(id, str) or not re.match('^(?i)[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be a non empty string matching ^(?i)[a-z0-9][a-z0-9_-]*$')
    self._length = length
    self._id = id
    self.bounds = bounds
    self.cbounds = cbounds
    self._build_feasible_region()
    self._keys = ['id', 'length', 'bounds', 'cbounds']
    for k, v in meta.items():
      setattr(self, k, v)
    self._keys += list(meta.keys())

  def __str__(self):
    ''' Print main settings. Dont print the actual min/max bounds vectors because its too verbose. '''
    _str = 'id=%s; length=%d; *bounds=%.3f/%.3f; cbounds=%s' % (self.id, len(self), self.lbounds.min(), self.hbounds.max(), self.cbounds)
    _str = '; '.join([_str] + ['{k}={v}'.format(k=k, v=getattr(self, k)) for k in self._keys if k not in ['id', 'length', 'bounds', 'cbounds']])
    return _str

  def __len__(self):
    return self._length

  def u(self, s, p):
    ''' Get scalar utility value for `s` consumption, at price (parameter) `p`. This base Device's
    utility function makes an assumption device cares linearly about costs. Generally all sub devices
    should do this too.
    '''
    return (-s*p).sum()

  def deriv(self, s, p):
    ''' Get jacobian vector of the utility at `s`, at price `p`, which is just -p. '''
    return -p*np.ones(len(self))

  def hess(self, s, p=0):
    ''' Get hessian vector of the utility at `s`, at price `p`. With linear utility for the numeriare
    price drops out.
    '''
    return np.zeros((len(self), len(self)))

  @property
  def id(self):
    return self._id

  @property
  def length(self):
    return self._length

  @property
  def shape(self):
    return (1, len(self))

  @property
  def shapes(self):
    return np.array([self.shape])

  @property
  def partition(self):
    ''' Returns array of (offset, length) tuples for each sub-device's mapping onto this device's
    flow matrix.
    '''
    return np.array([[0,1]])

  @property
  def bounds(self):
    return self._bounds

  @property
  def lbounds(self):
    return np.array(self.bounds[:, 0])

  @property
  def hbounds(self):
    return np.array(self.bounds[:, 1])

  @property
  def cbounds(self):
    return self._cbounds

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Contraints at this level are just cbounds. bounds constraints are generally handled separately.
    '''
    constraints = []
    if self.cbounds:
      constraints += [{
        'type': 'ineq',
        'fun': lambda s: s.dot(np.ones(len(self))) - self.cbounds[0],
        'jac': lambda s: np.ones(len(self))
      },
      {
        'type': 'ineq',
        'fun': lambda s: self.cbounds[1] - s.dot(np.ones(len(self))),
        'jac': lambda s: -1*np.ones(len(self))
      }]
    return constraints

  @property
  def params(self):
    ''' ~BWC '''
    return self.to_dict()

  @bounds.setter
  def bounds(self, bounds):
    ''' Set bounds. Convert to consistent format which is a (len(self), 2) shaped ndarray. Input
    format is lost. Input can be (len(self), 2) shape array, or a pair, each entry of which must be
    a scalar or arrary of len(self)
    '''
    if not hasattr(bounds, '__len__'):
      raise ValueError('bounds must be a sequence type')
    if len(bounds) == 2:
      bounds = list(bounds)
      if isinstance(bounds[0], numbers.Number):
        bounds[0] = np.repeat(bounds[0], len(self))
      if isinstance(bounds[1], numbers.Number):
        bounds[1] = np.repeat(bounds[1], len(self))
      if len(bounds[0]) == len(bounds[1]) == len(self):
        bounds = np.stack((bounds[0], bounds[1]), axis=1)
    if len(bounds) != len(self):
      raise ValueError('bounds has wrong length (%d). Require %d' % (len(bounds), len(self)))
    bounds = np.array(bounds)
    lbounds = np.array(bounds[:, 0])
    hbounds = np.array(bounds[:, 1])
    if not np.vectorize(lambda v: v is None)(bounds).all() and not (hbounds - lbounds >= 0).all():
      raise ValueError('max bound must be >= min bound for all min/max bound pairs: %s' % (str(hbounds - lbounds),))
    self._bounds = bounds
    self._build_feasible_region()

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' Set cbounds ensuring they are feasible wrt (l|h)bounds. '''
    if cbounds is None:
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
    ''' Convenience buld setter. '''
    for k, v in params.items():
      setattr(self, k ,v)

  def project(self, s):
    return self._feasible_region.project(s.reshape(len(self)))

  def to_dict(self):
    ''' Dump object as dict. Dict should allow re-init of instance via cls(**data). See __init__(),
    from_dict().
    '''
    data = {k: getattr(self, k) for k in self._keys}
    return data

  def _build_feasible_region(self):
    region = HyperCube(self.bounds)
    if self.cbounds is not None:
      region = Intersection(region, Slice(np.ones(len(self)), self.cbounds[0], self.cbounds[1]))
    self._feasible_region = region
