import re
import logging
import numpy as np
from pprint import pformat
from device_kit import BaseDevice
from device_kit.projection import *
from warnings import warn

class Device(BaseDevice):
  ''' BaseDevice implementation for single Device. '''
  _id = None              # The identifier of this device.
  _len = 0                # Fixed length of the following vectors / the planning window.
  _bounds = None          # Vector of 2-tuple min/max bounds on r.
  _cbounds = None         # list of 4-Tuple cummulative min/max bounds. Cummulative bounds are optional.
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
    if not isinstance(id, str) or not re.match('(?i)^[a-z0-9][a-z0-9()\\[\\]\\+_-]*$', id):
      warn('id should be a non empty string matching "^(?i)[a-z0-9][a-z0-9_-]*$" not "%s"' % (id,))
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
    _str = 'type=%s; id=%s; length=%d; bounds_bounds=%.3f/%.3f; cbounds=%s' % (
      self.__class__.__name__,
      self.id,
      len(self),
      self.lbounds.min(),
      self.hbounds.max(),
      self.cbounds
    )
    _str = '; '.join([_str] + ['{k}={v}'.format(k=k, v=getattr(self, k)) for k in self._keys if k not in ['id', 'length', 'bounds', 'cbounds']])
    return _str

  def __len__(self):
    return self._length

  def cost(self, s, p):
    ''' Get scalar cost (inverse utility) value for `s` consumption, at price (parameter) `p`. This base Device's
    cost function makes an assumption device cares linearly about costs. Generally all sub devices
    should do this too.
    '''
    return (s*p).sum()

  def deriv(self, s, p):
    ''' Get jacobian vector of the cost at `s`, at price `p`, which is just -p. '''
    return p*np.ones(len(self))

  def hess(self, s, p=0):
    ''' Get hessian vector of the cost at `s`, at price `p`. With linear cost for the numeriare
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
      for cbound in self.cbounds:
        l, h, s, e = cbound
        constraints += [{
          'type': 'ineq',
          'fun': lambda s: s.dot(np.ones(len(self))) - l,
          'jac': lambda s: np.ones(len(self))
        },
        {
          'type': 'ineq',
          'fun': lambda s: h - s.dot(np.ones(len(self))),
          'jac': lambda s: -1*np.ones(len(self))
        }]
    return constraints

  @property
  def params(self):
    ''' ~BWC '''
    return self.to_dict()

  @bounds.setter
  def bounds(self, bounds):
    ''' See _validate_bounds() '''
    bounds = self.validate_bounds(bounds)
    self._bounds = bounds
    self._build_feasible_region()

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' Set cbounds ensuring they are feasible wrt (l|h)bounds. cbounds is either a 2-tuple or a list of 4-tuples. Each
    4-tuple is (lower_bound, upper_bound, start_index, end_index). cbounds are always stored in 4-tuple form.
    The actual range is [s,e) just like a python range, so to be contiguous e_{i} == s_{i+1}, although continuity isn't
    checked here.
    '''
    def set_cbound(cbound):
      if not hasattr(cbound, '__len__') or len(cbound) != 4:
        raise ValueError(f'cbound must be a 4-tuple not "{cbound}"')
      if cbound[1] <= cbound[0]:
        raise ValueError('max cbound (%f) must be > min cbound (%f)' % (cbound[1], cbound[0]))
      if self.lbounds[cbound[2]:cbound[3]].sum() > cbound[1]:
        raise ValueError('cbounds infeasible; min possible sum (%f) is > max cbounds (%f)' % (self.lbounds[cbound[2]:cbound[3]].sum(), cbound[1]))
      if self.hbounds[cbound[2]:cbound[3]].sum() < cbound[0]:
        raise ValueError('cbounds infeasible; max possible sum (%f) is < min cbounds (%f)' % (self.hbounds[cbound[2]:cbound[3]].sum(), cbound[0]))
      self.cbounds.append(cbound)
    self._cbounds = []
    if cbounds is None:
      self._cbounds = None
      return
    elif not hasattr(cbounds, '__len__'):
      raise  ValueError('cbounds should be a list of 4-tuples or a 2-tuple')
    if len(cbounds) == 2 and not hasattr(cbounds[0], '__len__'):
      set_cbound(tuple(cbounds) + (0, len(self)))
    else:
      for cbound in cbounds:
        set_cbound(cbound)

  @params.setter
  def params(self, params):
    ''' Convenience buld setter. '''
    for k, v in params.items():
      setattr(self, k ,v)

  def project(self, s):
    return self._feasible_region.project(s.reshape(len(self))).reshape(self.shape)

  def to_dict(self):
    ''' Dump object as dict. Dict should allow re-init of instance via cls(**data). See __init__(),
    from_dict().
    '''
    data = {k: getattr(self, k) for k in self._keys}
    return data

  def _build_feasible_region(self):
    region = HyperCube(self.bounds)
    # if self.cbounds is not None:
    #   region = Intersection(region, Slice(np.ones(len(self)), self.cbounds[0], self.cbounds[1]))
    self._feasible_region = region
