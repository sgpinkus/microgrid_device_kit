import re
import uuid
import numbers
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from device_kit import *
from .utils import zero_mask
from logging import debug, info, warn, exception, error


class DeviceSet(BaseDevice):
  ''' A container for a set of sub-devices. Sub-devices have [c]bounds and other constraints. This
  DeviceSet base add sbounds for cummulative bounds accross the set of devices. This is a common
  device coupling.

  DeviceSet effectively serves as an internal node of a tree of devices.

  @todo sbounds can be a vector of two tuples like Device.bounds
  '''
  _id = None
  _length = None             # Number of time units (not number of devices).
  _sbounds = None         # min/max aggregate power consumption by this agent at ant timeslot.
  _devices = None         # Sub devices of this agent.

  def __init__(self, id, devices, sbounds=None):
    '''  '''
    if not (np.vectorize(lambda a: len(a))(np.array(devices)) == len(devices[0])).all():
      raise ValueError('Devices have miss-matched lengths')
    if id is not None and not re.match('^(?i)[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be string matching ^(?i)[a-z0-9][a-z0-9_-]*$ Given "%s"' % (id,))
    self._id = id
    self._devices = devices
    self._length = len(devices[0])
    self.sbounds = sbounds

  def __len__(self):
    return self._length

  def __str__(self):
    return '\n'.join([str(d) for d in self.devices])

  def __iter__(self):
    for d in self.devices:
      yield d

  def u(self, s, p):
    ''' Get the sum of the utility of current solution accross all devices as scalar. '''
    return self.uv(s, p).sum()

  def deriv(self, s, p):
    ''' Get deriv. Result is a (len(devices), len(self)) 2D vector. '''
    s = s.reshape(self.shape)
    return np.vstack(
      [d.deriv(s[i[0]:i[0]+i[1], :], p) for d, i in zip(self.devices, self.partition)]
    )

  def hess(self, s, p=0):
    ''' Get hessian for `s` consumption at price `p`. '''
    s = s.reshape(self.shape)
    return np.round(
      np.array([d.hess(s[i[0]:i[0]+i[1], :], p) for d, i in zip(self.devices, self.partition)]).sum(axis=0),
      6
    )

  def uv(self, s, p):
    ''' Get the utility accross all devices. Result is a 1D vector of len(devices) **not** a
    (len(devices), len(self)) 2D vector.
    '''
    s = s.reshape(self.shape)
    return np.array(
      [d.u(s[i[0]:i[0]+i[1], :], p)  for d, i in zip(self.devices, self.partition)],
    )

  @property
  def id(self):
    return self._id

  @property
  def length(self):
    return self._length

  @property
  def devices(self):
    return self._devices

  @property
  def shape(self):
    ''' Return absolute shape. This recurses down nested device to provide absolute shape. It is not
    merely (len(self.devices), len(self))
    '''
    return (self.shapes.sum(axis=0)[0], len(self))

  @property
  def shapes(self):
    ''' Return list with shapes of all child devices. '''
    return np.array(list([d.shape for d in self.devices]))

  @property
  def partition(self):
    ''' Returns list of tuples: (offset of each child row, number of rows of child), for each child
    device.
    '''
    offset = np.roll(self.shapes[:,0].cumsum(), 1)
    offset[0] = 0
    return np.array(list(zip(offset, self.shapes[:,0])), dtype=int)

  @property
  def sbounds(self):
    ''' sbounds is to Device.bounds, but applies to aggregate of all devices in this set. Would use
    `bounds` property name but it is already used. See bounds.
    '''
    return self._sbounds

  @property
  def lbounds(self):
    ''' lower bound of self.bounds. '''
    return np.array(self.bounds[:, 0])

  @property
  def hbounds(self):
    ''' upper bound of self.bounds. '''
    return np.array(self.bounds[:, 1])

  @property
  def bounds(self):
    ''' Return a len(self.s.flatten())x2 ndarray of min max bounds. This can be used in constraints
    when optimizing over `s`
    '''
    return np.concatenate([d.bounds for d in self.devices])

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this aggregate device. The constraints
    are to be applied to a matrix, but each device constraints only apply to a row of that matrix.
    Thus to reuse device constraints need to slice out correct row for input and zero pad other rows.
    '''
    constraints = []
    shape = self.shape
    flat_shape = shape[0]*shape[1]
    for d, i in zip(self.devices, self.partition):
      for constraint in d.constraints:
        constraints += [{
          'type': constraint['type'],
          'fun': lambda s, i=i, f=constraint['fun']: f(s.reshape(shape)[i[0]:i[0]+i[1], :]),
          'jac': lambda s, i=i, f=constraint['jac']: zero_mask(s.reshape(shape), f, row=i[0], cnt=i[1]).reshape(flat_shape)
        }]
    if self.sbounds is not None:
      for i in range(0, len(self)): # for each time
        constraints += [{
          'type': 'ineq',
          'fun': lambda s, i=i: s.reshape(shape)[:, i].dot(np.ones(shape[0])) - self.sbounds[i][0],
          'jac': lambda s, i=i: zero_mask(s.reshape(shape), lambda r: np.ones(shape[0]), col=i).reshape(flat_shape)
        },
        {
          'type': 'ineq',
          'fun': lambda s, i=i: self.sbounds[i][1] - s.reshape(shape)[:, i].dot(np.ones(shape[0])),
          'jac': lambda s, i=i: zero_mask(s.reshape(shape), lambda r: -1*np.ones(shape[0]), col=i).reshape(flat_shape)
        }]
    return constraints

  @sbounds.setter
  def sbounds(self, bounds):
    ''' Set bounds. Can be input as a pair which will be repeated len times. Conversion is one way currently. '''
    if bounds is None:
      self._sbounds = None
      return
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
      raise ValueError('max sbound must be >= min sbound for all min/max sbound pairs: %s' % (str(hbounds - lbounds),))
    self._sbounds = bounds

  def project(self, s):
    return np.vstack(
      [d.project(s[i[0]:i[0]+i[1], :]) for d, i in zip(self.devices, self.partition)]
    )

  def to_dict(self):
    ''' Dump object as a dict. '''
    return {
      'id': self.id,
      'sbounds': self.sbounds,
      'devices': self.devices,
    }
