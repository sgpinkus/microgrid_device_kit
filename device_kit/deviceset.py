import re
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from scipy.optimize import minimize
from device_kit import *
from .utils import zmm
from logging import debug, info, warn, exception, error


class DeviceSet(BaseDevice):
  ''' A container for a set of sub-devices. Sub-devices have [c]bounds and other constraints. This
  DeviceSet base add sbounds for cummulative bounds accross the set of devices. This is a common
  device coupling (example max capacity of bus connecting device).

  DeviceSet effectively serves as an internal node of a tree of devices.

  @todo sbounds can be a vector of two tuples like Device.bounds
  '''
  _id = None
  _length = None          # Number of time units (not number of devices).
  _sbounds = None         # min/max aggregate power consumption by this agent at ant timeslot.
  _devices = None         # Sub devices of this agent.

  def __init__(self, id, devices, sbounds=None):
    '''  '''
    if not (np.vectorize(lambda a: len(a))(np.array(devices)) == len(devices[0])).all():
      raise ValueError('Devices have miss-matched lengths')
    if not re.match('(?i)^[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be string matching ^(?i)[a-z0-9][a-z0-9_-]*$ Given "%s"' % (id,))
    self._id = id
    self._devices = devices
    self._length = len(devices[0])
    self.sbounds = sbounds

  def __len__(self):
    return self._length

  def __str__(self):
    return self.to_str()

  def __iter__(self):
    for d in self.devices:
      yield d

  def cost(self, s, p):
    ''' Get the sum of the cost of current solution accross all devices as scalar. '''
    return self.costv(s, p).sum()

  def costv(self, s, p):
    ''' Get the cost accross all devices. Result is a 1D vector of len(devices) **not** a
    (len(devices), len(self)) 2D vector. proces are exploded like this to handle the cases where a
    full prices are given in self.shape matrix (same for deriv and hess).
    '''
    s = s.reshape(self.shape)
    p = p*np.ones(self.shape)
    return np.array(
      [d.cost(s[i[0]:i[0]+i[1], :], p[i[0]:i[0]+i[1], :])  for d, i in zip(self.devices, self.partition)]
    )

  def deriv(self, s, p):
    ''' Get deriv. Result is a (len(devices), len(self)) 2D vector. '''
    s = s.reshape(self.shape)
    p = p*np.ones(self.shape)
    return np.vstack(
      [d.deriv(s[i[0]:i[0]+i[1], :], p[i[0]:i[0]+i[1], :]) for d, i in zip(self.devices, self.partition)]
    )

  def hess(self, s, p=0):
    ''' Get hessian for `s` consumption at price `p`. '''
    s = s.reshape(self.shape)
    p = p*np.ones(self.shape)
    return np.array(
      [d.hess(s[i[0]:i[0]+i[1], :], p[i[0]:i[0]+i[1], :]) for d, i in zip(self.devices, self.partition)]
    ).sum(axis=0)

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
  def slices(self):
    ''' Like partition but more useful. Maps devices on to slice flow matrix like `x[slice(_slice),:]`. '''
    offset = np.roll(self.shapes[:,0].cumsum(), 1)
    offset[0] = 0
    return list(zip(self.devices, list(zip(offset, offset+self.shapes[:,0]))))

  @property
  def sbounds(self):
    ''' sbounds is to Device.bounds, but applies to aggregate of all devices in this set. Would use
    `bounds` property name but it is already used. See bounds.
    '''
    return self._sbounds

  @sbounds.setter
  def sbounds(self, sbounds):
    self._sbounds = self.validate_bounds(sbounds) if sbounds is not None else None

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
        c = {
          'type': constraint['type'],
          'fun': lambda s, i=i, f=constraint['fun']: f(s.reshape(shape)[i[0]:i[0]+i[1], :]),
        }
        if 'jac' in constraint:
          c['jac'] = lambda s, i=i, f=constraint['jac']: zmm(s.reshape(shape), range(i[0], i[0]+i[1]), fn=f).reshape(flat_shape)
        constraints += [c]
    if self.sbounds is not None:
      for i in range(0, len(self)): # for each time
        if self.sbounds[i][0] == self.sbounds[i][1]:
          constraints += [{
            'type': 'eq',
            'fun': lambda s, i=i: s.reshape(shape)[:, i].dot(np.ones(shape[0])) - self.sbounds[i][0],
            'jac': lambda s, i=i: zmm(s.reshape(shape), i, axis=1, fn=lambda r: np.ones(shape[0])).reshape(flat_shape)
          }]
        else:
          constraints += [{
            'type': 'ineq',
            'fun': lambda s, i=i: s.reshape(shape)[:, i].dot(np.ones(shape[0])) - self.sbounds[i][0],
            'jac': lambda s, i=i: zmm(s.reshape(shape), i, axis=1, fn=lambda r: np.ones(shape[0])).reshape(flat_shape)
          },
          {
            'type': 'ineq',
            'fun': lambda s, i=i: self.sbounds[i][1] - s.reshape(shape)[:, i].dot(np.ones(shape[0])),
            'jac': lambda s, i=i: zmm(s.reshape(shape), i, axis=1, fn=lambda r: -1*np.ones(shape[0])).reshape(flat_shape)
          }]
    return constraints

  def project(self, s):
    return np.vstack([d.project(s[i[0]:i[0]+i[1], :]) for d, i in zip(self.devices, self.partition)])

  def to_str(self, indent=1):
    _str = 'type=%s; id=%s; length=%d; sbounds_bounds=%s' % (
      self.__class__.__name__,
      self.id,
      len(self),
      '%.3f/%.3f' % (self.sbounds.min(), self.sbounds.max()) if self.sbounds is not None else None
    )
    _str += ('\n' + '\t'*indent).join([''] + [d.to_str(indent+1) if hasattr(d, 'devices') else str(d) for d in self.devices])
    return _str


  def to_dict(self):
    ''' Dump object as a dict. '''
    return {
      'id': self.id,
      'sbounds': self.sbounds,
      'devices': self.devices,
    }