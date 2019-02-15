import re
import uuid
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from device import *
from logging import debug, info, warn, exception, error


class DeviceSet(BaseDevice):
  ''' A container for a set of sub-devices. Sub-devices have [c]bounds and other constraints. This
  DeviceSet base add sbounds for cummulative bounds accross the set of devices. This is a common
  device coupling.

  DeviceSet effectively serves as an internal node of a tree of devices.

  @todo sbounds can be a vector of two tuples like Device.bounds
  '''
  _id = None
  _len = None             # Number of time units (not number of devices).
  _sbounds = None         # min/max aggregate power consumption by this agent at ant timeslot.
  _devices = None         # Sub devices of this agent.

  def __init__(self, devices, sbounds=None, id=None):
    '''  '''
    if not (np.vectorize(lambda a: len(a))(np.array(devices)) == len(devices[0])).all():
      raise ValueError('Devices have miss-matched lengths')
    if id is not None and not re.match('^(?i)[a-z0-9][a-z0-9_-]*$', id):
      raise ValueError('id must be string matching ^(?i)[a-z0-9][a-z0-9_-]*$ of None. Given "%s"' % (id,))
    if id is None:
      id = str(uuid.uuid4())[0:6]
    self._id = id
    self._devices = devices
    self._len = len(devices[0])
    self.sbounds = sbounds

  def __len__(self):
    return self._len

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
  def devices(self):
    return self._devices

  @property
  def shape(self):
    return (self.shapes.sum(axis=0)[0], len(self))

  @property
  def shapes(self):
    return np.array(tuple([d.shape for d in self.devices]))

  @property
  def partition(self):
    p = self.shapes[:,0]
    ps = [0] + list(p.cumsum())[0:-1]
    return np.array(tuple(zip(ps, p)), dtype=int)

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
  def sbounds(self, sbounds):
    ''' Set bounds. Can be input as a pair which will be repeated len times. Conversion is one way currently. '''
    if sbounds is None:
      self._sbounds = None
      return
    if not hasattr(sbounds, '__len__'):
      raise ValueError('sbounds must be a sequence type')
    if len(sbounds) == 2:
      sbounds = np.array([sbounds for i in range(0, len(self))])
    if len(sbounds) != len(self):
      raise ValueError('sbounds has wrong length (%d)' % len(sbounds))
    sbounds = np.array(sbounds)
    lbounds = np.array(sbounds[:, 0])
    hbounds = np.array(sbounds[:, 1])
    if not np.vectorize(lambda v: v is None)(sbounds).all() and not (hbounds - lbounds >= 0).all():
      raise ValueError('max sbound must be >= min sbound for all min/max sbound pairs: %s' % (str(hbounds - lbounds),))
    self._sbounds = sbounds

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
