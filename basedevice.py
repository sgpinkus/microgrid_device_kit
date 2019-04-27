import re
import logging
import numpy as np
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseDevice(ABC):
  ''' Base class for any type of device including composite devices. Devices are all characterised
    by having:

      - A fixed length, `len`, which is the number of slots during which the device consumes or
        produces some resource.
      - A shape which is (`N`,`len`), N is 1 for atomic devices, but accounts for a device potentially
        being a composite.
      - If composite, a list of low/high resource consumption `bounds` of length `N`*`len`.
      - A differentiable utility function `u()`, which represents how much value the device gets
        from consuming / producing a given resource allocation (`N`,`len`) at some price. Note `u()`
        is also used to represent costs of production (cost is -ve utility).

    Device is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function and possibly additional
    constraints.

    Other notes:

      - This class declares the necessary interfaces to treat a BaseDevice as a composite. Mainly,
       __iter__(), shapes, partition, leaf_devices().
      - Utility functions should be concave (giving convex cost function), and constraints should be
        convex but this is not currently enforced.
      - Device was intended to be and should be treated as immutable but currently this is not enforced.
      - Because Device was intended to be immutable, while a Device represents flow flexibility
        it does not *have* a state  of flow.
      - Devices should be serializable and constructable from the serialization.
      - Python3 @properties have been used throughout these classes. They mainly serve as very
        verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.

    @todo rename this class DeviceSpace?
  '''

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def u(self, s, p):
    ''' Scalar utility for `s` at `p`. `s` should have the same shape as this Device. '''
    pass

  @abstractmethod
  def deriv(self, s, p):
    ''' Derivative of utility for `s` at `p`. `s` should have the same shape as this Device.
    Return value has same shape as `s`.
    '''
    pass

  @abstractmethod
  def hess(self, s, p=0):
    ''' Hessian of utility for `s` at `p` - normally p should fall out. `s` should have the same
    shape as this Device *but* the return value has shape (len, len).
    '''
    pass

  @property
  @abstractmethod
  def id(self):
    pass

  @property
  @abstractmethod
  def shape(self):
    ''' Return absolute shape of device flow matrix. '''
    pass

  @property
  @abstractmethod
  def shapes(self):
    ''' Array of shapes of sub devices, if any, else [shape]. '''
    pass

  @property
  @abstractmethod
  def partition(self):
    ''' Returns array of (offset, length) tuples for each sub-device's mapping onto this device's
    flow matrix.
    '''
    pass

  @property
  @abstractmethod
  def bounds(self):
    pass

  @property
  @abstractmethod
  def lbounds(self):
    pass

  @property
  @abstractmethod
  def hbounds(self):
    pass

  @property
  @abstractmethod
  def constraints(self):
    pass

  @abstractmethod
  def project(self, s):
    ''' project s into cnvx space of this device a return point. Projection should always be possible. '''
    pass

  @abstractmethod
  def to_dict(self):
    pass

  def leaf_devices(self):
    ''' Iterate over flat list of (fqid, device) tuples for leaf devices from an input BaseDevice.
    fqid is the id of the leaf device prepended with the dot separated ids of parents. The input device
    may be atomic or a composite. The function distinguishes between them via support for iteration.
    '''
    def _leaf_devices(device, fqid, s='.'):
      try:
        for sub_device in device:
          for item in _leaf_devices(sub_device, fqid + s + sub_device.id, s):
            yield item
      except:
          yield (fqid, device)
    for item in _leaf_devices(self, self.id, '.'):
      yield item

  def map(self, s):
    ''' maps rows of flow matrix `s` to identifiers of atomic devices under this device.
    Returns list of tuples. You can load this into Pandas like pd.DataFrame(dict(device.map(s)))
    '''
    s = s.reshape(self.shape)
    for i, d in enumerate(self.leaf_devices()):
      yield (d[0], s[i:i+1,:].reshape(len(self)))

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)
