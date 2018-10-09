import re
import numpy as np
from abc import ABC, abstractmethod

class BaseDevice(ABC):
  ''' Base class for any type of device. Devices are all characterised by having:

      - A fixed length, `len`, which is the number of slots during which the device consumes or
        produces some resource.
      - A shape which is (`N`,`len`), N is 1 for normal devices, but accounts for a device potentially
        being a composite.
      - A list of low/high resource consumption `bounds` of length `N`*`len`.
      - A concave differentiable utility function `u()`, which represents how much value the device
          gets from consuming / producing a given resource allocation (`N`,`len`) at some price.

    This class is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function.

    Constraints should be convex but this is not currently enforced. Try to maintain:

      - Device is stateless.
      - Device should be considered immutable (the currently available setters are all used on init).
      - Device is serializable and and constructable from the serialization.

    Note Python3 @properties have been used throughout these classes. They mainly serve as very
    verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.
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

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)
