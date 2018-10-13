import numpy as np
from copy import deepcopy


def base_soc(b, s, l):
  ''' Apply decay to scalar b over times l, at rate 1-s '''
  return b*(s**np.arange(1, l+1))


def soc(r, s, e):
  ''' Get "state of charge" or rather state of something. This is basically calculating a discrete
  integral for all values between [0,len(r)] given r, some sustainment `s` and efficiency `e` factors.
  '''
  r = np.array(r)
  if len(r.shape) != 1:
    raise ValueError('Input value must have vector shape not %s' % (r.shape,))
  sm = sustainment_matrix(s, len(r))
  return ((r*(e**np.sign(r)))*sm).cumsum(axis=1).diagonal()


def sustainment_matrix(s, l):
  ''' Returns a matrix with coefficients for basically thermal-ish decay. Note "sustainment" is
  the opposite of decay, sustainment (s) or 1 means zero loss.
  '''
  if s == 1:
    return np.tril(np.ones((l, l)))
  return np.tril(s**power_matrix(l))


def power_matrix(l):
  ''' Returns a lower triangular matrix, that can be used in a power series. '''
  return np.array([i.cumsum() for i in np.triu(np.ones((l, l)), 1)]).transpose()


def care2bounds(device):
  ''' The bounds style used by Device is same a scipy minimize, but it's annoying. This function
  converts `care` array, plus `bounds` 2-tuple to Device.bounds style bounds.
  '''
  device = deepcopy(device)
  care = device['care']
  bounds = device['bounds']
  del device['care']
  if len(bounds) == 2:
    device['bounds'] = np.stack((care*bounds[0], care*bounds[1]), axis=1)
  else:  # Assume bounds is a vector
    device['bounds'] = np.stack((care*bounds, care*bounds), axis=1)
  return device
