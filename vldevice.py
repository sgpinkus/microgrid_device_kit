import numpy as np
from powermarket.device import Device


class VLDevice(Device):
  ''' Virtual load device. Dumbly follows the load of some other thing with a r property. Since devices
  don't have an `r` property, the the load has to follow an agent or some other adaptor.

  This virtual load device implementation is of limited use. It won't work for modelling storage
  device efficiency factor - as was the orginal idea for it, because it is too far decoupled from
  the storage devices actual flow. I.e. you can't simultaneously optimize battery flow taking this
  load into account. So ... I'm not sure what this is good for. Keeping it around anyway.
  '''
  _factor = 1
  _follows = None

  @property
  def params(self):
    return {'factor': self.factor, 'follows': self.follows}

  @property
  def factor(self):
    return self._factor

  @property
  def follows(self):
    return self._follows

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if not isinstance(params, dict):
      raise ValueError('params incorrect type')
    if not hasattr(params['follows'], 'r'):
      raise ValueError('need an object with an \'r\' property to follow')
    (self._factor, self._follows) = (params['factor'], params['follows'])

  @property
  def bounds(self):
    v = np.zeros(len(self))
    if self.follows:
      v  = np.abs(self.follows.r)*self.factor
    return np.stack((v,v), axis=1)

  @bounds.setter
  def bounds(self, bounds):
    pass
