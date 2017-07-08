import numpy as np
from lcl.device import Device


class SDevice(Device):
  ''' Storage device. Behaviour is determined by settings but typically storage device draws and supplies resource.
  The utility function has a cost and a reward component:

    U(q(t)) = -D(q(t)) - p*q(t)

  Since q(t) can be -ve for a storage device it's always preferable for the device to discharge -
  assuming the price is always +ve. The higher the price the more preferable it is. The cost has 3
  components that model:

    - The cost of fast charging or discharging rates
    - The cost from a large number of charge discharge cycles.
    - The cost from deep discharge cycles.

  There are six parameters to IDevice:

    - c1,c2,c3: coefficients for the first, second, third cost terms.
    - capacity: the storage capacity of the device.
    - damage_depth: how deep is a deep discharge
    - reserve: how much charge must the device be storing at the end of the planning window. We
        assume the device starts with this reserve as well.
  '''
  _c1 = 1
  _c2 = 0.1
  _c3 = 0.01
  _capacity = 10
  _damage_depth = 0.2
  _reserve = 0.5

  def uv(self, r, p):
    return -1*self.charge_costs(r) - r*p

  def u(self, r, p):
    return self.uv(r, p).sum()

  def deriv(self, r, p):
    return -1*self.charge_costs_deriv(r) - p

  def charge_costs(self, r):
    cost1 = (self.c1*r**2)
    cost2 = self.flip_cost_at(r)
    cost3 = self.deep_damage_at(r)
    return cost1 + cost2 + cost3

  def charge_costs_deriv(self, r):
    cost1_deriv = self.c1*2*r
    cost2_deriv = self.c2*-1*np.hstack((r[1:],[r[len(r)-1]]))
    cost3_deriv = self.deep_damage_at_deriv(r)
    return cost1_deriv + cost2_deriv + cost3_deriv

  def charge_at(self, r):
    return self.reserve*self.capacity+r.cumsum()

  def flip_cost_at(self, r):
    return self.c2*-1*np.array([r[i]*r[i+1] for i in range(0, len(r)-1)] + [0])

  def deep_damage_at(self, r):
    return self.c3*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)**2

  def deep_damage_at_deriv(self, r):
    ''' Partial dervis of deep_damage_at(). Chain rule with outer minimum() and
    charge_at() w.r.t. r.
    '''
    d = np.zeros(len(r))
    c = self.c3*2*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)
    for i in range(0, len(c)):
      d += c[i]*np.hstack((np.ones(i+1), np.zeros(len(c)-i-1)))
    return d

  @property
  def c1(self):
    return self._c1

  @property
  def c2(self):
    return self._c2

  @property
  def c3(self):
    return self._c3

  @property
  def capacity(self):
    return self._capacity

  @property
  def reserve(self):
    return self._reserve

  @property
  def damage_depth(self):
    return self._damage_depth

  @property
  def params(self):
    return {
      'c1': self.c1,
      'c2': self.c2,
      'c3': self.c3,
      'capacity': self.capacity,
      'reserve': self.reserve,
      'damage_depth': self.damage_depth
    }

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Constraints ensure at every time-slot the current storage level is between 0 and `capacity`.
    At any point level is determined only by initial level and sum of consumption upto that point.
    Also impose that at EOD level is at least reserve.
    '''
    constraints = Device.constraints.fget(self)
    min_level = -1*self.reserve*self.capacity
    max_level = self.capacity*(1 - self.reserve)
    for i in range(0, len(self)):
      mask = np.concatenate((np.ones(i+1), np.zeros(len(self)-i-1)))
      constraints += [{
        'type': 'ineq',
        'fun': lambda r, mask=mask, min_level=min_level: r.dot(mask) - min_level,
        'jac': lambda r, mask=mask: mask
      },
      {
        'type': 'ineq',
        'fun': lambda r, mask=mask, max_level=max_level: max_level - r.dot(mask),
        'jac': lambda r, mask=mask: -1*mask
      }]
    constraints += [{ # At least reserve left at end of window.
        'type': 'ineq',
        'fun': lambda r, mask=np.ones(len(self)), min_level=0: r.dot(mask) - 0,
        'jac': lambda r, mask=np.ones(len(self)): mask
      },
      {
        'type': 'ineq',
        'fun': lambda r, mask=mask, max_level=max_level: max_level - r.dot(mask),
        'jac': lambda r, mask=mask: -1*mask
    }]
    return constraints

  @params.setter
  def params(self, params):
    ''' Sanity check params. '''
    if not isinstance(params, dict):
      raise ValueError('params to SDevice must be a dictionary')
    p = self.params
    p.update(params)
    if p['c1'] <= p['c2']:
      raise ValueError('c1 must be greater than c2')
    if p['c1'] < 0 or p['c2'] < 0:
      raise ValueError('cost coefficients must be non -ve')
    self._c1 = p['c1']
    self._c2 = p['c2']
    self.c3 = p['c3']
    self.capacity = p['capacity']
    self.reserve = p['reserve']
    self.damage_depth = p['damage_depth']

  @c1.setter
  def c1(self, c1):
    if c1 < 0:
      raise ValueError('cost coefficients must be non -ve')
    if c1 <= self.c2:
      raise ValueError('c1 must be greater than c2')
    self._c1 = c1

  @c2.setter
  def c2(self, c2):
    if c2 < 0:
      raise ValueError('cost coefficients must be non -ve')
    if c2 >= self.c1:
      raise ValueError('c2 must be less than c1')
    self._c2 = c2

  @c3.setter
  def c3(self, c3):
    if c3 < 0:
      raise ValueError('cost coefficients must be non -ve')
    self._c3 = c3

  @capacity.setter
  def capacity(self, capacity):
    if capacity <= 0:
      raise ValueError('capacity must be > 0')
    self._capacity = capacity

  @reserve.setter
  def reserve(self, reserve):
    if reserve < 0 or reserve > 1:
      raise ValueError('reserve must be in [0,1]')
    self._reserve = reserve

  @damage_depth.setter
  def damage_depth(self, damage_depth):
    if damage_depth < 0 or damage_depth > 1:
      raise ValueError('damage_depth must be in [0,1]')
    self._damage_depth = damage_depth
