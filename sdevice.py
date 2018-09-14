import numpy as np
import numdifftools as nd
from powermarket.device import Device


class SDevice(Device):
  ''' Storage device. Behaviour is determined by settings but typically storage device draws and supplies resource.
  The utility function has a cost and a reward component:

    U(q(t)) = -D(q(t)) - p*q(t)

  Since q(t) can be -ve for a storage device it's always preferable for the device to discharge -
  assuming the price is always +ve. The higher the price the more preferable it is. The cost has 3
  components that model:

    - The cost of fast charging or discharging rates.
    - The cost from a large number of charge discharge cycles over a day.
    - The cost from deep discharge cycles.

  There are seven parameters to SDevice:

    - c1,c2,c3: coefficients for the first, second, third cost terms listed above.
    - capacity: the storage capacity of the device.
    - damage_depth: how deep is a deep discharge.
    - reserve: how much charge must the device be storing at the end of the planning window. We
        assume the device starts with this reserve as well.
    - efficiency: The round trip efficiency factor - [0,1]. Presumed to apply symmetrically to in/out flow.
  '''
  _c1 = 1
  _c2 = 0.1
  _c3 = 0.01
  _capacity = 10
  _damage_depth = 0.0
  _reserve = 0.5
  _efficiency = 1.0
  _sustainment = 1.0
  _rate_clip = None

  def uv(self, r, p):
    return -1*self.charge_costs(r) - r*p

  def u(self, r, p):
    return self.uv(r, p).sum()

  def deriv(self, r, p):
    return -1*self.charge_costs_deriv(r) - p

  def hess(self, r, p=0):
    ''' Return hessian approximation. '''
    return nd.Hessian(lambda x: self.u(x,0))(r)

  def charge_costs(self, r):
    ''' Get total costs for flow vector `r`. '''
    r = r.reshape((len(self),))
    cost1 = (self.c1*r**2)
    cost2 = self.flip_cost_at(r)
    cost3 = self.deep_damage_at(r)
    return cost1 + cost2 + cost3

  def charge_costs_deriv(self, r):
    ''' Deriv of charge_costs(). '''
    r = r.reshape((len(self),))
    cost1_deriv = self.c1*2*r
    cost2_deriv = self.c2*-1*np.hstack((r[1:],[r[len(r)-1]]))
    cost3_deriv = self.deep_damage_at_deriv(r)
    return cost1_deriv + cost2_deriv + cost3_deriv

  def flip_cost_at(self, r):
    ''' Calculate total cost for flippyness in flow vector `r`. '''
    r = r.reshape((len(self),))
    return self.c2*-1*np.array([r[i]*r[i+1] for i in range(0, len(r)-1)] + [0])

  def deep_damage_at(self, r):
    ''' Calculate total cost for deep discharge in flow vector `r`. '''
    return self.c3*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)**2

  def deep_damage_at_deriv(self, r):
    ''' Partial dervis of deep_damage_at(). Chain rule with outer minimum() and charge_at() wrt `r`. '''
    d = np.zeros(len(r))
    c = self.c3*2*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)
    for i in range(0, len(c)):
      d += c[i]*np.hstack((np.ones(i+1), np.zeros(len(c)-i-1)))
    return d

  def charge_at(self, r):
    ''' Return SoC vector given the RoC vector `r`. No check is made to ensure `r` is feasible. '''
    sm = self.sustainment_matrix()
    e = self.efficiency
    s = self.sustainment
    soc =  self.base()*(s**np.arange(1,len(self)+1)) + ((r*(e**np.sign(r)))*sm).cumsum(axis=1).diagonal()
    return soc

  def charge_at_lossless(self, r):
    ''' Return SoC vector given the RoC vector `r`. No check is made to ensure r is feasible. '''
    return self.base()+r.cumsum()

  def sustainment_matrix(self):
    ''' Returns a matrix with coefficients for how much of the input from the i-th timeslot eff in
    the j-th timeslot, j >= i, when non perfect sustainment (i.e. some decay) is considered.
    To get the actual SoC given a efficiency corrected RoC vector, r, use  r*m
    '''
    l = len(self)
    exp = np.array([i.cumsum() for i in np.triu(np.ones((l, l)), 1)]).transpose()
    return np.tril(self.sustainment**exp)

  def base(self):
    return self.reserve*self.capacity

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
  def efficiency(self):
    return self._efficiency

  @property
  def sustainment(self):
    return self._sustainment

  @property
  def rate_clip(self):
    return self._rate_clip

  @property
  def params(self):
    return {
      'c1': self.c1,
      'c2': self.c2,
      'c3': self.c3,
      'capacity': self.capacity,
      'reserve': self.reserve,
      'damage_depth': self.damage_depth,
      'efficiency': self.efficiency,
      'sustainment': self.sustainment,
      'rate_clip': self.rate_clip
    }

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Constraints ensure at every time-slot the current storage level is between 0 and `capacity`.
    At any point level is determined only by initial level and sum of consumption upto that point.
    Also impose that at EOD level is at least reserve.
    '''
    constraints = Device.constraints.fget(self)
    reserve = self.capacity*self.reserve
    base = self.base()
    e = self.efficiency
    s = self.sustainment
    sustainment_matrix = self.sustainment_matrix()
    def soc(r, i):
      mask = sustainment_matrix[i]
      return base*(s**(i+1)) + ((e**np.sign(r))*r).dot(mask)
    # Discrete integral always within [0,capacity]
    for i in range(0, len(self)):
      mask = sustainment_matrix[i]
      constraints += [
        # SoC >=0
        {
          'type': 'ineq',
          'fun': lambda r, i=i: soc(r, i),
          'jac': lambda r, mask=mask: (e**np.sign(r))*mask
        },
        # SoC <= capacity
        {
          'type': 'ineq',
          'fun': lambda r, i=i: self.capacity - soc(r, i),
          'jac': lambda r, mask=mask: -1*(e**np.sign(r))*mask
        }
      ]
    # RoC varies linearly as SoC varies capacity. rate_clip must be >=1
    if self.rate_clip:
      for i in range(0, len(self)):
        mask = sustainment_matrix[i]
        constraints += [
          # RoC <= Rate_Clip*Max_RoC*((capacity-SoC)/capacity)
          {
            'type': 'ineq',
            'fun': lambda r, i=i: self.rate_clip*self.hbounds[i]*((self.capacity - soc(r, i))/self.capacity) - r[i],
            # 'jac': lambda r, mask=mask: (e**np.sign(r))*mask
          },
          # RoC >= Rate_Clip*Min_RoC*((SoC)/capacity)
          {
            'type': 'ineq',
            'fun': lambda r, i=i: r[i] - self.rate_clip*self.lbounds[i]*((soc(r, i))/self.capacity),
            # 'jac': lambda r, mask=mask: (e**np.sign(r))*mask
          },
        ]
    # At least reserve left at end of window.
    constraints += [
      {
        'type': 'ineq',
        'fun': lambda r: soc(r, len(self)-1) - reserve,
        'jac': lambda r, mask=sustainment_matrix[len(self)-1]: (e**np.sign(r))*mask
      },
    ]
    return constraints

  @params.setter
  def params(self, params):
    ''' Sanity check params. Constraints also applied in individual setters.'''
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
    self.efficiency = p['efficiency']
    self.sustainment = p['sustainment']
    self.rate_clip = p['rate_clip']

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
    if not 0 <= reserve <= 1:
      raise ValueError('reserve must be in [0,1]')
    self._reserve = reserve

  @damage_depth.setter
  def damage_depth(self, damage_depth):
    if not 0 <= damage_depth <= 1:
      raise ValueError('damage_depth must be in [0,1]')
    self._damage_depth = damage_depth

  @efficiency.setter
  def efficiency(self, efficiency):
    if not 0 < efficiency <= 1.0:
      raise ValueError('efficiency factor must be in range (0, 1]')
    self._efficiency = efficiency

  @sustainment.setter
  def sustainment(self, sustainment):
    if not 0 < sustainment <= 1.0:
      raise ValueError('sustainment must be in range (0, 1]')
    self._sustainment = sustainment

  @rate_clip.setter
  def rate_clip(self, rate_clip):
    if rate_clip is not None and not rate_clip >= 1.0:
      raise ValueError('rate_clip must be >1')
    self._rate_clip = rate_clip
