import numpy as np
import numdifftools as nd
from device_kit import Device, IDevice
from device_kit.utils import soc, base_soc, sustainment_matrix


class TDevice(Device):
  ''' Represents a heating or cooling device. Or really any device who's utility is based on a
  point that behaves like thermodynamics. The utility curve is the same as IDevice, but based on
  instaneous temperature not resource consumption directly, and with two restrictions on parameters:

    IDevice.a is always 0.
    IDevice.b must be even. This is just a quick fix. May change to allow odd in future.

  Recall IDevice.c is a scaling factor for utility. Thus if one doesn't care about the temperature
  at a given time, set IDevice.c to zero. If one cares a little, set IDevice.c a little.

  The instaneous temperature is dependent on resource consumption at all times before a given time.
  Under the hood, the thermodynamics are modelled by a simple discrete exponential equation:

    T(i) = T(i-1) + A(TE(i) - T(i-1)) + B(Q(t))

  Where T internal temperature, TE external temperature, Q(t) consumption at t, A, B constant params.

  Note that the min|max acceptable temps may be infeasible given TE, and consumption bounds and
  other parameters. So we haven't implemented them as hard constraints. The min|max temp inputs are
  actually just parameters to the utility function not hard constraints. Specifically the diff
  between t_optimal and t_min or t_max (symmetrically) is `c` units of utility (@see IDevice.c).

  @todo this doesn't really need to extend IDevice. It just uses a classs method from IDevice for its
  utility curve.
  '''
  t_external = None              # External temperature vectors of `len` length.
  t_init = None                  # The initial (and final) temperature T0.
  t_optimal = None               # The scalar ideal temperature TC. It's assumed ideal temperature is time invariant.
  t_range = None                 # The +/- range min/max temp.
  t_a = None                     # Thermal loss factor to external environment.
  t_b = None                     # Thermal efficiency factor.
  t_base = None                  # temperature without out any consumption by heat engine. Derived value.
  t_utility_base = 0             # utility of t_base pre calculated & used as offset.
  sustainment_matrix = None      # stashed for use in deriv.

  def u(self, s, p):
    return self.uv(s, p).sum()

  def uv(self, s, p):
    ''' @override uv() to do r to t conversion. '''
    return self.uv_t(self.r2t(s)) - s*p - self.t_utility_base

  def deriv(self, s, p):
    ''' @override deriv() to do r to t conversion. Chain rule to account for r2t(). '''
    dt = self.deriv_t(self.r2t(s))
    return (self.sustainment_matrix*dt.reshape(24,1)).sum(axis=0)*self.t_b - p

  def hess(self, s, p=0):
    ''' Return hessian diagonal approximation. nd.Hessian takes long time. In testing so far
    Hessdiag is an OOM faster and works just as good if not better.
    '''
    return np.diag(nd.Hessdiag(lambda x: self.u(x, 0))(s.reshape(len(self))))

  def uv_t(self, t):
    return np.vectorize(IDevice._u, otypes=[float])(t, 0, 2, 1, self.t_min, self.t_optimal)

  def deriv_t(self, t):
    return np.vectorize(IDevice._deriv, otypes=[float])(t, 0, 2, 1, self.t_min, self.t_optimal)

  def r2t(self, r):
    ''' Map `r` consumption vector to its effective heating or cooling effect, given heat transfer
    (t_base), thermal loss (t_a) and efficiency of device (t_b).
    '''
    return self.t_base + soc(r.reshape(len(self)), s=(1 - self.t_a), e=self.t_b)

  @property
  def params(self):
    return {
      't_external': self.t_external,
      't_init': self.t_init,
      't_optimal': self.t_optimal,
      't_range': self.t_range,
      't_a': self.t_a,
      't_b': self.t_b
    }
    return p

  @property
  def t_min(self):
    return self.t_optimal - self.t_range

  @property
  def t_max(self):
    return self.t_optimal + self.t_range

  @params.setter
  def params(self, params):
    ''' Set params and check validity. Set derived vars t_base, t_a_min|max based on params.
    t_act_min|max is the min max temperature change that the device itself must cause - not the
    thermodynamics, to bring be within bounds. Values in t_act_min|max may not actually be feasibile.
    For example, `t_act_min` is +ve while `t_b` is -ve. This is currently handled by relexing some
    unrealistic constraints. @see constraints.
    '''
    if not isinstance(params, dict):
      raise ValueError('params to IDevice must be a dictionary')
    p = self.params
    p.update(params)
    if len(p['t_external']) != len(self):
      raise ValueError('external temperature vector has wrong len (%s)' % (p['t_external'],))
    if p['t_range'] < 0:
      raise ValueError('acceptable temperature range (t_range) must be >= 0')
    if p['t_b'] == 0:
      raise ValueError('thermal efficiency must not be 0')
    if p['t_a'] < 0 or p['t_a'] > 1:
      raise ValueError('heat transfer coefficient must be in [0,1]')
    self.t_a = p['t_a']
    self.t_b = p['t_b']
    self.t_external = p['t_external']
    self.t_init = p['t_init']
    self.t_optimal = p['t_optimal']
    self.t_range = p['t_range']
    self.t_base = self._make_t_base(self.t_external, self.t_a, self.t_init)
    self.t_utility_base = self.uv_t(self.t_base)
    self.t_act_min = (self.t_optimal - self.t_range) - self.t_base  # Derived for convenience only.
    self.t_act_max = (self.t_optimal + self.t_range) - self.t_base  # Derived for convenience only.
    self.sustainment_matrix = sustainment_matrix((1 - self.t_a), len(self))

  def _make_t_base(self, t_external, t_a, t_init):
    ''' Calculate the base temperature, that occurs with no heat engine activity. This is used in
    utility calculation. Note `t_init` is the temperature in the last time-slot of last planning
    window, *not* the first time-slot of this planning window.
    '''
    t_base = base_soc(t_init, s=(1 - t_a), l=len(self)) + soc(t_external, s=(1 - t_a), e=t_a)
    return t_base


class ContrainedTDevice(TDevice):
  ''' This class overrides TDevice in an attempt to actually implement the temperature constraints,
  in addition to the usual power consumption constraints.
  '''
  _precision = 1e-8               # @see min_constraint(), max_constraint()
  _enforce_t_constraints = False  # Try and enforce  t_min <= t <= t_max constraint. @see constraints.

  def min_constraint(self, r, i):
    ''' The ineq constraint `T(r, i) >= t_min` except if it's not possible according to thermal
    parameters and/or consumption constraints. In that case return a small +ve number when the
    consumption is at the extreme at which the constraint is closest to being satisfied. This is a
    a bit if-y and doesn't actually work all the time!
    @see constraints
    '''
    v = self.t_b*(self.exponents(i)*r[0:i+1]).sum() + self.t_base[i] - self.t_min
    if v >= 0:
      return v
    elif self.t_b < 0 and self.lbounds[i] - r[i] + self._precision >= 0:
      return self._precision
    elif self.t_b > 0 and r[i] - self.hbounds[i] + self._precision >= 0:
      return self._precision
    return v

  def max_constraint(self, r, i):
    ''' The ineq constraint `T(r, i) <= t_max` except if it's not possible according to thermal
    parameters and/or consumption constraints.
    @see min_constraint(), constraints.
    '''
    v = self.t_max - self.t_b*(self.exponents(i)*r[0:i+1]).sum() - self.t_base[i]
    if v >= 0:
      return v
    elif self.t_b < 0 and r[i] - self.hbounds[i] + self._precision >= 0:
      return self._precision
    elif self.t_b > 0 and self.lbounds[i] - r[i] + self._precision >= 0:
      return self._precision
    return v

  def min_constraint_deriv(self, r, i):
    ''' Derivative of `min_constraint()`. '''
    return np.pad(self.t_b*self.exponents(i), (0, len(r)-i-1), 'constant')

  def max_constraint_deriv(self, r, i):
    ''' Derivative of `max_constraint()`. '''
    return np.pad(-1*self.t_b*self.exponents(i), (0, len(r)-i-1), 'constant')

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device. The only constraint is:

      t_optimal - t_range <= T(t) <= t_optimal + t_range for all t.

    The caveat is acheiving a given temperature change may not be feasible given the power constraints,
    and the thermal parameters. Thus the inequalities are a disjunctive that ensures the device
    acheives the temperature constraint OR tries as hard as possibe to acheive it (which entails the
    power consumption is at it's hard lower|upper bound depending). The caveat in that is the union
    of two convex region is not necessarily or even likely to be convex, and pretty sure convexity of
    ineq constraints is an underlying assumption of any minimizer capable of handling them (i.e. SLSQP).
    Never the less the the constraints min|max_constraint() try to implement this OR-ed constraint
    and it's seems to work in some cases, although it's know to fail in others.
    '''
    constraints = TDevice.constraints.fget(self)
    if self._enforce_t_constraints:
      for i in range(0, len(self)):
        constraints += [{
          'type': 'ineq',
          'fun': lambda r, i=i: self.min_constraint(r, i),
          'jac': lambda r, i=i: self.min_constraint_deriv(r, i)
        },
        {
          'type': 'ineq',
          'fun': lambda r, i=i: self.max_constraint(r, i),
          'jac': lambda r, i=i: self.max_constraint_deriv(r, i)
        }]
    return constraints
