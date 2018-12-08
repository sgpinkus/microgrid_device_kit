import re
import logging
import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseDevice(ABC):
  ''' Base class for any type of device. Devices are all characterised by having:

      - A fixed length, `len`, which is the number of slots during which the device consumes or
        produces some resource.
      - A shape which is (`N`,`len`), N is 1 for atomic devices, but accounts for a device potentially
        being a composite.
      - A list of low/high resource consumption `bounds` of length `N`*`len`.
      - A concave differentiable utility function `u()`, which represents how much value the device
          gets from consuming / producing a given resource allocation (`N`,`len`) at some price.

    This class is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function.

    This class declares the necessary interfaces to treat a BaseDevice as a composite, __iter__(),
    shapes(), partition().

    Constraints should be convex but this is not currently enforced. Rule should be:

      - Device should be considered immutable (the currently available setters are all used on init).
      - Device is serializable and and constructable from the serialization.

    Note Python3 @properties have been used throughout these classes. They mainly serve as very
    verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.

    @todo rename DeviceSpace
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
    ''' Overall shape of this device '''
    pass

  @property
  @abstractmethod
  def shapes(self):
    ''' Array of shapes of sub devices. '''
    pass

  @property
  @abstractmethod
  def partition(self):
    ''' Returns array of (offset, length) tuples for each sub-device's mapping onto this device's `s` '''
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

  def step(self, p, s, stepsize, solver_options={}):
    ''' Take one step towards optimal demand for price vector `p`, using stepsize plus limited
    minimization. This means stepsize sets the upper bound of change in x, but stepsize can be large
    since limited minimization limits severity of overshoots. This method does not modify the agent.
    @see Device.solve()
    '''
    # Step & Projection
    s_next = s + stepsize * self.deriv(s, p)
    (s_next, o) = project(s_next, s, self.bounds, self.constraints)
    if not o.success:
      if o.status == 8:
        logger.warn(o)
      else:
        raise OptimizationException(o)
    # Limited minimization
    ol = minimize(
      lambda x, p=p: -1*self.u(s + x*(s_next - s), p),
      0.,
      method='SLSQP',
      bounds = [(0, 1)],
      options = solver_options,
    )
    if not ol.success:
      if ol.status == 8:
        logger.warn(ol)
      else:
        raise OptimizationException(ol)
    s_next = (s + ol.x*(s_next - s)).reshape(self.shape)
    return (s_next, ol)

  def solve(self, p, s0=None, solver_options={}):
    ''' Find the optimal demand for price for the given self and return it. Works on any agent
    since only requires s and self.deriv(). This method does not modify the agent.
    Note AFAIK scipy.optimize only provides two methods that support constraints:
      - COBYLA (Constrained Optimization BY Linear Approximation)
      - SLSQP (Sequential Least SQuares Programming)
    Only SLSQP supports eq constraints. SLSQP is based on the Han-Powell quasi–Newton method. Apparently
    it uses some quadratic approximation, and the same method seems to be sometimes referred called
    SLS-Quadratic-Programming. This does not mean it is limited to quadratics. It should work with *any*
    convex nonlinear function over a convex set.

    SLSQP doesnt take a tol option, only an ftol options. Using this option in the context of this
    software implies tolerance is +/- $ not consumption.

    @see http://www.pyopt.org/reference/optimizers.slsqp.html
    @see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    '''
    class OptDebugCb():
      i = 0
      @classmethod
      def cb(cls, device, x):
        print('step=%d' % (cls.i,), device.u(x, 0))
        cls.i += 1
    _solver_options = {'ftol': 1e-6, 'maxiter': 500, 'disp': False}
    _solver_options.update(solver_options)
    logger.debug(_solver_options)
    s0 = s0 if s0 is not None else self.project(np.zeros(self.shape))

    if (self.bounds[:, 0] == self.bounds[:, 1]).all():
      return (self.lbounds, None)
    o = minimize(
      lambda s, p=p: -1*self.u(s, p),
      s0,
      jac=lambda s, p=p: -1*self.deriv(s, p),
      method='SLSQP',
      bounds = self.bounds,
      constraints = self.constraints,
      options = _solver_options,
      # callback=lambda x: OptDebugCb.cb(self, x),
    )
    if not o.success:
      raise OptimizationException(o)
    return ((o.x).reshape(self.shape), o)

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)


def leaf_devices(device: BaseDevice):
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
  for item in _leaf_devices(device, device.id, '.'):
    yield item


def zero_mask(x, fn=None, row=None, col=None, cnt=1):
  if row is None and col is None:
    raise ValueError('row or col argument must be supplied');
  if row is not None and col is not None:
    raise ValueError('row and col arguments are mutually exclusive')
  i = x[row:row+cnt,:] if row is not None else x[:, col:col+cnt]
  o = fn(i).reshape(i.shape)
  r = np.zeros(x.shape)
  if row is not None:
    r[row:row+cnt,:] = o
  else:
    r[:, col:col+cnt] = o
  return r


def project(p, x0, bounds=[], constraints=[], solver_options={}):
  ''' Find the point in feasible region closest to p. '''
  p = p.flatten()
  options = {
    'ftol': 1e-9,
    'disp': False,
    'maxiter': 200
  }
  options.update(solver_options)
  o = minimize(lambda s, p=p: ((s - p)**2).sum(), x0, method='SLSQP',
    jac=lambda s, p=p: 2*(s - p),
    options = options,
    bounds = bounds,
    constraints = constraints
  )
  return (o.x.reshape(x0.shape), o)


class OptimizationException(Exception):  # pragma: no cover
  ''' Some type of optimization error. '''
  o = None # An optional optimization method specific status report (i.e. OptimizeResult).

  def __init__(self, *args):
    self.o = args[0] if args else None
    super(Exception, self).__init__(*args)
