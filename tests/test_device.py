import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import numpy as np
from copy import deepcopy
import unittest
from unittest import TestCase
from device_kit import *
from device_kit import utils
from device_kit.utils import sustainment_matrix
from device_kit.functions import poly2d


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%0.6f' % (v,),
  },
)


test_dim = 24
test_choice_prices = np.array([3, 1, 3, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 1])
test_rand_choice_prices = np.random.choice([1, 3], 24)
mins = np.random.rand(24)
maxs = mins + np.random.rand(24)
ones = np.ones(24)
zeros = np.zeros(24)

# Device
test_device = [
  'test_device',
  24,
  [(0.11098505, 0.73401641), (0.16957934, 0.20297378), (0.06926077, 0.41622013), (0.36589927, 0.91650255), (0.24329407, 0.65472194), (0.29358734, 0.61163406), (0.08356067, 0.67125244), (0.10659574, 0.68789302), (0.36779274, 0.60779149), (0.40205272, 1.37443838), (0.77592892, 1.05536045), (0.80435839, 1.37691422), (0.34357759, 0.89649611), (0.36303363, 0.48074482), (0.97522713, 1.10429662), (0.50986841, 0.76206089), (0.86281828, 1.34641641), (0.22111678, 0.92726414), (0.68289869, 0.99868162), (0.01570868, 0.13552611), (0.80499837, 0.84452731), (0.28573255, 0.64946374), (0.71161904, 1.62050741), (0.62565335, 1.46114493)],
  (1, 24),
]
# Device Simple
test_device_simple = [
  'test_device_simple',
  24,
  np.stack((np.zeros(24), np.ones(24)*10), axis=1),
  (10, 24),
]
# CDevice
test_cdevice = [
  'test_cdevice',
  24,
  np.stack((np.ones(24)*-1, np.ones(24)), axis=1),
  (-100, 100),
  {'a': 2, 'b': 0}
]
# CDevice2
test_cdevice_2 = [
  'test_cdevice_2',
  24,
  np.stack((mins, maxs), axis=1),
  (15, 20),
  {'a': 1, 'b': 0}
]
# IDevice
test_idevice_mins = np.concatenate((
  np.zeros(6),
  np.ones(4)*0.1,
  np.zeros(6),
  np.ones(4)*0.1,
  np.ones(4)*0.2
))
test_idevice_maxs = test_idevice_mins*4
test_idevice = [
  'test_idevice',
  24,
  np.stack((test_idevice_mins, test_idevice_maxs), axis=1),
  None,
  {
    'a': 0,
    'b': [2, 4, 2, 3, 2, 2, 2, 2, 3, 2, 3, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2, 4, 3],
    'c': 1,
    'd': 0,
  }
]
# IDevice 2
test_idevice_cbound = deepcopy(test_idevice)
test_idevice_cbound[3] = (0, 4)
# IDevice 3 - Simple
test_idevice_simple = [
  'test_idevice',
  24,
  np.stack((zeros, ones), axis=1),
  None,
  {}
]
# SDevice
test_sdevice = [
  'test_sdevice',
  24,
  np.stack((np.ones(24)*-5, np.ones(24)*5), axis=1),
  None,
  {'c1': 1, 'c2': 0, 'c3': 0, 'capacity': 10, 'reserve': 0.5, 'damage_depth': 0.2}
]
# SDevice
test_sdevice_1 = [
  'test_sdevice',
  24,
  np.stack((np.ones(24)*-5, np.ones(24)*5), axis=1),
  None,
  {'c1': 1, 'c2': 0, 'c3': 0, 'capacity': 10, 'reserve': 0.5, 'damage_depth': 0.2}
]
# TDevice
test_tdevice = [
  'test_tdevice',
  24,
  np.stack((np.ones(24)*0, np.ones(24)*4), axis=1),
  None,
  {
    't_external': np.sin(np.linspace(0, np.pi, 24))*10+10,
    't_init': 10,
    't_optimal': 15,
    'care_time': np.ones(24),
    't_range': 1.5,
    't_a': 0.4,
    't_b': 0.5,
    'a': 0,
    'b': 2
  }
]
# GDevice
test_gdevice = [
  'gas',
  24,
  np.stack((-100*np.ones(24), np.zeros(24)), axis=1),
  None,
  {'cost': [1., 1., 0]}
]
# GDevice with time varying cost curve.
test_gdevice_tv = [
  'gas',
  24,
  np.stack((-100*np.ones(24), np.zeros(24)), axis=1),
  None,
  {'cost': np.concatenate((np.ones((12, 4))*[0.00045, 0.0058, 0.024, 0], np.ones((12, 4))*[0.73, 0.58, 0.024, 1]))}
]
# IDevice2
test_idevice2 = [
  'test_idevice2',
  24,
  np.stack((test_idevice_mins, test_idevice_maxs), axis=1),
  None,
  {'d_0': 0.1, 'd_1': 1.0}
]

# CDevice2
test_cdevice2 = [
  'test_cdevice2',
  24,
  np.stack((np.ones(24)*-1, np.ones(24)), axis=1),
  (-100, 100),
  {'d_0': 0.1, 'd_1': 1.0}
]



class TestBaseDevice(TestCase):
  ''' Test Device, the base class for all devices. '''

  def test_basic_properties(self):
    ''' Test basic getters. '''
    device = Device(*test_device)
    self.assertEqual(device.id, 'test_device')
    self.assertEqual(len(device), 24)
    self.assertEqual(device.cbounds, (1, 24))
    self.assertEqual(device.params, None)
    self.assertEqual(len(device.deriv(np.ones(len(device)), np.ones(len(device)))), len(device))

  def test_more_properties(self):
    device = Device(*test_device)
    self.assertEqual(device.shape, (1, 24))
    self.assertEqual(device.shapes.tolist(), [[1, 24]])
    self.assertEqual(device.partition.tolist(), [[0, 1]])

  def test_leaf_and_map(self):
    device = Device(*test_device)
    _map = list(device.map(np.ones(24)))
    self.assertEqual(len(_map), 1)
    self.assertEqual(_map[0][0], 'test_device')
    self.assertEqual(list(_map[0][1]), list(np.ones(24)))

  def test_invalid_settings(self):
    ''' Test creating Device with settings that are ill-formedevice. '''
    _test_device = deepcopy(test_device)
    _test_device[1] = 25
    with self.assertRaises(ValueError):
      device = Device(*_test_device)
    # len(bounds) == length
    _test_device = deepcopy(test_device)
    _test_device[2].append((2, 3))
    with self.assertRaises(ValueError):
      device = Device(*_test_device)
    # cbounds is a 2-tuple
    _test_device = deepcopy(test_device)
    _test_device[3] = (1, 24, 30)
    with self.assertRaises(ValueError):
      device = Device(*_test_device)

  def test_tuple_bounds(self):
    _test_device = deepcopy(test_device)
    _test_device[2] = [0, 1]
    device = Device(*_test_device)
    self.assertEqual(device.bounds.shape, (24, 2))
    self.assertEqual(device.bounds[0][0], 0)
    with self.assertRaises(ValueError):
      _test_device[2] = [1, 0]
      device = Device(*_test_device)

  def test_mixed_bounds(self):
    _test_device = deepcopy(test_device)
    _test_device[2] = (0, np.ones(24))
    device = Device(*_test_device)
    self.assertEqual(device.bounds.shape, (24, 2))
    self.assertEqual(device.bounds[0][0], 0)
    with self.assertRaises(ValueError):
      _test_device[2] = [1, 0]
      device = Device(*_test_device)

  def test_infeasible_settings(self):
    ''' Test creating Device with settings that don't satisfy basic feasiblilty constraints. '''
    # id required
    for i in (None, '.xxx', 'x.x', 'x.', ''):
      _test_device = deepcopy(test_device)
      _test_device[0] = i
      with self.assertRaises(ValueError):
        device = Device(*_test_device)
    # mins < maxs
    _test_device = deepcopy(test_device)
    _test_device[2][0] = (1.0, 0.9)
    with self.assertRaises(ValueError):
      device = Device(*_test_device)
    # if cbounds
    _test_device = deepcopy(test_device)
    _test_device[3] = None
    device = Device(*_test_device)
    low = device.lbounds.sum()
    high = device.hbounds.sum()
    # low cbounds is feasible wrt maxs
    _test_device[3] = (high+0.1, high+1.0)
    with self.assertRaises(ValueError):
      device = Device(*_test_device)
    # high cbounds is feasible wrt mins
    _test_device[3] = (low-1.0, low-0.1)
    with self.assertRaises(ValueError):
      device = Device(*_test_device)

  def test_quasilinear_base_utility(self):
    ''' '''
    d = Device(*deepcopy(test_device_simple))
    r = np.ones(24)
    p = np.ones(24)
    self.assertEqual(-24, d.u(r, p))
    self.assertTrue((d.deriv(r, p) == -1*p).all())
    r = d.solve(p)[0]
    self.assertTrue((np.abs((r - np.ones(24)*(10/24))) <= 1e-6).all())


class TestCDevice(TestCase):
  ''' Test cummulative utility device CDevice. '''

  def test_cdevice_utility(self):
    device = self.get_test_device()
    zeros = np.zeros(len(device))
    ones = np.ones(len(device))
    # The params
    self.assertEqual(device.a, 2)
    self.assertEqual(device.b, 0)
    self.assertEqual(device.u(zeros, zeros), 0)
    self.assertEqual(len(device.deriv(ones, ones)), len(device))
    self.assertTrue((device.deriv(ones, ones) == device.a - ones).all())
    device.params = {'a': 2, 'b': 1}
    self.assertEqual(device.u(zeros, zeros), 1)
    self.assertTrue((device.deriv(zeros, test_choice_prices) == (-1*test_choice_prices+2)).all())

  def test_solve(self):
    ''' test_device and price setting, mean the device should derive 1 utility from each time unit. '''
    device = CDevice(*deepcopy(test_cdevice))
    p = test_choice_prices
    x = device.solve(p)[0]
    self.assertTrue(((np.abs(x - 1) < 1e-8) | (np.abs(x + 1) < 1e-8)).all())
    self.assertTrue((np.abs(device.u(x, p) - 24.) < 1e-8).all())

  def test_solve_constrained(self):
    ''' Due to the test prices unconstrained optimal r.sum() is -6 '''
    device = CDevice(*deepcopy(test_cdevice))
    device.cbounds = (-6, 0)
    p = test_choice_prices
    r = device.solve(p)[0]
    self.assertTrue(abs(device.u(r, p) - 24) <= 1e-8)
    device.cbounds = (-5, 0)
    r = device.solve(p)[0]
    self.assertTrue(abs(device.u(r, p) - 23) <= 1e-8)
    device.cbounds = (0, 1)
    r = device.solve(p)[0]
    self.assertTrue(abs(device.u(r, p) - 18) <= 1e-8)

  def test_bounds(self):
    device = self.get_test_device()
    device.cbounds = (-10, 10)

  def get_test_device(self):
    _test_device = deepcopy(test_cdevice)
    device = CDevice(*_test_device)
    return device


class TestIDevice(TestCase):
  ''' Test instantaneous utility device IDevice. @todo more tests '''

  def test_invalid_settings(self):
    ''' Test creating Device with settings that are ill-formedevice. '''
    _test_device = deepcopy(test_idevice)
    # length
    _test_device[4]['a'] = np.zeros(23)
    with self.assertRaises(ValueError):
      device = Device(*_test_device)
    # non -ve
    for v in ('a', 'b', 'c', 'd'):
      _test_device = deepcopy(test_idevice)
      _test_device[4][v] = -0.1
      with self.assertRaises(ValueError):
        device = Device(*_test_device)
    # a in extent
    _test_device = deepcopy(test_idevice)
    _test_device[4] = 100
    with self.assertRaises(ValueError):
      device = Device(*_test_device)

  def test_utility(self):
    pass


class TestTDevice(TestCase):

  def test_invalid_settings(self):
    # pass through params
    _test_device = deepcopy(test_tdevice)
    _test_device[1] = 25
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # t_external
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['t_external'] = [1]
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # t_range
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['t_range'] = -1
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # t_a
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['t_a'] = 2
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # t_b
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['t_b'] = 0
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # a
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['a'] = 100
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)
    # b
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['b'] = 0
    with self.assertRaises(ValueError):
      device = TDevice(*_test_device)

  def test_t2r(self):
    device = TDevice(*deepcopy(test_tdevice))
    self.assertTrue((device.r2t(np.zeros(len(device))) == device.t_base).all())
    self.assertTrue((device.r2t(np.ones(len(device))) > device.t_base).all())
    device = self.get_test_device_heat()
    self.assertTrue((device.r2t(np.ones(len(device))) <= device.t_base).all())

  def test_t2r_shape(self):
    device = TDevice(*deepcopy(test_tdevice))
    self.assertTrue((device.r2t(np.zeros((1, len(device)))) == device.t_base).all())
    self.assertTrue((device.r2t(np.ones((1, len(device)))) > device.t_base).all())
    device = self.get_test_device_heat()
    self.assertTrue((device.r2t(np.ones((1, len(device)))) <= device.t_base).all())


  def test_solve(self):
    ''' Test solving. This is the reason ftol is set to such a high default values. '''
    p = np.ones(24)
    d = TDevice(*deepcopy(test_tdevice))
    try:
      (r, o) = d.solve(p)
    except OptimizationException as e:
      self.fail(e.o)
    self.assertTrue(o.success, msg=o)

  def test_solve_more(self):
    ''' nit: 155 '''
    p = np.ones(24)
    solver_options = {
        'ftol': 1e-6,
        'maxiter': 200,
    }
    device = self.get_test_device_heat()
    (r, o) = device.solve(p, solver_options=solver_options)
    self.assertTrue(o.success)

  def test_solve_more_more(self):
    ''' This will either succeed with nit < 200 or fail with iter exceeded (~1/12 times). '''
    p = np.random.random(24)
    solver_options = {
        'ftol': 1e-6,
        'maxiter': 500,
    }
    device = self.get_test_device_heat()
    (r, o) = device.solve(p, solver_options=solver_options)
    self.assertTrue(o.success)

  def get_test_device_heat(self):
    _test_device = deepcopy(test_tdevice)
    _test_device[4]['t_b'] = -1*_test_device[4]['t_b']
    return TDevice(*_test_device)


class TestConstrainedTDevice(TestCase):

    @unittest.skip('Not implemented')
    def test_constraints(self):
      device = self.get_test_device_agent()
      r = np.zeros(len(device))
      satisfied = np.array([device.min_constraint(r, i) for i in range(0, len(device))])
      self.assertTrue((satisfied[0:4] < 0).all())
      self.assertTrue((satisfied[4:22] > 0).all())
      satisfied = np.array([device.max_constraint(r, i) for i in range(0, len(device))])
      self.assertTrue((satisfied[7:20] >= 0.).all())


class TestGDevice(TestCase):

  def test_gdevice(self):
    d = self.get_test_device()
    self.assertEqual(d.u(zeros, zeros), 0)
    self.assertTrue((d.uv(zeros, zeros) == zeros).all())
    self.assertTrue((d.deriv(zeros, zeros) == 1).all(), d.deriv(zeros, zeros))
    x = np.vectorize(lambda r: d.u(-ones*r, zeros))(np.linspace(1, 10, 100))
    for i, v in enumerate(x):
      self.assertTrue(i == 0 or x[i] < x[i-1])
    x = np.vectorize(lambda r: d.deriv(-ones*r, zeros).sum())(np.linspace(1, 10, 100))
    for i, v in enumerate(x):
      self.assertTrue(i == 0 or x[i] > x[i-1])
    x = np.vectorize(lambda p: d.u(-1*ones, p))(np.linspace(1, 10, 100))
    for i, v in enumerate(x):
      self.assertTrue(i == 0 or x[i] > x[i-1])
    x = np.vectorize(lambda p: d.deriv(-ones, p).sum())(np.linspace(1, 10, 100))
    for i, v in enumerate(x):
      self.assertTrue(i == 0 or x[i] < x[i-1])

  def test_time_varying(self):
    d = self.get_test_device()
    d.cost = np.random.randint(1, 4, (24, 3))
    self.assertNotEqual(d.u(zeros, zeros), 0)
    self.assertNotEqual(d.u(ones, zeros), 0)

  def get_test_device(self):
    cost = [1, 1, 1, 0]
    device = GDevice(
      'x',
      24,
      np.stack((-20*np.ones(24), np.zeros(24)), axis=1),
      None,
      {'cost': cost}
    )
    return device


class TestPVDevice(TestCase):
  ''' Test PVDevice basics. '''
  @classmethod
  def test_basics(cls):
    d = cls.get_test_device()
    # print(d)
    # print(d.bounds, d.lbounds, d.hbounds)

  @classmethod
  def get_test_device(cls):
    max_rate = 2
    area = 2.5
    efficiency = 0.9
    solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
    lbounds = -1*np.minimum(max_rate, solar_intensity*efficiency*area)
    return PVDevice('solar', 24, np.stack((lbounds, np.zeros(24)), axis=1), None, None)


class TestADevice(TestCase):
  ''' Basic test of the basic ADevice, which is Device that takes a utility function, f '''

  def test_adevice(self):
    f = poly2d(np.random.randint(0, 4, (24, 4)))
    d = ADevice('s', 24, [0, 1], None, {'f': f})
    x = np.random.random(24)
    p = np.random.random(24)
    vu = d.u(x, p)
    vd = d.deriv(x, p)
    vh = d.hess(x, p)


class TestSDevice(TestCase):
  ''' Some basic testing of somet part of SDevice. Mainly want to test; that cost functions
  behave as expected, constraints are implemented correctly, and battery dynamics behave correctly
  under various input RoC vectors. Latter is testable via charge_at().
  '''

  def test_sdevice_charge_at(self):
    d = self.get_test_device()
    r = np.ones(24)
    self.assertTrue((d.charge_at(r) == np.arange(6, 24+6)).all())
    d.sustainment = 0.5
    self.assertTrue((np.abs(d.charge_at(r*2) - np.array([4.500000, 4.250000, 4.125000, 4.062500, 4.031250, 4.015625, 4.007812, 4.003906, 4.001953, 4.000977, 4.000488, 4.000244, 4.000122, 4.000061, 4.000031, 4.000015, 4.000008, 4.000004, 4.000002, 4.000001, 4.000000, 4.000000, 4.000000, 4.000000])) < 1e-5).all())

  @classmethod
  def get_test_device(cls):
    return SDevice(*deepcopy(test_sdevice))


class TestMostDevices(TestCase):
  ''' Test things on all known devices.
  @todo Maybe make this the base case, init a common device in sub-classes.
  '''
  test_devices = [
      Device(*test_device),
      CDevice(*test_cdevice),
      CDevice2(*test_cdevice2),
      TestPVDevice.get_test_device(),
      GDevice(*test_gdevice),
      IDevice(*test_idevice),
      IDevice2(*test_idevice2),
      SDevice(*test_sdevice),
      TDevice(*test_tdevice),
    ]

  def test_deriv(self):
    l = len(TestMostDevices.test_devices[0])
    r = np.random.random(l)
    for d in TestMostDevices.test_devices:
      self.assertEqual(len(d.deriv(r, 0)), l)
      self.assertEqual(len(d.deriv(r.reshape(1, l), 0)), l)

  def test_hess(self):
    l = len(TestMostDevices.test_devices[0])
    r = np.random.random(l)
    for d in TestMostDevices.test_devices:
      self.assertEqual(d.hess(r).shape, (l, l))
      self.assertEqual(d.hess(r.reshape(1, l)).shape, (l, l))


class TestUtil(TestCase):

  def test_sustainment_matrix(self):
    sustainment = 0.5
    m = sustainment_matrix(sustainment, 24)
    self.assertTrue(m.shape == (24, 24))
    self.assertTrue((m.diagonal() == np.ones(24)).all())


if __name__ == '__main__':
    unittest.main()
