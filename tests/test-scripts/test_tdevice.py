'''
Informal tests for TDevice. How to test TDevice??

  + Ensure the utility function and it's derivative have th right shape over temperature domain with uv_t, deriv_t.
  + Ensire the actual temperature and utility wrt to power have the right correspondance.

@todo Codify in formal tests.
'''
import random
import numpy as np
from numpy import ones, zeros, hstack, stack
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
from device_kit import TDevice, solve
from device_kit.tests.test_device import test_tdevice


colors = ['red', 'black', 'lime', 'navy', 'fuchsia', 'green', 'yellow', 'blue', 'orange', 'purple']
test_tdevice_more = {
  'id': 'test_tdevice',
  'length': 24,
  'bounds': (0, 10),
  't_external': np.sin(np.linspace(0, np.pi, 24))*20+10,  # np.sin(np.linspace(0,np.pi,24))*10+10,
  't_init': 10,
  't_optimal': 21,
  't_range': 2,
  'sustainment': 0.5,
  'efficiency': 0.5,
  'b': 2,
  'c': 1
}
test_tdevice_more_2 = {
  'id': 'test_tdevice',
  'length': 24,
  'bounds': (0, 10),
  't_external': np.sin(np.linspace((1/6.)*np.pi, (7/6.)*np.pi, 24))*70+15,
  't_init': 78,
  't_optimal': random.randrange(73, 77),
  't_range': random.randrange(1, 3),
  'sustainment': 0.99,
  'efficiency': random.random()*(0.008-0.011)-0.008,
  'b': 2
}


def get_test_device(flip=False):
  d = deepcopy(test_tdevice_more)
  if flip:
    d['efficiency'] = -1*d['efficiency']
  type = 'Heater' if d['efficiency'] > 0 else 'Cooler'
  d['id'] = type
  return TDevice(**d)


def get_test_device_2(flip=False):
  d = get_test_device(flip)
  d._c = hstack((zeros(4), ones(16), zeros(4)))*6
  return d


def plot_tdevice_params(d):
  ''' Plot the basic parameters of a TDevice - t_external, t_base, t_conf_(optimal|min|max) '''
  plt.axhline(0, color='k')
  plt.axhline(d.t_optimal, label='t_optimal', color='k')
  plt.axhline(d.t_optimal-d.t_range, label='t_minimum', color='k', ls='--')
  plt.axhline(d.t_optimal+d.t_range, label='t_maximum', color='k', ls='--')
  plt.plot(d.t_external, label='t_external')
  plt.plot(d.t_base, label='t_baseline')


def plot_tdevice(d, r, p):
  ra = np.average(r)
  plt.plot(d.r2t(r), label='t_actual (%.2f)' % (ra,))
  plt.plot(r[0], label='power (%.2f)' % (ra,))
  # Utility function peaks at one (in this cas and as general rule). Translate for visuals.
  plt.plot(d.uv(r, p)[0], label='utility (%.2f)' % (ra,))
  plt.plot(d.deriv(r, p), label='deriv (%.2f)' % (ra,))
  plt.plot(d.c, ls='--', label='care fctr')


def test_r2t():
  ''' Show r2t() increases temperature above t_base for a heater. '''
  d = TDevice(
    id='test_tdevice',
    length=24,
    bounds=(0, 10),
    **{
      't_external': np.sin(np.linspace(0, np.pi, 24))*20+10,  # np.sin(np.linspace(0,np.pi,24))*10+10,
      't_init': 10,
      't_optimal': 21,
      't_range': 2,
      'sustainment': 0.01,
      'efficiency': 5,
    }
  )
  plt.title('Heater power driving up internal temperature above t_base')
  plot_tdevice_params(d)
  r = np.concatenate((ones(12), zeros(12)))
  plt.plot(r, label='input power')
  plt.plot([i for i in d.r2t(r)], label='t_powered')
  plt.legend()
  plt.show()


def test_t_utility(d):
  print(d)
  ax = np.linspace(d.t_min - d.t_range, d.t_max + d.t_range, 100)
  plt.plot(ax, np.vectorize(lambda x: d.uv_t(x).sum())(ax))
  plt.axvline(d.t_optimal, label='set point', color='k')
  plt.axvline(d.t_optimal-d.t_range, label='minus 1', color='k', ls='--')
  plt.axvline(d.t_optimal+d.t_range, label='plus 1', color='k', ls='--')
  plt.xlabel('Temperature')
  plt.ylabel('Utility ($)')
  plt.ylim(-0.5, None)
  plt.legend()
  plt.grid()
  plt.show()


def test_utility(d, rs=1):
  ''' Show utility of temperature and it's derivative '''
  print(d)
  p = zeros(24)
  plt.cla()
  plot_tdevice_params(d)
  plot_tdevice(d, rs*ones(24).reshape(1,24), p)
  plt.xlim(0, 36)
  plt.ylim(None, 32)
  plt.title('''%s device taking a constant supply of power, the real temperatures,
and the device's utility, it's derivative over one day''' % (d.id,))
  plt.xlabel('Time')
  plt.ylabel('Temperature or Power')
  plt.legend()
  plt.grid()
  plt.show()


def test_solve(d):
  ''' Test solving '''
  p = ones(24)*0.001
  print(d.id, '-'*100)
  try:
    r = solve(d, p, solver_options={'maxiter': 500, 'ftol': 1e-4})[0]
  except Exception as e:
    print('An exception occured:', e)
    return
  print('r', r)
  print(r <= d.hbounds)
  print(r >= d.lbounds)
  print(d.r2t(r) >= d.t_min)
  print(d.r2t(r) <= d.t_max)
  print(d.t_min, ';', d.t_max)
  print(d.r2t(r))
  print(d.deriv(r, p))
  plt.cla()
  plot_tdevice_params(d)
  plot_tdevice(d, r, p)
  plt.xlim(0, 36)
  plt.ylim(-10, 32)
  plt.title('''%s device optimal solution, the real temperatures,
and the device's utility, it's derivative over one day''' % (d.id,))
  plt.xlabel('Time')
  plt.ylabel('Temperature or Power')
  plt.legend()
  plt.grid()
  plt.show()


def test_solve_more():
  p = ones(24)*0.01
  test_tdevice = {
    'id': 'test_tdevice',
    'length': 24,
    'bounds': (0, 4),
    **{
      't_external': np.sin(np.linspace(0, np.pi, 24))*10+10,
      't_init': 10,
      't_optimal': 16,
      't_range': 1.5,
      'sustainment': 0.1,
      'efficiency': 10,
    }
  }
  _test_device = deepcopy(test_tdevice)
  d = TDevice(**_test_device)
  print(d)
  print('-'*100)
  r = solve(d, p, solver_options={'maxiter': 10000, 'ftol': 1e-6})[0]
  print(r)
  print(d.r2t(r))


def test_show_utility_is_concave_over_r():
  a = TDevice("ac_1", dimension, stack((mins, maxs), axis=1), cbounds, **params_degenerate())
  plt.cla()
  for i in range(0, 40):
    v = np.random.rand(24)
    w = np.random.rand(24)
    plt.plot([a.u(v*k, p=zeros(24)) for k in np.linspace(0, 2, 20)])
  plt.show()

if __name__ == '__main__':
  print('-'*100)
  test_t_utility(get_test_device(flip=True))
  test_utility(get_test_device(flip=True), 0)
  test_utility(get_test_device(flip=True), 1)
  test_utility(get_test_device(flip=True), 2)
  test_utility(get_test_device(flip=False), 0)
  test_utility(get_test_device(flip=False), 1)
  test_utility(get_test_device(flip=False), 2)
  print('-'*100)
  test_solve(get_test_device(True))
  test_solve(get_test_device())
  test_solve(get_test_device_2(True))
  test_solve(get_test_device_2())
  test_solve_more()
