''' Some settings tend to break optimization in previous versions. Seems to have been fixed.
'''
import random
import numpy as np
from numpy import ones, zeros, hstack, stack
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
from powermarket.device_kit import TDevice, care2bounds, solve
from powermarket.device_kit.tests.test_device import test_tdevice


dim = 24
test_devices_break_me = [
  care2bounds({
    'id': 'x',
    'length': dim,
    'bounds': [0, 300],
    'cbounds': None,  # 2880
    'care': ones(dim),
    't_external': ones(dim)*20,
    't_init': 3,
    't_optimal': 3,
    't_range': 1,
    'sustainment': 0.1,
    'efficiency': -0.1,
    'c': 0.1
  })
]


def test_solve_and_plot(devices):
  for d in devices:
      p = zeros(dim)
      d = TDevice(**d)
      print(d)
      print('-'*100)
      (r, x) = solve(d, p, solver_options={'maxiter': 5000, 'ftol': 1e-2, 'disp': True})
      r = r.reshape(24,)
      print('-'*100)
      print(d)
      print('temp', d.r2t(r))
      print('uv  ', d.uv(r, p))
      print('-'*100)
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      ax1.axhline(0, color='k')
      ax1.axhline(d.t_optimal, label='t_optimal', color='k')
      ax1.axhline(d.t_optimal-d.t_range, label='t_minimum', color='k', ls='--')
      ax1.axhline(d.t_optimal+d.t_range, label='t_maximum', color='k', ls='--')
      ax1.plot(d.t_external, label='t_external', color='green')
      ax1.plot(d.t_base, label='t_baseline', color='purple')
      ra = np.average(r)
      ax1.plot(d.r2t(r), color='red', label='t_actual (%.2f)' % (ra,))
      # ax1.plot(d.deriv(r,p), label='deriv (%.2f)' %(ra,))
      ax2.plot(r, label='power (%.2f)' % (ra,))
      # Utility function peaks at one (in this case and as general rule). Translate for visuals.
      # ax2.plot(d.uv(r,p), label='utility (%.2f)' %(ra,))
      ax1.set_xlabel('Time (H)')
      ax1.set_ylabel('Temp (C)')
      ax2.set_ylabel('Power (W)')
      ax1.legend()
      ax2.legend()
      plt.show()


test_solve_and_plot(test_devices_break_me)
