'''
'''
import random
import numpy as np
from numpy import ones, zeros, hstack, stack
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
from powermarket.agent import *
from powermarket.device import *
from powermarket.tests.test_device import test_tdevice


dim = 24
test_devices_break_me = [
  care2bounds({
    'id': 'x',
    'length': dim,
    'bounds': [0, 300],
    'cbounds': None,  # 2880
    'care': ones(dim),
    'params': {
      't_external': ones(dim)*20,
      't_init': 3,
      't_optimal': 3,
      't_range': 1,
      't_a': 0.1,
      't_b': -0.1,
      'c': 1
    }
  })
]


def test_solve_and_plot(devices):
  for d in devices:
      p = zeros(dim)
      d = DeviceAgent(TDevice(**d))
      print(d)
      print('-'*100)
      r = d.solve(p, solver_options={'maxiter': 5000, 'ftol': 1e-2, 'disp': True})
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
      # Utility function peaks at one (in this cas and as general rule). Translate for visuals.
      # ax2.plot(d.uv(r,p), label='utility (%.2f)' %(ra,))
      ax1.set_xlabel('Time (H)')
      ax1.set_ylabel('Temp (C)')
      ax2.set_ylabel('Power (W)')
      ax1.legend()
      ax2.legend()
      plt.show()


test_solve_and_plot(test_devices_break_me)
