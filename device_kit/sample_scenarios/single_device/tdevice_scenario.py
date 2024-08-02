'''
Simple single TDevice test scenario:
  - HVAC is modelled.
  - Initial inside temp is 11.
  - External temp is 15+15*sin([0,pi]) so starts at 15 peak at t=12 at 30.
  - Utility is quadratic decreasing about optimal of 19 with a soft range of +/-3 degrees.
  - Cost functions are also sinusoidal, peaking at t=12.
  - t=0 would be roughly 5am and t=11 roughly 5pm.
  - User is assumed to only care about the temperature at t=10 (they get home).

Notes:

  - to model "don't care times" can set TDevice.c parameter. However currently this will break
  NPANetwork.
  - LCLNetwork raises optimization exception Iteration limit exceeded when t_a is ~ <= 0.6.
'''
import numpy as np
from device_kit import *

meta = {
  'title': 'A single TDevice test'
}


def make_deviceset():
  return DeviceSet('nw', make_devices())


def make_devices(dim=24):
  sinusoidal = np.sin(np.linspace(0, np.pi, dim))
  cost = np.stack([sinusoidal*0.2+0.01, np.ones(dim)*0.001, np.zeros(dim)], axis=1)
  caretime = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  params = {
    't_external': np.sin(np.linspace(0, np.pi, dim))*15+15,
    't_init': 11,
    't_optimal': 19,
    't_range': 3,
    'sustainment': 0.7,
    'efficiency': -1.8,
    # 'c': caretime
  }
  return [
    TDevice(
      'hvac',
      dim,
      np.stack((np.zeros(dim), np.ones(dim)*20), axis=1),
      **params
    ),
    GDevice(
      'supply',
      dim,
      np.stack((-20*np.ones(dim), np.zeros(dim)), axis=1),
      **{'cost': cost}
    )
  ]


# def matplot_network_writer_hook(event, fig, writer=None):
#   if event != 'after-update':
#     return
#   fig.axes[0].axhline(agents[0].t_optimal, label='t_opt', color='k')
#   fig.axes[0].axhline(agents[0].t_optimal-agents[0].t_range, label='t_min', color='k', ls='--')
#   fig.axes[0].axhline(agents[0].t_optimal+agents[0].t_range, label='t_max', color='k', ls='--')
#   fig.axes[0].plot(agents[0].t_external, label='t_external')
#   fig.axes[0].plot(agents[0].r2t(agents[0].r), label='t_'+agents[0].id)
#   fig.axes[0].ylim(fig.axes[0].ylim()[0], max(fig.axes[0].ylim()[1], agents[0].t_external.max()))
