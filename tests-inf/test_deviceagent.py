import numpy as np
from pprint import pprint
from numpy.random import random
from powermarket.agent import DeviceAgent
from powermarket.device import *
from powermarket.tests.test_device import *
from powermarket.agent.pointlimitedbidstrategy import PointLimitedBidStrategy


np_printoptions = {
    'linewidth': 1e6,
    'threshold': 1e6,
    'formatter': {
      'float_kind': lambda v: '%+0.3f' % (v,),
      'int_kind': lambda v: '%+0.3f' % (v,),
    },
  }
np.set_printoptions(**np_printoptions)


def test_basics():
  a = DeviceAgent(IDevice(*test_idevice))
  print(a)
  print('Basic Accessors')
  print(list(a.b), list(a.r), list(a.p), a.s, a.s.shape)
  print('Utility')
  print('u(r, 0)', a.u(a.r, 0))
  # Interestingly u(), and numpy in gen will work fine with (1,len) shaped 2D input.
  print('u(s, 0)', a.u(a.s, 0))
  print(a.constraints)


def test_solve():
  a = DeviceAgent(IDevice(*test_idevice))
  print(a)
  p = random(len(a))
  a.update(p)
  print(a)


def test_step():
  a = DeviceAgent(IDevice(*test_idevice))
  print(a)
  print('---')
  p = random(len(a))
  a.update(p)
  a0 = deepcopy(a)
  a.init()
  a.p = p
  for i in range(0, 10):
    # print('---\n%s\n%s\n' % (a.p, a.r))
    a.s = a.step(p, 5e-1)[0].reshape((1, len(a)))
  print(a)
  print(a0)


def test_step_strategy():
  a = DeviceAgent(IDevice(*test_idevice), strategy=PointLimitedBidStrategy(stepsize=1e-2))
  p = random(len(a))
  a.init()
  print(a.r)
  print('---')
  for i in range(0, 10):
    a.update(p)
    print(a.r)
    print('---')


def test_typing():
  ''' Typing does not do anything at runtime. Errors not expected. '''
  try:
    a = DeviceAgent(device={}, strategy={})
  except Exception as e:
    print(e)
  try:
    a = DeviceAgent(IDevice(*test_idevice), strategy=[])
  except Exception as e:
    print(e)


test_basics()
test_solve()
# test_step()
# test_step_strategy()
# test_typing()
