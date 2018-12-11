import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import numpy as np
from copy import deepcopy
import unittest
from unittest import TestCase
from powermarket.device import *
from powermarket.tests.test_device import test_device, test_cdevice, test_idevice


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%0.6f' % (v,),
  },
)


class TestDeviceSet(TestCase):
  devices = [
    Device(*test_device),
    Device(*test_device),
    CDevice(*test_cdevice),
    IDevice(*test_idevice),
  ]

  def test_basic_properties(self):
    d = DeviceSet(self.devices)
    self.assertEqual(len(d), 24)
    self.assertEqual(len(d.id), 8)
    self.assertEqual(d.devices, self.devices)
    self.assertEqual(d.shape, (4, 24))
    self.assertTrue((d.partition[:,1] == 1).all())
    self.assertEqual(d.bounds.shape, (4*24, 2))
    self.assertEqual(d.sbounds, None)
    self.assertEqual(len(d.constraints), 6)

  def test_preferences(self):
    self._test_preferences(DeviceSet(self.devices))

  def _test_preferences(self, d):
    x = np.random.random(d.shape)*10
    p = np.random.random(len(d))
    _u = np.array([d.u(x[i], p) for i, d in enumerate(self.devices)]).sum()
    _d = np.concatenate([d.deriv(x[i], p) for i, d in enumerate(self.devices)]).reshape(d.shape)
    self.assertEqual(d.u(x, p), _u)
    self.assertTrue((np.abs(d.deriv(x, p) - _d) < 1e-8).all())
    d.hess(x, p)

  def test_initializer(self):
    d = DeviceSet(self.devices, id="foo", sbounds=(0,100))
    self.assertEqual(len(d.constraints), 6+48)

  def test_set_of_sets(self):
    d1 = DeviceSet(self.devices)
    d2 = DeviceSet(self.devices)
    d = DeviceSet([d1, d2])
    self.assertEqual(d.shape, (8, 24))
    self.assertTrue((d.partition[:,1] == 4).all())
    self.assertEqual(d.bounds.shape, (8*24, 2))


if __name__ == '__main__':
    unittest.main()
