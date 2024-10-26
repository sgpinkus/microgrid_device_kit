import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import numpy as np
import unittest
from unittest import TestCase
from device_kit.functions import *


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%0.6f' % (v,),
  },
)

class TestFunctions(TestCase):
  def test_basic_properties(self):
    f = QuadraticCost2(-1, 0, 0, 1) # Turns out b = -1, a = 1/2.
    self.assertEqual(f(1), 0)
    self.assertEqual(f(0), 0.5)
    self.assertEqual(f.deriv()(0), -1)
    self.assertEqual(f.deriv()(1), 0.)
    self.assertEqual(f.hess()([0]), f.hess()(0), [[1.]])
    self.assertTrue((f.hess()([1,1]) == f.hess()([253,2])).all())


if __name__ == '__main__':
    unittest.main()
