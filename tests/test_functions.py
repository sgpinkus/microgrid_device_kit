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


class TestQuadraticCost2Function(TestCase):
  def test_basics(self):
    f = HLQuadraticCost(-1, 0, 0, 1) # Turns out b = -1, a = 1/2.
    self.assertEqual(f(1), 0)
    self.assertEqual(f(0), 0.5)
    self.assertEqual(f.deriv()(0), -1)
    self.assertEqual(f.deriv()(1), 0.)
    self.assertEqual(f.hess()([0]), f.hess()(0), [[1.]])
    self.assertTrue((f.hess()([1,1]) == f.hess()([253,2])).all())


class TestSumFunction(TestCase):
  def test_basics(self):
    f = SumFunction([NullFunction(), NullFunction(), NullFunction()])
    g = HLQuadraticCost(-1, 0, 0, 1)
    self.assertEqual(f(1), 0)
    self.assertEqual(f.deriv()(0), 0)
    self.assertEqual(f.hess()(0), 0)
    f = SumFunction([g, g, g])
    self.assertEqual(f(1), 0)
    self.assertEqual(f(0), 1.5)
    self.assertEqual(f.deriv()(0), -3)
    self.assertEqual(f.hess()(0), 3)


class TestPoly2D(TestCase):
  def test_basics(self):
    f = Poly2D([[0,0,0], [1,1,1], [1,2,3]])
    xy = [
      [[0,0,0], [0,1,3]],
      [[1,1,1], [0,3,6]],
      [[2,2,2], [0,4+2+1,4+4+3]]
    ]
    for [x, y] in xy:
      self.assertTrue((y == f(x)).all())


class TestX2D(TestCase):
  def test_basics(self):
    g = HLQuadraticCost(-1, 0, 0, 1)
    h = HLQuadraticCost(0, 1, 0, 1)
    f = X2D([g, h, g])
    xy = [
      [[0,0,0], [0.5,0,0.5]],
      [[1,1,1], [0,0.5,0]],
      [[2,2,2], [0.5,2,0.5]],
      [[3,4,5], [2,8,8]]
    ]
    for [x, y] in xy:
      self.assertTrue((y == f(x)).all())


if __name__ == '__main__':
    unittest.main()
