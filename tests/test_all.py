import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__ + '/../../')))
sys.path.append(dirname(realpath(__file__ + '/../')))
import unittest
from unittest import TestCase
from test_projection import *
from test_device import *


if __name__ == "__main__":
    unittest.main()
