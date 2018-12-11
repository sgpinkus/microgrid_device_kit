import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import unittest
from unittest import TestCase
from test_projection import *
from test_device import *


if __name__ == "__main__":
    unittest.main()
