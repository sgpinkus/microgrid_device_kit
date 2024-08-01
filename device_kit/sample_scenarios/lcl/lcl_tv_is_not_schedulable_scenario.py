''' LCL scenario has no unschedulable loads at all. They presume the tv / entertainment system has
some flex. This scenario just drops that assumption and makes the TV stuff unschedulable.
'''
import numpy as np
from powermarket.agent import *
from device_kit import *
from powermarket.scenario.lcl.lcl_scenario import *


meta = {
  'title': 'The LCL reference scenario with a non schedulable load profile'
}


def make_entertainment(type, id):
  care = np.hstack((np.zeros(4), np.ones(11), np.zeros(9)))  # {12-23}
  bounds = np.stack((0.4*care, 0.4*care), axis=1)
  if type == 2:
    care = np.hstack((np.zeros(10), np.ones(6), np.zeros(8)))  # {18-24}
    bounds = np.stack((0.4*care, 0.4*care), axis=1)
  return Device('tv-av', 24, bounds)
