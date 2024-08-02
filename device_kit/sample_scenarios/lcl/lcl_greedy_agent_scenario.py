from device_kit import *
from device_kit.sample_scenarios.lcl.lcl_scenario import *


meta = {
  'title': 'The LCL reference scenario with a greedy agent (home1)'
}


def make_deviceset():
  return DeviceSet('mg', [
    make_home(type=3, id='home1'),
    make_home(type=1, id='home2'),
    make_home(type=1, id='home3'),
    make_home(type=1, id='home4'),
    make_home(type=2, id='home5'),
    make_home(type=2, id='home6'),
    make_home(type=2, id='home7'),
    make_home(type=2, id='home8'),
    make_gas_gen(type=1, id='supply'),
  ])
