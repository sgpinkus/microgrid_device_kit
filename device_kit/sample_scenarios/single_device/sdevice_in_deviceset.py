'''
Test SDevice (storage/battery) in simple single_device scenario. The difference between this and
sdevice_scenario is in sdevice_scenario the battery is it's own device and tries to arbitrage. Thus
actually making costs greater for the demand side. In this scenario demand and the battery are
coupled via a DeviceSet.
See sdevice_scenario.py
'''
import numpy as np
from device_kit import *
from device_kit.sample_scenarios.single_device import idevice_scenario
from device_kit.sample_scenarios.single_device import sdevice_scenario

meta = {
  'title': 'Single IDevice scenario augmented with storage'
}


def make_deviceset():
  (demand, demand2, supply) = idevice_scenario.make_devices().values()
  battery = sdevice_scenario.make_devices()['battery']
  return DeviceSet('nw', [DeviceSet('demand-w-battery', [demand, battery]), supply])
