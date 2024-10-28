''' A GUI builder exist that makes devices with this format. '''
import json
from os.path import basename
import importlib
import numpy as np
import argparse
from scipy.optimize import minimize
import device_kit
from device_kit.functions import *
from pprint import pprint
import logging

logger = logging.getLogger()

def main():
  _self = importlib.import_module('__main__')
  parser = argparse.ArgumentParser(
    description='Load a device_kit scenario from some ambiguous builder export',
  )
  parser.add_argument('filename', action='store',
    help='name of JSON file containing scenario to load'
  )
  args = parser.parse_args()
  print(load(args.filename))


def load(filename):
  data = json.load(open(filename, 'r'))
  basis = data['basis']
  data = data['devices']
  devices = []
  for d in data:
    loader = globals()[f'load_{d['type']}_device']
    devices.append(loader(d, basis))
  return (device_kit.DeviceSet('site', devices), {}, None)


def load_load_device(d, basis: int):
  bounds = np.array(d['bounds']).reshape((basis, 2))
  cbounds = None
  device_id = d['title'] if 'title' in d else d['type']
  params = {}
  cost_function = load_cost_function(d, bounds, cbounds, basis)
  if cost_function:
    params = { 'f': cost_function }
  return device_kit.ADevice(device_id, basis, bounds, cbounds, **params)


def load_fixed_load_device(d, basis: int):
  bounds = np.array(d['bounds']).reshape((basis, 2))
  device_id = d['title'] if 'title' in d else d['type']
  cbounds = None
  if (bounds[:,0] != bounds[:,1]).all():
    raise Exception('Invalid fixed load device')
  return device_kit.ADevice(device_id, basis, bounds, cbounds)


def load_storage_device(d, basis: int):
  parameter_map = {
    'capacity': 'capacity',
    'efficiencyFactor': 'efficiency',
    'reserveRatio': 'reserve',
    'startingRatio': 'start',
    'fastChargeCostFactor': 'c1',
    'flipFlopCostFactor': 'c2',
    'deepDischargeCostFactor': 'c3',
    'deepDepthRatio': 'damage_depth',
    # chargeRateClippingFactor?: number,
    # disChargeRateClippingFactor?: number,
  }
  bounds = np.array(d['bounds']).reshape((basis, 2))
  device_id = d['title'] if 'title' in d else d['type']
  params = { parameter_map[k]: v for k, v in d['parameters'].items() }
  rate_clip = (None, None)
  if 'disChargeRateClippingFactor' in d['parameters']:
    rate_clip[0] = d['parameters']['disChargeRateClippingFactor']
  if 'chargeRateClippingFactor' in d['parameters']:
    rate_clip[1] = d['parameters']['chargeRateClippingFactor']
  params['rate_clip'] = rate_clip
  return device_kit.SDevice(device_id, basis, bounds, None, **params)


def load_supply_device(d, basis: int):
  bounds = -1*np.array(d['bounds']).reshape((basis, 2))
  bounds = np.array([bounds[:,1], bounds[:,0]])
  cbounds = None
  device_id = d['title'] if 'title' in d else d['type']
  params = {}
  cost_function = load_cost_function(d, bounds, cbounds, basis)
  if cost_function:
    params = { 'f': ReflectedFunction(cost_function) }
  return device_kit.ADevice(device_id, basis, bounds, cbounds, **params)


def load_cost_function(d, bounds, cbounds, basis):
  costs_data = d['costs']
  costs = []
  if 'flow' in costs_data:
    logger.info(f'Found flow costs for {d['type']}')
    coeffs = _reshape_offset_quad_coeffs(costs_data['flow'])
    costs += [Poly2DOffset(coeffs)]
  if 'cumulative_flow' in costs_data:
    logger.info(f'Found cumulative_flow for {d['type']}')
    raise Exception('Not implemented')
  if 'flow_bounds_relative' in costs_data:
    logger.info(f'Found flow_bounds_relative for {d['type']}')
    _functions = [HLQuadraticCost(v[0], v[1], bounds[i,0], bounds[i, 1]) for i, v in enumerate(costs_data['flow_bounds_relative'])]
    costs += [X2D(_functions)]
  if 'cumulative_flow_bounds_relative' in costs_data:
    logger.info(f'Found cumulative_flow_bounds_relative for {d['type']}')
    raise Exception('Not implemented')
  if 'peak_flow' in costs_data:
    logger.info(f'Found peak_flow for {d['type']}')
    raise Exception('Not implemented')
  return SumFunction(costs) if len(costs) else None


def _reshape_offset_quad_coeffs(x):
  x = np.array(x)
  basis = len(x)
  return np.concat((x[:, 0:2], np.zeros((basis, 1)), np.array(x[:, 2]).reshape((basis, 1))), axis=1)


if __name__ == '__main__':
  main()