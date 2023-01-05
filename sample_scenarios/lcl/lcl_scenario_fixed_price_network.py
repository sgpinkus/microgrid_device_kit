import numpy as np
from powermarket.agent import *
from device_kit import *
from powermarket.scenario.lcl.lcl_scenario import *


class FixedPriceNetwork(Network):
  ''' Just set the price to a constant always. The complication is if just we do that, supply and
  demand won't match, so need to force supply side to what ever the demand side wants
  '''
  _supply_side = None
  _fixed_price = None

  def __init__(self, *args, **kwargs):
    super(FixedPriceNetwork, self).__init__(*args, **kwargs)
    for i, a in enumerate(self.agents):
      if a.id.find('supply') >= 0:
        if self._supply_side:
          raise ValueError('Only one supply side supported.')
        self._supply_side = a
    if not self._supply_side:
      raise ValueError('Supply side agent not found')
    self._fixed_price = kwargs['fixed_price']

  def init(self, *args, **kwargs):
    super(FixedPriceNetwork, self).init(*args, **kwargs)

  def step(self):
    ''' Do one step of iterative method. Where demand exceeds supply self.r is positive. Price is
    adjusted proportionally to excess demand.
    '''
    (self._r_last, self._p_last) = (self.r, self.p)  # Stash for stability calculation.
    self.p = self._fixed_price
    for a in filter(lambda a: a != self._supply_side, self.agents):
      a.update(self.p)
    p = self._supply_side.deriv(r=-1*self.demand, p=0)
    self._supply_side.solve(p, solver_options={'ftol': 1e-09})  # Set p so Network.supply() works.
    self._steps += 1


def make_deviceset():
  raise Exception('Not implemented')
