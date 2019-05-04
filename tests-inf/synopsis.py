import numpy as np
import pandas as pd
import device_kit
from device_kit import *


def random_uncntrld():
  return np.repeat([np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24)))], 2, axis=0)

def main():
  np.random.seed(19)
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  model = DeviceSet('site1', [
      Device('uncntrld', 24, random_uncntrld(), None),
      IDevice2('scalable', 24, (0.5, 2), (0, 24), {'d_0': 0.3}),
      CDevice('shiftable', 24, (0, 2), (12, 24)),
      GDevice('generator', 24, (-10,0), None, {'cost': cost}),
      DeviceSet('sub-site1', [
          Device('uncntrld', 24, random_uncntrld(), None),
          SDevice('buffer', 24, (-7, 7), params={ 'capacity': 70, 'sustainment': 1, 'efficiency': 0.975})
        ],
        sbounds=(0,10)
      ),
    ],
    sbounds=(-0,0)
  )

  # Simple example of "solving". Solution ~meaningless w/o additional constraints such as a requirement for balanced supply and demand.
  (x, solve_meta) = device_kit.solve(model, p=0)
  print(solve_meta.message)
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)
  print(df.sort_index())
  print('Utility: ', model.u(x, p=0))

if __name__ == '__main__':
  main()
