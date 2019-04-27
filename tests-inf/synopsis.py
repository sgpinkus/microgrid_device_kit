import numpy as np
import pandas as pd
import device_kit
from device_kit import *

composite_device = DeviceSet([
    Device('uncntrld', 24, np.repeat([np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24)))], 2, axis=0), None),
    IDevice('scalable', 24, (0, 2), (0, 24)),
    CDevice('shiftable', 24, (0, 2), (0, 24)),
    GDevice('generator', 24, (-4,0), None, {'cost': 0.1+0.1*np.sin(np.linspace(0, np.pi, 24))}),
    DeviceSet([
      Device('uncntrld', 24, np.repeat([np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24)))], 2, axis=0), None),
      SDevice('buffer', 24, (-7, 7), params={ 'capacity': 70, 'sustainment': 0.95, 'efficiency': 0.975})
    ],
    id='sub-site1'
    )
  ],
  id='site1'
)

# Simple example of "solving". Solution ~meaningless w/o additional constraints such as a requirement for balanced supply and demand.
(x, solve_meta) = device_kit.solve(composite_device, p=0)
print(solve_meta.message)
print(pd.DataFrame.from_dict(dict(composite_device.map(x)), orient='index'))
print('Utility: ', composite_device.u(x, p=0))
