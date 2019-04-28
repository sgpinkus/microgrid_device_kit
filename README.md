

# Device Kit
`device_kit` is a Python package containing a collection of "device" models. It also defines a common interfaces for constructing new device models, and collections of devices.

## Synopsis
A simple example of using `device_kit` to model a collection of devices and then doing something with it:

    import numpy as np
    import pandas as pd
    from device_kit import *
    from device_kit.solve import solve

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

    # Simple example of "solving". Solution is meaningless w/o additional constraints such as a
    # requirement for balanced supply and demand. Note "p" parameter is prices.
    (x, solve_meta) = solve(composite_device, p=0)
    print(solve_meta.message)
    print(pd.DataFrame.from_dict(dict(composite_device.map(x)), orient='index'))
    print('Utility: ', composite_device.u(x, p=0))

# Overview
In the scope of this package, a "device" is something that consumes or produces (or both) some kind of scalar valued commodity (ex. electricity, gas, fluid) over a fixed, discrete, and finite future planning/scheduling horizon (ex. every hour of the next day, or every minute of the next hour, etc). What is modeled is merely the constraints on the commodity *flow* to/from the device and preferences for different feasible states of flow to/from the device. The library was created with the intent of providing simple models of electrical devices (gas generators, washing machines, batteries, etc. Since flow at a given instant is just represented as a scalar so the commodity could be anything that can be represented by a scalar).

Devices don't exist in isolation. They exist in some kind of network which acts as a conduit for commodity flow between devices. A device's operation may also be constrained with respect to another devices (example device A flow supplies device B). Most simple networks have a radial structure. The device_kit package also allows one to model, at a rudimentary level, a radially structured network of devices, and also flow couplings between sub-sets of devices. The way it does this is by organizing devices into a rooted tree.

## Class Structure
There is two important class, `Device` and `DeviceSet`. The UML class diagram in the figure below shows how they are related. All devices sub-class `BaseDevice` which is the key abstract representation of a device. A device's consumption/production/both (herein called it's *flow*), is a 2D `RxC` numpy array. For "atomic" `Device` class devices `R` is just 1, and `C` is the fixed length (\_\_len\_\_) of the Device. However, a collection of devices is also implemented as a sub-type of `Device` and this is how `R` can be greater than 1. The `DeviceSet` class is used to represent collections of devices, such as a network or sub-network. `DeviceSet` allows devices to be organized into an arbitrarily deep tree of devices. An example is shown in the figure below. Atomic device always occur at the leaves. All internal nodes, and entire tree itself are `DeviceSet` instances.

Note that all devices are intended stateless; they are not actively consuming producing anything. Rather they are modeling the preferences and constraints for the consumption/production/both of a commodity (the `map(flow)` method shown in the UML diagram allows you to map an actual flow matrix onto a collection of devices). Devices are also intended to be immutable (but technically they are not currently strictly immutable).

---

<table style="width: 100%; text-align: center;">
  <tr>
    <td style="width: 50%;">
        <figure>
            <a name='f1'><img width='460px' name='f1' src='docs/img/uml-cd.png'/></a><br/>
            <small>The core classes of device_kit. Most the model complexity is in sub classes of `Device`. A few are provided in this package such as `IDevice`. </small>
        </figure>
    </td>
    <td>
        <figure>
            <a name='f2'><img width='460px' name='f2' src='docs/img/tree-of-smart-homes-op-e-phys.jpg'/></a><br/>
            <small>DeviceSet allows devices to be connected to represent a radially connected network of devices, such as the Microgrid example shown in the figure. The Microgrid and Sub-Network nodes correspond to DeviceSet instances.</small>
        </figure>
    </td>
  </tr>
</table>


---

## Flexibility Modeling
Device's encapsulate a flexibility model. Flexibility has two components:

  1. *Preferences*. These are sometimes called soft constraints.
  2. *Constraint*. These are sometimes hard constraints.

For convenience, the `Device` base class provides for two very common rudimentary baked-in constraints, `Device.bounds` for *interval* (also called instantaneous) bounds and `Device.cbounds` for *cumulative* bounds.



Preferences are expressed via the `Device.u(flow)` utility function which expresses how much the device "likes" the given state of flow (note that this "utility" function is also used to express how much a producer likes producing a given flow). The Device base class has no preferences; Device.u() just returns 0. It is the main job of a `Device` sub-type to define preferences and/or additional more complex constraints that describe more nuanced device models. sub-types do this by overriding `Device.u()` (preferences) and `Device.constraints` (constraints).
