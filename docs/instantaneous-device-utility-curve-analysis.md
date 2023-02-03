# Instantaneous Utility Function Analysis
Instantaneous means utility is obtained from instaneous consumption. Example: dimmable ligthing. What are the high level and necessary requirements and desirata for instantaneous utility curve, for a device class like dimmable lighting?:

  + The curve *must* be concave.
  + The curve only has to be defined over [q_min(t), q_max(t)]
  + For the devices we are trying to model here, the maximum utility should coincide with `q_max(t)` and decrease towards `q_min(t)`. It doesn't make much sense for the curve to be eventually decreasing.

In practice a human user is going to define the parameters to the curve, so what they mean has to be intuitive. In general, a user cares about 3 things:

  + Where the peak occurs.
  + Where the curve intersects the `q_{t,min}` and `q_{t,max}`.
  + How the curvy-ness matches their understanding of their own utility.

# Alt 1 (IDevice)
`IDevice` instantaneous utility function tries to capture these things with 4 parameters. Refer to notes in [source](../device.py). [This figure](img/idevice-utility-curves.png) shows some example curves.

# Alt 2 (IDevice2)
User specifies `q_{t,min}`, `q_{t,max}` hard bounds. Consumption at `q_{t,max}` is utility maximizing and utility decreases towards `q_{t,min}`. It is unreasonable to expect the user to completely define the mathematical parameters to the utility function. An intuitive way for the user to specify flexibility is to specify at what price they wish to start compromising (from `q_{t,max}`) and at what price they are willing to stop compromising (at `q_{t,min}`). This corresponds to the derivative of the utility function at `q_{t,max}` and `q_{t,min}` respectively.

To acheive this, define a reference quadratic section and scale / fit it accordingly. Etc etc.

    u = np.poly1d([-1/2, 1, 0])
    print(u)
    u(0)
    u(1)
    u.deriv()(1)
    u.deriv()(0)

# Reference Instantaneous Utility Function
The device modeling this [paper][lcl] defined a utility function for two devices whose utility is defined independently at each time-slot as follows:

    U(q(t)) = a - (b - q(t)/q_bar)**(-3/2)

Where:

  - *a*, *b* are "positive constants".
  - *q_bar* is not defined, but we assume they mean be *q_max(t)*.

The above utility function didn't make sense to me so build IDevice, IDevice2 instead.

[lcl]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.650.167&rep=rep1&type=pdf
