# Scenarios
This directory contains various `device_kit` simulation scenarios as Python modules. A scenario provides domain model. A scenario module provides:

  - make_deviceset(): DeviceSet.
  - meta: optional dict holding meta-data about the scenario.
    - title: Title of scenario. Used in some outputs.
  - matplot_network_writer_hook(event, fig, writer): None
