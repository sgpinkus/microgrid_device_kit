# Scenarios
This directory contains various simulation scenarios for PowerMarket as Python modules. A scenario provides a `device_kit` domain model to which `powermarket` tacks on agents. A scenario provides:

  - make_deviceset(): DeviceSet.
  - meta: optional dict holding meta-data about the scenario.
    - title: Title of scenario. Used in some outputs.
  - matplot_network_writer_hook(event, fig, writer): None
