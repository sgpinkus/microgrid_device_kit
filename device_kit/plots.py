'''
Convenience functions for plotting tables of data.
'''

from os.path import splitext, basename
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from device_kit import DeviceSet

_colors = ['red', 'orange', 'yellow', 'purple', 'fuchsia', 'lime', 'green', 'blue', 'navy', 'black']
ylim  = (None, None)


def plot_dataframe_as_bars(df, title, filename, aggregation_level=2):
  df_sums = df.groupby(lambda l: '.'.join(l.split('.')[0:aggregation_level])).sum()
  cm=plt.get_cmap('Paired', len(df_sums)+1)
  f = plt.figure(0)
  for (i, (device_label, r)) in enumerate(df_sums.iterrows()):
    plt.bar(range(0, len(df.columns)), r, label=device_label, width=1, edgecolor=cm.colors[i], fill=False, linewidth=1)
  plt.xlim(0, len(df.columns)+12)
  plt.legend()
  plt.title(title)
  plt.ylabel('Power (kWh)')
  plt.xlabel('Time (H)')
  plt.savefig(filename)
  plt.clf()


def plot_dataframe_as_stacked_bars(df, title, filename=None, as_lines=False, fltr=None, aggregation_level=3):
  fig, ax = plt.subplots()
  ax.clear()

  _len = df.shape[1]

  # Sometimes nice to show some lines (ex price, total demand/supply line) with bars.
  if as_lines:
    for line in as_lines:
      ax.plot(range(0, _len), df.loc[line], label=line)

  # Plot possibly filtered list of items of network as stacked bars.
  if fltr:
    df_sums = df.filter(like=fltr, axis=0)
  else:
    df_sums = df.groupby(lambda l: '.'.join(l.split('.')[0:aggregation_level])).sum()
  if as_lines:
    df_sums = df_sums[~df_sums.index.isin(as_lines)]
  bottom = np.zeros(_len)
  neg_bottom = np.zeros(_len)
  for (i, (label, r)) in enumerate(df_sums.iterrows()):
    use_bottom = np.choose(np.array(r < 0, dtype=int), [bottom, neg_bottom])
    ax.bar(range(0, _len), r, color=_colors[i%len(_colors)], label=label, bottom=use_bottom)
    neg_bottom += np.minimum(np.zeros(_len), r)
    bottom += np.maximum(np.zeros(_len), r)

  # Setup ax meta.
  ax.set_xlim(-2, _len+2)
  ax.set_ylim(*ylim)
  ax.set_title(title)
  ax.legend(
    prop={'size': 12},
    # loc='upper right',
    framealpha=0.6,
    frameon=True,
    fancybox=True,
    borderaxespad=-3
  )

  if filename:
    fig.savefig(filename, format='png')

  return fig, ax