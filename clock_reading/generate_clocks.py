# NOTE: matplotlib is difficult to use inside a virtualenv (tensorflow) in OSX,
# recommend not using it that way (generate clock faces *outside* of the
# virtualenv on mac).

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# NOTE: This code is heavily sourced from the following StackOverflow answer:
# http://codegolf.stackexchange.com/a/20807

# Parameters of the clock hands.
widths = [.05, .05, .00]  # Hide seconds.
lengths = [.6, .9, 1]
colors = plt.cm.gray(np.linspace(0, 1, 4))[0:3]
factor = [12, 60, 60, 1]

# Mess with this to get bigger/smaller images (in pixels). Note that both the
# dpi and figure size matters. These settings produce a 57x57 image.
fig_size = 5
fig_dpi = 14


def _time_to_radians(time):
    rads = [0, 0, 0]
    time += (0, )  # Pad with zero in milliseconds place.
    for i in range(3):
        rads[i] = 2 * np.pi * (float(time[i]) / factor[i] +
                               float(time[i + 1]) /
                               factor[i + 1] / factor[i])

        rads[i] -= (widths[i] / 2)
    return rads


def _setup_axes():
    plt.rcParams['toolbar'] = 'None'
    fig = plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    ax = plt.subplot(111, polar=True)
    ax.get_yaxis().set_visible(False)

    # 12 labels, clockwise
    marks = np.linspace(360. / 12, 360, 12, endpoint=True)
    ax.set_thetagrids(marks, map(lambda m: int(m / 30), marks), frac=.85,
                      size='x-large')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.grid(None)

    # These are the clock hands. We update the coordinates later.
    bars = ax.bar([0.0, 0.0, 0.0], lengths,
                  width=widths, bottom=0.0, color=colors, linewidth=0)

    return fig, ax, bars


def _update_bars(bars, times):
    rads = _time_to_radians(times)
    map(lambda bt: bt[0].set_x(bt[1]), zip(bars, rads))


def init_clock():
    fig, ax, bars = _setup_axes()
    return fig, ax, bars


def save_clock(fig, directory, time):
    fname = os.path.join(directory, 'clock-{:02d}.{:02d}.{:02d}.png'.format(*time))

    fig.savefig(fname, bbox_inches='tight', dpi=14)
    return fname


def set_clock(bars, hours, minutes, seconds,
              show=False):
    time = (hours, minutes, seconds)
    _update_bars(bars, time)

    if show:
        plt.show()


def main(dir_name, index_fname):
    """

    :param dir_name: Directory where to save clocks.
    :param index_fname: File name of index.
    """

    #
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        print('Created directory {}.'.format(dir_name))

    fig, ax, bars = init_clock()

    hours = range(0, 12)
    minutes = range(0, 60)
    seconds = [0]
    times = [x for x in product(hours, minutes, seconds)]

    with open(index_fname, 'w') as index_file:

        for t in times:
            print('Generating clock for time: {}'.format(t))
            set_clock(bars, *t, show=False)
            clock_fname = save_clock(fig, dir_name, t)

            # Store the index string: the filename, hour, and minute
            index_str = '{}\t{}\t{}\n'.format(clock_fname, t[0], t[1])
            index_file.write(index_str)

    print('Created {} clocks.'.format(len(times)))


if __name__ == "__main__":
    dir_name = 'clocks'
    index_fname = 'clocks_all.txt'
    main(dir_name, index_fname)
