# Pretty plot configuration.
from matplotlib import rc, pyplot
rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
pyplot.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
pyplot.rc('axes', titlesize=BIGGER_SIZE)  # font size of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
pyplot.rc('legend', fontsize=SMALL_SIZE)  # legend font size
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
