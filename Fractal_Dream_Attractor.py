#===============================================================================================
#-----------------------------      Fractal Dream Attractor     --------------------------------
#===============================================================================================

#-----------------------       X = sin(b * Y) + c * sin(b * X)       ---------------------------
#-----------------------       Y = sin(a * X) + d * sin(a * Y)       ---------------------------

#                       This program must be run in a Jupyter notebook.
#-----------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
import pandas as pd
import sys
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import Greys9, inferno, viridis
from datashader.utils import export_image
from functools import partial
from numba import jit
import numba
from colorcet import palette

#-----------------------------------------------------------------------------------------------

background = "white"
img_map = partial(export_image, export_path="fractal_dream_maps", background=background)

n = 10000000

#-----------------------------------------------------------------------------------------------

@jit
def trajectory(fn, a, b, c, d, x0=0, y0=0, n=n):

    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(a, b, c, d, x[i], y[i])

    return pd.DataFrame(dict(x=x, y=y))

@jit
def fractal_dream(a, b, c, d, x, y):

    return np.sin(b*y) + c*np.sin(b*x),   np.sin(a*x) + d*np.sin(a*y)

#-----------------------------------------------------------------------------------------------

cmaps =  [palette[p][::-1] for p in ['bgy', 'bmw', 'bgyw', 'bmy', 'fire', 'gray', 'kbc', 'kgy']]
cmaps += [inferno[::-1], viridis[::-1]]
cvs = ds.Canvas(plot_width = 500, plot_height = 500)
ds.transfer_functions.Image.border=0

#-----------------------------------------------------------------------------------------------

# Parameter  :                   a=xxxxx,  b=xxxxx,  c=xxxxx,  d=xxxxx,
df = trajectory(fractal_dream,   -2.827,    1.371,    1.955,    0.597,   0.1,   0.1)
#df = trajectory(fractal_dream,   -1.895,   -1.142,   -3.120,    1.551,   0.1,   0.1)
#df = trajectory(fractal_dream,   -1.155,   -2.341,   -1.978,    2.181,   0.1,   0.1)
#df = trajectory(fractal_dream,   -1.155,   -1.651,   -2.822,    1.331,   0.1,   0.1)

# Try to put a value in xxxxx.
#df = trajectory(fractal_dream,    xxxxx,    xxxxx,    xxxxx,    xxxxx,   0.1,   0.1)

agg = cvs.points(df, 'x', 'y')
img = tf.shade(agg, cmap = cmaps[2], how='linear', span = [0, n/60000])
img_map(img,"attractor")

