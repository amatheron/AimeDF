#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:57:08 2024

@author: michal
"""
import sys
from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import numpy as np
#sys.path.append('/home/michal/hzdr/codes/python')import sys

import mmmUtils as mu
import rossendorfer_farbenliste as rofl
import diffra as df
from importlib import reload
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import LogNorm
import os
import time
import random
import yaml
import warnings

# proj='5438_focus'

# yaml='A14_5438_focus_seib_N04000'
# #yaml='A14_5438_focus_cut_N04000'
# yaml='A14_5438_focus_N04000'


# proj='long_M'
# yaml='A14_long_A1110_A2130_p000_s500_N04000'
# yaml='A14_long_A1110_A2130_p000_s500x_N01500'


# proj='11_everlong'
# yaml='A14_pos_everlong_far_A1160_A2160_N01000'


proj='pooyan'
#yaml='5438_N6000'
yaml='5438_1_N10000'
#fn=proj+'/pickles/'+yaml+'_figs.pickle'

fn=os.path.join(proj,'pickles',yaml+'_figs.pickle')
print(fn)
figs=mu.loadPickle(fn)
print(list(figs.keys()))
for fig in figs.keys():
    #if fig!='A1-': continue
    if fig!='A1-': continue
    print(fig)
    mu.figure()
    img=figs[fig][0]
    ps2=figs[fig][2]*1e6/2
    ex=(-ps2,ps2,-ps2,ps2)
    plt.imshow(img,cmap=rofl.cmap())#, norm=LogNorm())#,extent=ex)#, norm=LogNorm())
    L,H = img.shape
    side = 10
    rect_dim = [L//2-side, L//2+side, H//2-side, H//2+side]
    mu.drawRect(rect_dim)
    plt.title(yaml+'   '+ fig)
    plt.colorbar()
    ps3=ps2
    #ax=(-ps3,ps3,-ps3,ps3)
    #mu.savefig(proj+'/images/'+yaml+'_'+fig+'.png')
    mu.savefig(os.path.join(proj,'images',yaml+f'_{fig}.png'))
#    plt.axis(ax)
    plt.clim(0.01,0.35)
    plt.show()
    print("{:.1e}".format(np.sum(img)))

    
# %% profile
    prof=np.nansum(img,0)
    plt.semilogy(prof)
    plt.show()
#print(np.sum(img[140:160,140:160]))
print(np.sum(img))
#mu.drawRect([5,20,5,20])