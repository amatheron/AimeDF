import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg

import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils_v2 as mu

from importlib import reload

import time
import random
import yaml
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


#sys.path.append('/home/msmid/mmm_HED/')
#sys.path.append('/home/michal/hzdr/codes/python')
HOME = '/home/yu79deg/darkfield_p5438/'

yaml_template = 'LP_28_template.yaml' #Template of the YAML file we want to open
outdir = './generated_yamls'
#outdir2='../yamls'


########### PARAMETERS TO SET AND SCAN #############

#wide_factors= np.array([1,1.5,3,5,7,10])
#wide_factors= np.array([0.5,1.25,2,4,6])
wide_factors= np.array([0.5,1,1.25,1.5,2,3,4,5,6,7,10])
#wide_factors= np.array([1,3,10])
O2ins=[0,1]
#O2ins=[1]
O_mults=np.array([0.5,1,1.3])
A_mults=np.array([0.5,1,1.3])

#O_mults=np.array([0.5])
#A_mults=np.array([0.5])

####################################################

defect='def-20-2'
for wide_factor in wide_factors:
 for O_mult in O_mults:
  for A_mult in A_mults:
    for O2in in O2ins:
                f = open(yaml_template)
                ip = yaml.safe_load(f)

                ip['simulation']['N_negative'] = 8000
                ip['beam_shaper']['size']=float(ip['beam_shaper']['size']*wide_factor)
                ip['O1']['size']=float(ip['O1']['size']*wide_factor*O_mult)
                ip['L1']['size']=float(ip['L1']['size']*wide_factor)
                ip['O2']['size']=float(ip['O2']['size']*wide_factor*O_mult)
                ip['A1']['size']=float(ip['A1']['size']*wide_factor*A_mult)
                ip['L2']['size']=float(ip['L2']['size']*wide_factor)
                ip['A2']['size']=float(ip['A2']['size']*wide_factor*A_mult)
                ip['beam']['size']=float(ip['beam']['size']*wide_factor)

                if not O2in:
                    ip['O2']['in'] = 0
                    
                if defect!='':
                    defeected=['O1','O2','A1','A2']
                    for di,dd in enumerate(defeected):
                        ip[dd]['defect_type']='sine'
                        ip[dd]['defect_lambda']=float((20+di*0.2)*1e-6)
                        ip[dd]['defect_amplitude']=float(2e-6)

#filenames
                #fn = '/33j_factor-{:03.0f}_02in-{:}'.format(wide_factor*10,O2in)
                #fn = fn + '_Om-{:02.0f}'.format(O_mult*10)
                #fn = fn + '_Am-{:02.0f}'.format(A_mult*10)
                #fn = fn + '_'+defect
                #fn = fn + '.yaml'

                name = outdir + fn
                #name2=outdir2+fn
                print(fn)
                if os.path.exists(name):
                    print('--already exists')
#                    continue

                yaml.dump(ip,open(name, 'w'),sort_keys=False)
                #yaml.dump(ip,open(name2, 'w'),sort_keys=False) #going directly to que
