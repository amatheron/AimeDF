#!/usr/bin/env python3

#  %% imports
import sys
import os
import time
import random
import yaml
import argparse
import warnings

#---------- Import the good backend to work on both cluster and laptop VSCODE -----------
os.environ.pop("MPLBACKEND", None)
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    print("No display detected: using non-interactive 'Agg' backend.")
    matplotlib.use("Agg")  # Pour le cluster
else:
    print("Display detected: using 'TkAgg' backend.")
    matplotlib.use("TkAgg")  # Pour ton ordi local
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from LightPipes import *

import mmmUtils_v2 as mu
import rossendorfer_farbenliste as rofl
#import diffra_v2 as df
import VIBE as vibe
from importlib import reload
#reload(df)
reload(vibe)

warnings.filterwarnings("ignore")
version = 1


# Command-line arguments to make it SLURM-compatible
def parse_args():
    parser = argparse.ArgumentParser(description="Launch simulation with input YAML and resolution.")
    parser.add_argument("-N", type=int, default=1000, help="Number of simulation points.")
    parser.add_argument("--yaml", required=True, help="YAML input file for the simulation.")
    return parser.parse_args()

args = parse_args() # arguments from the bash command.

# Override default values with args
N_negative = args.N

file = args.yaml #name of the file from the bash command, ex : LP_54.yaml

# Other global parameters
#N_positive = N_squeezed = -1
#simulation_types = [0, 1, 2]
dont_move_sim_files = 1
compact_figure = 0
gauss_shift = 0
force_flow = None
force_break = None
force_flow_figs = None
plot_object = ''
map_object = ''
close_figure = 1
forcescatter = 0


cluster_path = Path("/home/yu79deg/darkfield_p5438")
local_path = Path("/Users/aimematheron/Dropbox/AimeDF")

if cluster_path.exists():
    basepath = cluster_path
else:
    basepath = local_path

yaml_folder = basepath / "yamls"
projectdir = basepath / "Aime"

yamlfile = yaml_folder / args.yaml

print(f"######## Doing folder: {yaml_folder}")
print(f"######## Doing file: {yamlfile}")

if not yamlfile.exists():
    print(f"Error: YAML file '{yamlfile}' does not exist.")
    sys.exit(1)


if not dont_move_sim_files:
    running_dir = yamlfile.parent / "running"
    running_dir.mkdir(exist_ok=True)
    new_path = running_dir / yamlfile.name
    yamlfile.rename(new_path)
    yamlfile = new_path
    print(f'Moved yaml file to "{new_path}".')
else:
    print("Not moving yaml file.")

with open(yamlfile, 'r') as f:
    ip = yaml.safe_load(f)


yamlname = file[:-5] #ex : LP_54 (without the extension .yaml)


sts = ['dark-field', 'positive', 'squeezed']
paramss = {}


f = open(yamlfile)
ip = yaml.safe_load(f)
N = int(args.N)
print(f"#### Starting simulation,resolution N={N}")

#--------------- CONSTRUCTING THE LIST OF OPTICAL ELEMENTS FROM YAML FILE --------------
Elements = []
for name in ip:
    if name in ['beam', 'simulation', 'meta']:
        continue
    obj = ip[name]
    Elements.append([obj['position'], name, obj]) #Elements is a dictionnary with : Elements(0) = position, Elements(1) = name, Elements(2) = dictionnary of object properties

#---------------------------------------------------------------------------------------


params = ip['simulation']
XFEL_photon_E = ip['beam']['photonenergy']
params['photons_total']  = float(ip['beam']['photons_total']) if 'photons_total' in ip['beam'] else None
params['pulse_duration'] = float(ip['beam']['pulse_duration']) if 'pulse_duration' in ip['beam'] else None
#params['intensity_units'] = df.yamlval('intensity_units', ip['simulation'], 'relative')
params['intensity_units'] = vibe.yamlval('intensity_units', ip['simulation'], 'relative')
params['Zoom_global']  = float(ip['simulation']['Zoom_global']) if 'Zoom_global' in ip['simulation'] else 1


params.update({
    'N': N,
    'simulation_type': 0,
    'photon_energy': XFEL_photon_E,
    'fig_rows': 4,
    'fig_cols': 5,
    'beamsize': float(ip['beam']['size']),
    'gauss_x_shift': float(ip['beam']['offset']),
    'gauss_x_tilt': vibe.yamlval('tilt', ip['beam'], 0),
    'remove_ticks': 1
})

if force_flow is not None:
    params['flow'] = force_flow
if force_break is not None:
    params['break_at'] = force_break

fn = str(yamlname)
params['filename'] = fn
params['compact_figure'] = compact_figure

figX = plt.figure(figsize=(3, 10), layout='constrained')
fig = plt.figure(figsize=(13, 10), layout='constrained')

figstart = 1
params['fig_start'] = figstart + 2
params['profiles_subfig'] = 1
params['ax_apertures'] = None

params['projectdir'] = projectdir  

#------------- RUN THE SIMULATION -----------
params, trans, figs = vibe.main_VIBE(params, Elements) 


mu.dumpPickle([ip, params], str(projectdir) + '/pickles/' + fn + '_res')


print('Simulation finished.')
