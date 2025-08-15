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
import diffra_v2 as df
from importlib import reload
reload(df)

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
N_positive = N_squeezed = -1
simulation_types = [0, 1, 2]
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

for simulation_type in simulation_types:
    f = open(yamlfile)
    ip = yaml.safe_load(f)

    if simulation_type == 1:
        N = N_positive if N_positive > 0 else df.yamlval('N_positive', ip['simulation'], 5000)
    elif simulation_type == 2:
        N = N_squeezed if N_squeezed > 0 else df.yamlval('N_squeezed', ip['simulation'], 5000)
    else:
        N = N_negative if N_negative > 0 else df.yamlval('N_negative', ip['simulation'], 5000)

    if N == 0:
        print('Skipping simulation with N = 0')
        continue

    print(f"#### Doing the simulation type {simulation_type}: {sts[simulation_type]}, N={N}")

    #--------------- CONSTRUCTING THE LIST OF OPTICAL ELEMENTS FROM YAML FILE --------------
    Elements = []
    for name in ip:
        if name in ['beam', 'simulation', 'meta']:
            continue
        obj = ip[name]
        Elements.append([obj['position'], name, obj]) #Elements is a dictionnary with : Elements(0) = position, Elements(1) = name, Elements(2) = dictionnary of object properties

    #---------------------------------------------------------------------------------------

    
    removable = ['O1', 'O2', 'O1wb'] # elements to remove for the positive simulation
    insertable = ['TCC', 'squeezer'] # elements to add for the squeezed simulation
    
    if simulation_type == 1: #Positive simulation : removing the elements in "removable"
        for el in Elements:
            if el[1] in removable:
                el[2]['in'] = False
    if simulation_type == 2: # Squeezed simulation : adding the elements in "insertable"
        for el in Elements:
            if el[1] in insertable:
                el[2]['in'] = 1 # Set 'in' to 1 in the dictionnary of object properties

    if forcescatter:
        for el in Elements:
            if el[1] in ['L1', 'L2']:
                el[2]['scatterer'] = 1

    params = ip['simulation']
    XFEL_photon_E = ip['beam']['photonenergy']
    params['photons_total']  = float(ip['beam']['photons_total']) if 'photons_total' in ip['beam'] else None
    params['pulse_duration'] = float(ip['beam']['pulse_duration']) if 'pulse_duration' in ip['beam'] else None
    params['intensity_units'] = df.yamlval('intensity_units', ip['simulation'], 'relative')


    
    params.update({
        'N': N,
        'simulation_type': simulation_type,
        'photon_energy': XFEL_photon_E,
        'fig_rows': 4,
        'fig_cols': 5,
        'beamsize': float(ip['beam']['size']),
        'gauss_x_shift': float(ip['beam']['offset']),
        'gauss_x_tilt': df.yamlval('tilt', ip['beam'], 0),
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
    params, trans, figs = df.doit(params, Elements) 


    mu.dumpPickle([ip, params], str(projectdir) + '/pickles/' + fn + '_res')


print('Simulation finished.')
