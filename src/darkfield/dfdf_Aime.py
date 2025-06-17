#!/usr/bin/env python3

#  %% imports
import sys
import os
import time
import random
import yaml
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from LightPipes import *

import mmmUtils as mu
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
positive_signal_simulations = [0, 1, 2]
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
#project = None


yaml_folder = Path('/home/yu79deg/darkfield_p5438/yamls') #hardcoded for simplicity
projectdir = Path('/home/yu79deg/darkfield_p5438/Aime') #hardcoded for simplicity

yamlfile = yaml_folder / args.yaml

print(f"######## Doing folder: {yaml_folder}")
print(f"######## Doing file: {yamlfile}")

if not yamlfile.exists():
    print(f"Error: YAML file '{yamlfile}' does not exist.")
    sys.exit(1)

#project = yamlfile.parts[0]
#print(f"######## In directory (project): {project}")


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


#f = open(yamlfile)
#ip = yaml.safe_load(f)
yamlname = file[:-5] #ex : LP_54 (without the extension .yaml)




sts = ['dark-field', 'positive', 'squeezed']
paramss = {}

for positive_signal_simulation in positive_signal_simulations:
    f = open(yamlfile)
    ip = yaml.safe_load(f)

    if positive_signal_simulation == 1:
        N = N_positive if N_positive > 0 else df.yamlval('N_positive', ip['simulation'], 5000)
    elif positive_signal_simulation == 2:
        N = N_squeezed if N_squeezed > 0 else df.yamlval('N_squeezed', ip['simulation'], 5000)
    else:
        N = N_negative if N_negative > 0 else df.yamlval('N_negative', ip['simulation'], 5000)

    if N == 0:
        print('Skipping simulation with N = 0')
        continue

    print(f"#### Doing the simulation type {positive_signal_simulation}: {sts[positive_signal_simulation]}, N={N}")

    Elements = []
    for name in ip:
        if name in ['beam', 'simulation', 'meta']:
            continue
        obj = ip[name]
        Elements.append([obj['position'], name, obj])

    removable = ['O1', 'O2', 'O1wb']
    insertable = ['TCC', 'squeezer']
    if positive_signal_simulation == 1:
        for el in Elements:
            if el[1] in removable:
                el[2]['in'] = False
    if positive_signal_simulation == 2:
        for el in Elements:
            if el[1] in insertable:
                el[2]['in'] = 1

    if forcescatter:
        for el in Elements:
            if el[1] in ['L1', 'L2']:
                el[2]['scatterer'] = 1

    params = ip['simulation']
    XFEL_photon_E = ip['beam']['photonenergy']
    #'projectdir': projectdir,
    params.update({
        'N': N,
        'positive_signal_simulation': positive_signal_simulation,
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

    #fn = ''
    fn = str(yamlname) #+'_figs'
    
    #if positive_signal_simulation == 1:
    #    fn += 'pos_'
    #elif positive_signal_simulation == 2:
    #    fn += 'sqe_'
    #fn += ip['simulation']['name'] if ip['simulation']['name'] != 'as_filename' else yamlname

    #if forcescatter:
    #    fn += '_fs'
    #fn += f'_N{N:05d}'
    params['filename'] = fn
    params['compact_figure'] = compact_figure

    figX = plt.figure(figsize=(3, 10), layout='constrained')
    fig = plt.figure(figsize=(13, 10), layout='constrained')

    figstart = 1
    params['fig_start'] = figstart + 2
    params['profiles_subfig'] = 1
    params['ax_apertures'] = None

    params, trans, figs = df.doit(params, Elements) ########## RUN THE SIMULATION ########

    axInfo = plt.subplot(params['fig_rows'], params['fig_cols'], 2)
    col = rofl.o() if positive_signal_simulation == 0 else rofl.g() if positive_signal_simulation == 1 else 'k'
    plt.semilogy(trans, '*-', color=col)
    plt.ylim(1e-10, 1)
    fs = 10
    plt.text(0, 0.5, f'N = {params["N"]:.0f}', fontsize=fs)
    plt.text(0, 1e-1, f'T = {params["integ"]:.1e}', fontsize=fs)

    centralelement = 'TCC' if 'TCC' in params['intensities'] else 'PH'
    if centralelement in params['intensities']:
        tr_scat = df.yamlval('transmission_of_scatterer_L2', params, 1)
        t1 = params['intensities'][centralelement] / params['intensities']['start']
        plt.text(0, 1e-2, f'start->{centralelement} = {t1:.1e}', fontsize=fs)
        if 'roi' in params['intensities']:
            t2 = params['intensities']['roi'] / params['intensities'][centralelement] / tr_scat
            if positive_signal_simulation == 0:
                t22 = params['intensities']['roi2'] / params['intensities'][centralelement] / tr_scat
                plt.text(0, 1e-3, f'SFA13 = {t2:.1e}', fontsize=fs, color='r')
                plt.text(0, 1e-8, f'SFA75 = {t22:.1e}', fontsize=fs, color='r')
            if positive_signal_simulation == 1:
                plt.text(0, 1e-3, f'{centralelement}->roi = {t2:.1e}', fontsize=fs)
                if 'roi2' in params['intensities']:
                    t22 = params['intensities']['roi2'] / params['intensities'][centralelement]
                    plt.text(0, 1e-6, f'{centralelement}->roi2 = {t22:.1e}', fontsize=fs)

    if 'A1' in ip:
        plt.text(0, 1e-4, f'1: {ip["A1"]["size"] / um:.1f} μm', fontsize=fs)
    if 'A3' in ip:
        plt.text(0, 1e-5, f'A3: {ip["A3"]["size"] / um:.1f} μm', fontsize=fs)
    if 'A2' in ip:
        plt.text(0, 1e-6, f'A2: {ip["A2"]["size"] / um:.1f} μm', fontsize=fs)
    if 'A4' in ip:
        plt.text(0, 1e-7, f'A4: {ip["A4"]["size"] / um:.1f} μm', fontsize=fs)
    if params['duration'] > 100:
        plt.text(0, 1e-9, f'Duration {params["duration"] / 60:.0f} min.', fontsize=fs)
    else:
        plt.text(0, 1e-9, f'Duration {params["duration"]:.0f} s', fontsize=fs)

    plt.suptitle(fn[:-7], color=rofl.b(), fontsize=16, y=0.95)
    dpi = df.yamlval('figure_dpi', ip['simulation'], 300)
    #mu.savefig(projectdir + 'figures/' + fn, dpi=dpi)
    mu.savefig(str (projectdir / 'figures' / fn) , dpi=dpi)
    mu.print_times()
    params['ax_profiles'] = None
    mu.dumpPickle([ip, params], str(projectdir) + '/pickles/' + fn + '_res')

    if 'flow' in params and isinstance(params['flow'], (list, np.ndarray)) and len(params['flow']) > 0:
        flow_figs = not df.yamlval('flow_auto_save', ip['simulation'], False)
        if force_flow_figs is not None:
            flow_figs = force_flow_figs
        flow_figs = 0
        gyax = df.yamlval('flow_plot_gyax', ip['simulation'], [-200, 1000, 10])
        clim = df.yamlval('flow_plot_clim', ip['simulation'], [1e-11, 50])
        print(str(projectdir.relative_to('/home/yu79deg')), fn )
        #df.flow_plot(str(projectdir), fn , flow_figs=flow_figs, gyax_def=gyax, cl=clim)
        df.flow_plot(str(projectdir.relative_to('/home/yu79deg')), fn , flow_figs=flow_figs, gyax_def=gyax, cl=clim)

print('Simulation finished.')
