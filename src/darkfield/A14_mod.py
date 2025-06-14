import os
from time import localtime, strftime
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from LightPipes import *

import darkfield.mmmUtils as mu
import darkfield.rossendorfer_farbenliste as rofl
import darkfield.diffra_v2 as df

#import darkfield.diffra as df

from darkfield.utils import read_yaml, write_yaml
from darkfield.diffra_v2 import HOME 
warnings.filterwarnings("ignore")
__version__ = 1

IDX_TO_SIM_TYPE = {
    0: 'dark-field',
    1: 'positive',
    2: 'squeezed',
}


def parse_args():
    """
    Parse command-line arguments.
    """
    argparser = argparse.ArgumentParser(description="parse arguments")
    argparser.add_argument(
        "-N", default=1000, help="Number of simulation points."
    )
    argparser.add_argument(
        "--yaml", required=True, help="YAML filename (e.g., LP_9.yaml)"
    )
    return argparser.parse_args()

#########################################################################################
# Define simulation constants
#########################################################################################
compact_figure = 0
sim_idxs = [0]
# N = 1000
#yamlfile = os.path.join(HOME,"yamls", "5438_1.yaml")
#yamlfile = os.path.join(HOME,"yamls", "5438_Paper_serrated.yaml")
#yamlfile = os.path.join(HOME,"yamls", "42_11_none.yaml")
#yamlfile = os.path.join(HOME,"yamls", "LP_9.yaml")
args = parse_args()
yamlfile = os.path.join(HOME, "yamls", args.yaml)

#########################################################################################
# Main body
#########################################################################################
def main(N):
    ip = read_yaml(yamlfile)
    yamlname = os.path.basename(yamlfile).split('.')[0]

    project = ip['simulation']['project']
    print(project)
    if project:
        print(f'Part of project {project}')
        projectdir = os.path.join(HOME, project)

    keys = ['figures','pickles','yamls']
    for k in keys:
        mu.mkdir(os.path.join(projectdir,k),0)

    for sim_idx in sim_idxs:
        print(f"#### Doing the simulation type {sim_idx}: {IDX_TO_SIM_TYPE[sim_idx]}")
        
        # building the objects
        Elements = []
        for name in ip:
            if name not in ['beam', 'simulation']: 
                obj = ip[name]
                Elements.append([obj['position'],name,obj,1])

        # determine number of plots depending on the number of elements  
        n_plots = len(Elements) + 3
        n_rows = 5 if n_plots > 20 else 4 if n_plots > 15 else 3

        params = ip['simulation']
        params.update({
            'N': N,
            'projectdir': projectdir,
            'positive_signal_simulation': sim_idx,
            'photon_energy': ip['beam']['photonenergy'],
            'fig_rows': n_rows,
            'fig_cols': 5,
            'beamsize': float(ip['beam']['size']),
            'max_pixels': int(ip.get("max_pixels", 300)),
            'gauss_x_shift': float(ip['beam']['offset']),
            'gauss_x_tilt': ip['beam'].get('tilt', 0),
            'break_at': 30,
            'remove_ticks': 1,
            'profiles_normalize': 0,
            'profiles_ylim': [1e-10,1e4],
        })

        mu.clear_times()
        mu.tick()

        # Depending on the type of simulation, choose color and filename
        match sim_idx:
            case 0:
                #fn = ''
                col = rofl.o()
            case 1:
                #fn = 'pos_'
                col = rofl.g()
            case 2:
                #fn = 'sqe_'
                col = 'k'

        # add suffixes and grid size to filename
        if ip['simulation']['name'] == 'as_filename':
            fn = yamlname
        else:
            fn = ip['simulation']['name']
        #fn = f'{fn}_N{N:04.0f}_{strftime("%d-%m-%Y_%H-%M", localtime())}'
        params.update({
            'filename': fn,
            'figs_to_save': ip['simulation'].get('figs_to_save', []),
            'figs_to_export': ['ft'],
            'export_size': float(ip['simulation'].get('export_size', 0)),
            'compact_figure': compact_figure,
        })

        ################# PLOT OF THE NORMALISED INTENSITY (=TRANSMISSION) AS A FUNCTION OF PROPAGATION ################
        plt.figure(figsize=(13,12))

        figstart = 3
        params['ax_apertures'] = None
        params['ax_profiles'] = None

        axInfo = plt.subplot(params['fig_rows'],params['fig_cols'],figstart)
        figstart += 1
        params['fig_start'] = figstart


        #params, trans = df.doit(params,Elements)   ##################################### DO IT ###########
        params, trans, figs = df.doit(params,Elements)   ##################################### DO IT ###########
        plt.sca(axInfo)

        plt.semilogy(trans,'*-',color=col)
        fs=10
        plt.text(0,1e-2,'T = {:.1e}'.format(params['integ']),fontsize=fs)
        t1=params['intensities']['TCC']/params['intensities']['start']
        plt.text(0,1e-4,'start->tcc = {:.1e}'.format(t1),fontsize=fs)
        t2=params['intensities']['roi']/params['intensities']['TCC']
        plt.text(0,1e-5,'tcc->roi = {:.1e}'.format(t2),fontsize=fs)
        if 'roi2' in params['intensities']:
            t22=params['intensities']['roi2']/params['intensities']['TCC']
            plt.text(0,1e-6,'tcc->roi2 = {:.1e}'.format(t22),fontsize=fs)

        plt.text(0,1e-8,'A1: {:0.1f} μm'.format(ip['A1']['size']/um),fontsize=fs)
        plt.text(0,1e-9,'A2: {:0.1f} μm'.format(ip['A2']['size']/um),fontsize=fs)
        if 'intensity_ylim' in ip['simulation']:
            plt.ylim(ip['simulation']['intensity_ylim'])
        else:
            plt.ylim(1e-12,1)
        title_fig = f'{fn}_N{N:04.0f}_{strftime("%d-%m-%Y_%H-%M", localtime())}'
        plt.title(title_fig[:-6])

        mu.savefig(os.path.join(projectdir,'figures',fn), dpi=150) #new version from Aime
        
        mu.print_times()

        mu.dumpPickle([ip,params],os.path.join(projectdir,fn+'_res'))


if __name__ == '__main__':
    #args = parse_args()
    main(int(args.N))


