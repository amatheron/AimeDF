#!/usr/bin/env python3
#  %% imports
import sys
#sys.path.append('/home/msmid/mmm_HED/')
sys.path.append('/home/michal/hzdr/codes/python')

from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import numpy as np
import sys

import mmmUtils as mu
import rossendorfer_farbenliste as rofl
import diffra_v2 as df
from importlib import reload
import os
import time
import random
import yaml
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
reload (df)
version=1

compact_figure=0
gauss_shift=0

dont_move_sim_files=0
N_negative=N_positive=N_squeezed=-1
positive_signal_simulations=[0,1,2]

force_flow=None
force_break=None
force_flow_figs=None
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ("arg0: {:}".format(sys.argv[0]))
if len(sys.argv)>1:  #this can be used for command line controll on server
    print ("arg1: {:}".format(sys.argv[1]))
    command=sys.argv[1]
else:
    command='playground'

if command=='onefile': #load +ě from one single file:
    yamlfile=sys.argv[2]

plot_object=''
map_object=''
close_figure=1
dira=''
forcescatter=0
project=None





#     P L A Y G R O U N D

if command=='playground': #The playground switch;
    print('Just playing, not a production simulation')
#Those settings are mostly for debugging and playing around.
#If the script is used in production 'batch' mode, other commands and settings
#shall be used.
    positive_signal_simulations=[0]
    N_negative=2500
    N_positive=500
    N_squeezed=500
    dont_move_sim_files=1
   # force_flow=np.array( [ -11, 9, 0.25 ])

    force_flow_figs=0
  #  force_flow=0
    yamls=1
#    file='16_long_01only_01Ni3-25_O1-100_A1-050_A2-990_O2fac-85.yaml'
 #   command='onefile'
    close_figure=0
    plot_object='A2'
    map_object='O1'

    ref='34_05_polylens'
    ref=''
    refO=''
    #command='29_damping/29_damping_E.yaml'
    command='32_CRL3/32_20d.yaml'
    command='35_scatterer/35_1_6436_matching2.yaml'
    command='36_CRL3_geometry/36.yaml'
    command='36_CRL3_geometry/test.yaml'
    command='40_CRL34/40.yaml'
    command='yamls'
    command='33_double-imaging/33l.yaml'
    command='42_5438/42_04.yaml'
    command='42_5438/42_10_all.yaml'
    

##################################



###################################


if len(command)>4 and len(command)<7 and command[:4]=='yaml': #load from todos queue
    #dira=sys.argv[3]
    print('Choosing a todo file from directory : yaml' + dira)

    cas=random.random()*1
    print('Going to wait {:.1f}s to randomize the start...'.format(cas))
    time.sleep(cas)
    print('..done')

    if command=='yamls2': dirx='yamls2'
    elif command=='yamls3': dirx='yamls3'
    elif command=='yamls_forcescatter':
        dirx='yamls_forcescatter'
        forcescatter=1
    elif command=='yamls_highres':
        dirx='yamls_highres'
        N_negative=8000
        N_positive=3000
    else: dirx='yamls'

    mu.mkdir(dirx+'/'+dira+'/running/',0)
    dirap = Path(dirx+'/'+dira)
    files= sorted(dirap.glob("*.yaml"))
    assert len(files)>0, 'No more cases for me to do'
    fi=random.randint(0,len(files)-1)
    file=str(files[fi])

    print("########  Choosing file {:} ({:}/{:})".format(file,fi+1,len(files)))

else: #assuming the command is the yaml file
    file=command
    print("########  Doing file {:} ".format(file))
    project=str(file.split('/')[0])
    print("########  In directory (project) {:} ".format(project))
    dirx=project

    if len(sys.argv)>2:  ### LAUNCHING A SIMULATION IN DEBUG MODE ###
        print ("arg2: {:}".format(sys.argv[2]))
        param = sys.argv[2]
        if param[0]=='N':
            N_negative = int(param[1:])
            print('Forcing N_neg = {:.0f}'.format(N_negative)) #Overwrites N positive
            dont_move_sim_files = 1


file=file.split('/')[-1]
if not dont_move_sim_files:
    mu.mkdir(dirx+'/'+dira+'/running/')
    os.rename(dirx+'/'+dira+'/'+file,dirx+'/'+dira+'/running/'+file)
    yamlfile=dirx+'/'+dira+'/running/'+file
    print('Moving yaml file to "running".')
else:
    yamlfile=dirx+'/'+dira+file
    print('Not moving yaml file.')

yamls=1

f=open(yamlfile)
ip=yaml.safe_load(f)
yamlname=yamlfile.split('/')[-1][:-5]

if project is None:
    project=df.yamlval('project',ip['simulation'],'00')
if project==425438: project='42_5438'
if project!=0:
    print(project)
    
    print('Part of project '+project)
    projectdir='./'+project+'/'
    yamlsdonedir=projectdir+'yamls/'
else:
    projectdir='./'
    yamlsdonedir=projectdir+'yamls/done/'

mu.mkdir(projectdir+'figures/',0)
mu.mkdir(projectdir+'pickles/',0)
mu.mkdir(projectdir+'yamls/',0)
mu.mkdir(projectdir+'flows/',0)
sts=['dark-field','positive','squeezed']
paramss={}
for positive_signal_simulation in positive_signal_simulations:
    f=open(yamlfile)
    ip=yaml.safe_load(f)


    if positive_signal_simulation==1:
        if N_positive>0: N=N_positive
        else: N=df.yamlval('N_positive',ip['simulation'],5000)
    if positive_signal_simulation==2:
        if N_squeezed>0: N=N_squeezed
        else: N=df.yamlval('N_squeezed',ip['simulation'],5000)
    if positive_signal_simulation==0:
        if N_negative>0: N=N_negative
        else: N=df.yamlval('N_negative',ip['simulation'],5000)
    if N==0:
        print('skipping')
        continue

    margin=1.
    print("#### Doing the simulation type {:}: {:}, N={:.0f}".format(positive_signal_simulation,sts[positive_signal_simulation],N))

    ################### LIST OF ELEMENTS ###################
    Elements=[]
    for name in ip:
        if name in ['beam', 'simulation','meta']: continue
        obj=ip[name]
        Elements.append([obj['position'],name,obj])

    ################# CHOOSING THE TYPE OF SIMULATION ##############
    removable=['O1','O2','O1wb']
    insertable=['TCC','squeezer']
    if positive_signal_simulation==1:  # POSITIVE
        for el in Elements:
            if el[1] in removable:
                el[2]['in']=False
    if positive_signal_simulation==2:  # SQUEEZED
        for el in Elements:
            if el[1] in insertable:
                el[2]['in']=1


                
    if forcescatter:
        scatt=['L1','L2']
        for el in Elements:
            if el[1] in scatt:
                el[2]['scatterer']=1
# %%
    params=ip['simulation']

    XFEL_photon_E=ip['beam']['photonenergy']
    params['N']=N
    params['projectdir']=projectdir
    params['positive_signal_simulation']=positive_signal_simulation
    params['photon_energy']=XFEL_photon_E
    params['fig_rows']=4
    params['fig_cols']=5
    params['beamsize']=float(ip['beam']['size'])#fwhm of the gaussian beam


    params['gauss_x_shift']=float(ip['beam']['offset'])
    params['gauss_x_tilt']=df.yamlval('tilt',ip['beam'],0)
    if force_flow is not None:
        params['flow']=force_flow
    if force_break is not None:
        params['break_at']=force_break
    params['remove_ticks']=1

# %% Branch for plottin objects only
    if map_object:
        for oname in map_object.split(' '):
          for ele in Elements:
              if oname==ele[1]:
                  objecta=ele[2]
                  print(ele)
                  break
    #                params['propsize']=xl[1]*1e-0
          params['pxsize']=params['propsize']/params['N']
          params['ax_apertures']=None
# %%
          transmap,phasemap,thickmap=df.doap(objecta,params,debug=1,return_thickness=1)
          #plt.xlim(0,100)
          plt.axis([-100,0,0,100])
          plt.xlabel('x [μm]')
          plt.ylabel('y [μm]')
          # %%
        continue
#            mu.figure()
 #           plt.imshow(thickmap)
  #          N2=int(params['N']/2)
   #         ap=thickmap[N2,:]
    #        ps=phasemap[N2,:]


    if plot_object!='':
        mu.figure(9,4)
        xl=[300,1200]
        #xl=[0,1000]
        total_tr=np.zeros(params['N'])+1
        total_th=np.zeros(params['N'])
        total_ps=np.zeros(params['N'])
        tryl=[1e-3,1.5]
        for oname in plot_object.split(' '):
            print(oname)
            for ele in Elements:
                if oname==ele[1]:
                    objecta=ele[2]
                    print(ele)
                    break
            thyl=[-0.1,10000]
            if oname[:1]=='A':
                xl=[0,30]
                thyl=[-0.1,400]
                tryl=[1e-15,2]
            if oname=='L1':
           #     objecta=L1
                xl=[0,800]
                tryl=[1e-2,1]
            if oname=='O1i':
                objecta=O1i
                tryl=[1e-8,1.5]
                xl=[60,110]
                thyl=[-5,20]
            if oname=='O1':
                thyl=[-2,100]
                xl=[-1000,1000]
            if oname=='O2':
                thyl=[0,1000]
                xl=[-250,250]

            if 'O1xxxx' in plot_object.split('_'):
                params['pxsize']=params['propsize']/params['N']
                Na,ap,ps=df.doap(O1i,params,debug=0)
                ap[ap==0]=np.nan
                l=plt.plot(Na*1e6,ap*1e6,label='O1')
                plt.plot(Na*1e6,ap*0,color=l[0].get_color())
                plt.grid()
                plt.legend()
                plt.gca().axis('equal')
                plt.xlim(-150,150)
            doref=0
            if ref!='':
                rNa,r1,r2,r3=mu.loadPickle(projectdir+'/objects/'+ref+'_'+oname)
                refname=ref
                doref=1
            elif refO!='':
                rNa,r1,r2,r3=mu.loadPickle(projectdir+'/objects/'+yamlname+'_'+refO)
                refname=refO
                doref=1
            l1=oname
            if 1:
#                params['propsize']=xl[1]*1e-0
                params['pxsize']=params['propsize']/params['N']
                params['ax_apertures']=None
                transmap,phasemap,thickmap=df.doap(objecta,params,debug=0,return_thickness=1)
                N2=int(params['N']/2)
                ap=thickmap[N2,:]
                ps=phasemap[N2,:]
                trans1=transmap[N2,:]
                Na=(np.arange(params['N'])-N2)*params['pxsize']
                total_th+=ap
                ap[ap==0]=np.nan

                plt.subplot(131)
                l=plt.plot(Na*1e6,ap*1e6,label=l1)
                plt.plot(Na*1e6,ap*0,color=l[0].get_color())
                plt.grid()
                if doref:
                    plt.plot(rNa*1e6,r1*1e6,label=refname)
                plt.subplot(132)
                total_tr=total_tr*trans1
                ap[ap==0]=np.nan
                plt.semilogy(Na*1e6,trans1,label=l1)
                sel=trans1<np.exp(-1)
                plt.semilogy(Na[sel]*1e6,trans1[sel],lw=2)
                try:
                    effectivesize=np.max(Na[sel]*1e6)-np.min(Na[sel]*1e6)
                    print('Effective size: {:.1f} μm'.format(effectivesize))
                except: pass
                if doref:
                    plt.plot(rNa*1e6,r2,label=refname)

                plt.subplot(133)
                plt.semilogy(Na*1e6,ps*-1,label=l1)
                total_ps+=ps

                if doref:
                    plt.plot(rNa*1e6,r3,label=refname)
                mu.mkdir(projectdir+'/objects',verbose=0)
                mu.dumpPickle([Na,ap,trans1,ps],projectdir+'/objects/'+yamlname+'_'+oname)
#            mu.savefig('objects/'+yamlname+'_'+oname)
        addtotal=len(plot_object.split('_'))>1
        addtotal=0
        plt.subplot(131)
        if addtotal: plt.plot(Na*1e6,total_th*1e6,label='total')
        plt.legend()
        plt.xlim(xl)
        plt.ylim(thyl)
#        plt.xlim(50,100)
        plt.title('Thickness')
        plt.xlabel('Position [μm]')
        plt.ylabel('Thickness [μm]')

        plt.subplot(132)
        if addtotal: plt.plot(Na*1e6,total_tr,label='total')
        plt.plot(Na*1e6,total_tr*0+np.exp(-1),'k:',label='1/e')
        plt.legend()
        plt.xlim(xl)
        plt.grid()
        plt.ylim(tryl)
        plt.title('Transmission')
        plt.xlabel('Position [μm]')

        plt.subplot(133)
        if addtotal: plt.plot(Na*1e6,total_ps*-1,label='total')
        plt.grid()
        plt.legend()
        plt.xlim(xl)
        plt.xlabel('Position [μm]')
        plt.ylim(1e-1,1300)
#                plt.ylim(1e-10,1)
        plt.title('Phase shift')
        plt.suptitle(yamlname)
        mu.savefig(projectdir+'/objects/'+yamlname+'_'+plot_object)
        continue


# %% This is a part where we possibly don't really need to touch.
#Maybe just adjust the naming or plotting of some parameters

    fn=''
    if positive_signal_simulation==1:
        fn=fn+'pos_'
    if positive_signal_simulation==2:
        fn=fn+'sqe_'
    if ip['simulation']['name']=='as_filename':
        fn=fn+yamlname
    else:
        fn=fn+ip['simulation']['name']
        if 1:
            import shutil
            file2=file[:-5]
            shutil.copyfile(dirx+'/'+dira+file,yamlsdonedir+file2+'_'+ip['simulation']['name']+'.yaml')
            print('... backing up yaml.')

    if forcescatter:
        fn=fn+'_fs'
    fn=fn+'_N{:05.0f}'.format(N)
    params['filename']=fn

    params['compact_figure']=compact_figure
    efs=3
    figX=plt.figure(figsize=(3,10),layout='constrained')
    fig=plt.figure(figsize=(13,10),layout='constrained')

    figstart=1
    if 0:
        params['ax_apertures']=plt.subplot(params['fig_rows'],params['fig_cols'],figstart)
        figstart+=1
    else: params['ax_apertures']=None

    if 1:
        #params['ax_profiles']=plt.subplot(params['fig_rows'],params['fig_cols'], figstart)
        params['profiles_subfig']=1#plt.subplot(params['fig_rows'],params['fig_cols'], figstart)
        figstart+=1
    else:  params['ax_profiles']=None

    figstart=figstart+1
    params['fig_start']=figstart


# %% The actual simulation
    params,trans,figs=df.doit(params,Elements)   ##################################### DO IT ###########
    #plt.figure(figX)
    if params['ax_apertures']!=None:
        plt.sca(params['ax_apertures'])
        plt.title('Apertures')
        plt.ylim(1e-10,1.5)
        #params['ax_apertures'].set_yscale('linear')
#        plt.legend()
        plt.grid()

#axINFO
    axInfo=plt.subplot(params['fig_rows'],params['fig_cols'],2)
    col=rofl.o()
    if positive_signal_simulation==1: col=rofl.g()
    if positive_signal_simulation==2: col='k'
    plt.semilogy(trans,'*-',color=col)
    plt.ylim(1e-10,1)
    fs=10
    plt.text(0,0.5,'N = {:.0f}'.format(params['N']),fontsize=fs)
    plt.text(0,1e-1,'T = {:.1e}'.format(params['integ']),fontsize=fs)
    centralelement='TCC'
    if centralelement not in params['intensities']:
        centralelement='PH'
    if centralelement in params['intensities']:
        tr_scat=df.yamlval('transmission_of_scatterer_L2',params,1)
        t1=params['intensities'][centralelement]/params['intensities']['start']
        plt.text(0,1e-2,'start->'+centralelement+' = {:.1e}'.format(t1),fontsize=fs)
        if 'roi' in params['intensities']:
            t2=params['intensities']['roi']/params['intensities'][centralelement]/tr_scat
            if positive_signal_simulation==0:
                t22=params['intensities']['roi2']/params['intensities'][centralelement]/tr_scat
                plt.text(0,1e-3,'SFA13 = {:.1e}'.format(t2),fontsize=fs,color='r')
                plt.text(0,1e-8,'SFA75 = {:.1e}'.format(t22),fontsize=fs,color='r')
            if positive_signal_simulation==1:
                plt.text(0,1e-3,centralelement+'->roi = {:.1e}'.format(t2),fontsize=fs)

                if 'roi2' in params['intensities']:
                    t22=params['intensities']['roi2']/params['intensities'][centralelement]
                    plt.text(0,1e-6,centralelement+'->roi2 = {:.1e}'.format(t22),fontsize=fs)


    if 'A1' in ip:
        plt.text(0,1e-4,'1: {:0.1f} μm'.format(ip['A1']['size']/um),fontsize=fs)
    if 'A3' in ip.keys():
        plt.text(0,1e-5,'A3: {:0.1f} μm'.format(ip['A3']['size']/um),fontsize=fs)
    if 'A2' in ip.keys():
        plt.text(0,1e-6,'A2: {:0.1f} μm'.format(ip['A2']['size']/um),fontsize=fs)
    if 'A4' in ip.keys():
        plt.text(0,1e-7,'A4: {:0.1f} μm'.format(ip['A4']['size']/um),fontsize=fs)
    if params['duration']>100:
        plt.text(0,1e-9,'Duration {:.0f} min.'.format(params['duration']/60),fontsize=fs)
    else:
        plt.text(0,1e-9,'Duration {:.0f} s'.format(params['duration']),fontsize=fs)


    paramss[positive_signal_simulation]=params
#    print(['PH']/paramss[0]['intensities']['PH'])
    if positive_signal_simulation==1 and 0 in paramss.keys():
        start_difference=paramss[0]['intensities']['start']/paramss[1]['intensities']['start']
        obstacles=paramss[0]['intensities']['PH']/paramss[1]['intensities']['PH']/start_difference
        sfb75=paramss[0]['intensities']['roi2']/paramss[1]['intensities']['roi2']/start_difference
        sfb13=paramss[0]['intensities']['roi']/paramss[1]['intensities']['roi']/start_difference
        plt.text(0,2e-8,'trans. obst.: {:0.0f} %'.format(obstacles*100),fontsize=fs,color='b')
        plt.text(0,2e-9,'SFB75: {:0.0e} '.format(sfb75),fontsize=fs,color='b')
        plt.text(0,2e-10,'SFB13: {:0.1e} '.format(sfb13),fontsize=fs,color='b',weight='bold')

    if positive_signal_simulation==2:
        try:
            si=paramss[2]['intensities']['roi2']/paramss[0]['intensities']['roi2']
            plt.text(0,2e-8,'Sq.inc.= {:.1e}'.format(si),fontsize=fs,color='g')
        except:
            print('cant say values, possibly not running with neg.sim')

    plt.suptitle(fn[:-7], color=rofl.b(),fontsize=16,y=0.95)
    dpi=df.yamlval('figure_dpi',ip['simulation'],300)
    mu.savefig(projectdir+'figures/'+fn,dpi=dpi)
    mu.print_times()
    if close_figure: plt.close() #To not save the figure into pickle
    params['ax_profiles']=None #To not save the figure into pickle
#    simparams['ax_profiles']=None #To not save the figure into pickle
    mu.dumpPickle([ip,params],projectdir+'pickles/'+fn+'_res')

    if 'flow' in params and type(params['flow'])!=int and len(params['flow'])!=0:
        flow_figs=np.logical_not(df.yamlval('flow_auto_save',ip['simulation'],False))
        if force_flow_figs is not None:
            flow_figs=force_flow_figs
        flow_figs=0
        flow_plot_gyax=df.yamlval('flow_plot_gyax',ip['simulation'],[-200,1000,10])
        flow_plot_clim=df.yamlval('flow_plot_clim',ip['simulation'],[1e-11,50])
        df.flow_plot(project,fn,flow_figs=flow_figs,gyax_def=flow_plot_gyax,cl=flow_plot_clim)

if yamls:
    print('Simulation finished.')

    mu.mkdir(yamlsdonedir)
    if not dont_move_sim_files:
        os.rename(dirx+'/'+dira+'/running/'+file,yamlsdonedir+file)
        print('... movig yaml away.')
# %% scattering figure
if 0:
    mu.figure(8,5)
    fig=figs['Det']
    boxsize=fig[2]
    img=fig[0]
    lineoutwidth=10
    lineoutwidth=75
    pxsize=boxsize/np.shape(img)[0]*1e6
    halfwidth=int((lineoutwidth/pxsize)/2)
    maxx=int(np.shape(img)[0]/2)
    xa=maxx-halfwidth
    xb=maxx+halfwidth
    imcut=img[:,xa:xb]
    vertical=np.sum(imcut,1)
    maxy=np.argmax(vertical)
    yax=np.arange(np.size(vertical))-maxy
    vertical=mu.normalize(vertical)
    yax=(yax)*pxsize

    # downsample
    dyax=np.arange(np.min(yax),np.max(yax),75)
    dprof=dyax*0
    off=-45
    off=45
    #off=-(np.min(np.abs(dyax)))
    dyax=dyax+off
    for di,dy in enumerate(dyax):
        sel=(yax>(dy-37.5))*(yax<=(dy+37.5))
        dprof[di]=np.sum(vertical[sel])
    dprof=mu.normalize(dprof)
    plt.loglog(yax,vertical,label='Simulation')
    plt.semilogy(np.abs(dyax),dprof,'*-',label='Simulation, binned')
    print(np.min(np.abs(dyax)))
# %
    #Experiment
    if 0:
        runNo=560
        pic='/home/michal/hzdr/5438 XFEL darkfield/2024-03-p5438-darkfield/pickles/scatteringprofile_0560_10um.pickle'.format(runNo,lineoutwidth)
        a,b=mu.loadPickle(pic)
        b=mu.normalize(b)
        plt.plot(a,b,label='Experiment 5438 /#{:}'.format(runNo))

    #Experiment
    pic='/home/michal/hzdr/XFEL6436 Darkfield_next_step/p6436-darkfield-next-step/pickles/lens_scattering_CRL4AB scatter, PH30.pickle'
    a,b=mu.loadPickle(pic)
    b=mu.normalize(b)
    plt.plot(a*1e3,b,label='Experiment 6436, 30μm PH')

    plt.xlabel('Position [μm]')
    plt.ylim(1e-8,3)
    plt.xlim(10,3000)
    plt.legend()
    plt.title(ip['simulation']['filename'])
    plt.title('Scattering, '+ip['simulation']['filename']+'')
    mu.savefig('eval/Scattering_{:}_{:}'.format(ip['simulation']['project'],ip['simulation']['filename']))
    mu.dumpPickle([yax,vertical],ip['simulation']['project']+'/'+'pickles/scatteringprofile_{:}_{:}um'.format(ip['simulation']['filename'],lineoutwidth))
    plt.grid()

