from LightPipes import *
import numpy as np
import sys
import os
import time
from LightPipes import Field

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from astropy.io import ascii
from PIL import Image
from scipy import signal
from scipy.ndimage import map_coordinates

from pathlib import Path

from skimage.transform import resize
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy.constants import e, epsilon_0, hbar, c, h, pi
from scipy.special import j1

import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils_v2 as mu
import darkfield.regularized_propagation_v2 as rp

from dataclasses import dataclass
from typing import Dict   # only for type hints

@dataclass
class FieldBundle:
    """
    Holds one or more LightPipes `Field` objects that propagate together.
    For the moment we keep only one channel called **'main'**; the extra
    VB channels will slot in later with no API change.
    """
    fields: Dict[str, Field]   # e.g. {"main": F}
    z_pos: float               # current longitudinal position [m]
    reg: Dict                  # options for regularised propagation

def propagate_bundle(bundle: 'FieldBundle',
                     dz: float,
                     method: str = 'FFT') -> 'FieldBundle':
    """
    Move *all* fields contained in *bundle* forward by *dz* using the same
    rules that the monolithic `doit` loop currently applies to `F`.
    In this first step we still have only one field, but we already
    iterate over the dict so the function won’t need touching again when
    we add `F_VB_parr` and `F_VB_perp` later.
    """
    if dz == 0:
        return bundle  # nothing to do

    for key, F in bundle.fields.items():
        if not bundle.reg.get("regularized_propagation", False):
            if method.lower() == 'fresnel':
                F = Fresnel(dz, F)
            elif method.upper() == 'FFT':
                F = Forvard(dz, F)
            else:
                raise ValueError(f"Unknown propagation method: {method}")
        else:
            # Local import avoids a hard dependency if you don’t use RP
            #import darkfield.regularized_propagation_v2 as rp
            F = rp.Forvard_reg(
                    F,
                    bundle.reg.get("reg_parabola_focus"),
                    dz,
                    False
                 )
     
        bundle.fields[key] = F   # write back the propagated field

    bundle.z_pos += dz
    return bundle
# ──────────────────────────────────────────────────────────────────────



def elem2Z(elem):
    if elem=='Be':   return 4
    if elem=='C':   return 6
    if elem=='CH':   return 5
    if elem=='CH6':   return 4.5
    if elem=='polymer':   return 4.15
    if elem=='O':   return 8
    if elem=='Al': return 13
    if elem=='SiO2': return 10
    if elem=='Si': return 14
    if elem=='Ti':  return 22
    if elem=='Cr':  return 24
    if elem=='Fe':   return 26
    if elem=='Ni':   return 28
    if elem=='Cu':   return 29
    if elem=='Zn':   return 30
    if elem=='Ge':   return 32
    if elem=='Zr':   return 40
    if elem=='Ag':   return 47
    if elem=='W':  return 74
    if elem=='Pt':   return 78
    if elem=='Au':   return 79
    if elem=='Pb':   return 82

    assert 1, "element not found"


def simparams2str(p):

    paramsstr="{:} {:03.0f} {:03.0f} {:03.0f} {:03.0f} {:} {:03.0f} {:.2f} {:.2f} {:s} {:03.0f} {:02.0f}".format(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11])
    return paramsstr

def simstr2params(simstr):
    pars= simstr.split('_')
    return pars


def newobject(shape='',size=0.0*mm,rot=0,smooth=0,invert=0,offset=0,thickness=0,elem='',profile=[],typ='aperture',num=0,defect=''):
    obj={}
    obj['shape']=shape
    obj['size']=size
    obj['smooth']=smooth
    obj['rot']=rot
    obj['invert']=invert
    obj['offset']=offset
    obj['thickness']=thickness
    obj['elem']=elem
    obj['profile']=profile
    obj['type']=typ
    obj['num']=num
    obj['defect']=defect
    #obj['dophaseshift']=1

    return obj

def newobject_from_yaml(name,ip):
    obj=newobject()
    for k in ip.keys():
        spl=k.split('_')
        if spl[0]!=name:continue
        val=ip[k]
        if mu.is_float(val):
            val=float(val)
        obj[spl[1]]=val
    return obj


def update_object_from_yaml(obj,name,ip):
    for k in ip.keys():
        spl=k.split('_')
        if spl[0]!=name:continue
        val=ip[k]
        if mu.is_float(val):
            val=float(val)
        obj[spl[1]]=val
    return obj

def get_n(elem,E):
    #assuming 9831 eV  10μm Zn: 0.17035
    #assert E==8766, print("forcing E as 8766eV, don't have other constants")
    beta=-1
    if elem=='Zn': k=-np.log(0.17035)/(10*um)
    if elem=='W':
        #k=-np.log(0.1522)/(10*um)
        delta=2.945e-5
        beta=1.889e-6
    if elem=='Hf':
        delta=1.988e-5
        beta=3.2887e-6
    if elem=='Au':
        delta=3.10255637E-05
        beta=2.34868139E-06
    if elem=='Al':
        delta=5.655e-6
        beta=7.05e-8
    if elem=='Be':
        delta=3.52e-6
        beta=1.09e-9
    if elem=='Fe':
        delta=1.5705e-5
        beta=1.412e-6
    if elem=='Pt':
        delta=3.407e-5
        beta=2.495e-6
    if elem=='H':  #9831eV, 0.07g/cm3
        delta=4.340e-07
        beta=3.0100e-11
    if elem=='diamond':  #9831eV, 3.5g/cm3
        delta=7.52725919E-06
        beta=8.13892242E-09
    if elem=='Cu':   # at 8200 eV
        E=8200
        delta=2.327e-5
        beta=5.079e-7

    if elem=='Ti':
    #k=-np.log(0.58629)/(10*um)
        delta=9.12e-6
        beta=5.359e-7
    if E==8766:       #for the Ge 440 case at 8766 eV
        if elem=='Be':   #4
            delta=4.4354183E-06
            beta=1.65277048E-09
        if elem=='CH6':  #"4.5"
            delta=7.90943523E-06
            beta=5.73043568E-09
        if elem=='polymer':  #"4.15"
        #https://www.nature.com/articles/s41467-022-28902-8
        #C14 H18 O8
        #6*14 + 18+ 8*8 =   166
        #166/(14*18*8)
        #density 1.2g/cm3
        #8766.    7.11041981E-09
            delta=3.43664874E-06
            beta=7.11041981E-09
        if elem=='CH':  #"5"
            delta=6.406176E-06
            beta=7.60660246E-09
        if elem=='O':
            delta=3.88238197E-09
            beta=1.37375606E-11
        if elem=='acrylate_resin':  #"5"
            delta=3.4461118E-06
            beta=6.87303858E-09
            #C14H18O7 Density=1.2
            #Energy(eV), Delta, Beta
            #  8766.  3.4461118E-06  6.87303858E-09
        if elem=='C':  #6
            delta=5.95414031E-06
            beta=8.17077517E-09
        if elem=='SiO2':  #10
            delta=5.99842178E-06
            beta=6.59105837E-08
        if elem=='Al':  #13
            delta=7.1283157E-06
            beta=1.10876357E-07
        if elem=='Si':  #14
            delta=6.37863104E-06
            beta=1.2381669E-07
        if elem=='Ti':  #22
            delta=1.14419981E-05
            beta=8.25563404E-07
        if elem=='Cr':  #24
            delta=1.79797535E-05
            beta=1.57515535E-06
        if elem=='Fe':   #26
            delta=1.94091281E-05
            beta=2.13984299E-06
        if elem=='Ni':   #28
            #delta=1.64529829E-5
            #beta=3.55248289E-7  probably wrong

            delta=2.11287716E-05
            beta=2.92094592E-06
        if elem=='Cu':   #29
            delta=1.94896511E-05
            beta=3.95634999E-07
        if elem=='Zn':   #30
            delta=1.64529829E-05
            beta=3.55248289E-07
        if elem=='Ge':   #32
            delta=1.21416979E-05
            beta=3.13029915E-07
        if elem=='Zr':   #40
            delta=1.52957E-05
            beta=7.57413659E-07
        if elem=='Ag':   #47
            delta=2.47911867E-05
            beta=1.93673986E-06
        if elem=='W':  #74
            delta=3.86332977E-05
            beta=2.85704482E-06
        if elem=='Pt':   #78
            delta=4.35562652E-05
            beta=3.76524736E-06
        if elem=='Au':   #79
            delta=3.95017742E-05
            beta=3.55916109E-06
        if elem=='Pb':   #82
            delta=2.30786445E-05
            beta=2.31635704E-06
    if E==8906:       #for the Ge 440 case at 8766 eV9
      if elem=='Be':
          delta=4.29694001E-06
          beta=1.55831559E-09
      if elem=='W':
          delta=3.73104158E-05
          beta=2.69871407E-06
      if elem=='Ni':
          delta=2.07341545E-05
          beta=2.7637459E-06

    assert beta!=-1, "Index of refraction not found ({:}, {:} eV)".format(elem,E)
    c=3e8#m/s

    lambd=12398/E*1e-10 #nmn go to ~<
    k=beta/lambd*4*3.14 #based on https://henke.lbl.gov/optical_constants/intro.html
    thickness_to_phaseshift=(delta)/(c*(1-delta)) *c /lambd* 2*3.14
    thickness_to_phaseshift=thickness_to_phaseshift*-1  #this is just as that shall be [March 2015]; Otherwise lenses do not lense.
    return beta,delta,k,thickness_to_phaseshift

_index_cache = {}

def get_index(elem, E, table_dir=None):
    """
    Return delta and beta by interpolating the Henke data for given element and energy.
    If table_dir is not specified, it is assumed to be next to diffra_v2.py
    """
    global _index_cache

    # Special case for Hafnium
    if elem == 'Hf':
        return 3.2887e-6 , 1.988e-5
    
    if elem=='W':  
        return 2.85704482E-06 , 3.86332977E-05
    
    # Locate optical_constants folder relative to the file location of diffra_v2.py
    if table_dir is None:
        table_dir = Path(__file__).parent / "optical_constants"

    filepath = table_dir / f"{elem}.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Set table_dir explicitly if needed."
        )

    if elem not in _index_cache:
        data = np.genfromtxt(filepath, comments='#', skip_header=1)
        if data.shape[1] < 3:
            raise ValueError(f"Expected ≥3 columns (E, delta, beta) in {filepath}")
        energies = data[:, 0]
        delta = data[:, 1]
        beta = data[:, 2]
        _index_cache[elem] = (energies, delta, beta)

    energies, delta_vals, beta_vals = _index_cache[elem]
    delta = np.interp(E, energies, delta_vals)
    beta  = np.interp(E, energies, beta_vals)
    return beta, delta


def thickness_to_phase_and_transmission(E_eV, delta, beta):
    """
    Compute phase shift and absorption coefficient per meter thickness
    for given photon energy and refractive index components.

    Parameters
    ----------
    E_eV : float
        Photon energy in eV
    delta : float
        Refractive index decrement (n = 1 - delta + i beta)
    beta : float
        Absorption index

    Returns
    -------
    thickness_to_phase : float
        Phase shift per meter thickness [rad/m]
    thickness_to_transmission : float
        Absorption coefficient per meter thickness [1/m]
        (so that transmission = exp(-thickness_to_transmission * thickness))
    """
    # Convert energy in eV to wavelength in meters
    E_J = E_eV * e
    lam = h * c / E_J

    # OLD VERSION FROM MICHAL :
    lambd=12398/E_eV*1e-10 #nmn go to ~<
    k_transmission = beta/lambd*4*3.14 #based on https://henke.lbl.gov/optical_constants/intro.html
    thickness_to_phase = (delta)/(c*(1-delta)) *c /lam* 2*3.14
    # Phase shift per meter
    #thickness_to_phase = -(2 * pi * delta) / lam

    # Transmission decay coefficient per meter
    #k_transmission = (4 * pi * beta) / lam

    return thickness_to_phase, k_transmission



def parabolic_lens_profile(xax,r,r0,minr0=0,plot=0):

    #r=0.5
    x=xax
    #r0=2r0/2
    a=2*r

    par=1/a * x**2

    circ=r-(r**2-x**2)**0.5

    max_thick=np.max(par[np.abs(x)<r0])
    par2=par*1
    par2[par2>max_thick]=max_thick


    if minr0>0:
        min_thick=np.max(par[np.abs(x)<minr0])
    #    par2=par*1
        par2=par2-min_thick
        par2[par2<0]=0

    if plot:
        mu.figure()
        plt.plot(x,par,label='parabole')
        plt.plot(x,circ,label='circle')
        plt.plot(x,par2,label='lensprofile')
        plt.ylim(0,2*max_thick)
        plt.xlabel('radius [mm]')
        plt.ylabel('thickness [mm]')
        plt.legend()
        mu.figure()
    return par2

####################### ADDS THE DEFECTS FOR CRLS FROM CELESTRE AND SEIBOTH ################
def do_phaseplate(el_dict,params,debug=0):

    from pathlib import Path #in order for the code to work on laptop and Cluster
    basepath = Path(params["projectdir"]).parent  #in order for the code to work on laptop and Cluster


    assert el_dict['type']=='phaseplate'
    defect = el_dict['defect']

    E = params['photon_energy']
    N = params['N']
    pxsize = params['pxsize']
    num = el_dict['num']
    N2 = int(N/2)
    mx = N2*pxsize

    Na = np.arange(-N2,N2)*pxsize
    xm,ym = np.meshgrid(Na,Na)
    r = ((xm**2)+(ym**2))**0.5
    thickness = np.zeros([N,N])

    if 'seiboth' in defect:
        #fia=ascii.read(f'{HOME}/Seiboth_Fig4') #ancienne version ou la version laptop n'est pas compatible
        fia = ascii.read(basepath / "Seiboth_Fig4")
        fiax=fia['col1']
        fiay=fia['col2']
        if 0:
            mu.figure()
            plt.plot(fiax,fiay)
            plt.xlabel('radial position [μm]')
            plt.ylabel('deformation [μm]')

        img = np.zeros([N,N])
        for xi,x in enumerate(Na):
            for yi,y in enumerate(Na):
                rh = r[xi,yi] / um
                deformation = np.interp(rh,fiax,fiay)
                img[xi,yi] = deformation
        thickness+=img*um

    if 'celestre' in defect:
        #image = Image.open(f'{HOME}/Celestre_Fig8.png') #ancienne version ou la version laptop n'est pas compatible
        image = Image.open(basepath / "Celestre_Fig8.png")
        image = image.resize((N,N))
        im = np.array(image)[:,:,0]
        im = im/255*24 #from values to μm in figure
        im/=11 #to go to one lens from 11
        if 0:
            mu.figure()
            plt.imshow(im)
            plt.colorbar()
        thickness+=im*um

    ############### Multiplying the phase defect by the number of lenses #########
    thickness = thickness * num
    #elem = 'Be'
    elem = params.get('lens_material', 'Be')
    print(f"Material in the do_phaseplate function = {elem}")
    beta ,delta  = get_index(elem, E)
    thickness_to_phaseshift, k_transmission = thickness_to_phase_and_transmission(E, delta, beta)
    phaseshiftmap = thickness * thickness_to_phaseshift

    if 0:
        mu.figure(10,5)
        ax=plt.subplot(121)
        plt.title('Thickness [μm]')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(thickness/um,extent=ex)
        plt.colorbar()
    #if 1:
        ax=plt.subplot(122)
        plt.title('phase shift')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(phaseshiftmap,extent=ex,cmap=rofl.cmap())
        plt.colorbar()
        
    return phaseshiftmap


def make_sphere(radius,pxsize):
    Ns=int(np.ceil(2*radius/pxsize))
    Is=np.zeros([Ns,Ns])
    Ns2=int(Ns/2)
    mx=Ns2*pxsize
    xx=np.arange(-Ns2,Ns2)*pxsize
    ones=xx*0+1
    xa=np.matmul(np.transpose(np.matrix(xx)),(np.matrix(ones)))
    ya=np.transpose(xa)
    ra2=(np.power(xa,2)+np.power(ya,2))
    circ1=np.power((radius**2-ra2),0.5)*2
    sel=np.isnan(circ1)
    circ1[sel]=0
    if 0:
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(circ1/um,extent=ex,cmap=rofl.cmap())
        plt.colorbar()
    return circ1
#    sp=
def add_sphere(radius,xr,yr,img,pxsize,positive):
    #s=np.shape(sph)[0]
    s=int(np.ceil(2*radius/pxsize))
    if (s%2)==1:
        s-=1
    x1=int(xr/pxsize)
    y1=int(yr/pxsize)
    point=img[int(x1+s/2),int(y1+s/2)]
    if point==0:return img
    if point>=4*radius:return img

    orig=img[x1:x1+s,y1:y1+s]
    sph=make_sphere(radius,pxsize)
    #new=orig+sph
    if positive:
#        if np.mean(orig)>10*radius:
 #           new=orig+sph
  #      else:
            new=np.power(orig**2+np.power(sph,2),0.5)
    else: #negative
        new=orig-sph
        new[new<0]=0

    img[x1:x1+s,y1:y1+s]=new
    return img




def do_edge_damping_aperture(params):
    N=params['N']
    edge_damping_shape=yamlval('edge_damping_shape',params,'square')
    trans=np.zeros([N,N])+1
    edge_damping_pixels=params['edge_damping']
    debug=0
    if np.size(edge_damping_pixels)==1: #doing sine damping
#first number is fraction of N where the damping starts
        N_edge=int(N*edge_damping_pixels[0])
        x=np.arange(N_edge)/N_edge*(np.pi/2)
        y=np.sin(x)
        if edge_damping_shape=='square':
    #        plt.plot(x,y)
     #       plt.ylim(0,1.1)
      #      plt.grid()
            for ri,mult in enumerate(y):
                trans[ri,:]*=mult
                trans[:,ri]*=mult
                trans[-1-ri,:]*=mult
                trans[:,-1-ri]*=mult

        if edge_damping_shape=='circular':
            mu.figure()
            N_through=(N/2)-N_edge
            rax=np.arange(N*0.8)
            prof=rax*0+0.5
            prof[rax<N_through]=1
            prof[rax>=N/2]=0
            xm=np.arange(N)
            prof2=prof*1.
            prof2[(rax>=N_through)*(rax<N/2)]=np.flip(y)

            if debug:
                mu.figure()
 #               plt.plot(rax,prof)
                x2=N_through+np.flip(np.arange(N_edge))
#                plt.plot(x2,y)
                plt.plot(rax,prof2,lw=3,alpha=0.5)

            N2=int(N/2)
            Na=(np.arange(N)-N2)*1
            xm,ym=np.meshgrid(Na,Na)
            r=((xm**2)+(ym**2))**0.5

            if debug:
                mu.figure()
                plt.imshow(r)
                plt.colorbar()
            for xi,x in enumerate(xm):
                for yi,y in enumerate(xm):
                    val=np.interp(r[xi,yi],rax,prof2)
                    trans[xi,yi]=val

    else: #doing silly pixel damping
        for ri,mult in enumerate(edge_damping_pixels):
            trans[ri,:]*=mult
            trans[:,ri]*=mult
            trans[-1-ri,:]*=mult
            trans[:,-1-ri]*=mult
  #  print(N_edge)
    if debug:
        mu.figure()
        plt.imshow(trans)
        plt.colorbar()
        plt.title('damping aperture')
        asdf
    return trans




def get_aperture_transmission_map(pars, params={}, debug=0):
    typ = pars['shape']
    pxsize = params['pxsize']  # pixel size in meters
    N = params['N']
    N2 = int(N / 2)

    # 2D coordinate grid centered at 0
    Na = (np.arange(N) - N2) * pxsize
    xm, ym = np.meshgrid(Na, Na)

    # Default map is fully transmissive
    trmap = np.ones((N, N))

    if typ == 'square':
        hs = pars['size'] / 2  # half side length
        sel = (np.abs(xm) <= hs) & (np.abs(ym) <= hs)
        trmap[sel] = 0

    elif typ == 'rectangle':
        hs = pars['size'] / 2
        vs = pars['sizevert'] / 2
        sel = (np.abs(xm) <= hs) & (np.abs(ym) <= vs)
        trmap[sel] = 0

    elif typ == 'wire':
        hs = pars['size'] / 2
        sel = (np.abs(xm) <= hs)
        trmap[sel] = 0

    elif typ == 'circle':
        r = np.sqrt(xm**2 + ym**2)
        rad = pars['size'] / 2
        trmap[r < rad] = 0

    elif typ == 'gaussian':
        r2 = xm**2 + ym**2
        fwhm = float(pars['size'])      # 'size' is FWHM
        P = float(pars.get('power', 2)) # order of super-Gaussian
    
        sigma = fwhm / (2 * np.sqrt(2) * (np.log(2))**(1 / (2 * P))) #wikipedia definition of Super Gaussian profile
    
        trmap = np.exp( - ( (r2 / (2 * sigma**2)) ** P ) )
        

    else:
        raise ValueError(f"Unknown aperture shape: {typ}")

    # Invert transmission if needed
    if yamlval('invert', pars):
        trmap = 1 - trmap

    return trmap



def get_aperture_thickness_map(pars,params=[],debug=0):

    typ = pars['shape']
    pxsize = params['pxsize']
    N = params['N']
    N2 = int(N/2)

    Na=(np.arange(N)-N2)*pxsize
    thicknessmap=np.zeros([N,N])+1  #that mean default is 1 m thick.

    xm,ym = np.meshgrid(Na,Na)
    if typ=='circle':
        r = ((xm**2)+(ym**2))**0.5
        rad = pars['size']/2
        thicknessmap = thicknessmap*0
        thicknessmap[r<rad] = pars['thickness']
        if yamlval('invert',pars):
            maxi = np.max(thicknessmap)
            thicknessmap = maxi-thicknessmap

    if typ in ['parabolic_lens','streichlens']:  #realistic 2D-depth maps
        if typ in ['parabolic_lens','streichlens']:
            r=((xm**2)+(ym**2))**0.5
            rax=np.arange(0,2*N2)*pxsize
            r0=pars['size']/2
            roc=pars['roc']
            prof=parabolic_lens_profile(rax,roc,r0,pars['minr0'],plot=0)
            for xi,x in enumerate(xm):
                for yi,y in enumerate(ym):
                    val=np.interp(r[xi,yi],rax,prof)
                    thicknessmap[xi,yi]=val
            if pars['double_sided']:
                thicknessmap=thicknessmap*2
            thicknessmap=thicknessmap*pars['num_lenses']

    if typ=='streichlens':

        half_gap=pars['gap_size']/2
        sel=np.abs(xm)<half_gap
        if pars['gap_fill']=='empty':
            thicknessmap[sel]=0
        if pars['gap_fill']=='flat':
            iedge=np.argmin(np.abs(Na-half_gap))
            edgeprof=thicknessmap[iedge,:]
            for i,x in enumerate(Na):
                if np.abs(x)<=half_gap:
                    thicknessmap[i,:]=edgeprof
        if pars['gap_fill']=='blade1':
            iedge=np.argmin(np.abs(Na+half_gap))
            iedge2=np.argmin(np.abs(Na-half_gap))
            edgeprof=thicknessmap[iedge,:]
            for i,x in enumerate(Na):
                if i==250:
                    print('asdf')
                horprof=thicknessmap[:,i]
                x1=Na[iedge]
                y1=horprof[iedge]
                x3=Na[iedge2]
                x2=0
                y2=horprof[0]
                blade=np.interp(Na,[x1,x2,x3],[y1,y2,y1])
                sel=blade>horprof
                horprof[sel]=blade[sel]
                thicknessmap[:,i]=horprof
    wls=['realwire','trapez','tent','customwire','pooyan','invpoo','invpar','par','wireslit','linearslit','wire_grating','wireslitup','wireslitdown']

    if typ in wls :
        wireprof = get_wire_like_profile(pars,params,debug) #get the 1D transmission profile

        wireprof[np.isnan(wireprof)] = 0
        if yamlval('smooth',pars)!=0:
            smpx=pars['smooth']/pxsize
            wireprof=mu.convolve_gauss(wireprof,smpx,1)
            ee=int(smpx*2)
            wireprof[0:ee]=wireprof[ee+1]
            wireprof[-ee:]=wireprof[-(ee+1)]

        if yamlval('invert',pars):
            maxi=np.max(wireprof)
            wireprof=maxi-wireprof

        ones = np.ones(N)
        orientate = pars.get('orientate_horizontal', 0)

        if orientate == 1:
            thicknessmap = np.outer(ones, wireprof)  # Horizontal structure (profile along y)
        else:
            thicknessmap = np.outer(wireprof, ones)  # Vertical structure (profile along x)

    defect_type=yamlval('defect_type',pars)
    #if defect_type=='sine':
    #    l = yamlval('defect_lambda',pars)
    #    a = yamlval('defect_amplitude',pars)
    #    #offsets=np.sin(Na/(l/2/3.1415))*a # Michal's version
    #    x = Na * 2 * np.pi / l
    #    offsets = np.sin(x)*a
    #    #offsets_px=(np.round(offsets/pxsize))
    #    offsets_px = np.round(offsets / pxsize).astype(int)
    #    tmp = thicknessmap*0
    #    for yi in np.arange(N):
    #        #op=int(offsets_px[yi])
    #        op = offsets_px[yi]
    #        #a=np.roll(np.transpose(np.array(thicknessmap[:,yi])),op)
    #        #tmp[:,yi]=np.transpose(a)
    #        tmp[:, yi] = np.roll(thicknessmap[:, yi], offsets_px[yi])
    #    thicknessmap = tmp*1



    if defect_type in ['sine', 'sawtooth', 'triangle']:
        wavelength = float(yamlval('defect_lambda', pars))
        amplitude = float(yamlval('defect_amplitude', pars))

        # Create phase array
        x = Na * 2 * np.pi / wavelength  # normalize to [0, 2π]

        if defect_type == 'sine':
            offsets_m = np.sin(x) * amplitude
        elif defect_type == 'sawtooth':
            offsets_m = signal.sawtooth(x) * amplitude
        elif defect_type == 'triangle':
            offsets_m = signal.sawtooth(x, width=0.5) * amplitude  # triangle waveform

        offsets_px = np.round(offsets_m / pxsize).astype(int)

        tmp = np.zeros_like(thicknessmap)
        for yi in range(N):
            
            defect_line = np.roll(np.transpose(np.array(thicknessmap[:,yi])),offsets_px[yi]) #Michal's and Pooyan's version (not optimal?)
            tmp[:, yi] = np.transpose(defect_line)

        thicknessmap = tmp.copy()

    return thicknessmap


def get_wire_like_profile(pars,params,debug):
    #Calculates a 1D profile of the transmission axis.
    r = yamlval('size',pars,0)/2
    off = float(yamlval('offset',pars,0))
    elem = pars['elem']
    pxsize = params['pxsize']
    N = params['N']
    N2 = int(N/2)

    Na = (np.arange(N)-N2)*pxsize
    x = Na-off
    typ = pars['shape']

    #here I need to find the wireprof - i.e. 1D profile of thickness on 'x' as an x-axis
    if typ.find('trapez')==0:
        thickness=pars['thickness']
        edge=pars['edge']
        g=r/edge
        wireprof=g*r-g*np.abs(x)
        wireprof=thickness/edge*(r-np.abs(x))
        wireprof[wireprof<0]=0
        wireprof[wireprof>thickness]=thickness
    elif typ.find('tent')==0:
        thickness=pars['thickness']
        wireprof=thickness*(1-np.abs(x)/r)
        wireprof[wireprof<0]=0
    elif typ.find('customwire')==0:
        wireprof=pars['profile']
    elif typ.find('invpoo')==0:
        l1=pars['l1']
        d1=pars['size']
        l2=pars['l2']

        d2=pars['size']-pars['d']*2

        d=(d1-d2)/2
        two_p=(l1**2+d**2)**0.5
        p=two_p/2
        alpha=np.arctan(l1/d)
        r=p/np.cos(alpha)
        wireprof=x*0
        assert d<=l1, 'The Pooyan shape does not work like this: d>l1 (d={:.0f},l1={:.0f})'.format(d/um,l1/um)
        circ_cen=0-d1/2+r
        wireprof[np.abs(x)>=d2/2]=l2+l1

        ss=np.logical_and(x<(-d2/2),x>(-d1/2))
        circ=(r**2-(x-circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

        ss=np.logical_and(x>(d2/2),x<(d1/2))
        circ=(r**2-(x+circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

    elif typ == 'wireslit':# before : typ.find('wireslit')==0:
        r = pars['r']
        wireprof = x*0
        halfsize = pars['size']/2
        off = halfsize+r
        circ1 = (r**2-(Na-off)**2)**0.5*2
        sel1 = (x>=halfsize) * (x<=off)
        wireprof[sel1] = circ1[sel1]
        circ2 = (r**2-(Na+off)**2)**0.5*2
        sel2 = (x<=halfsize) * (x>=-off) #wrong condition !! be carefull.
        wireprof[sel2] = circ2[sel2]
        wireprof[np.abs(x)>off] = 2*r
        wireprof[np.abs(x)<halfsize] = 0

    elif typ == 'wireslitup':
        r = pars['r']
        wireprof = x * 0
        halfsize = pars['size'] / 2
        off = halfsize + r
        circ1 = (r**2-(Na-off)**2)**0.5*2
        sel1 = (x>=halfsize) * (x<=off)
        wireprof[sel1] = circ1[sel1]
        wireprof[x > off] = 2 * r
        wireprof[x < halfsize] = 0

    elif typ == 'wireslitdown':
        r = pars['r']
        wireprof = x * 0
        halfsize = pars['size'] / 2
        off = halfsize + r
        circ2 = (r**2-(Na+off)**2)**0.5*2
        sel2 = (x<=halfsize) * (x>=-off) 
        wireprof[sel2] = circ2[sel2]
        wireprof[x < off] = 2 * r
        wireprof[x > halfsize] = 0

    elif typ.find('invpar')==0:
        l=pars['l']
        halfsize=pars['size']/2
        n=pars['n']
        d=pars['d']
        a=l/(d**n)
        wireprof=x*0
        wireprof[np.abs(x)>=halfsize]=l

        par=a*np.abs(x-(halfsize-d))**n
        ss=x>=halfsize-d
        wireprof[ss]=par[ss]

        par=a*np.abs(x+(halfsize-d))**n
        ss=x<=-halfsize+d
        wireprof[ss]=par[ss]

        wireprof[wireprof>l]=l

    elif typ.find('linearslit')==0:
        l=pars['l']
        d=pars['d']
        halfsize=pars['size']/2+d
        a=l/d
        angle=np.arctan(a)/np.pi*180
        print('  Angle of the {:} slit blade is {:.0f}˚'.format(pars['elem'],angle))
        wireprof=x*0
        wireprof[np.abs(x)>=halfsize]=l

        par=a*np.abs(x-(halfsize-d))
        ss=x>=halfsize-d
        wireprof[ss]=par[ss]

        par=a*np.abs(x+(halfsize-d))
        ss=x<=-halfsize+d
        wireprof[ss]=par[ss]

        wireprof[wireprof>l]=l
        if yamlval('thicksize',pars)>0:
            ss=np.abs(x)>=pars['thicksize']
            wireprof[ss]=pars['thickthickness']


    elif typ.find('par')==0:
        l=pars['l']
        halfsize=yamlval('d2',pars,0)/2
        n=pars['n']
        d=pars['d']
        a=l/(d**n)
        size=yamlval('size',pars,-1)
        if size>0:  #estimating the d2 from effecitve size;
            par=a*np.abs(x-d)**n
            beta, delta  = get_index(elem,params['photon_energy'])
            thickness_to_phaseshift,k = thickness_to_phase_and_transmission(params['photon_energy'], delta, beta)
            par_trans=np.exp(-k*par) #transmission
            sel=(par_trans>np.exp(-1))*(x>0)
            edgex=np.min(x[sel])
            halfsize=size/2-edgex

        wireprof=x*0
        wireprof[np.abs(x)<=halfsize]=l

        if 1:
            par=a*np.abs(x-(halfsize+d))**n
            print('Parabolic obstacle parameter a={:.2f} mm-1'.format(a*1e-6))
            if 0:
                print(x)
                print(par)
                mu.figure()
                plt.plot(x*1e6,par*1e6)
                mu.figure()
            ss=x>=halfsize-d
            wireprof[ss]=par[ss]

            par=a*np.abs(x+(halfsize+d))**n
            ss=x<=-halfsize+d
            ss=x<=0
            wireprof[ss]=par[ss]
            wireprof[wireprof>l]=l
            wireprof[np.abs(x)>halfsize+d]=0

        if yamlval('edge-r',pars)>0:
                print('doing the edge')

                r=pars['edge-r']
                off=np.abs(x[np.argmin(np.abs(wireprof-2*r))])
                print(off)
                circ1=(r**2-(Na-off)**2)**0.5*2
                sel1=(x>=off) * (x<=off+r)
                wireprof[sel1]=circ1[sel1]
                circ2=(r**2-(Na+off)**2)**0.5*2
                sel2=(x<=-off) * (x>=-off-r)
                wireprof[sel2]=circ2[sel2]
                wireprof[np.abs(x)>off+r]=0

    elif typ.find('pooyan')==0:
        l1=pars['l1']
        d1=pars['d1']
        l2=pars['l2']
        d2=pars['d2']
        d=(d1-d2)/2
        two_p=(l1**2+d**2)**0.5
        p=two_p/2
        alpha=np.arctan(l1/d)
        r=p/np.cos(alpha)
        print('Pooyans shape r is {:.0f} μm'.format(r/um))
        wireprof=x*0
        assert d<=l1, 'The Pooyan shape does not work like this: d>l1 (d={:.0f},l1={:.0f})'.format(d/um,l1/um)
        circ_cen=0-d2/2-r
        circ=(r**2-(x-circ_cen)**2)**0.5
        wireprof[np.abs(x)<=d2/2]=l2+l1

        ss=np.logical_and(x<(-d2/2),x>(-d1/2))
        wireprof[ss]=l1-circ[ss]

        ss=np.logical_and(x>(d2/2),x<(d1/2))
        circ=(r**2-(x+circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

        wireprof[wireprof<0]=0


    elif typ.find('realwire')==0: #round wire
            wireprof=(r**2-(Na-off)**2)**0.5*2
    elif typ=='mist':
        maxi=r*2*np.sqrt(2)
        wireprof=maxi-2*np.abs(x)
        wireprof[wireprof<0]=0

    elif typ.find('wire_grating')==0: #GRATING made of wires
    #paremters: spoacing, factor
        spacing=float(pars['spacing'])
        factor=float(pars['factor'])
        offset=float(pars['offset'])
        wireradius=spacing/factor/2
        numwires=int(np.ceil(N*pxsize/spacing))
        grating=Na*0
        for wi in np.arange(-numwires,numwires):
            wirecenter=wi*spacing+offset
            wireprof=(wireradius**2-(Na-wirecenter)**2)**0.5*2
            wireprof[wireprof<0]=0
            wireprof[np.logical_not(np.isfinite(wireprof))]=0
            grating=grating+wireprof
        wireprof=grating
    else:
        assert 1, "Obstacle type not found"
    return wireprof




'''
shapes available:
    with realistic 2D depth profiles:
        circle
        parabolic_lens
    with realistic 1D depth profils ('like wires'):
        realwire
        trapez
        tent
        customwire
        wire_grating
'''

def doap(pars,params=[],debug=0,return_thickness=0):
    axap = params['ax_apertures']
    E = params['photon_energy']
    N = params['N']
    N2 = int(N/2)
    typ = pars['shape']

    if typ in ['square','rectangle','wire','gaussian']:
        transmissionmap = get_aperture_transmission_map(pars,params,debug)
        phaseshiftmap = transmissionmap * 0
        thicknessmap = transmissionmap * 0
    else:
        thicknessmap = get_aperture_thickness_map(pars,params,debug)


    #General modifications of thickness map
        if yamlval('randomizeA',pars):  # Adding random defects. 'RandomizeA' is the maximal amplitude of the noise [m]. The spatial frequency is just given by pixel size
            ra = float(yamlval('randomizeA',pars))
            rand = np.random.random((N,N))*ra - ra/2
            img2 = thicknessmap+rand
            img2[thicknessmap==0] = 0
            img2[thicknessmap<=0] = 0
            thicknessmap = img2
            print('randomized')
        if yamlval('randomizeB',pars): # Adding random defects. in better way. 'RandomizeB' is the maximal radius of sphere added to the material[m].
            maxsize=float(yamlval('randomizeB',pars))
            density=float(yamlval('density',pars,2))
            print('Density: ',density)
            print(pars)
            boxsize=params['propsize']
            numsph=density*boxsize**2/maxsize**2 
            #10.6.2025 --formula above was dependent on box size, which is not numerically correct
            # I'm changing that to the size of the lens.
            #Density example: I have a size=400μm, spheres with 20μm max. diamter, density 1
               #  that makes 400 spheres. sounds sane.
               #If I had k=0.02, and num_lenses=6, then that would make 48 spheres, which is quite noisy
            numsph=density*pars['size']**2/maxsize**2 
            pxsize=params['pxsize']

            for i in np.arange(numsph):
                size=np.random.random()*maxsize
                xr=np.random.random()*(boxsize-2*size-2*pxsize)
                yr=np.random.random()*(boxsize-2*size-2*pxsize)
                positive=np.random.random()>0.3
                add_sphere(size,xr,yr,thicknessmap,pxsize,positive)

            print('randomized B')

        if yamlval('rot',pars)==0:
            thicknessmap=np.array(np.transpose(thicknessmap))
        elif yamlval('rot',pars)==90:
            thicknessmap=np.array(thicknessmap)
        else:
            from scipy.ndimage.interpolation import rotate
            rot=90-pars['rot']#
            thicknessmap= rotate(thicknessmap, angle=rot,reshape=0)
        if yamlval('crossed',pars,0):
            thicknessmap2=np.transpose(thicknessmap)
            thicknessmap=thicknessmap2*thicknessmap


        #CONVERTING THICKNESS MAP INTO TRANSMISSION AND PHASESHIFT MAP
        elem=pars['elem']
        beta ,delta  = get_index(elem,E)
        thickness_to_phaseshift,k = thickness_to_phase_and_transmission(E, delta, beta)
        transmissionmap = np.exp(-k*thicknessmap)
        phaseshiftmap = thicknessmap * thickness_to_phaseshift
        if debug:
            print('thickness_to_phaseshift =',thickness_to_phaseshift)
            print('max thickness = ',np.max(thicknessmap))
            print('max phaseshift = ',np.max(phaseshiftmap))
            print('min phaseshift = ',np.min(phaseshiftmap))

    if 0:
        plt.sca(axap)
        lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
        plt.semilogy(Na/um,trans,label=lab)
        plt.ylabel('Transmission [-]')
        plt.xlabel('Position [μm]')
    if debug and 0:
        mu.figure()
        plt.subplot(311)
        plt.plot(Na/um,wireprof/um)
        plt.ylabel('Thickness [μm]')
        plt.subplot(312)
        plt.semilogy(Na/um,trans)
        plt.ylabel('transmission [-]')
        plt.ylim(1e-30,1)
        plt.grid()
        plt.subplot(313)

        plt.plot(Na/um,phaseshift)
        plt.ylabel('phase shift [rad]')

        plt.xlabel('position [μm]')

    if axap!=None:# and pars['shape']!='circle':
        plt.sca(axap)
        drawthis=1
        if drawthis:
            lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
            lab=pars['shape']
            if yamlval('invert',pars): lab=lab+', inv.'
            if typ.find('trapez')==0:
                lab=lab+", {:.0f}/{:.0f}".format(pars['thickness']/um,pars['edge']/um)
            trans1=transmissionmap[N2,:]
            plt.semilogy(Na/um,trans1,label=lab)
            plt.ylabel('Transmission [-]')
            plt.xlabel('position [μm]')

    if  debug or 0:
        mx=N2*params['pxsize']
#halfsize of window

        mu.figure()
        ax=plt.subplot(121)
        plt.title('Transmission')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(transmissionmap,extent=ex,cmap=rofl.cmap())
        plt.colorbar()
        plt.clim(0,1)
        prof =transmissionmap[N2,:]
        prof=mu.normalize(prof)
 #       N=params['N']
#        #N2=int(N/2)
        Na=(np.arange(N)-N2)*params['pxsize']
        plt.plot(Na*1e6,prof*mx/um,'w')

        ax=plt.subplot(122)
        if 1:
            plt.title('Thickness')
            ax.set_facecolor("black")
            ex=[-mx/um,mx/um,-mx/um,mx/um]
            plt.imshow(thicknessmap,extent=ex)
            plt.colorbar()
            prof =thicknessmap[N2,:]
            prof=mu.normalize(prof)
            Na=(np.arange(N)-N2)*params['pxsize']
            plt.plot(Na*1e6,prof*mx/um,'w')
        else:
            plt.title('phase shift')
            ax.set_facecolor("black")
            ex=[-mx/um,mx/um,-mx/um,mx/um]
            plt.imshow(phaseshiftmap,extent=ex)
            plt.colorbar()
    if return_thickness:
        return transmissionmap,phaseshiftmap,thicknessmap
    else:
        return transmissionmap,phaseshiftmap


def prepare_image(img, ps=750, max_pixels=300, ZoomFactor=1, log=1,
                  norms=[0,0], el_dict=None, normalize=True):
    from scipy.interpolate import RegularGridInterpolator
    import cv2

    inte = np.max(img) * ps**2
    suma = np.sum(img) * ps**2

    if normalize:
        if np.sum(norms) == 0:
            norms[0] = inte
            norms[1] = suma
        inte = inte / norms[0]
        suma = suma / norms[1]

    # First: cut the central region to be shown, as given by zoom factor
    if ZoomFactor > 1:
        pxc = np.shape(img)[0]
        newpxcH = int(pxc / ZoomFactor / 2)
        c = int(pxc / 2)
        imgC = img[c - newpxcH : c + newpxcH, c - newpxcH : c + newpxcH]
    else:
        imgC = img

    # Second: make sure the result is not bigger than max_pixels
    pxc = np.shape(imgC)[0]
    if pxc > max_pixels:
        dsize = [max_pixels, max_pixels]
        imgC = cv2.resize(imgC, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return imgC, norms, [inte, suma]


def imshow(imgC,ps=750,ZoomFactor=1,log=1,measures=[0,0],el_dict=None):
    if log:
        norm=colors.LogNorm()
    else:
        norm=colors.Normalize()
    if ZoomFactor>1:
        ps=ps/ZoomFactor

    ps2=ps/2/um
    extent=(-ps2,ps2,-ps2,ps2)
    #imgC=imgC/np.max(imgC)
    plt.imshow(imgC,norm=norm,cmap=rofl.cmap(),extent=extent)
    #plt.clim(1e-3,1)
    #plt.clim(1e-5,1)
    ax=plt.gca()
    ########### PLOT OF THE TEXT AT THE TOP LEFT : Total size of the image in um ####################
    if ps/um >=10:
        plt.text(.01, .99, "{:.0f} μm".format(ps/um), ha='left', va='top', transform=ax.transAxes,color='w')
    else:
        plt.text(.01, .99, "{:.1f} μm".format(ps/um), ha='left', va='top', transform=ax.transAxes,color='w')
        #################################################################################################
    if el_dict is not None:
        goodkeys=['size','f','shape','roc']
        units={}
        units['size']='μm'
        units['roc']='μm'
        units['f']='m'
        formats={}
        formats['size']='{:.0f}'
        formats['roc']='{:.0f}'
        formats['shape']='{:}'
        row=1
        ########################### PLOT OF THE INFORMATION ABOUT OBJECT : SIZE, FOCAL, SHAPE etc... ##############
        for k in el_dict.keys():
            if k not in goodkeys: continue
            unit=yamlval(k,units,'')
            form=yamlval(k,formats,'{:.1f}')
            if unit=='μm': mult=1e6
            else: mult=1
            val=el_dict[k]*mult

            plt.text(.01, .99-row*0.1, ("{:}: "+form+" {:}").format(k,val,unit), ha='left', va='top', transform=ax.transAxes,color='w')
            row+=1
            
    ################ PLOT OF THE SUM OF PIXELS IN THE INTENSITY 2D MAP (TOP RIGHT) ####################
    plt.text(.99, .99, "M {:.1e}".format(measures[0]), ha='right', va='top', transform=ax.transAxes,color='w')
    ################ PLOT OF THE MAXIMUM OF THE INTENSITY 2D MAP (TOP RIGHT) ####################
    plt.text(.99, .89, "S {:.1e}".format(measures[1]), ha='right', va='top', transform=ax.transAxes,color='w')
    return imgC

def sort_elements(ele,debug=0):
    print('sorting')
    poss=np.zeros(len(ele))
    for ei,el in enumerate(ele):
        poss[ei]=el[0]
    ass=np.argsort(poss)
    El2=[]

    for ei,el in enumerate(ass):
        El2.append(ele[el])
    if debug:
        print('after sorting:')
        for el in El2 :
            print(el[1])
    return El2


def croping_to_odd(image):
    """
    Crops the image in the case where N is even such that the corresponding image has an odd number of points. This is better for air scattering convolutions
    """
    rows, cols = image.shape
    if rows % 2 == 0:
        image = image[:-1, :]
    if cols % 2 == 0:
        image = image[:, :-1]
    return image


def restore_even_shape_by_duplication(image, target_shape):
    """
    Add one row and/or column to the image by duplicating the last row/column,
    to restore original shape after cropping.
    """
    current_rows, current_cols = image.shape
    target_rows, target_cols = target_shape
    assert target_rows >= current_rows and target_cols >= current_cols

    # Start with the cropped image
    restored = image.copy()

    # If we need to add a row
    if target_rows > current_rows:
        last_row = restored[-1:, :]
        restored = np.vstack([restored, last_row])

    # If we need to add a column
    if target_cols > current_cols:
        last_col = restored[:, -1:]
        restored = np.hstack([restored, last_col])

    return restored
    

def build_symmetric_kernel_from_particles(x_particles, y_particles, e_particles, Initial_energy_Geant4, N, propsize, nbins=401, smooth_sigma=4.0, plot_debug=False):
    
    """
    Build a smooth, rotationally symmetric scattering kernel from Geant4 particle hits.

    Parameters:
    - x_particles, y_particles: arrays of particle hit positions [µm]
    - N: output resolution (LightPipes grid)
    - propsize: LightPipes physical window size [m]
    - nbins: number of radial bins
    - smooth_sigma: Gaussian filter sigma (in bins)
    - log_bins: if True, use logarithmic binning to better resolve the center
    - plot_debug: plot radial profile and final kernel if True

    Returns:
    - kernel_2D: (N x N) normalized scattering kernel
    """
    
    # Compute the radius of each particle
    r_particles = np.sqrt(x_particles**2 + y_particles**2)   # in [um]
    
    # Convert propsize [m] to micrometers
    half_size_um = propsize * 1e6 / 2
    r_max = 2**0.5 * half_size_um

    # Filter particles within the simulation window
    valid = r_particles <= r_max
    r_particles = r_particles[valid]
    e_particles = e_particles[valid]
    
    # Step 2: define bins
    r_bins = np.linspace(0, r_max, nbins + 1) # in [um]

    Energy_weights_radial = e_particles / Initial_energy_Geant4  # only keep weights for selected particles
    radial_hist, _ = np.histogram(r_particles, bins = r_bins, weights = Energy_weights_radial)

    bin_areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2) #in [um^2]
    radial_density = radial_hist / bin_areas  # [particles / µm²]
    
    # Smoothing part
    if smooth_sigma is None or smooth_sigma == 0:
        radial_density_smooth = radial_density.copy()
    else:
        n_unsmoothed = 1 # Number of points to exclude from smoothing
        tail_smoothed = gaussian_filter1d(radial_density[n_unsmoothed:], sigma=smooth_sigma) # Create a smoothed version of the tail of the profile
        radial_density_smooth = np.concatenate([ radial_density[:n_unsmoothed], tail_smoothed]) # Concatenate unsmoothed + smoothed parts


    pixel_size_um = 2 * half_size_um / N
    linspace_um = pixel_size_um * (np.arange(N) - N // 2)

    X, Y = np.meshgrid(linspace_um, linspace_um)
    R_grid = np.sqrt(X**2 + Y**2)
    kernel_2D = np.interp(R_grid, r_bins[:-1], radial_density_smooth, left=0, right=0)

    return kernel_2D, radial_hist, r_bins, radial_density , radial_density_smooth

def apply_air_scattering_and_debug_plot(F, params, propsize, N, plot_debug = False, use_symmetric_kernel = False, compute_transmission = False, test_identity_kernel = False , crop_to_odd = True):

    """
    Applies Geant4 air scattering to the LightPipes field via convolution.
    
    Parameters:
    - F: LightPipes field object
    - params: dict containing at least 'projectdir'
    - propsize: field size in meters (LightPipes simulation window)
    - N: number of pixels in LightPipes grid
    - plot_debug: if True, show and save a comparison figure
    - use_symmetric_kernel: if True, build rotationally symmetric smoothed kernel
    
    Returns:
    - I_after_air: convolved intensity (2D numpy array)
    """
    import time
    start_time = time.time() #starts the timer
    
    basepath = Path(params["projectdir"]).parent
    data = np.load(basepath / "Air_scattering/xray_Primaries2e9_50umKapton_Air_stats.npz")
    x_particles = data["x"] # in um
    y_particles = data["y"] # in um
    e_particles = data["e"] # in keV
    Initial_energy_Geant4 = 8.8 # in keV
    
    half_size_um = propsize * 1e6 / 2

    # Step 1: Force identity kernel first if requested
    if test_identity_kernel:
        kernel_2D = np.zeros((N, N))
        kernel_2D[N // 2, N // 2] = 1.0
        use_symmetric_kernel = 0  # Prevent rebuilding later
        compute_transmission = 0  # Prevent scaling

        
    if use_symmetric_kernel:
        kernel_2D, radial_hist, r_bins, radial_density , radial_density_smooth = build_symmetric_kernel_from_particles(
            x_particles, y_particles, e_particles, Initial_energy_Geant4,  N = N, propsize = propsize,
            nbins=1001, smooth_sigma=4.0, plot_debug=False)
    else:
        kernel_2D, _, _ = np.histogram2d(x_particles, y_particles, bins=N, range=[[-half_size_um, half_size_um], [-half_size_um, half_size_um]], 
                                         weights = e_particles / Initial_energy_Geant4 )
        radial_hist = radial_density = radial_density_smooth = None

    ################ SANITY CHECK WITH AN IDENTITY KERNEL ##############

    I_lp = F

    
    # Step 2: Crop both kernel and image together, if needed
    if crop_to_odd:
        kernel_2D = croping_to_odd(kernel_2D)
        I_lp = croping_to_odd(I_lp)

    


    if not test_identity_kernel:
        kernel_2D /= np.sum(kernel_2D) #normalizing such that the sum is 1
    
        nb_particles_after_scattering = len(x_particles) # total number of particles ending on the screen after scattering
        nb_primaries = 2e9 #number of primary particles (nb of particles intialised in the simulation)

        transmission_factor = nb_particles_after_scattering / nb_primaries #percentage of particles making it through
    
        if compute_transmission:
            kernel_2D *= transmission_factor  # Take into account that some particles are absorbed by air
    
    #I_lp = Intensity(0, F)
    

    
    

    I_after_air = fftconvolve(I_lp, kernel_2D, mode='same') #convolve the image with the Kernel

    I_after_air = restore_even_shape_by_duplication(I_after_air, (N, N)) # add back a line and a row to make it (NxN) again.
    #I_lp = restore_even_shape_by_duplication(I_lp, (N, N)) # add back a line and a row to make it (NxN) again.

    # Set the values outside the disk to 0 after the convolution
    Ymask, Xmask = np.indices(I_after_air.shape)
    rmask = np.sqrt((Xmask - N//2)**2 + (Ymask - N//2)**2)
    mask = rmask <= (N//2)  # or a more precise radius
    I_after_air[~mask] = 0


    if plot_debug:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].scatter(x_particles, y_particles, s=1, alpha=0.3)
        axes[0, 0].set_xlim(-half_size_um, half_size_um)
        axes[0, 0].set_ylim(-half_size_um, half_size_um)
        axes[0, 0].set_title("Geant4 raw scatter (scatter plot)")
        axes[0, 0].set_xlabel("x [µm]")
        axes[0, 0].set_ylabel("y [µm]")
        axes[0, 0].grid(True)
        axes[0, 0].set_aspect("equal")

        if use_symmetric_kernel and r_bins is not None:
            axes[0, 1].plot(r_bins[:-1], radial_density)
            axes[0, 1].set_title("Radial histogram (nb particles at a given r / area)")
            axes[0, 1].set_xlabel("r [µm]")
            axes[0, 1].set_ylabel("Photons/area")
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)

            axes[0, 2].plot(r_bins[:-1], radial_density_smooth)
            axes[0, 2].set_title("Radial density smoothed (nb particles / area at a given r)")
            axes[0, 2].set_xlabel("r [µm]")
            axes[0, 2].set_ylabel("Photons/area")
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)

        extent_um = [-half_size_um, half_size_um, -half_size_um, half_size_um]  # [µm]
        
        im3 = axes[1, 0].imshow(kernel_2D, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 0].set_title("2D Geant4 kernel")
        fig.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(I_lp, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 1].set_title("LightPipes Intensity")
        fig.colorbar(im4, ax=axes[1, 1])
        
        im5 = axes[1, 2].imshow(I_after_air, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 2].set_title("After convolution")
        fig.colorbar(im5, ax=axes[1, 2])
        
        for ax in axes[1, :]:
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")

        plt.tight_layout()
        plt.savefig(basepath / "AirScattering_DebugPanel_6plots.png", dpi=300)
        plt.show()

    elapsed_time = time.time() - start_time  # End timer
    print(f"Air scattering + convolution took {elapsed_time/60:.2f} minutes.")
    
    return I_after_air



def airy_disk_map(L, N,  P_peak, lam=800e-9, f=0.10, D=0.10, return_grid=False, debug=False):
    """
    Build a 2-D Airy-disk intensity map on a square grid of side-length L.

    The pixel count (N x N) and the physical window size (L x L) are
    supplied directly, so the pattern is binned identically to any other
    field defined on the same grid.

    Parameters
    ----------
    L : float
        Physical width of the square window [m].
    N : int
        Number of samples along each axis (array is NxN).
    lam : float, default 800e-9
        Wavelength [m].
    f : float,  default 0.10
        Focal length [m].
    D : float,  default 0.10
        Aperture diameter [m].
    return_grid : bool, default False
        If True, also return the centred 1-D coordinate vector `x` [m].
    P_peak : float
        Power of the IR Relax laser. Nominal is 200 TW. Maximum is 300 TW.

    Returns
    -------
    airy : ndarray  (N x N)
        Airy disk intensity map (normalised so peak = 1).
    x : ndarray, optional
        Coordinate vector (metres), length N, centred on 0.
    """
    dx = L / N
    x  = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    r   = np.hypot(X, Y)

    kr  = (np.pi * D * r) / (lam * f)
    A   = np.ones_like(r)
    mask = r != 0
    A[mask] = 2.0 * j1(kr[mask]) / kr[mask]
    airy = A**2  # PSF

    # Normalize so that integral ≈ 1 (finite window approximation)
    norm = np.sum(airy) * dx * dx
    if norm <= 0:
        raise RuntimeError("Airy normalization failed.")
    airy /= norm

    # Convert to intensity [W/cm^2]; treat airy as spatial distribution of power
    I_W_cm2 = (P_peak * airy) / 1e4

    I_max = float(I_W_cm2.max())
    a0_pk = a0_from_I_lambda(I_max, lam)

    if debug:
        plt.figure()
        plt.imshow(I_W_cm2, extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6], origin='lower')
        plt.xlabel("x [μm]"); plt.ylabel("y [μm]")
        plt.title("Airy: I [W/cm²]"); plt.colorbar(label="Intensity [W/cm²]")
        plt.show()
        print(f"[Airy] ∫ airy dx dy ≈ {np.sum(airy)*dx*dx:.6f}")
        print(f"[Airy] Peak intensity: {I_max:.3e} W/cm² ; a0_peak(λ={lam*1e9:.0f} nm) ≈ {a0_pk:.3f}")

    return (I_W_cm2, x) if return_grid else I_W_cm2





def gaussian_spot_map(
L, N, fwhm_diameter,                # FWHM of the spot *diameter* [m]
    P_peak, return_grid=False, debug=False
):
    """
    2-D circular Gaussian intensity map on an LxL window, NxN samples.
    fwhm_diameter: full width at half maximum (diameter) of the spot [m].
    Normalized to integrate to 1, then scaled by peak power. Returns W/cm^2.

    Uses I(r) ∝ exp(-2 r^2 / w0^2), with w0 the 1/e^2 radius.
    Relation: FWHM_diameter = sqrt(2 ln 2) * w0  =>  w0 = FWHM / sqrt(2 ln 2).
    Normalized Gaussian in 2D: G(r) = (2 / (π w0^2)) * exp(-2 r^2 / w0^2).
    """

    projectdir = Path("/home/yu79deg/darkfield_p5438/Aime")

    dx = L / N
    x  = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    r    = np.hypot(X, Y)

    if fwhm_diameter <= 0:
        raise ValueError("fwhm_diameter must be > 0.")
    w0 = fwhm_diameter / np.sqrt(2*np.log(2))  # 1/e^2 radius

    # Properly normalized 2D Gaussian that integrates to 1
    G = (2.0 / (np.pi * w0**2)) * np.exp(-2.0 * (r**2) / (w0**2))

    I_W_cm2 = (P_peak * G) / 1e4

    if debug:
        lambda_IR = 800e-9 # Hardcoded here : it is just for the debug plot.

        # ---------- 2D zoomed image (± 3 * λ) ----------
        #r_zoom = 3.0 * float(lambda_IR)  # metres
        x_um = x * 1e6
        extent_um = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(10, 4.2))
        im = ax2d.imshow(
            I_W_cm2,
            extent=extent_um, origin='lower'
        )
        ax2d.set_xlabel("x [µm]"); ax2d.set_ylabel("y [µm]")
        ax2d.set_title("Gaussian: I [W/cm²]")
        cb = plt.colorbar(im, ax=ax2d); cb.set_label("W/cm²")

        # Apply zoom limits, but clip to the window if the box is larger than L
        lim_um = 4.0  # µm
        ax2d.set_xlim(-lim_um, lim_um)
        ax2d.set_ylim(-lim_um, lim_um)

        # ---------- 1D lineout through the center & FWHM from data ----------
        j = N//2                                  # central column (≈ y=0)
        x_line = x.copy()
        I_line = I_W_cm2[:, j]
        Imax   = float(I_line.max())
        half   = 0.5 * Imax

        # Find FWHM by linear interpolation around half-maximum on both sides
        i0 = int(np.argmax(I_line))
        # left crossing
        il = np.where(I_line[:i0] < half)[0]
        if il.size:
            k = il[-1]
            xL = x_line[k] + (half - I_line[k]) * (x_line[k+1]-x_line[k]) / (I_line[k+1]-I_line[k])
        else:
            xL = x_line[0]
        # right crossing
        ir = np.where(I_line[i0:] < half)[0]
        if ir.size:
            k2 = i0 + ir[0] - 1
            xR = x_line[k2] + (half - I_line[k2]) * (x_line[k2+1]-x_line[k2]) / (I_line[k2+1]-I_line[k2])
        else:
            xR = x_line[-1]

        fwhm_meas = (xR - xL)         # metres; this is the *diameter*
        fwhm_theo = float(fwhm_diameter)

        ax1d.plot(x_line*1e6, I_line, lw=1.6)
        ax1d.axhline(half, color='gray', ls='--', lw=1)
        ax1d.axvline(xL*1e6, color='gray', ls=':', lw=1)
        ax1d.axvline(xR*1e6, color='gray', ls=':', lw=1)
        ax1d.set_xlabel("x [µm]"); ax1d.set_ylabel("I [W/cm²]")
        ax1d.set_title("Central lineout & FWHM check")
        ax1d.text(
            0.05, 0.95,
            f"Peak = {Imax:.3e} W/cm²\n"
            f"FWHM (meas) = {fwhm_meas*1e6:.3f} µm\n"
            f"FWHM (theo) = {fwhm_theo*1e6:.3f} µm",
            transform=ax1d.transAxes, ha='left', va='top'
        )
        ax1d.grid(True)
        ax1d.set_xlim(-lim_um, lim_um)

        fig.tight_layout()
        plt.savefig(projectdir / 'gaussian_debug.png', dpi=300)
        plt.show()

        # Console checks
        print(f"[Gauss] ∫ G dx dy (analytic) = 1.000000")
        print(f"[Gauss] ∫ G dx dy (numeric on window) ≈ {np.sum(G)*dx*dx:.6f}")
        print(f"[Gauss] Peak intensity (meas): {float(I_W_cm2.max()):.3e} W/cm²")
        print(f"[Gauss] FWHM (meas) = {fwhm_meas*1e6:.3f} µm ; FWHM (theo) = {fwhm_theo*1e6:.3f} µm")
        print("Saved figure: gaussian_debug.png")

    return (I_W_cm2, x) if return_grid else I_W_cm2





def peak_power(P_peak=None, E=None, tau_FWHM=None):
    """
    Return peak power [W]. If P_peak is given, use it.
    Otherwise compute it from pulse energy E [J] and FWHM duration tau_FWHM [s]
    assuming a Gaussian temporal envelope:
        E = P0 * tau_FWHM * sqrt(pi / (4 ln 2))
      => P0 = E * sqrt(4 ln 2 / pi) / tau_FWHM
    """
    if P_peak is not None:
        return float(P_peak)
    if (E is not None) and (tau_FWHM is not None):
        return float(E * np.sqrt(4*np.log(2)/np.pi) / tau_FWHM)
    raise ValueError("Provide P_peak or (E and tau_FWHM).")




def a0_from_I_lambda(I_W_cm2, lam_m):
    """
    a0 ≈ 0.855 * sqrt(I[10^18 W/cm^2]) * λ[μm]
    """
    lam_um = lam_m * 1e6
    return 0.855 * np.sqrt(I_W_cm2 * 1e-18) * lam_um




def handle_custom_CRL(F: Field,
                      el_dict: dict,
                      params: dict,
                      projectdir: Path) -> Field:
    """
    Build transmission & phase maps for a ‘Custom_CRL’ element
    and apply them to *F*.  Also creates the 2-D / 1-D diagnostic
    plots (optional aperture included).
    Returns the modified field.
    """
    # ────────────────────────────────────────────────
    # 1. Read geometry from YAML ---------------------
    # ────────────────────────────────────────────────
    ROC       = yamlval('ROC',   el_dict, None)                           # Radius of curvature of the parabolic surface          [m]
    L         = yamlval('L',     el_dict, None)                           # Total mechanical lens thickness                       [m]
    A         = yamlval('A',     el_dict, None)                           # Geometric aperture radius                             [m]
    t_wall    = yamlval('twall', el_dict, None)                           # Minimal lens thickness (apex-to-apex)                 [m]
    nb_lenses = yamlval('nb_lenses',   el_dict, None)                     # Number of lenses in the CRL stack
    focal_lenght_stack = yamlval('focal_lenght_stack',   el_dict, None)   # Focal length of the stack                             [m]
    add_aperture = yamlval('add_aperture',   el_dict, 0)                  # If we want to add a circular aperture around the lens [Boolean]


    if A is None and t_wall is not None:
        A = 2.0 * np.sqrt(ROC * (L - t_wall))              # derive aperture from wall thickness
    elif t_wall is None and A is not None:
        t_wall = L - (A**2) / (4.0 * ROC)                  # derive wall thickness from aperture
    else:
        raise ValueError("Custom_CRL: provide *either* A or twall (not both).")
    
    print(f"ROC = {ROC*1e6:.0f} um, L = {L*1e3:.1f} mm, Diameter Aperture A = {A*1e6:.0f} um, Apex thickness t_wall = {t_wall*1e6:.0f} um")


    # ────────────────────────────────────────────────
    # 2. Optical constants ---------------------------
    # ────────────────────────────────────────────────
    E_eV         = params['photon_energy']                   # photon energy [eV]
    wavelength_m = h * c / (E_eV * e)                        # λ = h·c / E

    lens_material     = yamlval('lens_material', el_dict, 'Be')   # default material is set to Be
    beta, delta       = get_index(lens_material, E_eV)            # from Henke tables (https://henke.lbl.gov/optical_constants/getdb2.html)
    print(delta,beta)

    phase_per_m       = -2.0 * np.pi * delta / wavelength_m       # radians of phase delay per meter
    absorption_factor = 4.0 * np.pi * beta / wavelength_m         # absorption exponent scale

    if focal_lenght_stack is None and nb_lenses is not None:
        focal_lenght_stack = ROC / (2 * nb_lenses * delta)       # focal length calculated from the number of lenses in the stack [m]
    elif focal_lenght_stack is not None and nb_lenses is None:
        nb_lenses = ROC / (2 * delta * focal_lenght_stack)      # number of lenses (could be a not integer number) needed to achieve a given focal length.
    else:
        raise ValueError("Custom_CRL: provide *either* focal_lenght_stack or nb_lenses (not both).")

    print(f"Number of lenses = {nb_lenses}, focal lenght of the stack = {focal_lenght_stack} m")
    
    # ────────────────────────────────────────────────
    # 3. Mesh & thickness map ------------------------
    # ────────────────────────────────────────────────
    N   = params['N']
    Na  = (np.arange(N) -  N // 2) * params['pxsize']       # axis in meters
    
    xm, ym = np.meshgrid(Na, Na)                        # xm, ym in [m]
    r = np.sqrt(xm**2 + ym**2)

    lens_half_thickness = np.full_like(r, L / 2)                # default to max thickness (outside aperture)
    core_mask = r < A / 2 
    print(core_mask)
    lens_half_thickness[core_mask] = (xm[core_mask]**2 + ym[core_mask]**2) / (2 * ROC) + t_wall / 2
    lens_thickness = 2 * lens_half_thickness

    # ────────────────────────────────────────────────
    # 4. Apply transmission and phase maps ----------
    # ────────────────────────────────────────────────

    if add_aperture == 1 : # Add an aperture around the lens
        custom_ap = {}
        custom_ap['elem'] = 'Hf'
        custom_ap['thickness'] = 0.0001
        custom_ap['shape'] = 'circle'
        custom_ap['size'] = A
        custom_ap['invert'] = 1
        Aperture_transmission,phasemap = doap(custom_ap,params)   # creating the transmission and phase map of the lens aperture
        F = MultIntensity(Aperture_transmission,F)                # multiplying intensity of the field by the lens aperture
    
    transmission_map = np.exp(-nb_lenses * absorption_factor * lens_thickness)         # intensity attenuation
    phase_map        = nb_lenses * phase_per_m * lens_thickness                        # phase delay [radians]

    F = MultIntensity(transmission_map, F)  # apply intensity mask
    F = MultPhase(phase_map, F)             # apply phase shift


    # ───────────────────────────────────────────────────────────────
    #  Combined 2-D & 1-D plots for Custom CRL  (+ aperture) ---------
    # ───────────────────────────────────────────────────────────────
    extent_um = [
        Na[0] * 1e6, Na[-1] * 1e6,
        Na[0] * 1e6, Na[-1] * 1e6
    ]

    center_idx = N // 2
    x_um = Na * 1e6                      # horizontal axis in µm

    thickness_1d     = lens_thickness[center_idx, :] * 1e6            # [µm]
    transmission_1d  = transmission_map[center_idx, :]
    phase_1d         = phase_map[center_idx, :]

    # ---------- NEW: include aperture 1-D profile if requested -----
    if add_aperture:
        aperture_1d = Aperture_transmission[center_idx, :]

    # ---------- make room for an extra column ----------------------
    ncols = 4 if add_aperture else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 7))      # scale width

    # --- 2-D thickness map -----------------------------------------
    im0 = axes[0, 0].imshow(nb_lenses * lens_thickness * 1e6,
                            cmap='inferno', extent=extent_um, origin='lower')
    axes[0, 0].set(title='2-D Thickness [µm]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # --- 2-D transmission map --------------------------------------
    im1 = axes[0, 1].imshow(transmission_map, cmap='viridis',
                            extent=extent_um, origin='lower')
    axes[0, 1].set(title='2-D Transmission', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # --- 2-D phase (wrapped) ---------------------------------------
    im2 = axes[0, 2].imshow(phase_map, cmap='twilight',
                            extent=extent_um, origin='lower')
    axes[0, 2].set(title='2-D Phase [rad]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # --- 2-D aperture map (only if requested) -----------------------
    if add_aperture:
        im3 = axes[0, 3].imshow(Aperture_transmission, cmap='gray',
                                extent=extent_um, origin='lower')
        axes[0, 3].set(title='2-D Aperture\n(transmission)',
                    xlabel='x [µm]', ylabel='y [µm]')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # ---------------------------------------------------------------
    # ---------------- 1-D profiles (lower row) ---------------------
    # ---------------------------------------------------------------
    axes[1, 0].plot(x_um, nb_lenses * thickness_1d)
    axes[1, 0].set(ylabel='Thickness [µm]', xlabel='x [µm]',
                title='1-D Thickness (centre cut)')
    axes[1, 0].grid()

    axes[1, 1].plot(x_um, transmission_1d, color='green')
    axes[1, 1].set(ylabel='Transmission', xlabel='x [µm]',
                title='1-D Transmission (centre cut)')
    axes[1, 1].grid()

    axes[1, 2].plot(x_um, phase_1d, color='purple')
    axes[1, 2].set(ylabel='Phase [rad]', xlabel='x [µm]',
                title='1-D Phase (centre cut)')
    axes[1, 2].grid()

    # --- 1-D aperture cut (optional) -------------------------------
    if add_aperture:
        axes[1, 3].plot(x_um, aperture_1d, color='black')
        axes[1, 3].set(ylabel='Aperture T', xlabel='x [µm]',
                    title='1-D Aperture (centre cut)')
        axes[1, 3].grid()

    # ---------- optional zoom of 2-D maps --------------------------
    lim_um = None          # set e.g. 500e-6 to zoom in
    if lim_um is not None:
        for ax in axes[0, :]:
            ax.set_xlim(-lim_um * 1e6, lim_um * 1e6)
            ax.set_ylim(-lim_um * 1e6, lim_um * 1e6)

    plt.suptitle(
        f"Custom CRL – {lens_material}, E = {E_eV:.1f} eV, λ = {wavelength_m*1e10:.2f} Å",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(projectdir / 'Lens_CRLCut_2D1D.png', dpi=300)
    plt.show()

    return F





def zoom_window_with_interp(F: Field, zoom: float) -> Field:
    """
    Return a *new* LightPipes Field whose window size is ``zoom`` × the
    original one, with the optical wavefront re-sampled so that the
    physical beam is unchanged.

    Parameters
    ----------
    F     : LightPipes Field
        The input field.
    zoom  : float
        Scale factor for the window.  ``zoom < 1`` zooms *in* (smaller
        window, finer sampling); ``zoom > 1`` zooms *out*.

    Returns
    -------
    Field
        A freshly allocated LightPipes Field (input is left untouched).
    """
    if zoom <= 0:
        raise ValueError("zoom must be a positive number.")

    # --- original grid ---------------------------------------------------
    N        : int   = F.N
    lam      : float = F.lam
    size_old : float = F.siz
    size_new : float = size_old * zoom          # new physical window [m]

    # --- physical coordinates of the target grid ------------------------
    x_new = np.linspace(-0.5 * size_new, 0.5 * size_new, N, endpoint=False)
    Xn, Yn = np.meshgrid(x_new, x_new, indexing="xy")

    # --- map these coordinates onto indices of the *old* grid -----------
    dx_old   = size_old / N                      # sample pitch of old grid
    coords_x = (Xn + 0.5 * size_old) / dx_old
    coords_y = (Yn + 0.5 * size_old) / dx_old
    # keep inside array bounds for cubic interpolation
    eps = 1e-3
    coords_x = np.clip(coords_x, 0, N - 1 - eps)
    coords_y = np.clip(coords_y, 0, N - 1 - eps)

    # --- cubic interpolation of real & imag parts -----------------------
    real_i = map_coordinates(np.real(F.field), [coords_y, coords_x],
                             order=3, mode="nearest")
    imag_i = map_coordinates(np.imag(F.field), [coords_y, coords_x],
                             order=3, mode="nearest")

    # --- allocate the new LightPipes field ------------------------------
    Fnew = Begin(size_new, lam, N, dtype=F._dtype)   # safer than raw Field()
    Fnew.field    = real_i + 1j * imag_i
    Fnew._IsGauss = False        # interpolated, no longer analytic Gaussian

    return Fnew



def _save_center_crop_debug(I_map, params, fname_tag, title_tag):
    """Save 2x2 debug figure (2D linear/log + central 1D linear/log) for a ±4 µm crop."""
    from pathlib import Path
    from matplotlib.colors import LogNorm

    # scale to photons/m² (requires scale_phot to exist)
    scale_ph = params.get("scale_phot", None)
    if scale_ph is None:
        print(f"[TCC] scale_phot not available → skipping {fname_tag} debug figure.")
        return

    I_ph = I_map * scale_ph  # photons / m²

    # Crop to ±4 µm
    pxsize = params["pxsize"]          # [m]/px
    half_win_m = 4e-6                  # 4 µm
    half_win_px = max(1, int(half_win_m / pxsize))

    Ny, Nx = I_ph.shape
    cy, cx = Ny // 2, Nx // 2
    y0, y1 = max(0, cy - half_win_px), min(Ny, cy + half_win_px)
    x0, x1 = max(0, cx - half_win_px), min(Nx, cx + half_win_px)
    Iw = I_ph[y0:y1, x0:x1]

    # Axes extent in µm (centered on 0)
    ext = (-(Iw.shape[1]*pxsize)*0.5*1e6, (Iw.shape[1]*pxsize)*0.5*1e6,
           -(Iw.shape[0]*pxsize)*0.5*1e6, (Iw.shape[0]*pxsize)*0.5*1e6)
    x_um = np.linspace(ext[0], ext[1], Iw.shape[1])

    # Central horizontal cut
    profile = Iw[Iw.shape[0] // 2, :]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 2D linear
    im0 = axes[0, 0].imshow(Iw, origin="lower", extent=ext,
                            interpolation="nearest", aspect="equal", cmap=rofl.cmap())
    axes[0, 0].set_title(f"{title_tag} (linear, photons/m²)")
    axes[0, 0].set_xlabel("x [µm]"); axes[0, 0].set_ylabel("y [µm]")
    c0 = plt.colorbar(im0, ax=axes[0, 0], shrink=0.9); c0.set_label("photons / m²")

    # 2D log
    Ipos = Iw[Iw > 0]
    vmin = max(Ipos.min(), Iw.max()*1e-12) if Ipos.size else 1e-30
    im1 = axes[0, 1].imshow(Iw, origin="lower", extent=ext,
                            interpolation="nearest", aspect="equal", cmap=rofl.cmap(),
                            norm=LogNorm(vmin=vmin, vmax=max(Iw.max(), vmin*1.01)))
    axes[0, 1].set_title(f"{title_tag} (log, photons/m²)")
    axes[0, 1].set_xlabel("x [µm]"); axes[0, 1].set_ylabel("y [µm]")
    c1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.9); c1.set_label("photons / m²")

    # 1D central cuts
    axes[1, 0].plot(x_um, profile)
    axes[1, 0].set_title("Central cut (linear)")
    axes[1, 0].set_xlabel("x [µm]"); axes[1, 0].set_ylabel("photons / m²")

    axes[1, 1].semilogy(x_um, profile)
    axes[1, 1].set_title("Central cut (log)")
    axes[1, 1].set_xlabel("x [µm]"); axes[1, 1].set_ylabel("photons / m²")

    outpath = Path(params["projectdir"]) / f"{params['filename']}_{fname_tag}_at_TCC.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[TCC] Saved {title_tag} 2D map + central cut to {outpath}")




def apply_element(bundle: FieldBundle,
                  el_name: str,
                  el_dict: dict,
                  params: dict,
                  reg_prop_dict: dict,
                  method: str,
                  do_edge_damping: bool,
                  edge_damping_aperture=None):
    """
    Apply *one* optical element to every field contained in *bundle*.
    In this step we still have a single channel 'main', but we already
    loop over bundle.fields so the code will work unchanged once we add
    more channels.
    """
    # ──────────────────────────────────────────────────────────────────────
    # 1. bookkeeping / folders / wavelength
    # --------------------------------------------------------------------
    projectdir  = Path(params.get("projectdir", "."))  # needed by Custom_CRL
    def_do_plot = 1                                    # If it plots the element or not. Its = 1 by default
    
    #F          = bundle.fields["main"]             # Extract the field F from the field bundle
    el_type    = el_dict["type"]
    N          = params['N']                       # grid size
    propsize   = params['propsize']                # current physical window
    wavelength = params['wavelength']

    #for ch_name, F in bundle.fields.items():
    for ch_name in list(bundle.fields.keys()):

        F = bundle.fields[ch_name]                # ① current field

        ######## REG and DEREG elements ########
        if el_type == "zoom_window":
            zoom = el_dict.get("zoom", 1.0)
            F = zoom_window_with_interp(F, zoom)
            bundle.fields[ch_name] = F
            def_do_plot = 0

        
        if el_type == 'reg':  # regularize propagation
            reg_prop_dict["regularized_propagation"] = True
            if 'reg-by-f' in el_dict:
                tmp = el_dict['reg-by-f']
            else:
                tmp = reg_prop_dict["reg_parabola_focus"]
            F = Lens(F, -tmp)
            bundle.fields[ch_name] = F

            def_do_plot = 0

        elif el_type == 'dereg':  # deregularize propagation
            if not reg_prop_dict["regularized_propagation"]:
                print("  You can't deregularize an already deregularized field!!!")
            else:
                reg_prop_dict["regularized_propagation"] = False
                tmp = reg_prop_dict["reg_parabola_focus"]
                F = Lens(F, reg_prop_dict["reg_parabola_focus"])
                bundle.fields[ch_name] = F
                
            def_do_plot = 0
        

        if "reg-by-f" in el_dict:
            f = el_dict['reg-by-f']
            if reg_prop_dict["reg_parabola_focus"] is None:
                reg_prop_dict["reg_parabola_focus"] = f
                reg_prop_dict["regularized_propagation"] = True
                #print(f"Regularizing by CRL in {F_pos} by value {f}")
            else:
                #for inserting second, that images the focus made by first CRL
                if reg_prop_dict["regularized_propagation"] == True:
                    f2_tmp = f
                    #thin lens formula (zobrazovaci rovnice), where focus is the object
                    reg_new_tmp = 1.0/(1.0/f2_tmp + 1.0/reg_prop_dict["reg_parabola_focus"])
                    reg_prop_dict["reg_parabola_focus"] = reg_new_tmp
                    print("Re-regularizing by CRL")
                else:
                    #I dont know if we ever need this, so this is just a guess of
                    #how it might look
                    reg_prop_dict["reg_parabola_focus"] = f
                    reg_prop_dict["regularized_propagation"] = True
                    print("Unexpected regularizing by CRL")
                    #print(f"..but still regularizing by CRL in {F_pos} by value {f}")


            ############# LENS ELEMENT ###############
        if 'lens' in el_type:
            ideal = yamlval('ideal', el_dict, 1)
            if ideal:
                f = el_dict['f']
                if "reg" in el_type:
                    if reg_prop_dict["reg_parabola_focus"] is None:
                        reg_prop_dict["reg_parabola_focus"] = f
                        reg_prop_dict["regularized_propagation"] = True
                        #print(f"Regularizing by CRL in {F_pos} by value {f}")
                    else:
                        #for inserting second, that images the focus made by first CRL
                        if reg_prop_dict["regularized_propagation"] == True:
                            f2_tmp = f
                            #thin lens formula (zobrazovaci rovnice), where focus is the object
                            reg_new_tmp = 1.0/(1.0/f2_tmp + 1.0/reg_prop_dict["reg_parabola_focus"])
                            reg_prop_dict["reg_parabola_focus"] = reg_new_tmp
                            print("Re-regularizing by CRL")
                        else:
                            #I dont know if we ever need this, so this is just a guess of
                            #how it might look
                            reg_prop_dict["reg_parabola_focus"] = f
                            reg_prop_dict["regularized_propagation"] = True
                            print("Unexpected regularizing by CRL")
                            #print(f"..but still regularizing by CRL in {F_pos} by value {f}")
                else:
                    F = Lens(f,0,0,F)
                    bundle.fields[ch_name] = F

            ####### APERTURE OF THE LENS ########
            aperture = yamlval('size',el_dict,0)
            if aperture==0 and 'CRL4' in el_type:
                aperture = 400e-6

            if aperture>0: #aperture
                ap_dict = {}
                ap_dict['elem'] = 'Hf' 
                ap_dict['thickness'] = 0.0001
                ap_dict['shape'] = 'circle'
                ap_dict['size'] = aperture
                ap_dict['invert'] = 1
                tmap,phasemap = doap(ap_dict,params)   # creating the transmission and phase map of the lens aperture
                F = MultIntensity(tmap,F)              # multiplying intensity of the field by the lens aperture
                bundle.fields[ch_name] = F

            ############# CRL4 default parameters ############
            if 'CRL4' in el_type: 
                Lroc = yamlval('roc',el_dict,5.0e-5) #radius of curvature
                ab_dict = {} #absorption dictionnary
                #ab_dict['elem'] = 'Be' # ------- TO BE CHANGED BACK -----
                ab_dict['elem'] = yamlval('lens_material', el_dict, params.get('lens_material', 'Be'))
                ab_dict['minr0'] = 0
                ab_dict['shape'] = 'parabolic_lens'
                ab_dict['size'] = aperture
                ab_dict['roc'] = Lroc
                ab_dict['double_sided'] = 1 #parabolic shape on both sides
                ab_dict['num_lenses'] = yamlval('num_lenses',el_dict,1)
                tmap2,phasemap = doap(ab_dict,params,debug=0)
                F = MultIntensity(tmap*tmap2,F)
                bundle.fields[ch_name] = F

                if not ideal:  
                    F = MultPhase(phasemap,F)
                    bundle.fields[ch_name] = F
                    print('doing real lens')


                if not ideal:  
                    F = MultPhase(phasemap,F)
                    bundle.fields[ch_name] = F
                    print('doing real lens')

            if yamlval('celestre',el_dict,1):
                cel_dict = {}
                cel_dict['defect'] = 'celestre'
                cel_dict['type'] = 'phaseplate'
                cel_dict['num'] = yamlval('num_lenses',el_dict,1)
                phaseshiftmap = do_phaseplate(cel_dict,params)
                F = MultPhase(phaseshiftmap,F)
                bundle.fields[ch_name] = F
                
            if yamlval('seiboth',el_dict,1):
                seib_dict = {}
                seib_dict['defect'] = 'seiboth'
                seib_dict['type'] = 'phaseplate'
                seib_dict['num'] = yamlval('num_lenses',el_dict,1)
                phaseshiftmap = do_phaseplate(seib_dict,params)
                F = MultPhase(phaseshiftmap,F)
                bundle.fields[ch_name] = F
                
            if yamlval('scatterer',el_dict,0):
                sc_dict={}
                if 0: #first attemtp
                    sc_dict['randomizeB']=2.e-6
                    sc_dict['type']='aperture'
                    sc_dict['shape']='circle'
                    sc_dict['size']=aperture
                    sc_dict['invert']=0
                    sc_dict['thickness']=2e-6
                    sc_dict['elem']='W'
                if 1: #second one
                    sc_dict['randomizeB'] = yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['type'] = 'aperture'
                    sc_dict['shape'] = 'circle'
                    sc_dict['size'] = aperture
                    default_k = 3 #3 comes from 5348
                    default_k = 0.02 #3 comes from 6436 ....10.6.2025: this seems too low!!
                    k = yamlval('lens_randomize_k',params,default_k)
                    if 'scatterer_k' in el_dict:
                            k = el_dict['scatterer_k']
                    sc_dict['density'] = k*yamlval('num_lenses',el_dict,1)
                    sc_dict['thickness'] = 3*yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['elem'] = yamlval('lens_randomize_elem',params,'Ti')
                    
                Ii1 = (np.nansum(Intensity(0,F)))
                tmap3,phasemap = doap(sc_dict,params,debug=0)
                F = MultIntensity(tmap3,F)
                bundle.fields[ch_name] = F
                F = MultPhase(phasemap,F)
                bundle.fields[ch_name] = F
                Ii2 = (np.nansum(Intensity(0,F)))
                loss_on_scatterer = Ii2/Ii1
                params['transmission_of_scatterer_'+el_name] = loss_on_scatterer

        ############# CUSTOM CRL ELEMENT ###############
        if el_type == 'Custom_CRL':
            F = handle_custom_CRL(F, el_dict, params, projectdir)
            bundle.fields[ch_name] = F



            
        ############# ELEMENT : PHASE PLATE ###########
        if el_type=='phaseplate':
            phaseshiftmap=do_phaseplate(el_dict,params)
            F=MultPhase(phaseshiftmap,F)
            bundle.fields[ch_name] = F

        ############# ELEMENT : PURE APERTURE ###########
        if 'aperture' in el_type:
            if len(el_dict)==0:
                do_nothing = 1
            num = yamlval('num',el_dict,1)
            merged = 1
            if merged:
                bt = np.zeros((N,N))+1.
                ph = np.zeros((N,N))
                for i in np.arange(num):
                    tmap,phasemap = doap(el_dict,params)
                    bt = bt*tmap
                    ph+=phasemap
                if yamlval('do_intensity',el_dict,1):
                    F = MultIntensity(bt,F)
                    bundle.fields[ch_name] = F
                if yamlval('do_phaseshift',el_dict,1):
                    F = MultPhase(ph,F)
                    bundle.fields[ch_name] = F
            else:
                for i in np.arange(num):
                    tmap,phasemap=doap(el_dict,params)
                    if yamlval('do_intensity',el_dict,1):
                        F = MultIntensity(tmap,F)
                        bundle.fields[ch_name] = F
                    if yamlval('do_phaseshift',el_dict,1):
                        F = MultPhase(phasemap,F)
                        bundle.fields[ch_name] = F

        ############## Air Scattering ##########
        
        if el_name == 'Det' and el_dict.get('AirScat', 0):
            
            compute_transmission = el_dict.get('compute_transmission', 0)
            use_symmetric_kernel = el_dict.get('use_symmetric_kernel', True)
            test_identity_kernel = el_dict.get('test_identity_kernel', 0)
            crop_to_odd = el_dict.get('crop_to_odd', 1)  # crops the image if N is even such that the image has an odd number of pixels (better for convolution)
            
            I_after_air = apply_air_scattering_and_debug_plot(F,
                                                                params, 
                                                                propsize, 
                                                                N, 
                                                                plot_debug = True, 
                                                                use_symmetric_kernel = use_symmetric_kernel, compute_transmission = compute_transmission,
                                                                test_identity_kernel = test_identity_kernel , crop_to_odd = crop_to_odd)
            

        # ---------------- edge damping block ----------------------------
        if do_edge_damping:
            F = MultIntensity(edge_damping_aperture, F)
            bundle.fields[ch_name] = F
        # ----------------------------------------------------------------

    ################## CREATE VB FIELDS AT TCC #################
    if el_name == 'TCC' and el_dict.get('VB_signal', 0) and "VB_parr" not in bundle.fields:

        I_cr = 4.7e29 # Critical intensity in [W/cm^2]
        alpha_cst = e**2 / (4 * np.pi * epsilon_0 * hbar * c)

        # Calculate the intensity of IR laser :
        f_IR = 0.1 # focal lenght of the final parabolla in m
        D_IR = 0.1 # diameter of the final parabolla in m
        wavelength_IR = 800e-9 # Wavelenght of the IR laser, in m
        phi_VB = np.pi / 4 #45 degrees angle between the IR and X beams.

        # Define E_pulse (OR) P_peak
        tau_FWHM = 30e-15 # Laser FWHM duration in second
        E_pulse  = 4.8  # Joules
        #P_peak_direct = 200e12 # Peak power of the laser, in Watt

        P_peak = peak_power(E=E_pulse, tau_FWHM=tau_FWHM) # In [Watt]
        #P_peak = peak_power(P_peak=P_peak_direct)

        #I_W_cm2, x = airy_disk_map(F.grid_size, F.N, wavelenght_IR, f_IR, D_IR, P_peak, return_grid=True) #OLD=. to remove

        # 1) ------- Option 1 = Airy Disk -------
        #I_W_cm2, x = airy_disk_map(F.grid_size, F.N, P_peak, lam=wavelength_IR, f=f_IR, D=D_IR,return_grid=True)

        # 2) ------- Option 2 = Gaussian --------
        FWHM_diam = 1.3e-6    # target FWHM diameter [m]
        I_W_cm2, x = gaussian_spot_map(F.grid_size, F.N, fwhm_diameter=FWHM_diam, P_peak=P_peak, return_grid=True, debug=True)
        # ---------------------------------------

        tau = tau_FWHM / np.sqrt(2*np.log(2))
        prefactor = (c * tau / wavelength) * (alpha_cst / 90) * np.sqrt(np.pi / 2)
        VB_mask_parr = (I_W_cm2 / I_cr) * prefactor * (11 - 3 * np.cos(2 * phi_VB))  #mask of the intensity of IR laser at TCC (unitless)
        VB_mask_perp = (I_W_cm2 / I_cr) * prefactor * ( 3 * np.sin(2 * phi_VB)) #mask of the intensity of IR laser at TCC (unitless)

        print(f"tau ={tau} s")
        print(f"prefactor VB = {prefactor}")
        print(f"I_W_cm2 / I_cr = {I_W_cm2 / I_cr}")
        print(f"11 - 3 * np.cos(2 * phi_VB) = {11 - 3 * np.cos(2 * phi_VB)}")
        print(f"3 * np.sin(2 * phi_VB) = {3 * np.sin(2 * phi_VB)}")



        bundle = maybe_spawn_VB_channels(bundle,
                                        params,
                                        VB_mask_parr,
                                        VB_mask_perp)

    return bundle, bundle.fields["main"], def_do_plot
# ────────────────────────────────────────────────────────────────────

"""
def _center_crop(img2d, half_win_um, pxsize_m):
    if not (img2d.ndim == 2 and half_win_um and half_win_um > 0):
        return img2d, None  # no crop

    half_px = int((half_win_um * 1e-6) / pxsize_m)
    if half_px < 1:
        return img2d, None

    ny, nx = img2d.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_px), min(ny, cy + half_px)
    x0, x1 = max(0, cx - half_px), min(nx, cx + half_px)

    cropped = img2d[y0:y1, x0:x1]
    # Physical window size in meters (use the actual cropped pixel count)
    win_size_m = float(min(y1 - y0, x1 - x0)) * pxsize_m
    return cropped, win_size_m
"""

# ────────────────────────────────────────────────────────────────────
def maybe_spawn_VB_channels(bundle: FieldBundle,
                            params: dict,
                            VB_mask_parr,
                            VB_mask_perp):
    """
    At the TCC element we branch the existing field into two vacuum-
    birefringence (VB) channels by multiplying the intensity masks.
    Called exactly once; afterwards `propagate_bundle()` carries all
    three channels through the beamline.
    """
    if "VB_parr" in bundle.fields:        # already spawned
        return bundle

    F_main = bundle.fields["main"]
    #bundle.fields["VB_parr"] = MultIntensity(VB_mask_parr, F_main)
    #bundle.fields["VB_perp"] = MultIntensity(VB_mask_perp, F_main)

    bundle.fields["VB_parr"] = MultIntensity(VB_mask_parr**2, F_main)
    bundle.fields["VB_perp"] = MultIntensity(VB_mask_perp**2, F_main)

    return bundle
# ────────────────────────────────────────────────────────────────────




def doit(params,elements):
    """
    Simulate the beam line described by *elements* and *params*.

    Parameters
    ----------
    params   : dict   – global run settings (see YAML template)
    elements : list of optical elements   – [(z_pos, element name, yaml_dict), …] in *any* order

    Returns
    -------
    params   : dict   – updated with results (transmission, intensities, …)
    trans    : 1-D np.ndarray – I(z)/I(start) for each element plane
    figs     : dict   – pickled images requested via `figs_to_save`

    Others
    -------
    norms = [0,0] :  in the first passage of the loop and then norm=[integral / normalised , maximum] which gets updated at each passage in the loop.
    im, or imC    :  image prepared by the "prepare_image" function. 
    F             :  Electric field
    I             :  2D map of intensity
    measures      :  an array with the maximum and sum of the intensity map
    """

    # ──────────────────────────────────────────────────────────────────────
    # 0. Initialisation
    # --------------------------------------------------------------------
    mu.clear_times(); mu.tick()      # timing clear
    import pprint
    from pathlib import Path
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as mticker

    # ──────────────────────────────────────────────────────────────────────
    # 1. bookkeeping / folders / wavelength
    # --------------------------------------------------------------------

    pprint.pprint(params)

    projectdir = Path(params.get("projectdir", "."))
    basepath = Path(params["projectdir"]).parent
    
    N = params['N']
    propsize = params['propsize']
    params['initial_propsize'] = params['propsize']            # store the initial propsize of the simulation for the flow_plot
    params['pxsize'] = params['propsize'] / params['N']
    Na = ( np.arange(N) - 0.5 * N ) * params['pxsize'] / um
    max_pixels = yamlval('subfigure_size_px',params,300)
    if max_pixels>N: max_pixels=N
    if max_pixels == 0: max_pixels = N                        # If we put the parameter "subfigure_size_px" = 0 in the yaml, it means that we want to take the original number of cells N 

    dtype = np.complex64

    E_eV      = params['photon_energy']          # photon energy [eV]
    wavelength = h * c / (E_eV * e)             # λ = h·c / E     [m]
    params['wavelength'] = wavelength           # make it globally visible

    # ──────────────────────────────────────────────────────────────────────
    # 2. auto-flow (inject extra imager planes) & element sorting
    # --------------------------------------------------------------------
    

    auto_flow = yamlval('flow_auto_save',params)
    if 'flow' in params:
        flowdef=params['flow']
        flowposs= None
        if np.ndim(flowdef)==2:
            flowposs=np.array([])
            for r in np.arange(np.shape(flowdef)[0]):
                ff=flowdef[r]
                flowp1=np.arange(ff[0],ff[1],ff[2])
                flowposs=np.concatenate((flowposs,flowp1))

        if np.size(flowdef)==3:
            flowposs=np.arange(flowdef[0],flowdef[1],flowdef[2])
        if flowposs is not None:
            for fi,flowpos in enumerate(flowposs):
                fe={}
                fe['type']='imager'
                fe['position']=flowpos
                fe['plot']=yamlval('flow_images',params,0)
                fe['zoom']=yamlval('flow_zoom',params,1)
                flowname='flow_{:.2f}'.format(flowpos)
                flowname='flow_{:03.0f}_{:.2f}'.format(fi,flowpos)
                EE=[flowpos,flowname,fe]
                elements.append(EE)
        if auto_flow:
            ffdir = Path(params["projectdir"]) / "flow_figs" / f"{params['filename']}_auto"
            mu.mkdir(ffdir,0)

    elements = sort_elements(elements)          # sort the elements with their longitudinal position


    # ──────────────────────────────────────────────────────────────────────
    # 3. initial field  & regularisation
    # --------------------------------------------------------------------

    F = Begin(params['propsize'], wavelength, N, dtype=dtype)
    F = GaussBeam(
            F,
            params['beamsize'],
            x_shift=params['gauss_x_shift'],
            tx=params['gauss_x_tilt'])


    reg_prop_dict = {
        "regularized_propagation": params.get("regularized_propagation", False),
        "reg_parabola_focus":      params.get("reg_parabola_focus_mm", None),}

    bundle = FieldBundle(
                fields={"main": F},
                z_pos=elements[0][0],
                reg=reg_prop_dict
            )

    # ──────────────────────────────────────────────────────────────────────
    # 4. global stores
    # ----------------------------------------------------------------------

    figs_to_save = yamlval('figs_to_save',params,[])
    figs_to_export = yamlval('figs_to_export',params,[])

    figs = {}       # Definition of the pickle "figs"
    export = {}
    intensities = {}
    integ = 0.0
    profi = 0
    norms = [0,0]
    numel = len(elements)

    # --- traces for post-run normalization/plotting ---
    trace_z, trace_I, trace_names = [], [], []
    
    
    if 'edge_damping' in params:
        do_edge_damping = 1
        edge_damping_aperture = do_edge_damping_aperture(params)
    else:
        do_edge_damping = 0
    
    
    # ──────────────────────────────────────────────────────────────────────
    # 5. plotting infrastructure
    # ---------------------------------------------------------------------
    mosaic_cbar_dynamic = True    # Whether the colorbar axis change from plot to plot in the mosaic figure.

    max_panels     = params['fig_rows'] * params['fig_cols']
    fig_by_ch      = {}
    pi_by_ch       = {}          # current subplot number inside that figure
    step_x, step_y = [], []
    fig_summary    = plt.figure(figsize=(params['fig_cols'] * 3, 3))   # wide & short
    ax_prof        = fig_summary.add_subplot(1, 2, 1)    # left panel
    ax_steps       = fig_summary.add_subplot(1, 2, 2)    # right panel
    profiles_done  = False                               # helper flag
    unit_sel = params['intensity_units']                 # Intensity unit is "photons" or "relative"
    Zoom_global = params['Zoom_global']                  # Zoom global of all the outputs of the simulation
    print(f"unit read from yaml : {unit_sel}")
    print(f"Zoom global ={Zoom_global}")
    
    save_parts = yamlval('save_parts', params, 0)
    if save_parts:
        mu.mkdir('part')
        mu.savefig('part/'+params['filename']+'__start')

    # ──────────────────────────────────────────────────────────────────────
    # 5 bis. Find the active beam shape (if any)
    # --------------------------------------------------------------------
    beam_shaper_index = None
    for i, E in enumerate(elements):
        name = E[1]
        dct  = E[2]
        if name == "beam_shaper" and yamlval('in', dct, 1):
            beam_shaper_index = i
            break
    

    # ──────────────────────────────────────────────────────────────────────
    # 6. main loop over elements
    # --------------------------------------------------------------------
    method = yamlval('method',params,'FFT')
    
    for ei,E in enumerate(elements):      # ei = element index ([0,1,2,3...])   E = element dictionnary
        z       = E[0]                    # position of the element
        el_name = E[1]                    # element name
        el_dict = E[2]                    # element aperture
        el_type = el_dict['type']
        
        print('{:} (elem.n.{:.0f})   ###'.format(el_name,ei))

        # ---- skip inactive planes ---------------------------------------
        if 'off' in el_type: continue
        if not yamlval('in', el_dict, 1): continue  # Skip inactive elements

        # ---- propagate to element position ------------------------------
        delta_z = z - bundle.z_pos

        if delta_z == 0:
            print('skipping zero propagation')
        else:
            bundle = propagate_bundle(
                        bundle,
                        dz = delta_z,
                        method='Fresnel' if method.lower() == 'fresnel' else 'FFT'
                    )
            F = bundle.fields["main"]                                                 # keep the legacy variable alive
        
            propsize           = F.siz                                                # update physical size of grid
            params['propsize'] = propsize
            params['pxsize']   = F.siz / N                                            # update pixel size
            Na                 = (np.arange(N) - 0.5 * N) * params['pxsize'] / um     # Because the grid changes size with propagation we need to recalculate Na each time.
            
            if reg_prop_dict["reg_parabola_focus"] is not None:
                reg_prop_dict["reg_parabola_focus"] -= delta_z

        # ---------- apply optical element ---------------------------------
        bundle, F, def_do_plot = apply_element(
                bundle,
                el_name,
                el_dict,
                params,
                reg_prop_dict,
                method,
                do_edge_damping,
                edge_damping_aperture if do_edge_damping else None)
        
        all_channels = bundle.fields.items()     # [('main', F), ('VB_parr', F1) …]

        # ---- Prepare the plot variables ------------------------------
        #ZoomFactor    = yamlval('zoom',el_dict,1)
        ZoomFactor = Zoom_global if Zoom_global != 1 else yamlval('zoom', el_dict, 1) # If Zoom Global exist, it is prioritized over the Zoom_factor (local for each plane)
        print(f"Zoom factor = {ZoomFactor}")
        do_plot       = yamlval('plot', el_dict, def_do_plot)
        plot_phase    = yamlval('plot_phase', params, 0)
        logg          = yamlval('figs_log', params, 1)
        
        if auto_flow and 'flow' in el_name:         # Decide once whether this element is an auto-flow plane
            fi       = int(el_name.split('_')[1])   # e.g. flow_005_…
            position = float(el_name.split('_')[2])
        else:
            fi = position = None                    # will be ignored below 

        # ──────────────────────────────────────────────────────────────────────
        # 4. Loop over channels (compute, store and plot for every active channel)
        # ----------------------------------------------------------------------

        for ch_name, Fch in all_channels: # Channels are : main, VB_parr, VB_perp

            # ---- decide whether to plot this channel ----
            if ch_name == "main":
                do_plot_ch = do_plot                    # unchanged
            else:                                       # VB channels
                do_plot_ch = do_plot and yamlval('plot_VB', params, 1)

            # ---- choose phase vs intensity ----
            if yamlval('plot_phase', params, 0):
                I_ch = Phase(Fch)
            elif 'I_after_air' in locals() and ch_name == "main":
                I_ch = I_after_air
            else:
                I_ch = Intensity(0, Fch)
            
            # ----- Choice of unit for intensity -----

            if ch_name == "main" and ((beam_shaper_index is not None and ei == beam_shaper_index) or (beam_shaper_index is None and ei == 0)):

                photons_tot = params.get('photons_total')
                tau         = params.get('pulse_duration')

                if photons_tot:
                    params['scale_phot'] = photons_tot / np.nansum(I_ch) / ((propsize / N)**2) # Factor to scale the map to photons/m^2. Unit is [photons / m^2]
                    print(f"scale photons {params['scale_phot']}")
                    print(f"∑I_ch = {np.nansum(I_ch):.3e}")
                    print(f"∑I_ch x dx^2 = {np.nansum(I_ch)*((propsize / N)**2):.3e}")


                if photons_tot and tau:
                    E_J   = params['photon_energy'] * e
                    params['scale_Wcm2'] = (photons_tot * E_J / tau) \
                                        / np.nansum(I_ch) / propsize**2 / 1e4

                    
            # --- optional rectangular ROI integrals (for SFA labels) -------------
            if 'roi' in el_dict and ch_name == "main":
                s_um               = 0.5 * el_dict['roi']                 # half-width [µm]
                mask               = np.abs(Na) <= s_um                   # Na is 1-D μm-axis
                intensities['roi'] = (np.nansum(I_ch[np.ix_(mask, mask)])* propsize**2)

            if 'roi2' in el_dict and ch_name == "main":
                s_um                = 0.5 * el_dict['roi2']
                mask                = np.abs(Na) <= s_um
                intensities['roi2'] = (np.nansum(I_ch[np.ix_(mask, mask)]) * propsize**2)
                
            # ------ bookkeeping ---------------------------------------------------
            Iint = np.nansum(I_ch) * propsize**2            # Summed intensity
            intensities[f"{el_name}_{ch_name}"] = Iint      # e.g. "Det_main"

            # ──────────────────────────────────────────────────────────────────────
            # 4. Summary figure for the main field (Intensity as a function of propagation)
            # ----------------------------------------------------------------------
            if ch_name == "main":
                # store raw intensity vs z; we will normalize after the loop
                trace_z.append(z)
                trace_I.append(Iint)
                trace_names.append(el_name)

                # left panel: transverse profile at this plane
                prof = np.sum(I_ch, axis=0)
                if yamlval('profiles_normalize', params, 1):
                    prof = mu.normalize(prof)

                profiles_xlim = yamlval('profiles_xlim', params, [0, 200])
                mask          = (Na >= profiles_xlim[0]) & (Na <= profiles_xlim[1])
                ax_prof.semilogy(Na[mask], prof[mask], color=rofl.cmap()(ei/numel))
                profiles_done = True
            # -----------------------------------------------------------------------
            # --- DEBUG FIGURE @ TCC: 2D main field in photons/m² over ±4 µm (x,y) --
            # -----------------------------------------------------------------------

            # --- DEBUG FIGURES @ TCC ---
            if el_name == "TCC":
                if ch_name == "main":
                    _save_center_crop_debug(I_ch, params, fname_tag="Xray",    title_tag="Main @ TCC")
                elif ch_name == "VB_perp":
                    _save_center_crop_debug(I_ch, params, fname_tag="VB_perp", title_tag="VB_perp @ TCC")

            # ----------------------------------------------------------------------
            # -------------------------- end DEBUG FIGURE --------------------------
            # ----------------------------------------------------------------------

            # ---- prepare image (log, zoom, …) ----

            im, norms, measures = prepare_image(
                I_ch, ps=propsize, max_pixels=max_pixels,
                ZoomFactor=ZoomFactor, log=logg,
                norms=norms,
                el_dict=el_dict,
                normalize=(unit_sel == 'relative')  # only normalize in relative mode
            )

            # --------------------------------------------------------------
            # Always keep *every* automatic flow slice for *all* channels
            if el_name.startswith("flow") and "flow" in figs_to_save:
                figs[f"{el_name}_{ch_name}"] = [im, ei, propsize/ZoomFactor, z]
            # --------------------------------------------------------------


            # ---- auto-save figs / flow images ----
            key = f"{el_name}_{ch_name}"
            if el_name.startswith(tuple(figs_to_save)):
                figs[key] = [im, ei, propsize/ZoomFactor, z]
            if fi is not None:
                flow_savefig(I_ch, ffdir, fi, propsize, f"{params['filename']}_{ch_name}", position)

            # ---- plotting (subfigures) -------------------------------------
            if do_plot_ch:

                # 1. get or create the figure for this channel
                if ch_name not in fig_by_ch:
                    fig_size           = (params['fig_cols'] * 3, params['fig_rows'] * 3)
                    fig_by_ch[ch_name] = plt.figure(figsize=fig_size)
                    pi_by_ch[ch_name]  = 1
                fig_ch = fig_by_ch[ch_name]
                pi_ch  = pi_by_ch[ch_name]

                # 2. open a new page if the grid is full
                if pi_ch > max_panels:
                    fig_by_ch[ch_name] = plt.figure(figsize=fig_size)
                    fig_ch             = fig_by_ch[ch_name]
                    pi_ch              = 1

                # Intensity unit selection
                if unit_sel == 'photons':
                    scale_ph = params.get('scale_phot', 1.0)      # 1.0 if not defined yet
                    vmin, vmax = [c * scale_ph for c in [1e-11, 50]]
                    label_unit = "photons / m²"
                    scale_plot = scale_ph
                elif unit_sel == 'Wcm2':
                    vmin, vmax = [c * params['scale_Wcm2'] for c in [1e-11, 50]]
                    label_unit = "W cm⁻²"
                    scale_plot = 1.0
                else:                             # relative
                    vmin, vmax = [1e-11, 50]
                    label_unit = "relative"
                    scale_plot = 1.0

                # 3. ----- Draw the Mosaic --------
                lab_ch = f"{el_name}"
                plt.figure(fig_ch.number)             # activate this channel's fig
                ax = plt.subplot(params['fig_rows'], params['fig_cols'], pi_ch)
                ax.set_facecolor("black")

                im_plot = im * scale_plot

                # --- choose color limits ---
                if mosaic_cbar_dynamic:
                    # per-image dynamic limits (robust to zeros/NaN)
                    pos = im_plot[im_plot > 0]
                    if pos.size:
                        vmin = float(np.nanmin(pos))
                        vmax = float(np.nanmax(pos))
                        if vmax <= vmin:  # edge case
                            vmax = vmin * 1.01
                    else:
                        vmin, vmax = 1.0, 1.0
                else:
                    # fixed range (your current constants)
                    base = [1e-11, 50]
                    if unit_sel == 'photons':
                        vmin, vmax = [c * scale_ph for c in base]
                    elif unit_sel == 'Wcm2':
                        vmin, vmax = [c * scale_plot for c in base]
                    else:
                        vmin, vmax = base

                # --- draw ---
                half_span_m = 0.5 * propsize / ZoomFactor   # [m]
                img = ax.imshow(
                    im_plot,
                    extent=[-half_span_m*1e6, +half_span_m*1e6,-half_span_m*1e6, +half_span_m*1e6],
                    origin='lower',
                    cmap=rofl.cmap(),
                    norm=LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=vmax) if logg else None
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(img, cax=cax)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_label(label_unit, fontsize=7)
                
                ax.set_title(f"{lab_ch}", fontsize=9)
                ax.set_xlabel('x  [µm]', fontsize=8)
                ax.set_ylabel('y  [µm]', fontsize=8)
                
                # --------- Add the squares and Shadow factor on the Det pannel ----------
                if ch_name == "main" and el_name == "Det":
                    # draw the rectangles corresponding to the size of 1 camera pixel
                    if 'roi2' in el_dict:                       # SFA-75 (green, larger box)
                        s_um = 0.5 * el_dict['roi2']
                        ax.add_patch(plt.Rectangle((-s_um, -s_um), 2*s_um, 2*s_um,
                                                edgecolor='lime',  facecolor='none',
                                                lw=1.3))
                    if 'roi' in el_dict:                        # SFA-13 (red, smaller box)
                        s_um = 0.5 * el_dict['roi']
                        ax.add_patch(plt.Rectangle((-s_um, -s_um), 2*s_um, 2*s_um,
                                                edgecolor='red',  facecolor='none',
                                                lw=1.3))

                    # text at bottom-left
                    central_key = "TCC_main" if "TCC_main" in intensities else (
                                "PH_main"  if "PH_main"  in intensities else "start")
                    den    = intensities[central_key]
                    tr_scat = yamlval("transmission_of_scatterer_L2", params, 1)

                    if 'roi2' in intensities:
                        t75 = intensities['roi2'] / den / tr_scat
                        ax.text(0.02, 0.12, f"SF75 {t75:.1e}",
                                transform=ax.transAxes, color='lime', fontsize=8,
                                va='bottom', ha='left')

                    if 'roi' in intensities:
                        t13 = intensities['roi'] / den / tr_scat
                        ax.text(0.02, 0.02, f"SF13 {t13:.1e}",
                                transform=ax.transAxes, color='red', fontsize=8,
                                va='bottom', ha='left')

                # ----------------------------------------------------------------

                pi_by_ch[ch_name] = pi_ch + 1    # Advance the panel counter for this channel

        # ────────────────────────────────────────────────────────────────────
        
        if save_parts:
            for ch_name, fig_ch in fig_by_ch.items():
                fname = f"part/{params['filename']}__{ch_name}__{ei:02d}.jpg"
                plt.figure(fig_ch.number)
                mu.savefig(fname)
        
        if yamlval('end_after',params,'asdfasdfasdf')==el_name: break


    # ──────────────────────────────────────────────────────────────────────
    # 6 bis. Choose normalisation of intensity for summary plot
    # ----------------------------------------------------------------------
    if beam_shaper_index is not None:
        norm_idx = beam_shaper_index            # intensity is evaluated *after* element
    else:
        norm_idx = 0

    trace_z = np.asarray(trace_z, float)
    trace_I = np.asarray(trace_I, float)

    I0int = trace_I[norm_idx]
    trans  = trace_I / I0int

    # keep for later / pickling
    params['trace_z']       = trace_z
    params['trace_I']       = trace_I
    params['trace_trans']   = trans
    params['norm_index']    = int(norm_idx)
    params['norm_ref_name'] = trace_names[norm_idx]
    params['norm_ref_z']    = float(trace_z[norm_idx])

    # make 'start' consistent with this normalization choice
    intensities['start'] = float(I0int)

    # ──────────────────────────────────────────────────────────────────────
    # 7. stash results in params
    # ----------------------------------------------------------------------
    params['transmission'] =  trans
    params['intensities']  =  intensities
    params['integ']        =  integ

    if params['ax_apertures']!=None:
        plt.sca(params['ax_apertures'])
        plt.title('Apertures')
        plt.xlim(yamlval('profiles_xlim',params,[0,200]))
        plt.ylim(yamlval('apertures_ylim',params,[1e-10,2]))

    duration           = mu.print_times() # secondes
    params['duration'] = duration         # secondes

    if np.size(figs)>0:
        pkl_name = projectdir / "pickles" / f"{params['filename']}_figs"
        mu.dumpPickle(figs, str(pkl_name))
    if len(export)>0:
        pkl_name = projectdir / "pickles" / f"{params['filename']}_figs"
        mu.dumpPickle(figs, str(pkl_name))


    #------------ Save the .res pickle -----------
    elements_dict = {el[1]: el[2] for el in elements}

    res_name = projectdir / "pickles" / f"{params['filename']}_res"
    mu.dumpPickle((elements_dict, params), str(res_name))
    print(f"Saved RES pickle to {res_name}")

    # ─── finalise the summary figure (if anything was drawn) ───
    if profiles_done:
        # left panel cosmetics
        ax_prof.set_xlabel('Position [μm]')
        ax_prof.set_ylabel('Intensity')
        ax_prof.set_title('Intensity profiles')

        # right panel – cumulative transmission normalized at norm_ref
        ax_steps.clear()
        ax_steps.step(params['trace_z'], params['trace_trans'], where='post', color='tab:orange')
        ax_steps.set_xlabel('z  [m]')
        ax_steps.set_ylabel('∑ I  /  I(ref)')
        ax_steps.set_yscale('log')
        ax_steps.set_title('Cumulative transmission')


        # ----------------------------------------------------------
        # textual annotations
        info_lines = [f'N = {N}', f'I_ref = {I0int:.1e}', f'ref: {params["norm_ref_name"]} @ z={params["norm_ref_z"]:.2f} m']

        if 'TCC_main' in intensities and 'start_main' in intensities:
            r = intensities['TCC_main']/intensities['start_main']
            info_lines.append(f'start→TCC = {r:.1e}')
        
        I_start = params['intensities']['start']

        for i, txt in enumerate(info_lines):
            ax_steps.text(0.02, 0.98 - i*0.08, txt,
                        transform=ax_steps.transAxes,
                        va='top', ha='left',
                        fontsize=10,
                        color=('black'))
            
        # ------ Figure Summary saving ------------------------------
        fig_summary.tight_layout(pad=1.2)
        plt.figure(fig_summary.number)
        out_sum = projectdir / "figures" / f"{params['filename']}_summary.jpg"
        mu.savefig(str(out_sum))

    # ---------- Mosaic figure saving ------------------------------

    for ch_name, fig_ch in fig_by_ch.items():
        fig_ch.suptitle(f"Channel: {ch_name}, intensity unit: {unit_sel}", fontsize=14, y=0.97) # Add a title to the full figure
        out_name = projectdir / "figures" / f"{params['filename']}_{ch_name}.jpg"
        plt.figure(fig_ch.number)
        mu.savefig(str(out_name))
    # ----------------------------------------------------------------

    # -----------------------------------------------------------------
    # 8. flow-plot for every channel that may exist
    # -----------------------------------------------------------------
    # ─── Define horizontal range for flow plot: xl = [start, end] ───
    beam_shaper_pos = None
    detector_pos     = None

    # Search for beam_shaper and Det positions (no 'in' check needed)
    for z, name, dct in elements:
        if name == "beam_shaper":
            beam_shaper_pos = z
        if name == "Det":
            detector_pos = z

    # Set xl if either boundary exists
    xl = None
    if beam_shaper_pos is not None or detector_pos is not None:
        xl = [beam_shaper_pos if beam_shaper_pos is not None else None,
            detector_pos     if detector_pos     is not None else None]
        print(f"[FLOW] Setting xl = {xl} based on beam_shaper and/or detector.")

    # ─── Define vertical range for flow plot: gyax ───
    initial_propsize_m = params.get("initial_propsize", 1000)  # fallback if not set earlier
    half_box_um = initial_propsize_m * 1e6 / 2  # Convert to µm
    print(f"[FLOW] Using initial simulation box size = {2 * half_box_um:.1f} µm")    
    gyax_step = 1  # resolution in µm
    gyax_def = [-half_box_um, half_box_um, gyax_step]
    params["flow_plot_gyax_def"] = gyax_def



    for ch in ("main", "VB_parr", "VB_perp"):
        try:
            print(f"[DEBUG] flow_plot() for channel={ch} → gyax_def = {gyax_def}")

            flow_plot(projectdir,
                    params['filename'],
                    vertical_type="center",
                    channel=ch,
                    gyax_def=gyax_def,xl=xl)
        except AssertionError:
            # channel wasn’t recorded – simply skip
            pass
    # -----------------------------------------------------------------

    return params, trans, figs





def CRL4_get_length(number_of_lenses,Energy):
    f=CRL_get_length(0.05,number_of_lenses,Energy)
    return f

def CRL_get_length(radius_mm,number,Energy):
    dn=340/Energy**2
    n=1+dn
    f1=radius_mm/2/(n-1)
    f_calc=f1/number*1e-3
    return f_calc #focal length [m]


def yamlval(key,ip,default=0):
    if not key in ip.keys() :
        return default
    else:
        return ip[key]


def flow_plot(project_dir, file, cl=[1e-11,50], gyax_def=[-1000,1000,1], vertical_type='center', log=1, xl=None, flow_figs=0, flow_plot_crange=1e-5, channel="main", include_flow=True, unit=None):
    """
    Visualizes 2D flow maps along z, with optional export for movie making.

    Parameters
    ----------
    project_dir : str         – Path to simulation folder.
    file        : str         – Base name of the pickle files.
    cl          : list        – Color limits for plotting.
    gyax_def    : list        – Vertical axis definition [start, end, step] in µm.
    vertical_type : str       – How to reduce 2D → 1D (center, average_horiz, etc.)
    log         : bool        – Use log scale in plot.
    xl          : list or None – Horizontal axis limits in meters.
    flow_figs   : bool        – If True, saves each flow slice individually.
    flow_plot_crange : float  – Color scale fraction for flow slice plots.
    channel     : str         – Channel to plot ("main", "VB_parr", ...).
    include_flow : bool       – Whether to include "flow" auto-slices.
    unit        : str or None – Intensity unit ("relative", "photons", "Wcm2").

    Returns
    -------
    params : dict
    res    : dict
    fixedfall : np.ndarray – final interpolated waterfall image
    """
    import os
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # ─── Load Pickles ────────────────────────────────────────────────

    fn_figs = f"{file}_figs"
    fn_res  = fn_figs.replace('figs', 'res').replace('export', 'res')

    pic_path = Path(project_dir) / 'pickles' / f'{fn_figs}.pickle'
    res_path = Path(project_dir) / 'pickles' / f'{fn_res}.pickle'

    pic = mu.loadPickle(str(pic_path), strict=1)
    res = mu.loadPickle(str(res_path))
    partial = (res == 0)
    
    # ─── Safety checks ───────────────────────────────────────────────

    assert len(pic.keys()) > 0, 'No images found in the pickle!'
    if not partial:
        params = res[1]
    else:
        params = {}


    # ─── Setup core variables ────────────────────────────────────────
    first_key = sorted(pic.keys())[0]
    picc, _, _, _ = pic[first_key]
    N = np.shape(picc)[0]

    gyax = np.arange(*gyax_def)  # y-axis in µm

    numfigs = sum(1 for k in pic if k.endswith(f"_{channel}"))

    assert numfigs > 0, f"No flow slices found for channel '{channel}'"

    #print(f"[FLOW] Found {numfigs} flow slices for channel '{channel}'")


    # ─── Filter relevant flow slices ────────────────────────────────
    ffigs = []
    figs = pic.keys()

    for fig in figs:
        if not fig.endswith(f"_{channel}"):
            continue

        if channel == "main":
            if fig.startswith("flow"):
                ffigs.append(fig)
        else:
            el_name = fig.split('_')[0]
            wanted  = params.get("figs_to_save", [])
            if (el_name == "flow" and include_flow) or (el_name in wanted and el_name != "flow"):
                ffigs.append(fig)

    print(f"[DEBUG] ffigs for channel '{channel}': {len(ffigs)} found")

    # ─── Intensity unit handling ─────────────────────────────────────
    unit_sel = unit or params.get('intensity_units', 'relative')
    scale_ph  = params.get('scale_phot',  1.0)
    scale_Wcm = params.get('scale_Wcm2', 1.0)

    if unit_sel == 'photons':
        scale = scale_ph
        y_label = "photons / px"
    elif unit_sel == 'Wcm2':
        scale = scale_Wcm
        y_label = "W cm⁻²"
    else:
        scale = 1.0
        y_label = "Normalized intensity"


    try:
        cl = [float(c) * scale for c in cl]
    except Exception as e:
        raise ValueError(f"Invalid color limits (cl): {cl}. Make sure it's a list of two floats.") from e
    
    # --- Id of the run ------------------------------------------------
    run_id = file if channel in (None, "", "main") else f"{file}_{channel}"

    # ─── Initialize arrays for waterfall plot ────────────────────────
    numfigs = len(ffigs)
    waterfall  = np.zeros((numfigs, N))
    fixedfall  = np.zeros((numfigs, len(gyax)))
    propsizes  = np.zeros((numfigs,))
    zax        = np.zeros((numfigs,))


    scatterer_L2_position=1e9
    scatterer_L1_position=1e9
    skip_existing=1

    if not partial:
    #extracting scatterers and theirloses
        if 'L1' in res[0]:
            scatterer_L1_position=res[0]['L1']['position']
            scatterer_L1_loss=yamlval('transmission_of_scatterer_L1',params,1)
        if 'L2' in res[0]:
            scatterer_L2_position=res[0]['L2']['position']
            scatterer_L2_loss=yamlval('transmission_of_scatterer_L2',params,1)
        N=res[1]['subfigure_size_px']
    else: params=[]
    assert len(pic.keys())>0, 'There are no pictures in the pickle!'

    if flow_figs:
        ffdir = Path(project_dir) / 'flow_figs' / run_id
        #mu.mkdir(ffdir,0)
        mu.mkdir(str(ffdir), 0)


    # ─── Loop over flow slices ───────────────────────────────────────
    for fi, fig in enumerate(ffigs):
        picc, elemi, propsize, position = pic[fig]
        #print(f"[DEBUG] {fig} – propsize = {propsize:.3e} m")

        imsize  = picc.shape[0]
        pxsize  = propsize * 1e6 / imsize  # µm per pixel
        half_px = imsize // 2
        ps2     = propsize / 2             # half size in meters

        # Compute horizontal axis in μm
        xax = (np.arange(imsize) - 0.5 * imsize) * pxsize  # μm

        # Compute vertical lineout based on `vertical_type`
        if vertical_type == 'center':
            w = 2
            start = max(0, half_px - w)
            end   = min(imsize, half_px + w + 1)
            lineout = np.mean(picc[start:end, :], axis=0)
        elif vertical_type == 'average_horiz':
            lineout = np.mean(picc, axis=0)
        elif vertical_type == 'vert-center':
            lineout = picc[:, half_px]
        elif vertical_type == 'vert-integral':
            lineout = np.mean(picc, axis=1)
        else:
            raise ValueError(f"Unknown vertical_type: {vertical_type}")

        # Apply scatterer correction if needed
        if position > scatterer_L2_position:
            lineout /= (scatterer_L2_loss * scatterer_L1_loss)
        elif position >= scatterer_L1_position:
            lineout /= scatterer_L1_loss

        # Interpolate onto fixed gyax grid
        interp_line = np.full(len(gyax), np.nan, dtype=np.float64)
        valid = (gyax >= np.min(xax)) & (gyax <= np.max(xax))
        interp_line[valid] = np.interp(gyax[valid], xax, lineout)
        fixedfall[fi, :] = interp_line * scale


        # Store
        waterfall[fi, :] = lineout * scale
        fixedfall[fi, :] = interp_line * scale
        propsizes[fi] = propsize
        zax[fi] = position

        # ─── Optional: export movie frame if requested ───
        if flow_figs:
            ffdir = Path(project_dir) / 'flow_figs' / run_id
            ffdir.mkdir(parents=True, exist_ok=True)

            ff_fn = ffdir / f"fixed_{fi:04d}.jpg"

            # Skip existing if flag is set
            if skip_existing and ff_fn.exists():
                print(f"[SKIP] Frame {fi:04d} already exists.")
            else:
                print(f"[MOVIE] Exporting frame {fi:04d} at z = {position:.2f} m")

                # Plot the full image slice
                fig_movie, ax_movie = plt.subplots(figsize=(10, 10))
                npix = picc.shape[0]
                xc = (np.arange(npix) - npix / 2) * pxsize
                cmax = np.max(picc)
                cl1 = [cmax * flow_plot_crange, cmax]

                picc_T = picc.T  # transpose for correct orientation

                mu.pcolor(picc_T, xc=xc, yc=xc, ticks=0, log=1, cl=cl1, background=[0, 0, 0])
                plt.axis('equal')

                h = 100 / 2  # box size = 100 μm
                plt.plot([-h, -h, h, h, -h], [-h, h, h, -h, -h], 'r.', alpha=1, markersize=7)

                plt.xlabel('X [μm]')
                plt.ylabel('Y [μm]')
                plt.title(f"{file}, z = {position * 100:.0f} cm")

                # zoom into 100 μm × 100 μm box
                plt.xlim(-h, h)
                plt.ylim(-h, h)

                plt.tight_layout()
                plt.savefig(ff_fn)
                plt.close(fig_movie)


    fig, ax = plt.subplots(figsize=(14, 8))

    mu.pcolor(
        xc=zax,
        yc=gyax,
        data=fixedfall,
        log=log,
        ticks=None,
        cl=cl,
        colorbar=False
    )

    # Add colorbar with label
    cb = plt.colorbar()
    if vertical_type in ("average_horiz", "center"):
        cb_label = {
            "photons": "photons / m²",
            "Wcm2": "W / m²",
            "relative": "relative units"
        }.get(unit_sel, "relative units")
    else:
        cb_label = y_label
    cb.set_label(cb_label)

    # Overlay propagation box profile (normalized to gyax)
    profile = mu.normalize(propsizes) * np.max(gyax)
    plt.plot(zax, profile, 'r-')

    # Draw optical element markers
    if not partial:
        maxy = np.min(gyax)
        row = 0
        for el_name, el in res[0].items():
            if (
                    'position' not in el or
                    not mu.yamlval('in', el, 1) or
                    el_name.startswith("flow_")
                ):
                    continue
            pos = el['position']
            if len(el_name) == 2:  # short names like L1, L2
                yline = maxy * (0.8 if 'L' in el_name else 0.72)
                col = [1, 0.5, 0.9] if 'L' in el_name else [1, 0.9, 0.7]
            else:
                yline = maxy * (0.95 - row * 0.05)
                col = 'w'
                row = (row + 1) % 4

            plt.plot([pos, pos], [maxy, yline], color=col)
            mu.text(pos + 0.05, yline, el_name, color=col, fs=16, zorder=50, background=None)

    # Detector marker (white vertical line)
    det_pos = params.get('elements', {}).get('Det', {}).get('position', 7.0)
    roi = 13
    plt.plot([det_pos, det_pos], [-roi/2, roi/2], 'w-', lw=5)

    # Axes labels and limits
    plt.xlabel('Position [m]')
    plt.ylabel('Horizontal position [μm]')
    plt.xlim(xl if xl else [np.min(zax), np.max(zax)])
    plt.ylim(np.min(gyax), np.max(gyax))  # ← enforce correct y-range
    plt.title(f"{file} cut: {vertical_type}")

    plt.tight_layout()


    # ─── Total photon count near beam shaper ───
    elements_dict = res[0]
    beam_shaper_pos = None
    if "beam_shaper" in elements_dict:
        if yamlval("in", elements_dict["beam_shaper"], 1):
            beam_shaper_pos = elements_dict["beam_shaper"]["position"]

    if beam_shaper_pos is not None:
        idx = np.argmin(np.abs(zax - beam_shaper_pos))
        idx += 10  # margin to be clearly downstream

        # Select raw image or interpolated lineout
        if vertical_type in ("center", "vert-center"):
            fig_key = ffigs[idx]
            raw_img, _, propsize, _ = pic[fig_key]
            img_N = raw_img.shape[0]
            dx = dy = propsize / img_N
            img_scaled = {
                "photons": raw_img * scale_ph,
                "Wcm2": raw_img * scale_Wcm,
                "relative": raw_img
            }.get(unit_sel, raw_img)
            total_photons = np.nansum(img_scaled) * dx * dy
        else:
            # for 1D vertical lineouts
            dy = (gyax[1] - gyax[0]) * 1e-6  # m
            Lx = propsizes[idx]             # m
            total_photons = np.nansum(fixedfall[idx]) * dy * Lx

        # Print results if in photon mode
        photons_target = params.get("photons_total", None)
        if unit_sel == "photons":
            print(f"\n✅ Beam shaper at z = {beam_shaper_pos:.2f} m → closest slice z = {zax[idx]:.2f} m")
            print(f"→ Total photons from flow = {total_photons:.3e}")
            if photons_target:
                rel_err = (total_photons - photons_target) / photons_target
                print(f"→ Target photons_total = {photons_target:.3e}")
                print(f"→ Relative error = {rel_err:.2%}")

    if not partial:
        res[1]['propsizes'] = propsizes

    # --------- Shadow Factors and Detector Marker -------
    centralelement = "TCC"
    if f"{centralelement}_{channel}" not in params.get("intensities", {}):
        centralelement = "PH"
    key_central = f"{centralelement}_{channel}"

    intens = params.get("intensities", {})
    if key_central in intens:
        t1 = intens[key_central] / intens.get("start", 1.0)
        tr_scat = yamlval("transmission_of_scatterer_L2", params, 1)

        if "roi" in intens and "roi2" in intens:
            t13 = intens["roi"] / intens[key_central] / tr_scat
            t75 = intens["roi2"] / intens[key_central] / tr_scat
            print(f"SFA13 = {t13:.1e}, SFA75 = {t75:.1e}, Ratio = {t75/t13:.2f}")

            ax = plt.gca()
            ax.text(0.98, 0.85, f"SFA13 = {t13:.1e}",
                    transform=ax.transAxes, color='red', fontsize=14,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            ax.text(0.98, 0.77, f"SFA75 = {t75:.1e}",
                    transform=ax.transAxes, color='black', fontsize=14,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            ax.text(0.98, 0.69, f"SFA75/SFA13 = {t75/t13:.0f}",
                    transform=ax.transAxes, color='black', fontsize=11,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))


    # ─── Save main flow plot ───
    outdir = Path(project_dir) / "flows"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{file}_flowplot_{channel}.jpg"
    plt.savefig(outfile)
    print(f"[FLOW] Saved main flow plot to {outfile}")

    return params, res, fixedfall

"""
def flow_plot(project_dir, file, cl=[1e-11,50], gyax_def=[-1000,1000,1], vertical_type='center', log=1, xl=None, flow_figs=0, flow_plot_crange=1e-5, channel="main", include_flow=True, unit=None):    
    from pathlib import Path
    cols=['g','r','k','b',[0.5,1,0.8],[1,0.3,0.8],'r']

    gyax=np.arange(gyax_def[0],gyax_def[1],gyax_def[2]) #μm
    fn=str(file)+'_figs'
    fns=fn

    pic_path = Path(project_dir) / 'pickles' / f'{fn}.pickle'
    pic = mu.loadPickle(str(pic_path), strict=1)

    p2 = fn.replace('figs', 'res').replace('export', 'res')
    res_path = Path(project_dir) / 'pickles' / f'{p2}.pickle'
    res = mu.loadPickle(str(res_path))
    
    #------ Choosing the intensity unit ------
    sim_cfg   = yamlval('simulation', res[1], {})
    unit_sel = unit or res[1].get('intensity_units', 'relative')
    scale_ph  = res[1].get('scale_phot',  1.0)
    scale_Wcm = res[1].get('scale_Wcm2', 1.0)
    #----------------------

    partial=(res==0)
    fn2=fns[:-5]
    l=fn2
    scatterer_L2_position=1e9
    scatterer_L1_position=1e9
    skip_existing=1
    if not partial:
        params=res[1]
    #extracting scatterers and theirloses
        if 'L1' in res[0]:
            scatterer_L1_position=res[0]['L1']['position']
            scatterer_L1_loss=yamlval('transmission_of_scatterer_L1',params,1)
        if 'L2' in res[0]:
            scatterer_L2_position=res[0]['L2']['position']
            scatterer_L2_loss=yamlval('transmission_of_scatterer_L2',params,1)
        N=res[1]['subfigure_size_px']
    else: params=[]
    assert len(pic.keys())>0, 'There are no pictures in the pickle!'

    akey=sorted(pic.keys())[0]
    picc,b,c,d=pic[akey]
    N=np.shape(picc)[0]
    figs=pic.keys()

    ffigs=[]
    if flow_figs:
        ffdir=project_dir+'/flow_figs/'+fn2+'/'
        mu.mkdir(ffdir,0)

    for fig in figs:
        if not fig.endswith(f"_{channel}"):
            continue

        if channel == "main":            # main keeps its old behaviour
            if fig.startswith("flow"):   # include every flow slice
                ffigs.append(fig)

        else:                            # VB channels
            el_name = fig.split('_')[0]
            wanted  = params.get("figs_to_save", [])
            if (el_name == "flow" and include_flow) or (el_name in wanted and el_name != "flow"):
                ffigs.append(fig)

    print(f"[DEBUG] ffigs for channel {channel}:\n", ffigs)


    numfigs=len(ffigs)
    assert numfigs > 0, f"No flow slices found for channel '{channel}'"
    waterfall=np.zeros((numfigs,N))
    propsizes=np.zeros((numfigs))
    fixedfall=np.zeros((numfigs,np.size(gyax)))

    zax=np.zeros(numfigs)
    for fi,fig in enumerate(ffigs):
        picc,elemi,propsize,position=pic[fig]
        print(f"[DEBUG] {fig} – propsize = {propsize:.3e} m")
       # tr=res[1]['transmission']
        pxsize=propsize*1e6/np.shape(picc)[0]
        imsize=np.shape(picc)[0]
        halfsize=int(imsize/2)
        tit=file[:-13]
    #lineout
        ZoomFactor=1
        ps2=propsize/2/ZoomFactor

        # ---- New center averaging option ----
        center_average_width_px = 2  # <- average over ± this many pixels around center (set by user)

        halfsize = int(imsize / 2)
        if vertical_type == 'center':
            w = center_average_width_px
            start = max(0, halfsize - w)
            end = min(imsize, halfsize + w + 1)
            lineout = np.mean(picc[start:end, :], axis=0)
        if vertical_type == 'average_horiz':
            lineout = np.mean(picc, axis=0)  # average across x → result has unit per m²
        if vertical_type=='vert-center':
            lineout=picc[:,halfsize]
        if vertical_type=='vert-integral':
            lineout=np.mean(picc,1)
        xax=np.arange(np.shape(picc)[0])
        xax=(xax/np.size(xax)*ps2*2-ps2)*1e6 #um


        ############# PLOT OF THE FLOW SUBFIGS FOR THE MOVIE ############
        if flow_figs:
            ff_fn='./'+ffdir+'fixed_{:04.0f}.jpg'.format(fi)
            plot=1
            if skip_existing:
                import os
                if fi!=-135 and os.path.isfile(ff_fn):
                    plot=0
                    print(fi,'    ',fig,' skipping')

            if plot:
                print(fi,'    ',fig,' plotting')
                boxsize=100
                mu.figure(10,10,safe=1)
                npix=np.shape(picc)[0]
                xc=(np.arange(npix)-npix/2)*pxsize
                cmax=np.max(picc)
                cl1=[cmax*flow_plot_crange,cmax]
                picc=np.transpose(picc)
                print(cl1)
                mu.pcolor(picc,xc=xc,yc=xc,ticks=0,log=1,cl=cl1,background=[0,0,0])
                plt.axis('equal')
                h=boxsize/2
                #plt.plot([-h,-h,h,h,-h],[-h,h,h,-h,-h],'r',alpha=0.3)
                plt.plot([-h,-h,h,h,-h],[-h,h,h,-h,-h],'r.',alpha=1,markersize=7)
                plt.xlabel('X [μm]')
                plt.ylabel('Y [μm]')
                plt.title(l + ', {:.0f} cm'.format(position*100))
                plt.savefig(ffdir+'ff_{:04.0f}'.format(fi))

                fff=50
                fffx=fff*1.
                plt.ylim(-fffx,fffx)
                plt.xlim(-fff,fff)
                plt.savefig(ff_fn)
        if position>scatterer_L2_position:
            lineout=lineout/scatterer_L2_loss/scatterer_L1_loss
        elif position>=scatterer_L1_position:
            lineout=lineout/scatterer_L1_loss
        waterfall[fi,:]=lineout
        zax[fi]=position

        #inteprpprof=np.interp(gyax,xax,lineout)
        #inteprpprof[gyax<np.min(xax)]=np.nan
        #inteprpprof[gyax>np.max(xax)]=np.nan

        inteprpprof = np.full_like(gyax, np.nan)
        # Determine indices within the xax bounds
        valid = (gyax >= np.min(xax)) & (gyax <= np.max(xax))
        inteprpprof[valid] = np.interp(gyax[valid], xax, lineout)



        #fixedfall[fi,:]=inteprpprof #old way
        xaxis_um = np.arange(gyax_def[0], gyax_def[1], gyax_def[2])
        inteprpprof = np.interp(xaxis_um, xax, lineout)
        fixedfall[fi,:] = inteprpprof
        propsizes[fi]=propsize

    fixedfall[fixedfall<=0]=1e-30
    waterfall[waterfall<=0]=1e-30

    l2=l
    l2+=' cut: '+vertical_type
    if 0:
        mu.figure(16,9)
        nn=np.shape(picc)[0]
        rax=np.arange(nn)/nn*100-50
        pxsizes=propsizes*1e6/nn
        boundary_200=200/pxsizes#px
        boundary_200=boundary_200/nn*2*50 #%
        boundary_200[boundary_200>65]=np.nan
        mu.pcolor(xc=zax,yc=rax,data=waterfall,log=1,ticks=0,cl=cl)
        plt.plot(zax,boundary_200,'r-')
        plt.xlabel('Position [m]')
        plt.ylabel('Box size [%]')
        plt.title(l2)
        plt.ylim(-50,50)
        mu.savefig('./flows/boxflow_{:}_{:}'.format(l,vertical_type))

    # ──── NEW: rescale to photons or W/cm² *once* ──────────────────
    scale = 1.0                       # default → relative units
    if unit_sel == 'photons':
        scale = scale_ph              # read from the pickle header
        y_label = "photons / px"
    elif unit_sel == 'Wcm2':
        scale = scale_Wcm
        y_label = "W cm⁻²"
    else:
        y_label = "Normalised Intensity"

    waterfall *= scale                # apply the same factor to
    fixedfall *= scale                #   both data sets
    cl = [float(c) * scale for c in cl]      # rescale colour-bar limits
    #print("DEBUG flow scale =", scale, "unit =", unit_sel)
    # ----------------------------------------------------------------

    ################## PLOT OF THE MAIN FLOW FIG #####################
    #mu.figure(16,9)
    fig, ax = plt.subplots(figsize=(14, 8))

    #print(cl)
    linearize = 0 #linearize option is used in the case where the X and Y axis are not linear. (Don't activate it)
    if linearize:
        mu.pcolor(xc=zax,yc=gyax,data=fixedfall,log=1,ticks=0,cl=cl,linearize=1) #,xtics_spacing=1
        pos=np.arange(0,np.size(zax),15)
        vals=[]
        for va in zax[pos]:
            vals.append('{:.1f}'.format(va))
        plt.xticks(pos,vals)
        plt.yticks(np.arange(0,np.size(gyax)+1,50))
        plt.ylabel(f"Horizontal position [μm]\n({y_label})")
    else:
        mu.pcolor(xc=zax,yc=gyax,data=fixedfall,log=log,ticks=None,cl=cl,colorbar=False)

        # ─── Add colorbar with correct units ───
        cb = plt.colorbar()
        if vertical_type in ("average_horiz", "center"):
            if unit_sel == "photons":
                cb.set_label("photons / m²")
            elif unit_sel == "Wcm2":
                cb.set_label("W / m²")
            else:
                cb.set_label("relative units")
        else:
            cb.set_label(y_label)


    profile=mu.normalize(propsizes)*np.max(gyax)
    maxy=np.min(gyax)
    if not partial:

        ip=res[0]
        row=0
        plt.xlim(xl)
        for el_name in ip:
            el=ip[el_name]
            if 'position' not in el: continue
            if not mu.yamlval('in',el,1): continue
            if len(el_name)==2:
                yline=maxy*0.72
                col=[1,0.9,0.7]
                if 'L' in el_name:
                    yline=maxy*0.8
                    col=[1,0.5,0.9]

            else:
                yline=maxy*(0.95-row*0.05)
                col='w'
            mu.text(el['position']+0.05,yline,el_name,color=col,fs=16,zorder=50,background=None)
            plt.plot([el['position'],el['position']],[maxy,yline],color=col)
            if len(el_name)!=2:
                row+=1
                if row>3:row=0

        plt.xlabel('Position [m]')
        plt.ylabel('Horizontal position [μm]')
        if xl==None:
            plt.xlim(np.min(zax),np.max(zax))
        else:
            plt.xlim(xl)
        plt.plot(zax,profile,'r-')

    # -------- Ploting a small vertical white line at the location of the detector and of size = roi pixel ---------
    roi=13
    det_pos = params.get('elements', {}).get('Det', {}).get('position', 7.0)  # fallback to 7.0 if missing
    plt.plot([det_pos, det_pos], [-roi/2, roi/2], 'w-', lw=5)
    

    plt.title(l2)

    #print("Available keys in intensities:", params.get('intensities', {}).keys())
    centralelement = "TCC"
    if f"{centralelement}_{channel}" not in params["intensities"]:
        centralelement = "PH"
    key_central = f"{centralelement}_{channel}"


    intens = params["intensities"]

    if key_central in intens:
        t1 = intens[key_central] / intens["start"]
        tr_scat = yamlval("transmission_of_scatterer_L2", params, 1)

        if "roi" in intens and "roi2" in intens:
            t13 = intens["roi"] / intens[key_central] / tr_scat
            t75 = intens["roi2"] / intens[key_central] / tr_scat
            print(f"SFA13 = {t13:.2e}, SFA75 = {t75:.2e}, Ratio = {t75/t13:.2f}")
            # Add text annotations using axis-relative coordinates
            ax = plt.gca()

            ax.text(0.98, 0.85, f"SFA13 = {t13:.2e}",
                    transform=ax.transAxes, color='red', fontsize=14,
                    ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            ax.text(0.98, 0.77, f"SFA75 = {t75:.2e}",
                    transform=ax.transAxes, color='black', fontsize=14,
                    ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            ax.text(0.98, 0.69, f"SFA75/SFA13 = {t75/t13:.0f}",
                    transform=ax.transAxes, color='black', fontsize=11,
                    ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
    flow_path = Path(project_dir) / 'flows' / f'flow_{l}_{channel}_{vertical_type}'
    plt.tight_layout()

    mu.savefig(str(flow_path))

    # Get beam_shaper position from the YAML
    elements_dict = res[0]
    beam_shaper_pos = None
    if "beam_shaper" in elements_dict:
        if yamlval("in", elements_dict["beam_shaper"], 1):
            beam_shaper_pos = elements_dict["beam_shaper"]["position"]


    if beam_shaper_pos is not None:
        idx = np.argmin(np.abs(zax - beam_shaper_pos))
        idx = idx + 10 # Just to be sure that we are after the beam_shaper and now before.
        # Use full 2D field for "center" mode
        if vertical_type in ("center", "vert-center"):
            fig_beamshaper = ffigs[idx]
            raw_picc, _, propsize, _ = pic[fig_beamshaper]
            dx_m = propsize / N
            dy_m = dx_m  # assume square pixels for now

            # Apply scaling
            if unit_sel == "photons":
                picc_scaled = raw_picc * scale_ph
            elif unit_sel == "Wcm2":
                picc_scaled = raw_picc * scale_Wcm
            else:
                picc_scaled = raw_picc  # relative units

            total_photons = np.nansum(picc_scaled) * dx_m * dy_m

        else:
            # Original computation (works only for average profiles)
                dy_m = (gyax[1] - gyax[0]) * 1e-6
                Lx_m = propsizes[idx]  # total horizontal width in meters
                total_photons = np.nansum(fixedfall[idx, :]) * dy_m * Lx_m

        photons_yaml = params.get("photons_total", None)

        if unit_sel == "photons":
            print(f"\n✅ Flow-slice index matching beam_shaper position {beam_shaper_pos:.2f} m (raw index n°{idx})→ closest = z = {zax[idx]:.2f} m")
            print(f"→ Total photons from flow profile = {total_photons:.3e} (integrated over x and y)")

            if photons_yaml:
                print(f"→ Target photons_total from YAML = {photons_yaml:.3e}")
                print(f"→ Relative error = {(total_photons - photons_yaml)/photons_yaml:.2%}")

    res[1]["propsizes"] = propsizes


    return params, res, fixedfall
"""


def flow_savefig(I,ffdir,fi,propsize,label,position,flow_plot_crange=1e-5):
    picc=I
    ff_fn='./'+ffdir+'fixed_{:04.0f}.png'.format(fi)
    boxsize=100
    f2=mu.figure(14,10,safe=1)

    npix=np.shape(picc)[0]
    pxsize=propsize*1e6/np.shape(picc)[0]
    xc=(np.arange(npix)-npix/2)*pxsize
    cmax=np.max(picc)
    cl1=[cmax*flow_plot_crange,cmax]

    picc=np.transpose(picc)
    mu.pcolor(picc,xc=xc,yc=xc,ticks=0,log=1,cl=cl1,background=[0,0,0])
    plt.axis('equal')
    h=boxsize/2
    plt.plot([-h,-h,h,h,-h],[-h,h,h,-h,-h],'r.',alpha=1,markersize=7)
    plt.xlabel('X [μm]')
    plt.ylabel('Y [μm]')
    plt.title(label + ', {:.0f} cm'.format(position*100))
    plt.savefig(ffdir+'ff_{:04.0f}'.format(fi))

    fff=50
    fffx=fff*1.
    plt.ylim(-fffx,fffx)
    plt.xlim(-fff,fff)
    plt.savefig(ff_fn)
    #plt.close(f2)
