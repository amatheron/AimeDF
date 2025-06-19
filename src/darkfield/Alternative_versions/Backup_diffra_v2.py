from LightPipes import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import ascii
from PIL import Image
from scipy import signal

#import darkfield.regularized_propagation as rp
import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils_v2 as mu
#Simon start----
import darkfield.regularized_propagation_v2 as rp
#Simon end----
HOME = '/home/yu79deg/darkfield_p5438/'
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
#    assert E==8766, print("forcing E as 8766eV, don't have other constants")
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
#            k=-np.log(0.58629)/(10*um)
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
#        deltat=wireprof*(delta)/(c*(1-delta)) #derived by me...
 #       deltas=deltat*c #how much does it shift
  #      phaseshift=deltas/lambd* 2*3.14
    return beta,delta,k,thickness_to_phaseshift

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


def do_phaseplate(el_dict,params,debug=0):
    assert el_dict['type']=='phaseplate'
    defect=el_dict['defect']

    E=params['photon_energy']
    N=params['N']
    pxsize=params['pxsize']
    num=el_dict['num']
    N2=int(N/2)
    mx=N2*pxsize

    Na=np.arange(-N2,N2)*pxsize
    xm,ym=np.meshgrid(Na,Na)
    r=((xm**2)+(ym**2))**0.5
    thickness=np.zeros([N,N])

    if 'seiboth' in defect:
        fia=ascii.read(f'{HOME}/Seiboth_Fig4')
        fiax=fia['col1']
        fiay=fia['col2']
        if 0:
            mu.figure()
            plt.plot(fiax,fiay)
            plt.xlabel('radial position [μm]')
            plt.ylabel('deformation [μm]')

        img=np.zeros([N,N])
        for xi,x in enumerate(Na):
            for yi,y in enumerate(Na):
                rh=r[xi,yi]/um
                deformation=np.interp(rh,fiax,fiay)
                img[xi,yi]=deformation
        thickness+=img*um

    if 'celestre' in defect:
        image = Image.open(f'{HOME}/Celestre_Fig8.png')
        image=image.resize((N,N))
        im = np.array(image)[:,:,0]
        im=im/255*24 #from values to μm in figure
        im/=11 #to go to one lens from 11
        if 0:
            mu.figure()
            plt.imshow(im)
            plt.colorbar()
#            plt.xlabel('radial position [μm]')
            #plt.ylabel('deformation [μm]')
        thickness+=im*um
#        img=img*um

    thickness=thickness*num
    elem='Be'
    beta,delta,k,thickness_to_phaseshift=get_n(elem,E)
    phaseshiftmap=thickness*thickness_to_phaseshift#*-1

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


# %%
def get_aperture_transmission_map(pars,params=[],debug=0):
    typ=pars['shape']
    pxsize=params['pxsize']
    N=params['N']
    N2=int(N/2)

    Na=(np.arange(N)-N2)*pxsize
    trmap=np.zeros([N,N])+1  #that mean default is 1 = transmissive

    xm,ym=np.meshgrid(Na,Na)

#the flat ideal things
    if typ in ['square','rectangle','wire','circle','gaussian']:
        if typ=='square':
            hs=pars['size']/2
            sel=(xm>=-hs)*(xm<=hs)*(ym>=-hs)*(ym<=hs)
            trmap[sel]=0
        if typ=='rectangle':
            hs=pars['size']/2
            vs=pars['sizevert']/2
            sel=(xm>=-hs)*(xm<=hs)*(ym>=-vs)*(ym<=vs)
            trmap[sel]=0
        if typ=='wire':
            hs=pars['size']/2
            sel=(xm>=-hs)*(xm<=hs)
            trmap[sel]=0
    #    if typ=='circle':
#            trmap=trmap*0
     #       r=((xm**2)+(ym**2))**0.5
      #      rad=pars['size']/2
       #     trmap[r<rad]=0

        if typ=='gaussian':
            r=((xm**2)+(ym**2))**0.5

            rax=np.arange(0,2*N2)*pxsize# to be in m
            power=float(pars['power'])
            fwhm=float(pars['size'])
            gauss=mu.gauss(rax,0,1,0,fwhm,power=power)
         #   gauss=1-gauss
            for xi,x in enumerate(xm):
                for yi,y in enumerate(ym):
                    val=np.interp(r[xi,yi],rax,gauss)
                    trmap[xi,yi]=val

        if yamlval('invert',pars):
            trmap=1-trmap
    return trmap
# %%

def get_aperture_thickness_map(pars,params=[],debug=0):

    typ = pars['shape']
    pxsize = params['pxsize']
    N = params['N']
    N2 = int(N/2)

    Na = (np.arange(N)-N2) * pxsize #list of coordinates in [m], centered around 0.
    thicknessmap = np.zeros([N,N]) + 1  #that mean default is 1 m thick.

    xm,ym = np.meshgrid(Na,Na)
    if typ=='circle':
#            trmap=trmap*0
        r=((xm**2)+(ym**2))**0.5
        rad=pars['size']/2
        thicknessmap=thicknessmap*0
        thicknessmap[r<rad]=pars['thickness']
        if yamlval('invert',pars):
            maxi=np.max(thicknessmap)
            thicknessmap=maxi-thicknessmap

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
                
    wls=['realwire','trapez','tent','customwire','pooyan','invpoo','invpar','par','wireslit','linearslit','wire_grating']

    if typ in wls :
        wireprof = get_wire_like_profile(pars,params,debug)

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


        ##INVERTING ---might not work anymore (March 2025)
        if yamlval('invertphaseshift',pars):  #inverting phase-shift-wise
            if yamlval('inversiontype',pars)=='thickness':
                print('inverting by thickness')
                maxth=np.max(wireprof)
                wireprof=maxth-wireprof
                newelem=pars['newelem']
                beta,delta,k,thickness_to_phaseshift=get_n(newelem,E)
                phaseshift=wireprof*thickness_to_phaseshift

            else:
                beta,delta,k,thickness_to_phaseshift=get_n(elem,E)
                phaseshift=wireprof*thickness_to_phaseshift
                newelem=pars['newelem']
                if yamlval('invertphaseshift-maxps',pars):
                    maxpp=yamlval('invertphaseshift-maxps',pars)
                else:
                    maxpp=np.max(phaseshift)
                newpp=maxpp-phaseshift
                newpp[newpp<0]=0
                newbeta,newdelta,k,newthickness_to_phaseshift=get_n(newelem,E)
                wireprof=newpp/newthickness_to_phaseshift

                phaseshift=newpp
                thickness_to_phaseshift=newthickness_to_phaseshift

        ones=Na*0+1
        thicknessmap=np.matmul(np.transpose(np.matrix(wireprof)),(np.matrix(ones)))
        

    defect_type = yamlval('defect_type',pars)
    
    if defect_type in ['sine', 'sawtooth', 'triangle']: #Adds defects at the edge of apertures
        wavelength = float(yamlval('defect_lambda', pars))
        amplitude = float(yamlval('defect_amplitude', pars))

        x = Na * 2 * np.pi / wavelength  # common normalized phase array

        if defect_type == 'sine':
            offsets_m = np.sin(x) * amplitude
        elif defect_type == 'sawtooth':
            offsets_m = signal.sawtooth(x) * amplitude
        elif defect_type == 'triangle':
            offsets_m = signal.sawtooth(x, width=0.5) * amplitude

        offsets_px = np.round(offsets_m / pxsize).astype(int)

        tmp = np.zeros_like(thicknessmap)
        for yi in range(N):
            tmp[:, yi] = np.roll(thicknessmap[:, yi], offsets_px[yi])

        thicknessmap = tmp.copy()
    
    #print("thickness map in get_apetture_thickness_map:", thicknessmap)
    return(thicknessmap)


def get_wire_like_profile(pars,params,debug):
 #realistic wire-like structures
    r = float(yamlval('size', pars, 0)) / 2
    off=float(yamlval('offset',pars,0))
    elem=pars['elem']
    pxsize=params['pxsize']
    N=params['N']
    N2=int(N/2)

    Na=(np.arange(N)-N2)*pxsize
    x=Na-off
    typ=pars['shape']

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

    elif typ.find('wireslit')==0:
        r=pars['r']
        wireprof=x*0
        halfsize=pars['size']/2
        off=halfsize+r
        circ1=(r**2-(Na-off)**2)**0.5*2
        sel1=(x>=halfsize) * (x<=off)
        wireprof[sel1]=circ1[sel1]
        circ2=(r**2-(Na+off)**2)**0.5*2
        sel2=(x<=halfsize) * (x>=-off)
        wireprof[sel2]=circ2[sel2]
        wireprof[np.abs(x)>off]=2*r
        wireprof[np.abs(x)<halfsize]=0

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
        halfsize = float( yamlval('d2',pars,0)/2 )
        n=pars['n']
        d=pars['d']
        a=l/(d**n)
        size = float(yamlval('size',pars,-1))
        if size>0:  #estimating the d2 from effecitve size;
            par=a*np.abs(x-d)**n
            beta,delta,k,thickness_to_phaseshift=get_n(elem,params['photon_energy'])
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
    axap=params['ax_apertures']
    E=params['photon_energy']
    N=params['N']

    N2=int(N/2)
    typ=pars['shape']

    if typ in ['square','rectangle','wire','gaussian']:
        transmissionmap=get_aperture_transmission_map(pars,params,debug)
        phaseshiftmap=transmissionmap*0
        thicknessmap=transmissionmap*0
    else:
        thicknessmap = get_aperture_thickness_map(pars,params,debug)
        #print("thickness map in doap:", thicknessmap)


    #General modifications of thickness map
        if yamlval('randomizeA',pars):            #adding random defects.
            #'randomizeA' is the maximal amplitude of the noise [m] . The spatial frequency is just given by pixel sizera
            ra=float(yamlval('randomizeA',pars))
            rand=np.random.random((N,N))*ra - ra/2
            img2=thicknessmap+rand
            img2[thicknessmap==0]=0
            img2[thicknessmap<=0]=0
            thicknessmap=img2
            print('randomized')
        if yamlval('randomizeB',pars):            #adding random defects. in better way
            #'randomizeB' is the maximal radius of sphere added to the material[m].
            maxsize=float(yamlval('randomizeB',pars))
            density=float(yamlval('density',pars,2))
            print('Density: ',density)
            print(pars)
            boxsize=params['propsize']
            numsph=density*boxsize**2/maxsize**2
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
            rot_rad=pars['rot']#
            thicknessmap= rotate(thicknessmap, angle=rot_rad,reshape=0)
        if yamlval('crossed',pars,0):
            thicknessmap2=np.transpose(thicknessmap)
            thicknessmap=thicknessmap2*thicknessmap


    #CONVERTING THICKNESSMAPP INTO TRANSMISSION AND PHASESHIFTMAP
        elem=pars['elem']

        beta,delta,k,thickness_to_phaseshift=get_n(elem,E)
        #print(thicknessmap)
        print(k)
        transmissionmap=np.exp(-k*thicknessmap)
        phaseshiftmap=thicknessmap*thickness_to_phaseshift
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
        mu.figure()
    if return_thickness:
        return transmissionmap,phaseshiftmap,thicknessmap
    else:
        return transmissionmap,phaseshiftmap

def prepare_image(img,ps=750,max_pixels=300,ZoomFactor=1,log=1,norms=[0,0],el_dict=None):
    from scipy.interpolate import RegularGridInterpolator
    # pxc = pixel count : number of pixels in the image.
    #imgC = image croped : zooomed image

    import cv2
    inte=np.max(img)*ps**2 #Maximum de l'image
    suma=np.sum(img)*ps**2 #Somme de l'image
    if np.sum(norms)==0: # First passage in the loop
        norms[0]=inte # Saves the first value of the integral
        norms[1]=suma # Saves the first value of the maximum
    inte=inte/norms[0] #normalize the current value of the integral to the first one in the loop.
    suma=suma/norms[1] #normalize the current value of the maximum to the first one in the loop.
    
    #### First: cut the central region to be shown, a given by zoom factor #####
    if ZoomFactor>1:
        pxc=np.shape(img)[0]
        newpxcH=int(pxc/ZoomFactor/2)
        c=int(pxc/2)
        imgC=img[c-newpxcH:c+newpxcH,c-newpxcH:c+newpxcH]
    else:
        imgC=img
    #Second: make sure the result is not bigger then 300px size (max_pixel)
    pxc=np.shape(imgC)[0]
    if pxc>max_pixels:
        #dsize=max_pixels/pxc
        dsize=[max_pixels,max_pixels]
#        dsize=(np.array(np.shape(img))/downscale).astype('int')
        imgC= cv2.resize(imgC, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    return imgC,norms,[inte,suma] #measures = [inte,suma]

def imshow(imgC,ps=750,ZoomFactor=1,log=1,measures=[0,0],el_dict=None):
    #Plots the image normalised to 1
    if log:
        norm=colors.LogNorm()
    else:
        norm=colors.Normalize()
    if ZoomFactor>1:
        ps=ps/ZoomFactor

    ps2=ps/2/um
    extent=(-ps2,ps2,-ps2,ps2)
    imgC=imgC/np.max(imgC)
    plt.imshow(imgC,norm=norm,cmap=rofl.cmap(),extent=extent)
    plt.clim(1e-3,1)
    plt.clim(1e-5,1)
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


#############################################################################

#############################################################################

#############################################################################


## Params = all the parameters coming from the yaml file.
## F = Electric field
## I = 2D map of intensity
## measures = an array with the maximum and sum of the intensity map
## im, or imC = image prepared by the "prepare_image" function. 
## norms = [0,0] in the first passage of the loop and then norm=[integral / normalised , maximum] which gets updated at each passage in the loop.
## elements : Dictionnary of the optical elements : elements = { z position, element name, properties dictionnary}

def doit(params,elements):
    #method=params['method']
    mu.clear_times()
    mu.tick()
    fig=plt.gcf()

    method='FFT' #Simon: I like Forvard ('FFT') more. Regularize propagation also uses FFT method
    method=yamlval('method',params,'FFT')
    norms=[0,0]
    dtype=np.complex64
    params['pxsize'] = params['propsize'] / params['N']
    N=params['N']
    max_pixels=yamlval('subfigure_size_px',params,300)
    if max_pixels>N: max_pixels=N
    auto_flow=yamlval('flow_auto_save',params)
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
#            ffdir=project_dir+'/flow_figs/'+fn2+'/'
            ffdir=params['projectdir']+'flow_figs/'+params['filename']+'_auto/'
            mu.mkdir(ffdir,0)
    elements = sort_elements(elements)
    wavelength = 12398/params['photon_energy']/10*nm
    integ = 0
    F = Begin(params['propsize'],wavelength,N,dtype=dtype)
    propsize = params['propsize']
    F = GaussBeam(F, params['beamsize'],x_shift=params['gauss_x_shift'],tx=params['gauss_x_tilt'])
    F_pos = elements[0][0] #the starting position of my beamline
    figs_to_save = yamlval('figs_to_save',params,[])
    figs_to_export = yamlval('figs_to_export',params,[])
    if 'edge_damping' in params:
        do_edge_damping=1
        edge_damping_aperture = do_edge_damping_aperture(params)
    else:
        do_edge_damping=0

    figs={}
    export={}

    pi=params['fig_start']
    #trans=np.zeros(np.shape(Elements)[0]+1)
    trans=np.zeros(len(elements)+1)*np.nan
    numel=len(elements)
    intensities={}
    N2=int(N/2)
    #Na=np.arange(-N2,N2)*params['pxsize']/um
    Na=(np.arange(N)-0.5*N)*params['pxsize']/um
    profi=0
    save_parts=yamlval('save_parts',params,0)
    if save_parts:
        mu.mkdir('part')
        mu.savefig('part/'+params['filename']+'__start')
    #Simon start----
    reg_prop_dict = {"regularized_propagation": False,
                     "reg_parabola_focus": None
                     }
    #Simon end----
    for ei,E in enumerate(elements):
        z=E[0]
        el_name=E[1]  #el
        el_dict=E[2] #aperture
        print('{:} (elem.n.{:.0f})   ###'.format(el_name,ei))
    #    mu.tick('step {:.0f}: {:}'.format(ei,el))

        el_type=el_dict['type']
        if el_type=='blank':
            pi+=1
        if 'off' in el_type:
            continue
        if not yamlval('in',el_dict,1):
            if yamlval('plot',el_dict,1):
                pi+=1
            print('  skipping because out')
            continue

        delta_z=z-F_pos
    #First: propagate until the element
        if delta_z==0:
            print('skipping zero propagation')
        else:
            #Simon start----
            simple=1
            if not reg_prop_dict["regularized_propagation"]: #normal propagation
                if method=='fresnel':
                    F=Fresnel(delta_z,F)
                elif method=='FFT':
                    F=Forvard(delta_z,F)
           #     elif method=='direct':
             #       F=Forward(delta_z,propsize,N,F)
                else:
                    print("Beam did not propagate, even though it should have.")
            else: #regularized propagation
                F = rp.Forvard_reg(F, reg_prop_dict["reg_parabola_focus"],
                                   delta_z, False)
            F_pos=z
            params['pxsize'] = F.siz/N #update pixel size
            propsize = F.siz #update physical size of grid
            params['propsize'] = propsize
            if reg_prop_dict["reg_parabola_focus"] is not None:
                reg_prop_dict["reg_parabola_focus"] -= delta_z
            #Simon end----
        if np.abs(z)/mm >=2000:
            lab='{:.1f} m' .format(z/m)
        elif np.abs(z)/mm >=2:
            lab='{:.0f}mm' .format((z)/mm)
        else:
            lab='{:.1f}mm' .format((z)/mm)
        ZoomFactor=yamlval('zoom',el_dict,1)
        if ZoomFactor!=1:
            lab=lab+' ({:.0f}x)'.format(ZoomFactor)

    #Second: do the element
        def_do_plot = 1
        #Simon start----
        if el_type=='reg': #regularize propagation
            reg_prop_dict["regularized_propagation"] = True
            if 'reg-by-f' in el_dict:
                tmp = el_dict['reg-by-f']
            else:
                tmp = reg_prop_dict["reg_parabola_focus"]
            F = Lens(F, -tmp)
      #      print("Regularizing at {:.0f} m by value {:.2e}".format(F_pos,tmp))
            def_do_plot = 0
        if el_type=='dereg': #deregularize propagation
            if not reg_prop_dict["regularized_propagation"]:
                print("  You can't deregularize an already deregularized field!!!")
            else:
                reg_prop_dict["regularized_propagation"] = False
                tmp = reg_prop_dict["reg_parabola_focus"]
#                print(reg_prop_dict["reg_parabola_focus"])
                F = Lens(F, reg_prop_dict["reg_parabola_focus"])
            #    print("   Deregularizing in {:.0f} by value {:.2e}".format(F_pos,tmp))
            def_do_plot = 0
        #Simon end----

        ##extracgin regularizign for lens outside of lens
        if "reg-by-f" in el_dict:
            f=el_dict['reg-by-f']
            if reg_prop_dict["reg_parabola_focus"] is None:
                reg_prop_dict["reg_parabola_focus"] = f
                reg_prop_dict["regularized_propagation"] = True
                print(f"Regularizing by CRL in {F_pos} by value {f}")
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
                    print(f"..but still regularizing by CRL in {F_pos} by value {f}")


        if 'lens' in el_type:
            #Simon start----
            ideal=yamlval('ideal',el_dict,1)
            if ideal:
                f=el_dict['f']
                if "reg" in el_type:
                    if reg_prop_dict["reg_parabola_focus"] is None:
                        reg_prop_dict["reg_parabola_focus"] = f
                        reg_prop_dict["regularized_propagation"] = True
                        print(f"Regularizing by CRL in {F_pos} by value {f}")
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
                            print(f"..but still regularizing by CRL in {F_pos} by value {f}")
                else:
                    F = Lens(f,0,0,F)
                    #Simon end----
            aperture = yamlval('size',el_dict,0)
            if aperture==0 and 'CRL4' in el_type:
                aperture = 400e-6

            if aperture > 0: #aperture
                ap_dict={}
                ap_dict['elem']='Hf'
                ap_dict['thickness']=0.0001
                ap_dict['shape']='circle'
                ap_dict['size']=aperture
                ap_dict['invert']=1
                print(ap_dict)
                print(params)
                tmap,phasemap = doap(ap_dict,params) #Definition of the transmission map
                F = MultIntensity(tmap,F)

            if 'CRL4' in el_type: #creating all the decorations:
                Lroc = yamlval('roc',el_dict,5.0e-5)
                ab_dict = {}
                ab_dict['elem'] = 'Be'
                ab_dict['minr0'] = 0
                ab_dict['shape'] = 'parabolic_lens'
                ab_dict['size'] = aperture
                ab_dict['roc'] = Lroc
                ab_dict['double_sided'] = 1
                ab_dict['num_lenses'] = yamlval('num_lenses',el_dict,1)
                tmap2,phasemap = doap(ab_dict,params,debug=0)
                F = MultIntensity(tmap*tmap2,F)

                if not ideal:  #tohle prostě nefunguje...
                    F = MultPhase(phasemap,F)
                    print('doing real lens')
                    
            if yamlval('celestre',el_dict,1):
                cel_dict = {}
                cel_dict['defect']='celestre'
                cel_dict['type']='phaseplate'
                cel_dict['num']=yamlval('num_lenses',el_dict,1)
                phaseshiftmap=do_phaseplate(cel_dict,params)
                F=MultPhase(phaseshiftmap,F)
            if yamlval('seiboth',el_dict,1):
                seib_dict={}
                seib_dict['defect']='seiboth'
                seib_dict['type']='phaseplate'
                seib_dict['num']=yamlval('num_lenses',el_dict,1)
              #  seib_dict['num']=1
                phaseshiftmap=do_phaseplate(seib_dict,params)
                F=MultPhase(phaseshiftmap,F)
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
                    sc_dict['randomizeB']=yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['type']='aperture'
                    sc_dict['shape']='circle'
                    sc_dict['size']=aperture
                    default_k=3 #3 comes from 5348
                    default_k=0.02 #3 comes from 6436
                    k=yamlval('lens_randomize_k',params,default_k)
                    if 'scatterer_k' in el_dict:
                            k=el_dict['scatterer_k']
                    sc_dict['density']=k*yamlval('num_lenses',el_dict,1)
                    sc_dict['thickness']=3*yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['elem']=yamlval('lens_randomize_elem',params,'Ti')


                Ii1=(np.nansum(Intensity(0,F)))
                tmap3,phasemap=doap(sc_dict,params,debug=0)
                F=MultIntensity(tmap3,F)
                F=MultPhase(phasemap,F)
                Ii2=(np.nansum(Intensity(0,F)))
                loss_on_scatterer=Ii2/Ii1
                params['transmission_of_scatterer_'+el_name]=loss_on_scatterer

        if el_type=='phaseplate':
            phaseshiftmap=do_phaseplate(el_dict,params)
            F=MultPhase(phaseshiftmap,F)

        if 'aperture' in el_type:
            if len(el_dict)==0:
                do_nothing=1
            num=yamlval('num',el_dict,1)
            merged=1
            if merged:
                bt=np.zeros((N,N))+1.
                ph=np.zeros((N,N))
                for i in np.arange(num):
                    tmap,phasemap=doap(el_dict,params)
                    bt=bt*tmap
                    ph+=phasemap
                if yamlval('do_intensity',el_dict,1):
                    F = MultIntensity(bt,F)
                if yamlval('do_phaseshift',el_dict,1):
                    F = MultPhase(ph,F)
            else:
                for i in np.arange(num):
                    tmap,phasemap=doap(el_dict,params)
                    if yamlval('do_intensity',el_dict,1):
                        F = MultIntensity(tmap,F)
                    if yamlval('do_phaseshift',el_dict,1):
                        F = MultPhase(phasemap,F)



#edge damping
        if do_edge_damping:
            F=MultIntensity(edge_damping_aperture,F)

        do_plot=yamlval('plot',el_dict,def_do_plot)
        plot_phase=yamlval('plot_phase',params,0)

        ############################ COMPUTE THE INTENSITY I FROM THE FIELD F ####################
        if plot_phase:
            I=Phase(F)
            print(np.max(I))
            print("plotting phase instead of intensity")
            if 0:
                mu.figure()
                plt.imshow(ph)
                plt.colorbar()
        else:
            I=Intensity(0,F)
        ####################################################################################

        
        letts=['a','b','c','d','e','f','g','h','i']
        Iint=(np.nansum(I))*propsize**2
        intensities[el_name]=Iint
        logg=yamlval('figs_log',params,1)

        ########################### CALCULATION OF THE MAXIMUM AND INTEGRAL OF THE IMAGE #############
        #norms=[0,0] initially (1rst passage in the loop). Then, norms=[integral, sum] normalized.
        im,norms,measures=prepare_image(I,ps=propsize,max_pixels=max_pixels,ZoomFactor=ZoomFactor,log=logg,norms=norms,el_dict=el_dict) #prepare l'image
        if el_name.startswith(tuple(figs_to_save)):

            print('Saving figure: {:}'.format(el_name))
            figs[el_name]=[im,ei,propsize/ZoomFactor,z]

            if np.mod(ei,20)==0:
                print('Dumping figures')
                if np.size(figs)>0:
                    pkl_name = f"{HOME}/Aime/pickles/{params['filename']}_figs"
                    mu.dumpPickle(figs, pkl_name)
                    
        if auto_flow and 'flow' in el_name:
            fi=int(el_name.split('_')[1])
            position=float(el_name.split('_')[2])
            flow_savefig(I,ffdir,fi,propsize,params['filename'],position)
            plt.figure(fig)

        if ei==0: I0int=Iint #Initial integral of signal
        trans[ei]=Iint/I0int
        
        ######################## PLOTING THE FIGURE ######################
        if do_plot: #plotting
            lab=lab+', '+el_name
            lab="({:}) ".format(ei)+lab
    
            if np.isnan(Iint):
                print('Something wrong here (nan integral)')
                break


            ################# SUBPLOT OF EACH ELEMENT ###############
            plt.figure(fig)
            ax=plt.subplot(params['fig_rows'],params['fig_cols'],pi)
            ax.set_facecolor("black")
            pi+=1
            plt.title(lab)
            ###################### PLOT OF THE IMAGE USING IMSHOW CUSTOMIZED FUNCTION ####################
            imshow(im,ps=propsize,ZoomFactor=ZoomFactor,log=logg,measures=measures,el_dict=el_dict)
            
            if 'roi' in el_dict: #If the property "ROI" is defined in the yaml file for the detector.
                s=el_dict['roi']/2
                rect=np.array([-s,s,-s,s])
                mu.drawRect(rect,color='r')
                psum=propsize/um
                rect2=np.round((rect/psum + 0.5)*N)
                rect2=rect2.astype('int')
                ic=mu.cutRect(rect2,I)
                integ=np.sum(ic)*propsize**2 #integral of signal inside ROI
                roin=integ/norms[1]
                rp1=roin/(np.sum(I)*propsize**2/norms[1])*100
                plt.text(.02, .09, "ROI {:.1e} ({:.0f}%)".format(roin,rp1), ha='left', va='top', transform=ax.transAxes,color=[1,0.7,0.7])
                intensities['roi']=integ
            if 'roi2' in el_dict:
                s=el_dict['roi2']/2
                rect=np.array([-s,s,-s,s])
                mu.drawRect(rect,color='g')
                psum=propsize/um
                rect2=np.round((rect/psum + 0.5)*N)
                rect2=rect2.astype('int')
                ic=mu.cutRect(rect2,I)
                roi2=np.sum(ic)*propsize**2
                roi2n=roi2/norms[1]
                rp2=roi2n/(np.sum(I)*propsize**2/norms[1])*100
                plt.text(.02, .2, "ROI2 {:.1e} ({:.0f}%)".format(roi2n,rp2), ha='left', va='top', transform=ax.transAxes,color=[0.7,1,0.7])
                intensities['roi2']=roi2

            if el_name.startswith(tuple(figs_to_export)):

                print('Exporting data for : {:}'.format(el_name))
                export_size=yamlval('export_size',params,300)
                esel=np.abs(Na)<=export_size/um
                selI=I[esel,:][:,esel]
                export[el_name]=[selI,ei,z]
                export[el_name]=[I,ei,propsize]
                params['export_axis']=Na[esel]

            if not yamlval('axes',el_dict,1):
                plt.xticks([])
                plt.yticks([])

        if yamlval('profiles_subfig',params,None) is not None:
            plt.figure(fig)
            plt.subplot(params['fig_rows'],params['fig_cols'],params['profiles_subfig'])
            prof=np.sum(I,0)
            lab=el_name
            col=rofl.cmap()(1.*ei/numel)

            if yamlval('profiles_normalize',params,1):
                prof=mu.normalize(prof)
            l=plt.plot(Na,prof,label=lab,color=col)
            profi+=1
            plt.title('Intensity profiles')
            plt.ylabel('Intensity')
            plt.xlabel('Position [μm]')
            plt.gca().set_yscale('log')
            plt.xlim(yamlval('profiles_xlim',params,[0,200]))
            plt.ylim(yamlval('profiles_ylim',params,[1e-12,1e3]))


        if save_parts:
            mu.savefig('part/'+params['filename']+'__{:02.0f}'.format(ei))
        if pi>=yamlval('break_at',params,10000):break
        if yamlval('end_after',params,'asdfasdfasdf')==el_name: break
    trans[-1]=integ/I0int #Integral of signal inside ROI / initial intensity

    params['transmission']=trans
    params['intensities']=intensities
    params['integ']=integ

    if params['ax_apertures']!=None:
        plt.sca(params['ax_apertures'])
        plt.title('Apertures')
        plt.xlim(yamlval('profiles_xlim',params,[0,200]))
        plt.ylim(yamlval('apertures_ylim',params,[1e-10,2]))

    duration=mu.print_times()
    params['duration']=duration

#    plt.legend()
    if np.size(figs)>0:
        pkl_name = f"{HOME}/Aime/pickles/{params['filename']}_figs"
        mu.dumpPickle(figs, pkl_name)
    if len(export)>0:
        pkl_name = f"{HOME}/Aime/pickles/{params['filename']}_figs"
        mu.dumpPickle(figs, pkl_name)

    return params,trans,figs

# those functions copied from
#/home/michal/hzdr/XFEL2806_spectroscopy/focus_tracing/focusing2.py

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


def flow_plot(project_dir,file,cl=[1e-11,50],gyax_def=[-200,100,5],vertical_type='center',log=1,xl=None,flow_figs=0,flow_plot_crange=1e-5):
    
    cols = ['g','r','k','b',[0.5,1,0.8],[1,0.3,0.8],'r']
    mu.figure(8,4)

    gyax = np.arange(gyax_def[0],gyax_def[1],gyax_def[2]) #μm
    fn = str(file)+'_figs'
    fns = fn
    pic = mu.loadPickle('./'+project_dir+'/pickles/'+fn+'.pickle',strict=1) #loading the images
    p2 = fn.replace('figs','res')
    p2 = p2.replace('export','res')
    res = mu.loadPickle('./'+project_dir+'/pickles/'+p2+'.pickle') #loading the general parameters
    partial=(res==0)
    fn2 = fns[:-5]
    l = fn2
    scatterer_L2_position = 1e9
    scatterer_L1_position = 1e9
    skip_existing = 1
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
#    else:
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
        if fig[:4]=='flow':
            ffigs.append(fig)

    numfigs=len(ffigs)
    waterfall=np.zeros((numfigs,N))
    propsizes=np.zeros((numfigs))
    fixedfall=np.zeros((numfigs,np.size(gyax)))

    zax=np.zeros(numfigs)
    for fi,fig in enumerate(ffigs):
        picc,elemi,propsize,position=pic[fig]
       # tr=res[1]['transmission']
        pxsize=propsize*1e6/np.shape(picc)[0]
        imsize=np.shape(picc)[0]
        halfsize=int(imsize/2)
        tit=file[:-13]
    #lineout
        ZoomFactor=1
        ps2=propsize/2/ZoomFactor
        if vertical_type=='integral':
            lineout=np.mean(picc,0)
        if vertical_type=='center':
            lineout=picc[halfsize,:]
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
                mu.figure(7,7,safe=1)
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
        waterfall[fi,:] = lineout
        zax[fi] = position
        inteprpprof = np.interp(gyax,xax,lineout)
        inteprpprof[gyax<np.min(xax)]=np.nan
        inteprpprof[gyax>np.max(xax)]=np.nan
        fixedfall[fi,:] = inteprpprof
        propsizes[fi] = propsize

    fixedfall[fixedfall<=0]=1e-30
    waterfall[waterfall<=0]=1e-30

    l2=l
    l2+=' cut: '+vertical_type
    if 0:
        mu.figure(13,6)
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

    ################## PLOT OF THE MAIN FLOW FIG #####################
    mu.figure(13,6)
    linearize = 0 #linearize option is used in the case where the X and Y axis are not linear. (Don't activate it)
    if linearize:
        mu.pcolor(xc=zax,yc=gyax,data=fixedfall,log=1,ticks=0,cl=cl,linearize=1) #,xtics_spacing=1
        pos=np.arange(0,np.size(zax),15)
        vals=[]
        for va in zax[pos]:
            vals.append('{:.1f}'.format(va))
        plt.xticks(pos,vals)
        plt.yticks(np.arange(0,np.size(gyax)+1,50))
    else:
        mu.pcolor(xc=zax,yc=gyax,data=fixedfall,log=1,ticks=0,cl=cl) ########### MAIN PLOT ##########
    profile=mu.normalize(propsizes)*np.max(gyax)
    maxy=np.min(gyax)
    print(partial)
    if not partial:

        ip = res[0]
        row = 0
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
    # % SFA
        centralelement='TCC'
        if centralelement not in params['intensities']:
            centralelement='PH'
        infox=(np.max(zax)-np.min(zax))*0.8+np.min(zax)
        fs=14
        if centralelement in params['intensities']:
            tr_scat=yamlval('transmission_of_scatterer_L2',params,1)
            t1=params['intensities'][centralelement]/params['intensities']['start']
            #plt.text(infox,maxy*0.5,'start->'+centralelement+' = {:.1e}'.format(t1),fontsize=12,color='w')
            if 'roi' in params['intensities']:
                t2=params['intensities']['roi']/params['intensities'][centralelement]/tr_scat
                t75=params['intensities']['roi2']/params['intensities'][centralelement]/tr_scat
                plt.text(infox,maxy*0.6,'SFA13 = {:.2e}'.format(t2),fontsize=18,color='r')                
                mu.text(infox,maxy*0.55,'SFA75 = {:.2e}'.format(t75),fs=18,color='k')
                
                mu.text(infox,maxy*0.5,'SFA75/SFA13 = {:.0f}'.format(t75/t2),fs=12,color='k')

        plt.xlabel('Position [m]')
        plt.ylabel('Horizontal position [μm]')
        if xl==None:
            plt.xlim(np.min(zax),np.max(zax))
        else:
            plt.xlim(xl)
        plt.plot(zax,profile,'r-')
    roi=13
    plt.plot([zax[-1],zax[-1]],[-roi/2,roi/2],'w-',lw=5)
    plt.title(l2)
    mu.savefig('./'+project_dir+'/flows/flow_{:}_{:}'.format(l,vertical_type))
    return params,res



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
