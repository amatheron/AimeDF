from LightPipes import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from astropy.io import ascii
from PIL import Image

import darkfield.regularized_propagation as rp
import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils as mu


HOME = '/home/qo38soh/darkfield_p5438/'
def elem2Z(elem):
    if elem=='Be':   return 4
    if elem=='C':   return 6
    if elem=='CH':   return 5
    if elem=='CH6':   return 4.5
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
    obj['dophaseshift']=1

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
    assert E==8766, print("forcing E as 8766eV, don't have other constants")
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
        if elem=='CH':  #"5"
            delta=6.406176E-06
            beta=7.60660246E-09
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


    assert beta!=-1, "Index of refraction not found ({:}, {:} eV)".format(elem,E)
    c=3e8#m/s

    lambd=12398/E*1e-10 #nmn go to ~<
    k=beta/lambd*4*3.14 #based on https://henke.lbl.gov/optical_constants/intro.html
    thickness_to_phaseshift=(delta)/(c*(1-delta)) *c /lambd* 2*3.14
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
        print(par)
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
    print('phaseplate')

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
        fia=ascii.read("Seiboth_Fig4")
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
    phaseshiftmap=thickness*thickness_to_phaseshift*-1

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




'''
shapes available:
    with realistic 2D depth profiles:
        circle
        parabolic_lens
    with realistic 1D depth profils ('like wires'):
        realwire
        trapez
        pooyan
        tent
        customwire
'''

def doap(pars,params=[],debug=0,just_return_profile=0):
    axap=params['ax_apertures']
    E=params['photon_energy']
    N=params['N']
    pxsize=params['pxsize']
    print(pxsize)
    if not just_return_profile:
        img=np.zeros([N,N])+1
        phaseshiftmap=img*0

    wls=['realwire','trapez','tent','customwire','pooyan','invpoo','invpar','par','wireslit','linearslit', 'realslit']
    wls2=['circle','parabolic_lens']


    N2=int(N/2)
    Na=np.arange(-N2,N2)*pxsize

    mx=N2*pxsize
    typ=pars['shape']

    if just_return_profile and typ not in wls and typ not in wls2:
        print('Not going to return your profile')
        return Na,Na*0,Na*0
    xm,ym=np.meshgrid(Na,Na)
    r=((xm**2)+(ym**2))**0.5

    if typ=='square': #the old ideal things
        hs=pars['size']/2
        sel=(xm>=-hs)*(xm<=hs)*(ym>=-hs)*(ym<=hs)
        img[sel]=0
    if typ=='rectangle': #the old ideal things
        hs=pars['size']/2
        vs=pars['sizevert']/2
        sel=(xm>=-hs)*(xm<=hs)*(ym>=-vs)*(ym<=vs)
        img[sel]=0
    if typ=='wire':
        hs=pars['size']/2
        sel=(xm>=-hs)*(xm<=hs)
        img[sel]=0
    if yamlval('invert',pars):
        img=1-img

    if yamlval('rot',pars,0)!=0:
        from scipy.ndimage.interpolation import rotate
        img= rotate(img, angle=pars['rot'],reshape=0)
    if yamlval('smooth',pars,0)!=0:
        smpx=pars['smooth']/pxsize
        img=mu.smooth2d(img,smpx)


    if typ in ['circle','gaussian','parabolic_lens']:  #realistic 2D-depth maps
        img=np.zeros([N,N])+0
        debug=0
        if typ=='circle':
            rad=pars['size']/2
            img[r<rad]=1
            if pars['invert']:
                img=1-img
            img=img*pars['thickness']


        if typ=='gaussian':
            rax=np.arange(0,2*N2)*pxsize# to be in m
            print(pxsize)
            power=pars['power']
            fwhm=pars['size']
            gauss=mu.gauss(rax,0,1,0,fwhm)
            gauss=gauss**power

            for xi,x in enumerate(xm):
                for yi,y in enumerate(ym):
                    val=np.interp(r[xi,yi],rax,gauss)
                    img[xi,yi]=val
            if pars['invert']:
                img=1-img
            if 0:
                mu.figure()
#                plt.plot(rax,gauss)
                plt.imshow(img)
                plt.clim(0,1)
                plt.colorbar()
                mu.figure()
            phaseshiftmap=img*0
            if params['ax_apertures']!=None:
                plt.sca(params['ax_apertures'])
                lab=pars['shape']
                trans1=img[N2,:]
                plt.semilogy(rax/um,gauss,label=lab)
            return img,phaseshiftmap

        if typ=='parabolic_lens':
            rax=np.arange(0,2*N2)*pxsize
            r0=pars['size']/2
            roc=pars['roc']
            prof=parabolic_lens_profile(rax,roc,r0,pars['minr0'],plot=0)
            for xi,x in enumerate(xm):
                for yi,y in enumerate(ym):
                    val=np.interp(r[xi,yi],rax,prof)
                    img[xi,yi]=val
#            if pars['invert']:
 #               img=1-img
  #          img=img*pars['thickness']
            if pars['double_sided']:
                img=img*2
            img=img*pars['num_lenses']
        if yamlval('smooth',pars)!=0:
            smpx=pars['smooth']/pxsize
            img=mu.smooth2d(img,smpx)
        elem=pars['elem']
        beta,delta,k,thickness_to_phaseshift=get_n(elem,E)
        phaseshiftmap=img*thickness_to_phaseshift#*-1
        imgL=img*1
        img=np.exp(-k*img) #transmission

        if just_return_profile:
            print('Tu jsem')
            prof=imgL[N2,:]
            phaseshiftprof=phaseshiftmap[N2,:]
            return Na,prof,phaseshiftprof

    #realistic wire-like structures
#    if typ.find('realwire')==0 or typ.find('trapez')==0 or typ.find('tent')==0 or typ.find('customwire')==0 or typ.find('pooyan')==0 or typ.find('invpoo')==0 or typ.find('invpar')==0 :
    if typ in wls :
        r=pars['size']/2
        off=yamlval('offset',pars,0)
        shape='circle'
        elem=pars['elem']
        crossed=0
        x=Na-off
#here I need to find the wireprof - i.e. 1D profile of thickness on 'x' as an x-axis
        # if typ.find('trapez')==0:
        #     thickness=pars['thickness']
        #     edge=pars['edge']
        #     g=r/edge
        #     wireprof=g*r-g*np.abs(x)
        #     wireprof=thickness/edge*(r-np.abs(x))
        #     wireprof[wireprof<0]=0
        #     wireprof[wireprof>thickness]=thickness
        if typ.find('trapez')==0:
            thickness=pars['thickness']
            edge=pars['edge']
            g=r/edge
            wireprof=thickness/edge*(r-np.abs(x))
            wireprof[wireprof<0]=0
            wireprof[wireprof>thickness]=thickness
            sel1=(x>=halfsize) * (x<=off)       
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
            #circ1=(r**2-(Na-off)**2)**0.5*2
            circ1=r-(Na-off))
            sel1=(x>=halfsize) * (x<=off)
            wireprof[sel1]=circ1[sel1]
            #circ2=(r**2-(Na+off)**2)**0.5*2
            circ2=r-(Na+off)
            sel2=(x<=halfsize) * (x>=-off)
            wireprof[sel2]=circ2[sel2]
            wireprof[np.abs(x)>off]=2*r
            wireprof[np.abs(x)<halfsize]=0

#            wireprof[x>r]=0


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
        elif typ.find('realslit')==0:
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
#            mu.figure()
 #           plt.plot(x,circ)
            wireprof[np.abs(x)<=d2/2]=l2+l1

            ss=np.logical_and(x<(-d2/2),x>(-d1/2))
            wireprof[ss]=l1-circ[ss]

            ss=np.logical_and(x>(d2/2),x<(d1/2))
            circ=(r**2-(x+circ_cen)**2)**0.5
            wireprof[ss]=l1-circ[ss]

            wireprof[wireprof<0]=0

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
            halfsize=pars['d2']/2
            n=pars['n']
            d=pars['d']
            a=l/(d**n)
            wireprof=x*0
            wireprof[np.abs(x)<=halfsize]=l

            par=a*np.abs(x-(halfsize+d))**n
            #print('Parabolic obstacle parameter a={:.2f} m-1'.format(a))
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

#            wireprof[ss]=l1-circ[ss]

        elif typ.find('realslit')==0:
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
#            mu.figure()
 #           plt.plot(x,circ)
            wireprof[np.abs(x)<=d2/2]=l2+l1

            ss=np.logical_and(x<(-d2/2),x>(-d1/2))
            wireprof[ss]=l1-circ[ss]

            ss=np.logical_and(x>(d2/2),x<(d1/2))
            circ=(r**2-(x+circ_cen)**2)**0.5
            wireprof[ss]=l1-circ[ss]

            wireprof[wireprof<0]=0
            

        elif typ.find('realwire')==0: #round wire
            if shape=='circle':
                wireprof=(r**2-(Na-off)**2)**0.5*2
            elif shape=='mist':

                maxi=r*2*np.sqrt(2)
                wireprof=maxi-2*np.abs(x)
                wireprof[wireprof<0]=0
        else:
            assert 1, "Obstacle type not found"


        wireprof[np.isnan(wireprof)]=0
        if yamlval('smooth',pars)!=0:
            smpx=pars['smooth']/pxsize
            wireprof=mu.convolve_gauss(wireprof,smpx,1)
            ee=int(smpx*2)
            wireprof[0:ee]=wireprof[ee+1]
            wireprof[-ee:]=wireprof[-(ee+1)]


        if yamlval('invert',pars):
            maxi=np.max(wireprof)
            wireprof=maxi-wireprof

        beta,delta,k,thickness_to_phaseshift=get_n(elem,E)
        phaseshift=wireprof*thickness_to_phaseshift
##INVERTING
        if yamlval('invertphaseshift',pars):  #inverting phase-shift-wise
            if yamlval('inversiontype',pars)=='thickness':
                print('inverting by thickness')
                maxth=np.max(wireprof)
                wireprof=maxth-wireprof
                newelem=pars['newelem']
                beta,delta,k,thickness_to_phaseshift=get_n(newelem,E)
                phaseshift=wireprof*thickness_to_phaseshift

            else:
                newelem=pars['newelem']
                if yamlval('invertphaseshift-maxps',pars):
                    maxpp=yamlval('invertphaseshift-maxps',pars)
                else:
                    maxpp=np.max(phaseshift)
                newpp=maxpp-phaseshift
                newpp[newpp<0]=0
                newbeta,newdelta,k,newthickness_to_phaseshift=get_n(newelem,E)
                wireprof=newpp/newthickness_to_phaseshift
                #modifying new prof
     #           if 1:
#                    mu.figure()
    #                off=yamlval('offset',pars)
   #                 print(pars)
  #                  print(off)
 #                   off=0e-6
#                    wireprof=np.interp(x,x-off,wireprof)
 #                   plt.plot(x,wireprof)
  #                  mu.figure()

                print(np.shape(wireprof))
                phaseshift=newpp
                thickness_to_phaseshift=newthickness_to_phaseshift

        if just_return_profile:
            return Na,wireprof,phaseshift

        ones=Na*0+1

        if yamlval('randomizeA',pars):            #adding random defects.
            #'randomizeA' is the maximal amplitude of the noise [m] . The spatial frequency is just given by pixel sizera
            ra=float(yamlval('randomizeA',pars))
            rand=np.random.random((N,N))*ra - ra/2
            thicknessmap=np.matmul(np.transpose(np.matrix(wireprof)),(np.matrix(ones)))
            img2=thicknessmap+rand
            img2[thicknessmap==0]=0
            img2[thicknessmap<=0]=0
            thicknessmap=img2
            if 0:
                mu.figure()
                print(np.shape(thicknessmap))
                plt.imshow(thicknessmap)
                mu.figure()
            img=np.exp(-k*thicknessmap)
            phaseshiftmap=thicknessmap*thickness_to_phaseshift
            print('randomized')
        if yamlval('randomizeB',pars):            #adding random defects. in better way
            #'randomizeB' is the maximal radius of sphere added to the material[m].
            maxsize=float(yamlval('randomizeB',pars))
            density=2
#            density=1e-1
            boxsize=params['propsize']

            numsph=density*boxsize**2/maxsize**2
            thicknessmap=np.matmul(np.transpose(np.matrix(wireprof)),(np.matrix(ones)))
            for i in np.arange(numsph):
                size=np.random.random()*maxsize
                xr=np.random.random()*(boxsize-2*size-2*pxsize)
                yr=np.random.random()*(boxsize-2*size-2*pxsize)
                positive=np.random.random()>0.3
                add_sphere(size,xr,yr,thicknessmap,pxsize,positive)

            if 0:
                mu.figure()
                mx=N2*pxsize
                ex=[-mx/um,mx/um,-mx/um,mx/um]
                plt.imshow(thicknessmap/um,extent=ex,cmap=rofl.cmap())
                plt.colorbar()
                plt.clim(0,20)
                mu.figure()
            img=np.exp(-k*thicknessmap)
            phaseshiftmap=thicknessmap*thickness_to_phaseshift
            print('randomized B')
        else: #doing this the 1D way
            trans=np.exp(-k*wireprof)
            #Expanding 1D profile into 2D:
            img=np.matmul(np.transpose(np.matrix(trans)),(np.matrix(ones)))
            phaseshiftmap=np.matmul(np.transpose(np.matrix(phaseshift)),(np.matrix(ones)))
            phaseshiftmap=phaseshiftmap*-1 #minus was tested to be good


        if yamlval('rot',pars)==0:
            img=np.array(np.transpose(img))
            phaseshiftmap=np.array(np.transpose(phaseshiftmap))
        elif yamlval('rot',pars)==90:
            img=np.array(img)
            phaseshiftmap=np.array(phaseshiftmap)
        else:
            print('nooooo way now')

        if crossed:
            img2=np.transpose(img)
            img=img2*img
        if 0:
            plt.sca(axap)
            lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
            plt.semilogy(Na/um,trans,label=lab)
            plt.ylabel('Transmission [-]')
            plt.xlabel('Position [μm]')
        if debug:
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

#            plt.plot(Na/um,wireprof/(c)*1e15)
 #           plt.ylabel('wire penetration time [fs]')
#            plt.plot(Na/um,deltat*1e15)
 #           plt.ylabel('time shift [fs]')
 #           plt.plot(Na/um,deltas/nm)
#            plt.ylabel('position shift [nm]')
            plt.plot(Na/um,phaseshift)
            plt.ylabel('phase shift [rad]')

#            plt.ylabel('phase shift [-]')
            plt.xlabel('position [μm]')


    img[img<0]=0
    img[img>1]=1

    if axap!=None:# and pars['shape']!='circle':
        plt.sca(axap)
        drawthis=1
        if drawthis:
            lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
            lab=pars['shape']
            if yamlval('invert',pars): lab=lab+', inv.'
            if typ.find('trapez')==0:
                lab=lab+", {:.0f}/{:.0f}".format(pars['thickness']/um,pars['edge']/um)
            trans1=img[N2,:]
            plt.semilogy(Na/um,trans1,label=lab)
            plt.ylabel('Transmission [-]')
            plt.xlabel('position [μm]')

    if  0:
        mu.figure()
        ax=plt.subplot(121)
        plt.title('Transmission')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(img,extent=ex)
        plt.colorbar()
        plt.clim(0,1)

        ax=plt.subplot(122)
        plt.title('phase shift')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(phaseshiftmap,extent=ex)
        plt.colorbar()
        #plt.clim(0,10)
        mu.figure()

    if not yamlval('dophaseshift',pars):
        phaseshiftmap=phaseshiftmap*0
    return img,phaseshiftmap


def imshow(img,ps=750,max_pixels=300,ZoomFactor=1,log=1):
    from scipy.interpolate import RegularGridInterpolator

    import cv2
    inte=np.max(img)*ps**2
    suma=np.sum(img)*ps**2
    img=img/np.max(img)
    #First: cut the central region to be shown, a given by zoom factor

    if ZoomFactor>1:
        pxc=np.shape(img)[0]
        newpxcH=int(pxc/ZoomFactor/2)
        c=int(pxc/2)
        imgC=img[c-newpxcH:c+newpxcH,c-newpxcH:c+newpxcH]
        ps=ps/ZoomFactor
    else:
        imgC=img
    ps2=ps/2/um
    #Second: make sure the result is not bigger then 300px size (max_pixel)
    pxc=np.shape(imgC)[0]
    if pxc>max_pixels:
        #dsize=max_pixels/pxc
        dsize=[max_pixels,max_pixels]
#        dsize=(np.array(np.shape(img))/downscale).astype('int')
        imgC= cv2.resize(imgC, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    if log:
        norm=colors.LogNorm()
    else:
        norm=colors.Normalize()

    extent=(-ps2,ps2,-ps2,ps2)

    plt.imshow(imgC,norm=norm,cmap=rofl.cmap(),extent=extent)
    plt.clim(1e-3,1)
    ax=plt.gca()
    if ps/um >=10:
        plt.text(.01, .99, "{:.0f} μm".format(ps/um), ha='left', va='top', transform=ax.transAxes,color='w')
    else:
        plt.text(.01, .99, "{:.1f} μm".format(ps/um), ha='left', va='top', transform=ax.transAxes,color='w')
    plt.text(.99, .99, "M {:.1e}".format(inte), ha='right', va='top', transform=ax.transAxes,color='w')
    plt.text(.99, .89, "S {:.1e}".format(suma), ha='right', va='top', transform=ax.transAxes,color='w')
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



def doit(params,elements):
    #method=params['method']
    method='FFT' #Simon: I like Forvard ('FFT') more. Regularize propagation also uses FFT method
    dtype=np.complex64
    params['pxsize']=params['propsize']/params['N']
    N=params['N']
#    downscale=np.max([(N/300),1])
    elements=sort_elements(elements)
    wavelength=12398/params['photon_energy']/10*nm
    integ=0
    F=Begin(params['propsize'],wavelength,N,dtype=dtype)
    propsize=params['propsize']
    F=GaussBeam(F, params['beamsize'],x_shift=params['gauss_x_shift'],tx=params['gauss_x_tilt'])
    F_pos=elements[0][0] #the starting position of my beamline
    figs_to_save=params['figs_to_save']
    figs_to_export=params['figs_to_export']
    figs={}
    export={}

    pi=params['fig_start']
    #trans=np.zeros(np.shape(Elements)[0]+1)
    trans=np.zeros(len(elements)+1)
    intensities={}
    N2=int(N/2)
    Na=np.arange(-N2,N2)*params['pxsize']/um
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
            pi+=1
            print('  skipping because out')
            continue
        #Simon start----
        if el_type=='reg': #regularize propagation
            reg_prop_dict["regularized_propagation"] = True
            tmp = reg_prop_dict["reg_parabola_focus"]
            F = Lens(F, -tmp)
            print(f"   Regularizing in {F_pos} by value {tmp}")
            #pi+=1 #comment so plots dont get skipped
            continue
        if el_type=='dereg': #deregularize propagation
            if not reg_prop_dict["regularized_propagation"]:
                print("  You can't deregularize an already deregularized field!!!")
            else:
                reg_prop_dict["regularized_propagation"] = False
                tmp = reg_prop_dict["reg_parabola_focus"]
                print(reg_prop_dict["reg_parabola_focus"])
                F = Lens(F, reg_prop_dict["reg_parabola_focus"])
                print(f"   Deregularizing in {F_pos} by value {tmp}")
            #pi+=1 #comment so plots dont get skipped
            continue
        #Simon end----

        delta_z=z-F_pos
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
        if 'lens' in el_type:
            #Simon start----
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
            else:
                F=Lens(f,0,0,F)
                    #Simon end----
        if el_type=='phaseplate':
            phaseshiftmap=do_phaseplate(el_dict,params)
            F=MultPhase(phaseshiftmap,F)

        if 'aperture' in el_type:
            if len(el_dict)==0:
                do_nothing=1
#            elif type(el_dict)!=dict:
 #               for i,ap in enumerate(el_dict):
  #                  if i==0:
   #                     tmap,phasemap=doap(ap,params)
    #                else:
     #                   tmap2,phasemap2=doap(ap,params)
      #                  tmap=tmap*tmap2
       #                 phasemap=phasemap+phasemap2
        #        do_nothing=0
         #   else:
            tmap,phasemap=doap(el_dict,params)
            do_nothing=0
            if not do_nothing:
                F=MultIntensity(tmap,F)
                if yamlval('do_phaseshift',el_dict,1):
                    F=MultPhase(phasemap,F)

        I=Intensity(0,F)
        letts=['a','b','c','d','e','f','g','h','i']
        do_plot=yamlval('plot',el_dict,True)
        Iint=(np.nansum(I))*propsize**2
        intensities[el_name]=Iint
        logg=yamlval('figs_log',params,1)
        if do_plot: #plotting
            if ei==0: I0int=Iint
            lab=lab+', '+el_name
       #     if not params['compact_figure']:
        #        lab=lab+',  {:.1e}'.format(Iint/I0int)
         #   else:
          #      lab=letts[pi-1]+ ') '+lab
            trans[ei]=Iint/I0int
            if np.isnan(Iint):
                print('Something wrong here (nan integral)')
                break

            ax=plt.subplot(params['fig_rows'],params['fig_cols'],pi)
            ax.set_facecolor("black")
            pi+=1
            plt.title(lab)

    # %%
            print('    prop. size: {:.0f} μm'.format(propsize*1e6))
            im=imshow(I,ps=propsize,max_pixels=params['max_pixels'],ZoomFactor=ZoomFactor,log=logg)
            if not yamlval('axes',el_dict,0):
                plt.xticks([])
                plt.yticks([])
            if 'roi' in el_dict:
                s=el_dict['roi']
                rect=np.array([-s,s,-s,s])
                mu.drawRect(rect,color='r')
                psum=propsize/um
                rect2=np.round((rect/psum + 0.5)*N)
                rect2=rect2.astype('int')
                ic=mu.cutRect(rect2,I)
                integ=np.sum(ic)*propsize**2
                plt.text(.02, .09, "ROI {:.1e}".format(integ), ha='left', va='top', transform=ax.transAxes,color=[1,0.7,0.7])
                intensities['roi']=integ
            if 'roi2' in el_dict:
                s=el_dict['roi2']
                rect=np.array([-s,s,-s,s])
                mu.drawRect(rect,color='g')
                psum=propsize/um
                rect2=np.round((rect/psum + 0.5)*N)
                rect2=rect2.astype('int')
                ic=mu.cutRect(rect2,I)
                roi2=np.sum(ic)*propsize**2
                plt.text(.02, .2, "ROI2 {:.1e}".format(roi2), ha='left', va='top', transform=ax.transAxes,color=[0.7,1,0.7])
                intensities['roi2']=roi2
            if el_name.startswith(tuple(figs_to_save)):
                print('Saving figure: {:}'.format(el_name))
                figs[el_name]=[im,ei,propsize/ZoomFactor]
            if el_name.startswith(tuple(figs_to_export)):

                print('Exporting data for : {:}'.format(el))
#                Na=np.arange(-N2,N2)*params['pxsize']/um
                export_size=params['export_size']
                esel=np.abs(Na)<=export_size/um
#                selI=I[esel,esel]
                selI=I[esel,:][:,esel]
                #cutting
                export[el]=[selI,ei,z,el]
                params['export_axis']=Na[esel]

        if yamlval('ax_profiles',params) and params['ax_profiles']!=None: #horizontal profiles
            plt.sca(params['ax_profiles'])
            prof=np.sum(I,0)
            lab=el_name
            col=mu.colors[profi]
            if params['profiles_normalize']:
                prof=mu.normalize(prof)
            l=plt.plot(Na,prof,label=lab,color=col)
            plt.ylim(params['profiles_ylim'])
            mu.text_at_plot(l,-30+profi*10,lab,fs=10,background=[1,1,1,0.6])
            profi+=1
            plt.title('Intensity profiles')
            plt.ylabel('Intensity')
            plt.xlabel('Position [μm]')
            params['ax_profiles'].set_yscale('log')
            plt.xlim(yamlval('profiles_xlim',params,[0,200]))
            #plt.xlim(0,200)



#        mu.savefig(params['filename'])
        if save_parts:
            mu.savefig('part/'+params['filename']+'__{:02.0f}'.format(ei))
        if pi>=params['break_at']:break
    trans[-1]=integ/I0int

    params['transmission']=trans
    params['intensities']=intensities
    params['integ']=integ

    if params['ax_apertures']!=None:
        plt.sca(params['ax_apertures'])
        plt.title('Apertures')
        plt.xlim(yamlval('profiles_xlim',params,[0,200]))
        plt.ylim(yamlval('apertures_ylim',params,[1e-10,2]))

#    plt.legend()
    if np.size(figs)>0:
        #mu.dumpPickle(figs,params['projectdir']+'pickles/'+params['filename']+'_figs')
        fig_path = os.path.join(params['projectdir'],'pickles',params['filename']+'_figs')
        mu.dumpPickle(figs,fig_path)
    if len(export)>0:
        print(export)
        print(len(export))
        print(np.size(export))
        exp_path = os.path.join(params['projectdir'],'pickles',params['filename']+'_export')
        mu.dumpPickle([export,params],exp_path)
        
        #mu.dumpPickle([export,params],params['projectdir']+'pickles/'+params['filename']+'_export')



    return params,trans

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
