"""
Calculator for CRL focus positions and optimisation for the HED instrument.
Based on the work of Thomas Preston (c) 2020, European X-Ray Free-Electron Laser Facility GmbH
Updated by Oliver Humphries 2022

Updated 2024-12-09 for Diamond lenses in CRL3 arm 7
"""

import sys
import numpy as np
import pandas as pd
import copy
import itertools
from scipy.optimize import minimize
import os

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, \
QGridLayout, QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout, QSpinBox, \
QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QComboBox, \
QTabWidget, QFormLayout, QPushButton, QProgressBar
from PyQt5.QtGui import QIcon, QDoubleValidator, QGuiApplication

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

"""Physical Constants"""
classical_elec = 2.8179e-15 # m classical electron radius
z_Be = 4. # Be charge
rho_Be = 1.85 # g/cc
z_C = 6
rho_C = 3.5
atomw_C = 12
atomw_Be = 9.012 # g/mol
Avogadro = 6.022e23 # mol-1
natom_Be = rho_Be*Avogadro/atomw_Be # cm-3 ion density of Be
natom_C = rho_C * Avogadro / atomw_C


"""Fixed lens parameters and component positions
# Reference - https://confluence.desy.de/pages/viewpage.action?pageId=137171886
"""

lens_radius1 = [np.inf,  5.8,  5.0,  4.0,  3.5,  5.8,  4.0,  4.0,  2.0, np.inf]
num_lenses1  = [     0,    1,    1,    1,    1,    2,    3,    7,    7,      0]
aperture1    = [     4, 3.80, 3.53, 3.16, 2.96, 3.80, 3.16, 3.16, 2.76,      4]

lens_radius2 = [   5.8,  5.0,  4.0,  3.5,  5.8,  5.8,  5.8,  4.0,  3.5,    2.0]
num_lenses2  = [     1,    1,    1,    1,    2,    4,    7,   10,   10,      8]
aperture2    = [   3.8, 3.53, 3.16, 2.96, 3.80, 3.80, 3.80, 3.16, 2.96,   2.76]
# lens_radius3 = [   5.8,  5.8,  5.8,  4.0,  2.0,  1.0,  0.5,  0.5,  0.5,    5.8]
# num_lenses3  = [     1,    3,    4,   10,   10,   10,   10,   10,   10,      2]
# lens_radius3 = [   5.8,    np.array([   4, 0.5]),  5.8,  4.0,  2.0,  1.0,  0.5,  0.5,  0.5,    5.8]
# num_lenses3  = [     1,    np.array([   4,   4]),    4,   10,   10,   10,   10,   10,   10,      2]

# lens_radius3 = [   5.8,  5.8,  5.8,  4.0,  2.0,  1.0,  0.6,  0.5,  0.5,    5.8]
# num_lenses3  = [     1,    7,    4,   10,   10,   10,    3,   10,   10,      2]
# aperture3    = [   3.8,  3.8,  3.8, 3.16, 2.76, 1.95,  1.2, 1.38, 1.38,    3.8]
# lens_material3 = ['Be', 'Be', 'Be', 'Be', 'Be', 'Be',  'C', 'Be', 'Be',   'Be']

lens_radius3 = [   5.8,  5.8,  5.8,  4.0,  2.0,  0.5,  0.2,  0.5,  5.8,    5.8]
lens_radius3 = [   58,    58,   58,   40 ,  40,   40,  0.2,  40,  40 ,    40]
num_lenses3  = [     0,    7,    4,   10,   10,    5,    1,    10,    1,      2]
aperture3    = [   3.8,  3.8,  3.8, 3.16, 2.76, 1.95,  1.2,  1.38, 1.38,    3.8]
lens_material3 = ['Be', 'Be', 'Be', 'Be', 'Be', 'Be',  'C',  'Be', 'Be',   'Be']
    # Updated 2025-03-07
    #  - Arm 1 removed
    #  - Lenses from arm 1 moved to arm 9
    #  - Arm 6 exchanged for 5 x 0.5mm
    #  - Arm 7 exchanged for strongly focusing diamond lenses (1 lens less than original stack)

lens_radius4 = [ .125]
num_lenses4  = [ 3]
aperture4    = [.35]
lens_material4= ['C']

data = [[lens_set+1, arm+1, radius, num_lenses, aperture*1e3, ('Be' if lens_set+1 != 3 else lens_material3[arm])]  \
    for lens_set, (r, n, a) in enumerate(zip([lens_radius1, lens_radius2, lens_radius3, lens_radius4],
                                             [num_lenses1, num_lenses2, num_lenses3, num_lenses4],
                                             [aperture1, aperture2, aperture3, aperture4])) 
    for arm, (radius, num_lenses, aperture) in enumerate(zip(r,n,a))]

data[-1][-1]=lens_material4[0]    

CRL = pd.DataFrame(data, columns=['Set', 'Arm', 'Radius', 'NumberOfLenses', 'Aperture', 'Material'])
CRL = CRL.set_index(['Set', 'Arm'])

mu = chr(0x03bc)

DefaultValues = {
                'Energy': (8766, 'eV'),   # default energy (eV)
                'Bandwidth': (0.2, '%'),
                'Source position': (-41, 'm'),
                'Beam divergence': (2.1, mu+'rad'),
                'CRL3 z shift': (0, 'mm'),
                'CRL4 z shift': (-1250, 'mm'),
                'CRL4 lenses': (3, 'mm'),
                'FEL imager size': (200, mu+'m'),
                'HED imager size (w/ lens)': (500, mu+'m'),
                }

ComponentPositions = pd.Series({
                    'CRL1': 229,
                    'CRL2': 857.8,
                    'CRL3': 962.325,
                    'TCC1': 971.3,
                    'TCC2': 974.985,
                    'TCC-DAC': 975.415,
                    'FEL imager': 242,
                    'Mirror 1': 290,
                    'Mirror 2': 301.36,
                    'Mirror 3': 390,
                    'Pop-in 1': 303.682,
                    'Pop-in 2': 400,
                    'Mono': 853.6,
                    'HR-mono': 855.8,
                    'HED pop-in': 939,
                    'XTD6 shutter': 940,
                    'OPT shutter': 968,
                    'IBS Beamstop': 980,
                    })


"""Formulae to calculate focal length for radius of curvature for Beryllium"""

def image_dist(obj, foc):
    """Return image distance for object distance and focal length."""
    if foc == 0.:
        foc = np.inf
    return (1./foc - 1./obj)**-1

# https://en.wikipedia.org/wiki/Diffraction-limited_system
def diffr_lim(energy, beam_div):
    """Returns diffraction limited size in um for a beam of energy (eV) and divergence (urad)."""
    return 0.5*nmtoev(energy)*1e-3/np.sin(beam_div*1e-6)

# https://en.wikipedia.org/wiki/Rayleigh_length
def rayl_len(energy, beam_sz):
    """Returns Rayleigh length in mm for diffraction limited beam size in um for a beam of energy (eV)."""
    return np.pi*(0.5*beam_sz)**2/nmtoev(energy)


def free_space_matrix(dist):
    """https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis for free space propogation"""
    return np.array([[1.0, dist], 
                     [0.0, 1.0]])

def lens_matrix(flen):
    """https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis for thin lens."""
    if flen != 0:
        return np.array([[1.0, 0.0], 
                         [-1.0/flen, 1.0]])
    return np.array([[1.0, 0.0], 
                     [0.0, 1.0]])

"""Ray propogation with matrix formalism."""
def ray_trans_matrix(position, source_pos, CRL_f, CRL_pos):
    """Only accepts single positions (not arrays) and calculates the ray transfer matrix:
    https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis"""

    mat = np.eye(2)
    for crl, crl_pos in CRL_pos.items():
        mat = free_space_matrix(min(position, crl_pos) - source_pos) @ mat
        source_pos = crl_pos
        if position < crl_pos:
            break
        mat = lens_matrix(CRL_f[crl]) @ mat
    else:
        mat = free_space_matrix(position - source_pos) @ mat
    return mat


def ray_propogation(beam, CRL_pos, CRLs_in, TestPosition):
    """Ray propogation and calculation of beam sizes along beamline. Returns beam sizes at positions along beamline."""
    
    key_comps = copy.deepcopy(ComponentPositions)
    key_comps['Chosen position'] = TestPosition
    Bandwidth_eV = 0.5*1e-2*beam.Bandwidth*beam.Energy # Convert to HWHM in eV
    Bandwidth_eV += 1e-3*(Bandwidth_eV==0)
    Energies = beam.Energy + Bandwidth_eV*np.arange(-1,2)
    init_vec = np.array([0, beam['Beam divergence']]) # Initial beam vector in um, urad
    
    CRL_f, ImageDist, Rayleigh, DiffLim = dict(), dict(), dict(), dict()

    SamplePositions = np.hstack((sorted(ComponentPositions),
                                np.arange(beam['Source position'],ComponentPositions['IBS Beamstop'],100),
                                sorted(CRL_pos.values())))

    crls_in = [crl for crl, lenses in CRLs_in.items() if len(lenses)]

    LensPosition = [CRL_pos[crl] for crl in crls_in] + [ComponentPositions['IBS Beamstop']]
    BeamSizeAtLens, isFocusing = dict(), dict()
    for E in Energies:
        CRL_f[E] = ReturnAllCRLF(E, CRLs_in)
        ImageDist[E] = CheckFocPositions(beam['Source position'], CRL_f[E], CRL_pos)
        vecs = pd.Series({crl: np.dot(ray_trans_matrix(pos, beam['Source position'], CRL_f[E], CRL_pos), init_vec) for crl, pos in CRL_pos.items()})

        DiffLim[E] = {crl: diffr_lim(E, abs(v[1])) for crl, v in vecs.items()}
        Rayleigh[E] = {crl: rayl_len(E,v[0]) for crl, v in vecs.items()}

        BeamSizeAtLens[E] = [abs(vecs.loc[crl][0]) for crl in crls_in]
        FocusPosition = [ImageDist[E][crl] for crl in crls_in]
        DiffLimLens = [DiffLim[E][crl] for crl in crls_in]

        SamplePositions = np.append(SamplePositions, list(ImageDist[E].values()))
        isFocusing[E] = list()
        for i, crl in enumerate(crls_in):
            z_crl = np.array([LensPosition[i], LensPosition[i+1]]) - FocusPosition[i]
            Rlen = np.sqrt(BeamSizeAtLens[E][i]**2 - DiffLimLens[i]**2) / z_crl[0]
            zLinBounds = z_crl/np.sqrt((z_crl/Rlen)**2 + DiffLimLens[i]**2)
            zLin = np.linspace(zLinBounds[0], zLinBounds[1], 100)
            isFocusing[E].append(np.prod(np.sign(z_crl)) == -1)
            zSamp = DiffLimLens[i] * zLin * Rlen / np.sqrt( Rlen**2 - zLin**2 ) + FocusPosition[i]
            if isFocusing[E][i]:
                zSamp = np.append(zSamp,[FocusPosition[i]])
            SamplePositions = np.append(SamplePositions, zSamp)
    SamplePositions = SamplePositions[SamplePositions <= ComponentPositions['IBS Beamstop']]
    SamplePositions = SamplePositions[SamplePositions >= beam['Source position']]
    SamplePositions = np.array(sorted(set(SamplePositions)))

    previous_crl = np.argmin(SamplePositions[:,np.newaxis] > np.array(LensPosition + [np.inf])[np.newaxis,:],axis=1)

    BeamRadius, BeamMin = dict(), dict()
    crls_in = [0] + crls_in
    for E in Energies:
        vecs = np.array([np.dot(ray_trans_matrix(pos, beam['Source position'], CRL_f[E], CRL_pos), init_vec) for pos in SamplePositions])
        r = vecs[:,0]
        BSz = [1] + BeamSizeAtLens[E]
        r_lens = np.array([BSz[crl] for crl in previous_crl])

        DiffLim[E][0] = 0
        DiffLimLenses = [DiffLim[E][crl] for crl in crls_in]
        DiffLimLens = np.array([DiffLimLenses[crl] for crl in previous_crl])

        isFocused = np.array([([False] + isFocusing[E])[crl] for crl in previous_crl])

        cond = DiffLimLens>r_lens
        DiffLimLens[cond] = 0

        BeamRadius[E] = np.sqrt((1-DiffLimLens**2/r_lens**2)*r**2 + DiffLimLens**2)
        cond2 = cond * (BeamRadius[E]<r_lens)
        BeamRadius[E][cond2] = r_lens[cond2]

        sq_val = (1+DiffLimLens**2/r_lens**2)*r**2 - DiffLimLens**2
        BeamMin[E] = np.sign(r)*np.sqrt(np.abs(sq_val))
        BeamMin[E][sq_val<0] = 0
                # Do the negative values manually to avoid spamming warnings

        BeamRadius[E][~isFocused] = abs(r[~isFocused])
        BeamMin[E][~isFocused] = abs(r[~isFocused])
        
        
    LowerBound = np.min(np.abs(np.vstack((BeamMin[Energies[0]], BeamMin[Energies[2]]))),axis=0)
    LowerBound[np.prod(np.sign(np.vstack((BeamMin[Energies[0]], BeamMin[Energies[2]]))),axis=0) == -1] = 0
    Beam = np.vstack((
        LowerBound,
        BeamRadius[Energies[1]],
        BeamRadius[Energies[1]] + np.sqrt( (BeamRadius[Energies[0]]-BeamRadius[Energies[1]])**2 * ((BeamRadius[Energies[0]]-BeamRadius[Energies[1]])>0) + \
                 (BeamRadius[Energies[2]]-BeamRadius[Energies[1]])**2 * ((BeamRadius[Energies[2]]-BeamRadius[Energies[1]])>0))
        )).T
    if beam.Bandwidth == 0:
        Beam = Beam[:, [1,1,1]]
    Beam[Beam[:,0]>Beam[:,1],0] = 0
    # {position: (HWHM (um), divergence (urad))}
    return SamplePositions, Beam, CRL_f, ImageDist, DiffLim

def nmtoev(nm):
    """Converts eV to nm and vice versa"""
    return 1239.84193/nm

# https://en.wikipedia.org/wiki/Refractive_index#Complex_refractive_index
def delta(energy, material):
    """Returns refractive index delta of Be for photon energy (eV)."""

    if material == 'Be':
        return classical_elec * (nmtoev(energy)*1e-9)**2 * z_Be * natom_Be*1e6 / (2.*np.pi)
    elif material == 'C': 
        return classical_elec * (nmtoev(energy)*1e-9)**2 * z_C * natom_C*1e6 / (2.*np.pi)

# OPTICS LETTERS / Vol. 27, No. 9 / May 1, 2002
def foc_length(roc, energy, material):
    """Returns focal length of a spherical biconcave lens for roc (mm) and photon energy (eV)."""
    return 0.5 * roc * 1e-3 / delta(energy, material)

def calc_crlfoc(Energy, crl):
    """Convert radii of curvature to focal lengths at fixed photon energy (eV)."""
    if isinstance(crl.Radius, float):
        return foc_length(crl.Radius, Energy, crl.Material)/crl.NumberOfLenses
    return 1 / sum([n/foc_length(r, Energy, m) for r, n, m in zip(crl.Radius, crl.NumberOfLenses, crl.Material)])

def CRLF(Energy, CRLs_in):
    return {s: [calc_crlfoc(Energy, CRL.loc[(s, arm)]) for arm in crl_stack] \
                 for s, crl_stack in CRLs_in.items()}

def ReturnAllCRLF(Energy, CRLs_in):
    """Convert lens settings into focal lengths for energy."""
    return {lens_set: (1 / np.sum(1/np.array(flength)) if len(flength) else np.inf) for lens_set, flength in CRLF(Energy, CRLs_in).items()}

def CheckFocPositions(SourcePosition, CRL_f, CRL_pos):
    """Check position of foci for each CRL lens set)"""
    prev_image = SourcePosition
    CRL_Image = dict()
    for crl in CRL_f:
        CRL_Image[crl] = CRL_pos[crl] + image_dist(obj=CRL_pos[crl]-prev_image, foc=CRL_f[crl])
        prev_image = CRL_Image[crl]
        # calculate image distance for focal length
    return CRL_Image

def calculate(BeamPars, CRLs_in, CRL_pos, TestPosition):
    """First calculates focal lengths for this energy (eV). Then the total focal length of each lens set for the 
    chosen lens configuration crl1lens, crl2lens, crl3lens. Then propogates beam sizes through this for a source 
    size, position, and beam divergence."""

    SamplePositions, beam_vecs, CRL_f, ImageDist, DiffLim = dict(), dict(), dict(), dict(), dict()
    for ind, beam in BeamPars.iterrows():
        SamplePositions[ind], beam_vecs[ind], CRL_f[ind], ImageDist[ind], DiffLim[ind] = ray_propogation(beam, CRL_pos, CRLs_in, TestPosition)
    return SamplePositions, beam_vecs, CRL_f, ImageDist, DiffLim


def calc_div(Energy, CRL1, FELBeamSize, HEDBeamSize):
    """Calculates beam divergence and source position for the chosen set-up)"""
    f_crl1 = ReturnAllCRLF(Energy, {1: CRL1}) # chosen config
    f_crl1 = f_crl1[1]
    posz = np.array([ComponentPositions['FEL imager'], ComponentPositions['HED pop-in']])
    beamszs = np.array([FELBeamSize, HEDBeamSize])
    errszs = 0.1*beamszs # set to be 10%
    # y = fit0 * x + fit1
    fit0 = (beamszs[1] - beamszs[0])/(posz[1] - posz[0])
    fit1 = (beamszs[0]*posz[1] - beamszs[1]*posz[0])/(posz[1] - posz[0])
    errfit0 = abs(np.sqrt(errszs[0]**2+errszs[1]**2)/(posz[1] - posz[0]))
    errfit1 = abs(np.sqrt((posz[0]*errszs[1])**2 + (posz[1]*errszs[0])**2)/(posz[1] - posz[0]))
    # Calculate image distance, i.e. where y=0
    imagedist = -fit1/fit0 - ComponentPositions['CRL1']
    errimagedist = abs(np.sqrt(errfit1**2 + (fit1*errfit0/fit0)**2)/fit0) # convert to error in distance
    objdist = image_dist(obj=imagedist, foc=f_crl1)
    errobjdist = abs(errimagedist*(objdist/imagedist)**2) # Convert to error in source point from lens formula
    newsourcepos = ComponentPositions['CRL1'] - objdist
    # Calculate beam size and error at CRL1 lens
    crl1_beamsz = fit0*ComponentPositions['CRL1'] + fit1
    errcrl1sz = abs(np.sqrt((errfit0*ComponentPositions['CRL1'])**2 + errfit1**2))
    # To give the divergence from source point to CRL1
    newbeamdiv = crl1_beamsz/(ComponentPositions['CRL1']-newsourcepos)
    errbeamdiv = abs(np.sqrt(errcrl1sz**2 + (errobjdist/(ComponentPositions['CRL1']-newsourcepos))**2)/(ComponentPositions['CRL1']-newsourcepos))
    
    return np.round(newsourcepos, 1), np.round(newbeamdiv, 2), np.round(errobjdist, 1), np.round(errbeamdiv, 2)

def BeamSize(s, z, beam, CRLs_in):
    CRL_pos = {1: ComponentPositions['CRL1'],
               2: ComponentPositions['CRL2'],
               3: ComponentPositions['CRL3'] + s}
    init_vec = np.array([0, beam['Beam divergence']])
    Bandwidth_eV = 0.5*1e-2*beam.Bandwidth*beam.Energy # Convert to HWHM in eV
    Energies = beam.Energy + Bandwidth_eV*np.arange(-1,2)
    loop_ind = [0, 2, 1]
    bs = dict()
    for E_ind in loop_ind:
        E = Energies[E_ind]
        CRL_f = ReturnAllCRLF(E, CRLs_in)
        lin_sz = ray_trans_matrix(z, beam['Source position'], CRL_f, CRL_pos) @ init_vec
        bs[E] = np.sqrt(lin_sz[0]**2 + diffr_lim(E, lin_sz[1])**2)
    
    ChromaticWidth = np.sign(lin_sz[0]) * (bs[Energies[1]] + np.sqrt((bs[Energies[0]]-bs[Energies[1]])**2 * ((bs[Energies[0]]-bs[Energies[1]]) > 0) + \
                                    (bs[Energies[2]]-bs[Energies[1]])**2 * ((bs[Energies[2]]-bs[Energies[1]]) > 0)))
    return ChromaticWidth

def BeamSizeCRL4(s, z, beam, CRLs_in):
    CRL_pos = {1: ComponentPositions['CRL1'],
               2: ComponentPositions['CRL2'],
               3: ComponentPositions['CRL3'] + s[0],
               4: z + s[1]}
    init_vec = np.array([0, beam['Beam divergence']])
    Bandwidth_eV = 0.5*1e-2*beam.Bandwidth*beam.Energy # Convert to HWHM in eV
    Energies = beam.Energy + Bandwidth_eV*np.arange(-1,2)
    loop_ind = [0, 2, 1]
    bs = dict()
    for E_ind in loop_ind:
        E = Energies[E_ind]
        CRL_f = ReturnAllCRLF(E, CRLs_in)
        lin_sz = ray_trans_matrix(z, beam['Source position'], CRL_f, CRL_pos) @ init_vec
        bs[E] = np.sqrt(lin_sz[0]**2 + diffr_lim(E, lin_sz[1])**2)
    
    ChromaticWidth = np.sign(lin_sz[0]) * (bs[Energies[1]] + np.sqrt((bs[Energies[0]]-bs[Energies[1]])**2 * ((bs[Energies[0]]-bs[Energies[1]]) > 0) + \
                                    (bs[Energies[2]]-bs[Energies[1]])**2 * ((bs[Energies[2]]-bs[Energies[1]]) > 0)))
    return ChromaticWidth

def ReducedLensCombos(BeamPars, CRL_pos, z):

    CRLlenses = {1: list(range(2,10)), 2: list(range(1,11)), 3: list(range(1,11))}
    LensCombos = dict()
    for lens in CRLlenses:
        lenses = dict()
        for L in range(len(CRLlenses[lens]), -1, -1):
            for subset in itertools.combinations(CRLlenses[lens], L):
                f = [CRL.loc[(lens,arm)].NumberOfLenses / CRL.loc[(lens,arm)].Radius for arm in subset]
                f = [a if isinstance(a, float) else sum(a) for a in f]
                f = sum(f)
                if f not in lenses:
                    lenses[f] = subset
                elif len(subset) < len(lenses[f]):
                    lenses[f] = subset
        LensCombos[lens] = tuple(x for (_, x) in sorted(lenses.items()))
        # Construct dict of all possible unique lens combinations
        # Unique = unique combined focal length of lenses
        # iterated in order most arms -> least arms, where configs are overwritten, so fewer lenses are preferred
    
    f1 = pd.Series([np.array([ReturnAllCRLF(E,{1: lc})[1] for lc in LensCombos[1]]) for E in BeamPars.Energy])
    BeamSizeCRL1 = BeamPars['Beam divergence']*(CRL_pos[1]-BeamPars['Source position'])
    BeamDivAfterCRL1 = BeamPars['Beam divergence'] - BeamSizeCRL1/f1

    BeamSizeAtBeamStop = BeamSizeCRL1 + (ComponentPositions['IBS Beamstop']-CRL_pos[1])*BeamDivAfterCRL1
    BeamSizeAtCRL2 = BeamSizeCRL1 + (CRL_pos[2]-CRL_pos[1])*BeamDivAfterCRL1
    BeamSizeAtM3 = BeamSizeCRL1 + (ComponentPositions['Mirror 3']-CRL_pos[1])*BeamDivAfterCRL1

    BeamSizeAtBeamStop, BeamSizeAtCRL2, BeamSizeAtM3 = (np.array(s.values.tolist()) for s in [BeamSizeAtBeamStop, BeamSizeAtCRL2, BeamSizeAtM3])
    cond = np.prod(((BeamSizeAtBeamStop > 200) + \
                    (BeamSizeAtCRL2<-200)*(BeamSizeAtM3>200)) * \
                   (BeamSizeAtCRL2<CRL.loc[2].Aperture.max()),axis=0).astype(bool)
        # All Beams must be either intermediate focus between CRL1 & CRL2, without being < 200um on M3 or CRL2,
        # or still be >200um at the beam stop (i.e. not strongly focused)
    LensCombos[1] = tuple(lc for lc, c in zip(LensCombos[1],cond) if c)

    # Given the limited CRL1 configs, limit CRL2
    f1 = pd.Series([np.array([ReturnAllCRLF(E,{1: lc})[1] for lc in LensCombos[1]]) for E in BeamPars.Energy])
    BeamSizeCRL1 = BeamPars['Beam divergence']*(CRL_pos[1]-BeamPars['Source position'])
    BeamDivAfterCRL1 = BeamPars['Beam divergence'] - BeamSizeCRL1/f1
    ImagePositionCRL1 = CRL_pos[1] - BeamSizeCRL1 / BeamDivAfterCRL1

    SizeAtCRL2 = (BeamSizeCRL1 + (CRL_pos[2]-CRL_pos[1])*BeamDivAfterCRL1).map(abs)
    ImageDistCRL2_forSpotSize = CRL_pos[2] + (z-CRL_pos[2])*SizeAtCRL2/(SizeAtCRL2+BeamPars['Spot size'])
    f_CRL2lim = (1/(CRL_pos[2]-ImagePositionCRL1) + 1/(ImageDistCRL2_forSpotSize - CRL_pos[2]))**-1
    f2_bound_lower = f_CRL2lim.map(min)
    f2 = pd.Series([np.array([ReturnAllCRLF(E,{2: lc})[2] for lc in LensCombos[2]]) for E in BeamPars.Energy])
    cond = np.prod(np.vstack(f2)>np.vstack(f2_bound_lower), axis=0)
    LensCombos[2] = tuple(lc for lc, c in zip(LensCombos[2],cond) if c)

    CRL3pos = CRL_pos[3]-.5
    SizeAtCRL3 = (BeamSizeCRL1 + (CRL3pos-CRL_pos[1])*BeamDivAfterCRL1).map(abs)
    ImageDistCRL3_forSpotSize = CRL3pos + (z-CRL3pos)*SizeAtCRL3/(SizeAtCRL3 + BeamPars['Spot size'])
    f_CRL3lim = (1/(CRL3pos-ImagePositionCRL1) + 1/(ImageDistCRL3_forSpotSize - CRL3pos))**-1
    f3_bound_lower = f_CRL3lim.map(min)
    f3 = pd.Series([np.array([ReturnAllCRLF(E,{3: lc})[3] for lc in LensCombos[3]]) for E in BeamPars.Energy])
    cond = np.prod(np.vstack(f3)>np.vstack(f3_bound_lower), axis=0)
    LensCombos[3] = tuple(lc for lc, c in zip(LensCombos[3],cond) if c)

    return LensCombos

BeAttnLength = np.loadtxt('BeAttnLength.dat',skiprows=2)
FeAttnLength = np.loadtxt('FeAttnLength.dat',skiprows=2)
FeImpurityFraction = 1.5e-3
LensAttnLength = np.vstack((BeAttnLength[:,0], (1/BeAttnLength[:,1] + FeImpurityFraction/ FeAttnLength[:,1])**-1)).T
    # Attenuation length of CRL lenses, including Fe impurities
LensMinThickness = 50
    # Minimum thickness (in um) of bi-concave CRLs
def EstimateLensTransmission(BeamPars, CRLs_in, CRL_pos):
    """
    Estimates the transmission of each beam through the CRLs, assuming beam is uniformly
    illuminating every lens within its' spot size at each lens. Also assumes that lenses
    are bi-concave CRL's, with on-axis thickness = LensMinThickness, as defined above.
    If a beam over-illumnates a lens, the fraction within the aperture is considered.
    """
    BeamPars = BeamPars.copy()
    BeamPars['Transmission'] = [1]*BeamPars.shape[0]
    for ind, beam in BeamPars.iterrows():
        init_vec = np.array([0, beam['Beam divergence']])
        CRL_f = ReturnAllCRLF(beam.Energy, CRLs_in)
        BeamSizeAtLenses =  pd.Series({crl: np.dot(ray_trans_matrix(pos, beam['Source position'], CRL_f, CRL_pos), init_vec) for crl, pos in CRL_pos.items()})
        MinAverageLensThickness = 0
        BeamFraction = 1
        for crl, lenses in CRLs_in.items():
            for lens in lenses:
                r = CRL.loc[(crl, lens), 'Radius'] * 2e3
                Nlenses = CRL.loc[(crl, lens), 'NumberOfLenses']
                Aperture = CRL.loc[(crl, lens), 'Aperture']
                a = min(abs(BeamSizeAtLenses[crl][0]), Aperture) / 2
                AverageCapThickness = a**2 / r
                # Thickness of a biconcave lens with given radius, and illuminated aperture
                """
                    2    /a     x^2
                ------  |      --- 2*pi*x dx
                pi*a^2  /-a     2r

                where a is radius of illumination
                r is biconcave radius, = 2*R0
                Initial factor 2 is from biconcavity
                integral over illuminated aperture of parabola, integrated over angle
                """
                MinAverageLensThickness += LensMinThickness + Nlenses * AverageCapThickness
                BeamFraction = min(BeamFraction, (Aperture / BeamSizeAtLenses[crl][0])**2)
        BeamPars.loc[ind, 'Transmission'] = BeamFraction * np.prod(np.exp( -MinAverageLensThickness / np.interp(beam.Energy, LensAttnLength[:,0], LensAttnLength[:,1])))
    return BeamPars

"""Here is the GUI part"""
class MainWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("CRL Calculator")
        self.setWindowIcon(QIcon('logo.png'))


        StaticMessageLayout = QGridLayout()
        Message = QLabel("""
Calculator for CRL focus positions and beam sizes for the HED instrument.
Copyright Thomas Preston (c) 2021
European X-Ray Free-Electron Laser Facility GmbH.
Updated for multiple beams by Oliver Humphries, 2023
All rights reserved.
        """
        )
        StaticMessageLayout.addWidget(Message, 0, 0)

    
        grid_layout = QGridLayout(self)
        grid_layout.addLayout(StaticMessageLayout, 30, 0, 1, 1)

        Tabs = QTabWidget()
        Tab1 = QWidget()
        Tab2 = QWidget()
        Tabs.addTab(Tab1,'CRL selection')
        Tabs.addTab(Tab2,'Optimisation')

        cl = Tab2.palette().color(Tab2.backgroundRole()).getRgbF()
        
        Tab1.setLayout(self.ControlPanel())
        Tab2.setLayout(self.OptimisationLayout(cl))
        grid_layout.addWidget(Tabs, 1, 0, 29, 1)
        grid_layout.addLayout(self.PlotLayout(), 1, 1, 30, 3)
        grid_layout.setColumnStretch(0, 0)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setRowStretch(30, 0)
        self.setLayout(grid_layout)
        self.calculate_click()
        self.show()

    def OptimisationLayout(self,cl):
        VBox = QVBoxLayout()
        Form = QFormLayout()

        hbox = QHBoxLayout()
        self.optMono = QCheckBox('Mono')
        self.optHRMono = QCheckBox('HR-mono')
        hbox.addWidget(self.optMono)
        hbox.addWidget(self.optHRMono)
        Form.addRow(QLabel('Optional components'), hbox)

        self.optTarget = QComboBox()
        self.optTarget.addItems([c for c in ComponentPositions.index if 'TCC' in c])
        Form.addRow(QLabel('Target'), self.optTarget)

        self.optDefaultValues = {'Spot size': 0, 'Weight': 1, 'Focusing': 'Either'}
        self.optBeamParameters = QTableWidget(3,1)
        self.optBeamParameters.setFixedHeight(92)
        self.optBeamParameters.horizontalHeader().hide()
        self.optBeamParameters.setVerticalHeaderLabels(list(self.optDefaultValues))
        self.NumberOfColours.valueChanged.connect(self.optInputTable)
        self.optInputTable()
        
        self.optButton = QPushButton('Begin optimisation')
        self.optButton.clicked.connect(self.optimize)
        self.optCancel = QPushButton('Cancel')
        self.optCancel.clicked.connect(self.optInterrupt)
        self.optIsCancelled = False
        hbox = QHBoxLayout()
        hbox.addWidget(self.optButton)
        hbox.addWidget(self.optCancel)

        self.optProgress = QProgressBar()
        self.optOutput = QLabel('')
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.optOutput)
        self.scrollArea.setWidgetResizable(True)
        self.optLensConfigOut = QLabel('')

        VBox.addLayout(Form)
        VBox.addWidget(self.optBeamParameters)
        VBox.addLayout(hbox)
        VBox.addWidget(self.optProgress)
        lab = QLabel("""
Beam focus optimisation:
Set beam parameters on "CRL selection" tab
Enter desired spot size and relative weight for each beam
Optimal lens configuration is found according to:
        """)
        VBox.addWidget(lab)
        fig = Figure(edgecolor=cl, facecolor=cl,tight_layout=True,figsize=(1,.4))
        fig.clear()
        fig.suptitle(r'$min_{Config, shift} \sum_{i}W_i\left(y_i(z_{TCC}) - s_i\right)^2$',
                        x=0.5, y=0.5, 
                        horizontalalignment='center',
                        verticalalignment='center')
        canvas = FigureCanvas(fig)      
        canvas.draw()
        VBox.addWidget(canvas)
        lab = QLabel(f"""
where y(z) is the chromatic beam diameter at selected TCC. For non-chromatic minimisation, set bandwidth = 0. Each beam is checked against all critical components.
To include CRL4, set the number of lenses used in "CRL selection". CRL4 shift optimisation is limited to [-1000, 0]mm from target position, but is highly constrained by focal length.
Including mono will automatically adjust bandwidth.
When optimising CRL3, the maximum shift is limited to {chr(0x00b1)}200mm, to allow tuning.
Weight values are relative, and can be 0 (i.e. that beam is just checked against critical components, and not optimised). At least one weight value must be positive.
If desired focus size is found, loop will exit.
There are ~87.5 million unique lens combinations, they are iterated in order of decreasing focal length for each CRL. Higher photon energies take longer to optimise due to increased redundancy of focusing optics.
        """)
        lab.setWordWrap(True)
        VBox.addWidget(lab)
        VBox.addWidget(self.scrollArea)
        VBox.addWidget(self.optLensConfigOut)
        return VBox

    def optInterrupt(self):
        if not self.optIsCancelled:
            self.optAppendOutput('\nCancelling optimization')
        self.optIsCancelled = True
    
    def optAppendOutput(self,text):
        self.optOutput.setText(self.optOutput.text() + '\n' + text)
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
        QGuiApplication.processEvents()

    def optUpdateLensConfig(self,text):
        self.optLensConfigOut.setText(text)
        QGuiApplication.processEvents()

    def optimize(self):
        self.optIsCancelled = False
        self.optOutput.setText('Beginning optimisation')
        QGuiApplication.processEvents()
        Mono, HRMono = self.optMono.isChecked(), self.optHRMono.isChecked()

        BeamPars = self.GetBeamPars()
        BeamPars['Spot size'] = 0
        BeamPars['Weight'] = 1
        BeamPars['Focusing'] = [np.array([-1, 1])]*BeamPars.shape[0]
        if Mono:
            BeamPars.BandWidth = 100/BeamPars.Energy
            bw = ', '.join([f'{s:.2g}%' for s in BeamPars.BandWidth])
            self.optAppendOutput('Mono bandwidth set to ' + bw)
        if HRMono:
            BeamPars.BandWidth = 4/BeamPars.Energy
            bw = ', '.join([f'{s:.2g}%' for s in BeamPars.BandWidth])
            self.optAppendOutput('HR-Mono bandwidth set to ' + bw)
        

        ncols = self.NumberOfColours.value()
        for col in range(ncols):
            for row, (rowname, default) in enumerate(self.optDefaultValues.items()):
                if row==2:
                    try:
                        val = self.optBeamParameters.cellWidget(row,col)
                        val = val.currentIndex()
                        a = [[1], [-1], [-1, 1]][val]
                        BeamPars[rowname].update(pd.Series([a], index=[col]))
                    except: pass
                else:
                    try:
                        val = self.optBeamParameters.item(row,col)
                        val = val.text() if val is not None else str(default)
                        BeamPars.loc[col,rowname] = float(val)
                    except: pass
        if BeamPars.shape[0] == 1:
            if BeamPars.Weight[0] != 1:
                self.optAppendOutput('Single beam, setting relative weight = 1')
                BeamPars.loc[0,'Weight'] = 1
                self.optBeamParameters.item(1,0).setText('1')

        if (BeamPars.Weight<0).any():
            self.optAppendOutput('ERROR: weight cannot be negative!')
            return
        if sum(BeamPars.Weight) == 0:
            self.optAppendOutput('ERROR: At least 1 weight value must be positive')
            return

        z = ComponentPositions[self.optTarget.currentText()]
        if (((z - BeamPars['Source position'])*BeamPars['Beam divergence']) < BeamPars['Spot size']).any():
            self.optAppendOutput('ERROR: Spot size is not achievable, enter smaller value')
            return

        CRL_pos = {1: ComponentPositions['CRL1'],
                   2: ComponentPositions['CRL2'],
                   3: ComponentPositions['CRL3']}
        comps = ComponentPositions.sort_values()
        LensComps = {1: comps[['CRL2', 'Mirror 1', 'Mirror 2', 'Mirror 3', 'CRL3'] + ['Mono']*Mono + ['HR-mono']*HRMono],
                     2: comps[['CRL3', 'IBS Beamstop']],
                     3: comps[['IBS Beamstop']]}
        init_vec = {1: np.hstack([free_space_matrix(CRL_pos[1]-beam['Source position']) @ np.array([[0], [beam['Beam divergence']]]) for _, beam in BeamPars.iterrows()]),
                    2: np.zeros((2, BeamPars.shape[0])),
                    3: np.zeros((2, BeamPars.shape[0]))}

        LensCombos = ReducedLensCombos(BeamPars, CRL_pos, z)
        self.optAppendOutput(f'{len(LensCombos[1])} / 256 valid CRL1 configurations')
        self.optAppendOutput(f'{len(LensCombos[2])} / 1024 valid CRL2 configurations')
        self.optAppendOutput(f'{len(LensCombos[3])} / 1024 valid CRL3 configurations')


        if CRL.NumberOfLenses[(4,1)]:
            self.optimise_CRL4(BeamPars, z, CRL_pos, LensCombos, LensComps, init_vec)
        else:
            self.optimise_CRL3(BeamPars, z, CRL_pos, LensCombos, LensComps, init_vec)

    def optimise_CRL4(self,BeamPars, z, CRL_pos, LensCombos, LensComps, init_vec):

        LensComps[3]['CRL4 lower'] = z - 1
        LensComps[3]['CRL4 upper'] = z
        
        Total_configs = np.prod([len(v) for v in LensCombos.values()])

        ComponentSizeLimit = 200
        Valid_configs = 0
        Running_min = np.inf
        optVal = pd.Series({1: tuple(), 2: tuple(), 3: tuple(), 'Cost func': Running_min, 'Shift': 0})
        for isub1, subset in enumerate(LensCombos[1]):
            if self.optIsCancelled:
                self.optIsCancelled = False
                return
            self.optProgress.setValue(100*isub1/len(LensCombos[1]))
            QGuiApplication.processEvents()
            CRLs_in = {1: subset}
            for ind, E in enumerate(BeamPars.Energy):
                CRL_f = ReturnAllCRLF(E, CRLs_in)
                res = np.array([free_space_matrix(pos - CRL_pos[1]) @ lens_matrix(CRL_f[1]) @ init_vec[1][:,ind] for pos in LensComps[1]])
                if (np.abs(res[:,0])<ComponentSizeLimit).any() or abs(res[0,0])>3800:
                    break
                init_vec[2][:,ind] = res[0,:]
            else:
                for isub2, subset2 in enumerate(LensCombos[2]):
                    if self.optIsCancelled:
                        self.optIsCancelled = False
                        return
                    aperture = CRL.loc[2].loc[list(subset2)].Aperture.min()
                    if (init_vec[2][0,:] > aperture).any():
                        continue
                    CRLs_in[2] = subset2
                    for ind, beam in BeamPars.iterrows():
                        CRL_f = ReturnAllCRLF(beam.Energy, CRLs_in)
                        res = np.array([free_space_matrix(pos - CRL_pos[2]) @ lens_matrix(CRL_f[2]) @ init_vec[2][:,ind] for pos in LensComps[2]])
                        if (np.abs(res[:,0])<ComponentSizeLimit).any() or abs(res[0,0]) > 3800:
                            break
                        ImageDist = CheckFocPositions(beam['Source position'], CRL_f, CRL_pos)
                        if ImageDist[2] > CRL_pos[2] and ImageDist[2]<CRL_pos[3]:
                            break
                        init_vec[3][:,ind] = res[0,:]
                    else:
                        for isub3, subset3 in enumerate(LensCombos[3]):
                            aperture = CRL.loc[3].loc[list(subset3)].Aperture.min()
                            if (init_vec[3][0,:] > aperture).any():
                                continue
                            CRLs_in[3] = subset3
                            for ind, beam in BeamPars.iterrows():
                                CRL_f = ReturnAllCRLF(beam.Energy, CRLs_in)
                                res = np.array([free_space_matrix(pos - CRL_pos[3]) @ lens_matrix(CRL_f[3]) @ init_vec[3][:,ind] for pos in LensComps[3]])
                                if (np.abs(res[:-2,0])<ComponentSizeLimit).any():
                                    break
                                CRL4_sizes = np.abs(res[-2:,0])
                                if (CRL4_sizes < 200).all() or (CRL4_sizes > (2*CRL.Aperture[(4,1)])).all():
                                    break
                                ImPos = LensComps[3][0] + (np.sign(res[0,0])*beam['Spot size'] - res[0,0]) / res[0,1]
                                if ImPos < z:
                                    break
                            else:
                                ConfigNumber = 1 + isub3 + len(LensCombos[3])*(isub2 + isub1*len(LensCombos[2]))
                                Valid_configs += 1

                                CRLs_in[4] = (1,)
                                InterFocus = [-1, 1][ImageDist[1]<CRL_pos[2]]
                                CostFunc = lambda s: sum([np.min(beam.Weight*(BeamSizeCRL4(s, z, beam, CRLs_in)-InterFocus*np.array(beam['Focusing'])*beam['Spot size'])**2) for _, beam in BeamPars.iterrows()])
                                res = minimize(CostFunc, [0, -.3], bounds=[(-.5, .5), (-1, 0)])
                                del CRLs_in[4]

                                cf = res.fun
                                shift = res.x*1e3
                                if cf < Running_min:
                                    Running_min = cf
                                    optVal = pd.Series({1: CRLs_in[1], 2: CRLs_in[2], 3: CRLs_in[3],
                                    'Cost func': Running_min, 'Shift': shift})
                                    if Running_min < 1e-6:
                                        self.optPrintResults(f'{ConfigNumber} / {Total_configs}\n{Valid_configs} valid configurations tested\nFocus found',optVal)
                                        self.optProgress.setValue(100)
                                        self.optIsCancelled = False
                                        return
                                    self.optPrintResults(f'{ConfigNumber} / {Total_configs}\n{Valid_configs} valid configurations tested\nCurrent minimum',optVal)
        self.optProgress.setValue(100)
        self.optPrintResults(f'Optimum of {Valid_configs} valid configurations',optVal)
        self.optIsCancelled = False


    def optimise_CRL3(self,BeamPars, z, CRL_pos, LensCombos, LensComps, init_vec):
        
        Total_configs = np.prod([len(v) for v in LensCombos.values()])

        ComponentSizeLimit = 200
        Valid_configs = 0
        Running_min = np.inf
        optVal = pd.Series({1: tuple(), 2: tuple(), 3: tuple(), 'Cost func': Running_min, 'Shift': np.float64(0)})
        for isub1, subset in enumerate(LensCombos[1]):
            if self.optIsCancelled:
                self.optIsCancelled = False
                return
            self.optProgress.setValue(100*isub1/len(LensCombos[1]))
            QGuiApplication.processEvents()
            CRLs_in = {1: subset}
            for ind, E in enumerate(BeamPars.Energy):
                CRL_f = ReturnAllCRLF(E, CRLs_in)
                res = np.array([free_space_matrix(pos - CRL_pos[1]) @ lens_matrix(CRL_f[1]) @ init_vec[1][:,ind] for pos in LensComps[1]])
                init_vec[2][:,ind] = res[0,:]
            else:
                for isub2, subset2 in enumerate(LensCombos[2]):
                    if self.optIsCancelled:
                        self.optIsCancelled = False
                        return
                    aperture = CRL.loc[2].loc[list(subset2)].Aperture.min()
                    if (init_vec[2][0,:] > aperture).any():
                        continue
                    CRLs_in[2] = subset2
                    for ind, beam in BeamPars.iterrows():
                        CRL_f = ReturnAllCRLF(beam.Energy, CRLs_in)
                        res = np.array([free_space_matrix(pos - CRL_pos[2]) @ lens_matrix(CRL_f[2]) @ init_vec[2][:,ind] for pos in LensComps[2]])
                        if (np.abs(res[:,0])<ComponentSizeLimit).any() or abs(res[0,0]) > 3800:
                            break
                        ImageDist = CheckFocPositions(beam['Source position'], CRL_f, CRL_pos)
                        if ImageDist[2] > CRL_pos[2] and ImageDist[2]<CRL_pos[3]:
                            break
                        init_vec[3][:,ind] = res[0,:]
                    else:
                        for isub3, subset3 in enumerate(LensCombos[3]):
                            aperture = CRL.loc[3].loc[list(subset3)].Aperture.min()
                            if (init_vec[3][0,:] > aperture).any():
                                continue
                            CRLs_in[3] = subset3
                            for ind, beam in BeamPars.iterrows():
                                CRL_f = ReturnAllCRLF(beam.Energy, CRLs_in)
                                res = np.array([free_space_matrix(pos - CRL_pos[3]) @ lens_matrix(CRL_f[3]) @ init_vec[3][:,ind] for pos in LensComps[3]])
                                ImPos = LensComps[3][0] + (beam['Spot size']*np.array([1, -1]) - res[0,0]) / res[0,1]
                                if (np.abs(ImPos - z) > .2).all():
                                    break
                                if (np.abs(res[:,0])<ComponentSizeLimit).any():
                                    break
                            else:
                                ConfigNumber = 1 + isub3 + len(LensCombos[3])*(isub2 + isub1*len(LensCombos[2]))
                                Valid_configs += 1
                                InterFocus = [-1, 1][ImageDist[1]<CRL_pos[2]]
                                CostFunc = lambda s: sum([np.min(beam.Weight*(BeamSize(s, z, beam, CRLs_in)-InterFocus*np.array(beam['Focusing'])*beam['Spot size'])**2) for _, beam in BeamPars.iterrows()])
                                if not len(CRLs_in[3]):
                                    shift = 0
                                    cf = CostFunc(0)
                                else:
                                    res = minimize(CostFunc,0, bounds=[(-.2, .2)])
                                    cf = res.fun
                                    shift = res.x[0]*1e3
                                if cf < Running_min:
                                    Running_min = cf
                                    optVal = pd.Series({1: CRLs_in[1], 2: CRLs_in[2], 3: CRLs_in[3], 'Cost func': Running_min, 'Shift': shift})
                                    if Running_min < 1e-6:
                                        self.optPrintResults(f'{ConfigNumber} / {Total_configs}\n{Valid_configs} valid configurations tested\nFocus found',optVal)
                                        self.optProgress.setValue(100)
                                        self.optIsCancelled = False
                                        return
                                    self.optPrintResults(f'{ConfigNumber} / {Total_configs}\n{Valid_configs} valid configurations tested\nCurrent minimum',optVal)
        self.optProgress.setValue(100)
        if optVal['Cost func'] == np.inf:
            self.optUpdateLensConfig('No optimum found\nLinear focusing does not intercept spot size\nwithin 200mm shift')
        else:
            self.optPrintResults(f'Optimum of {Valid_configs} valid configurations',optVal)
        self.optIsCancelled = False
    
    def optPrintResults(self, text,optVal):
        if isinstance(optVal.Shift, np.float64):
            self.optUpdateLensConfig(f"{text}, Cost function = {optVal['Cost func']:.4g}:\n\tShift = {optVal.Shift:.1f}mm\n\t" + \
                '\n\t'.join([f'{k}: {v}' for k, v in optVal[[1,2,3]].to_dict().items()]))
        else:
            self.optUpdateLensConfig(f"{text}, Cost function = {optVal['Cost func']:.4g}:" +\
                ''.join([f"\n\tShift CRL{i+3} = {s:.1f}" for i, s in enumerate(optVal.Shift)]) +"\n\t" + \
                '\n\t'.join([f'{k}: {v}' for k, v in optVal[[1,2,3]].to_dict().items()]))

    def optInputTable(self):
        ncols = self.NumberOfColours.value()
        self.optBeamParameters.setColumnCount(ncols)
        for row, (rowname, default) in enumerate(self.optDefaultValues.items()):
            for col in range(ncols):
                val = self.optBeamParameters.item(row,col)
                if not val:
                    if row == 2:
                        val == self.optBeamParameters.cellWidget(row,col)
                        if not val:
                            item = QComboBox()
                            item.addItems(['Pre', 'Post', 'Either'])
                            item.setCurrentText('Either')
                            self.optBeamParameters.setCellWidget(row, col, item)
                    else:
                        item = QTableWidgetItem(str(default))
                        self.optBeamParameters.setItem(row,col,item)
        for col in range(ncols):
            header = self.optBeamParameters.horizontalHeader()
            header.setSectionResizeMode(col, QHeaderView.Stretch)


    def PlotLayout(self):
        
        plotlayout = QVBoxLayout()
        self.plot = Canvas(self, width=20, height=2, dpi=100)
        self.ax1 = self.plot.axes[0]
        self.ax2 = self.plot.axes[1]

        NaviBarLayout = QHBoxLayout()
        navi =  NavigationToolbar(self.plot, self)
        self.FixXAxes = QCheckBox('x')
        self.FixYAxes = QCheckBox('y')
        NaviBarLayout.addWidget(navi)
        NaviBarLayout.addWidget(QLabel('Fix axes limits:'))
        NaviBarLayout.addWidget(self.FixXAxes)
        NaviBarLayout.addWidget(self.FixYAxes)

        plotlayout.addLayout(NaviBarLayout)
        plotlayout.addWidget(self.plot)
        return plotlayout


    def ControlPanel(self):

        ControlLayout = QVBoxLayout()
        ControlLayout.setSpacing(5)
        
        ControlLayout.addLayout(self.BeamDivergence())
        ControlLayout.addLayout(self.BeamParametersLayout())
        ControlLayout.addLayout(self.CRLControls())

        WarningsLayout = QGridLayout()
        
        self.CRLOut = QTableWidget(0,0)
        self.CRLOut.setEditTriggers(QTableWidget.NoEditTriggers)
        self.CRLOut.setFixedHeight(122)

        self.Warnings = QLabel("\n"*10)
        scrollArea = QScrollArea()
        scrollArea.setWidget(self.Warnings)
        scrollArea.setWidgetResizable(True)
        #scrollArea.setFixedHeight(200)
        WarningsLayout.addWidget(self.CRLOut, 0, 0)
        WarningsLayout.addWidget(QLabel("Output and warnings:"), 1, 0)
        WarningsLayout.addWidget(scrollArea, 2, 0, 5, 1)
        WarningsLayout.setRowStretch(1,0)
        WarningsLayout.setRowStretch(2,1)
        ControlLayout.addLayout(WarningsLayout)

        return ControlLayout

    
    def BeamDivergence(self):
        BeamDivergenceLayout = QGridLayout()
        BeamDivergenceLayout.setSpacing(2)

        BeamDivergenceLayout.addWidget(QLabel("Beam divergence and source size calculator:"), 0, 0, 1, 2)

        self.FEL_beamsize_um = QLineEdit(str(DefaultValues['FEL imager size'][0]))
        self.FEL_beamsize_um.setValidator(QDoubleValidator())
        self.FEL_beamsize_um.editingFinished.connect(self.calculatebd_click)
        BeamDivergenceLayout.addWidget(QLabel(f"FEL imager size [{DefaultValues['FEL imager size'][1]}]:"), 1, 0)
        BeamDivergenceLayout.addWidget(self.FEL_beamsize_um, 1, 1)
        
        self.HED_beamsize_um = QLineEdit(str(DefaultValues['HED imager size (w/ lens)'][0]))
        self.HED_beamsize_um.setValidator(QDoubleValidator())
        self.HED_beamsize_um.editingFinished.connect(self.calculatebd_click)
        BeamDivergenceLayout.addWidget(QLabel(f"HED imager size (w/ lenses [{DefaultValues['HED imager size (w/ lens)'][1]}]:"), 2, 0)
        BeamDivergenceLayout.addWidget(self.HED_beamsize_um, 2, 1)

        self.SourcePositionOutput = QLabel("")
        BeamDivergenceLayout.addWidget(QLabel("Source position (m):"), 3, 0)
        BeamDivergenceLayout.addWidget(self.SourcePositionOutput, 3, 1)
        
        self.BeamDivergenceOutput = QLabel("")
        BeamDivergenceLayout.addWidget(QLabel(f"Beam divergence [{mu}rad]:"), 4, 0)
        BeamDivergenceLayout.addWidget(self.BeamDivergenceOutput, 4, 1)

        return BeamDivergenceLayout
    
    def BeamParametersLayout(self):
        
        BeamParameterLayout = QVBoxLayout()
        self.NumberOfColours = QSpinBox()
        self.NumberOfColours.setRange(1,3)
        self.NumberOfColours.valueChanged.connect(self.InputTable)
        self.NumberOfColours.valueChanged.connect(self.calculate_click)
        self.NumberOfColours.setPrefix("Number of colours: ")
        BeamParameterLayout.addWidget(self.NumberOfColours)

        self.header_names = ['Energy', 'Bandwidth', 'Source position', 'Beam divergence']
        self.BeamParameters = QTableWidget(4,1)
        self.BeamParameters.setFixedHeight(122)
        self.BeamParameters.horizontalHeader().hide()
        self.BeamParameters.setVerticalHeaderLabels([f'{a} [{DefaultValues[a][1]}]' for a in self.header_names])
        self.BeamParameters.cellChanged.connect(self.calculate_click)
        self.InputTable()

        BeamParameterLayout.addWidget(self.BeamParameters)

        return BeamParameterLayout

    def InputTable(self):
        self.BeamParameters.cellChanged.disconnect()
        ncols = self.NumberOfColours.value()
        self.BeamParameters.setColumnCount(ncols)
        for row, rowname in enumerate(self.header_names):
            for col in range(ncols):
                val = self.BeamParameters.item(row,col)
                if not val:
                    item = QTableWidgetItem(str(DefaultValues[rowname][0]))
                    self.BeamParameters.setItem(row,col,item)
        for col in range(ncols):
            header = self.BeamParameters.horizontalHeader()
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        
        self.BeamParameters.cellChanged.connect(self.calculate_click)

    def CRLControls(self):
        
        CRLLayout = QGridLayout()
        CRLLayout.setSpacing(2)

        self.CRLStatus = dict()
        for crl_set in range(1,4):
            self.CRLStatus[crl_set] = dict()
            CRLLayout.addWidget(QLabel(f"CRL{crl_set} lens arms:"), crl_set-1, 0)
            groupBox = QGroupBox()
            grid = QGridLayout()
            grid.setSpacing(0)
            for ind, arm in enumerate(CRL.loc[crl_set,:].index):
                if np.array_equal(CRL.loc[(crl_set, arm)].NumberOfLenses, 0):
                    continue
                CheckBox = QCheckBox(str(arm))
                self.CRLStatus[crl_set][arm] = CheckBox
                CheckBox.stateChanged.connect(self.calculate_click)
                grid.addWidget(CheckBox,ind//5, ind%5)
            groupBox.setLayout(grid)
            CRLLayout.addWidget(groupBox, crl_set-1, 1, 1, 2)
        
        self.CRL3_shift = QLineEdit(str(DefaultValues["CRL3 z shift"][0]))
        self.CRL3_shift.setValidator(QDoubleValidator(-500., 500.,1))
        self.CRL3_shift.editingFinished.connect(self.calculate_click)
        CRLLayout.addWidget(QLabel("CRL3 z shift [mm]:"), 3, 0)
        CRLLayout.addWidget(self.CRL3_shift, 3, 1, 1, 2)

        self.CRL4_lenses = QSpinBox()
        #self.CRL4_lenses = QLineEdit(str(DefaultValues["CRL4 lenses"][0]))
        self.CRL4_lenses.setRange(0, 40)
        self.CRL4_lenses.valueChanged.connect(self.updateCRL4Lenses)
        self.CRL4_lenses.valueChanged.connect(self.calculate_click)
        CRLLayout.addWidget(QLabel("CRL4 lens stack (# 0.125mm RoC C):"), 4, 0)
        CRLLayout.addWidget(self.CRL4_lenses, 4, 1, 1, 2)
        
        self.CRL4_shift = QLineEdit(str(DefaultValues["CRL4 z shift"][0]))
        self.CRL4_shift.setValidator(QDoubleValidator(-1000.,20000,1))
        self.CRL4_shift.editingFinished.connect(self.calculate_click)
        CRLLayout.addWidget(QLabel("CRL4 dist from TCC1 [mm]:"), 5, 0)
        CRLLayout.addWidget(self.CRL4_shift, 5, 1, 1, 2)
        
        
        self.TestPosition = QLineEdit(str(ComponentPositions["TCC1"]))
        self.TestPosition.editingFinished.connect(self.calculate_click)
        self.TestPosition.setValidator(QDoubleValidator())

        self.LoadTestPosition = QComboBox()
        PositionList = sorted(list(ComponentPositions.index) + ['CRL4'])
        self.LoadTestPosition.addItems(PositionList)
        self.LoadTestPosition.setCurrentText('TCC1')
        self.LoadTestPosition.currentIndexChanged.connect(self.setTestPosition)

        CRLLayout.addWidget(QLabel("Chosen pos. [m]:"), 6, 0)
        CRLLayout.addWidget(self.LoadTestPosition,6,1)
        CRLLayout.addWidget(self.TestPosition, 6, 2)

        return CRLLayout
    
    def setTestPosition(self):
        key = self.LoadTestPosition.currentText()
        if 'CRL' in key:
            CRL_pos = self.GetCRLPos()
            val = CRL_pos[int(key[-1])]
        else:
            val = ComponentPositions[key]
        self.TestPosition.setText(str(val))
        self.calculate_click()

    def update_plot(self):
        
        Styles = {'CRL': {'color': 'grey', 'linestyle': '--', 'alpha': .5},
                  'Apertures': {'color': 'grey', 'linestyle': '-'},
                  'TCC': {'color': 'red', 'linestyle': '--'},
                  'Mirrors': {'color': 'pink', 'linestyle': '--'},
                  'Monos': {'color': 'lightgreen', 'linestyle': '--'},
                  'Shutter': {'color': 'purple', 'linestyle': '--'}}
        CRLs_in = self.GetCRLStatus()

        axes = [self.ax1, self.ax2]
        Titles = ['XTD1/XTD6 Tunnel', 'OPT/EXP Hutch']
        xlims = [[-50, 1000], [935, 985]]
        self.ax1.set_xlim(-50, 1000)

        BeamColors = ['black', 'orange', 'maroon']
        for ax, title, xl in zip(axes, Titles, xlims):
            xlim = ax.get_xlim()
            if not self.FixXAxes.isChecked():
                xlim = xl
            ylim = ax.get_ylim()
            ax.cla()  # Clear the canvas.
            ax.set_title(title)
            ax.set_ylabel("Beam diameter y (um)")
            yminlim, ymaxlim = 0, 0
            Beamline = list()
            for ind, x in self.SamplePositions.items():
                y = self.BeamPositions[ind]
                Beamline.append(ax.plot(x, y[:,1], color=BeamColors[ind%len(BeamColors)], label=f"Beam {ind+1}, {self.BeamPars.Energy[ind]:.0f}eV")[0])

                ax.fill_between(x,y[:,0],y[:,2],linestyle='None',color=BeamColors[ind%len(BeamColors)], alpha=.2)
                
                final_crl = max([1] + [crl for crl, lenses in CRLs_in.items() if len(lenses)])
                final_focus = self.ImageDist[ind][self.BeamPars.Energy[ind]][final_crl]
                final_focus = max(self.CRL_pos[3],final_focus)
                cond = self.SamplePositions[ind] < final_focus
                ymaxlim = np.max(np.append(ymaxlim, y[cond,1]*1.2))
            if self.FixYAxes.isChecked():
                yminlim, ymaxlim = ylim
            ax.set_xlim(xlim)
            ax.set_ylim(yminlim,ymaxlim)
            ax.yaxis.set_ticks_position('both')

            for crl, pos in self.CRL_pos.items():
                if ax == self.ax1 and crl == 4:
                    continue
                ax.axvline(pos, **Styles['CRL'])
                ax.text(pos, yminlim + .99*(ymaxlim-yminlim), f'CRL{crl}', color=Styles['CRL']['color'],rotation=90, verticalalignment='top', horizontalalignment='right')
                aperture = max(CRL.loc[crl]['Aperture'])
                if len(CRLs_in[crl]):
                    index = pd.MultiIndex.from_product([[crl],CRLs_in[crl]],names=['Set', 'Arm'])
                    aperture = min(CRL.loc[index]['Aperture'])
                ax.vlines(pos, 0, aperture, **Styles['Apertures'])
        self.ax1.axvline(ComponentPositions['TCC1'], **Styles['TCC'])

        MonoPositions = sorted([position for component, position in ComponentPositions.items() if component in ['Mono', 'HR-mono']])
        for pos in MonoPositions:
            self.ax1.axvline(pos, **Styles['Monos'])
        self.ax1.text(pos, yminlim + .1*(ymaxlim-yminlim), 'Monos', color=Styles['Monos']['color'], rotation=90, verticalalignment='bottom', horizontalalignment='right')

        MirrorPositions = sorted([position for component, position in ComponentPositions.items() if 'Mirror' in component])
        for pos in MirrorPositions:
            self.ax1.axvline(pos, **Styles['Mirrors'])
        self.ax1.text(pos, yminlim + .98*(ymaxlim-yminlim), 'Mirrors', color=Styles['Mirrors']['color'], rotation=90, verticalalignment='top', horizontalalignment='right')

        ShutterPositions = {component: position for component, position in ComponentPositions.items() if component in ['XTD6 shutter', 'OPT shutter','IBS Beamstop']}
        for component, pos in ShutterPositions.items():
            self.ax2.axvline(pos, **Styles['Shutter'])
            self.ax2.text(pos, yminlim + .98*(ymaxlim-yminlim), component.split(' ')[0], color=Styles['Shutter']['color'], rotation=90, verticalalignment='top', horizontalalignment='right')

        TCCPositions = {component: position for component, position in ComponentPositions.items() if 'TCC' in component}
        for component, pos in TCCPositions.items():
            self.ax2.axvline(pos, **Styles['TCC'])
            if len(component) == 4:
                self.ax2.text(pos, yminlim + .8*(ymaxlim-yminlim), component, color=Styles['TCC']['color'], rotation=90, verticalalignment='top', horizontalalignment='right')

        if len(Beamline)>1:
            self.ax2.legend(handles=Beamline, loc='lower left')
        self.ax2.set_xlabel("Distance from nominal source (m)")

        self.plot.draw()
    
    def updateCRL4Lenses(self):
        CRL.loc[(4,1),'NumberOfLenses'] = self.CRL4_lenses.value()

    def GetCRLStatus(self):
        CRLs_in = {crl_set: [arm for arm, box in crl_stack.items() if box.isChecked()] for crl_set, crl_stack in self.CRLStatus.items()}
        CRLs_in[4] = []
        if CRL.loc[(4,1),'NumberOfLenses']:
            CRLs_in[4] = [1]
        return CRLs_in

    def GetCRLPos(self):
        CRL4_shift = float(self.CRL4_shift.text())*1e-3
        CRL3_shift = float(self.CRL3_shift.text())*1e-3
        return {1: ComponentPositions['CRL1'],
                2: ComponentPositions['CRL2'],
                3: ComponentPositions['CRL3'] + CRL3_shift,
                4: ComponentPositions['TCC1'] + CRL4_shift
        }


    def GetBeamPars(self):
        ncols = self.NumberOfColours.value()
        BeamPars = list()
        for col in range(ncols):
            BeamPars.append(dict())
            for row, rowname in enumerate(self.header_names):
                val = self.BeamParameters.item(row,col)
                val = val.text() if val is not None else str(DefaultValues[rowname][0])
                try:
                    BeamPars[col][rowname] = float(val)
                except ValueError:
                    BeamPars[col][rowname] = DefaultValues[rowname][0]
        BeamPars = pd.DataFrame(BeamPars)
        return BeamPars



    def calculate_click(self):

        self.CRL_pos = self.GetCRLPos()
        TestPosition = float(self.TestPosition.text())

        CRLs_in = self.GetCRLStatus()

        self.BeamPars = self.GetBeamPars()

        self.SamplePositions, self.BeamPositions, self.CRL_f, self.ImageDist, self.DiffLim = calculate(self.BeamPars, CRLs_in, self.CRL_pos, TestPosition)
        
        self.UpdateWarnings(CRLs_in)
        self.update_plot()
        
    def UpdateWarnings(self,CRLs_in):
        crls_in = [crl for crl, lenses in CRLs_in.items() if len(lenses)]
        Ncrls = len(crls_in)
        self.CRLOut.setRowCount(3*Ncrls)
        self.CRLOut.setColumnCount(len(self.CRL_f))
        
        for col in range(len(self.CRL_f)):
            header = self.CRLOut.horizontalHeader()
            header.setSectionResizeMode(col, QHeaderView.Stretch)

        Text = ''
        if Ncrls:
            self.CRLOut.setHorizontalHeaderLabels([f'Beam {i+1}' for i in self.BeamPositions])
            self.CRLOut.setVerticalHeaderLabels([f'CRL{crl} f' for crl in crls_in] + \
                                       [f'CRL{crl} image' for crl in crls_in] + \
                                       [f'CRL{crl} spot' for crl in crls_in])
            for beam_ind in self.CRL_f:
                for ind, crl in enumerate(crls_in):
                    Energies = sorted(self.CRL_f[beam_ind])
                    item = QTableWidgetItem(f"{self.CRL_f[beam_ind][Energies[1]][crl]:.4g} m")
                    self.CRLOut.setItem(ind,beam_ind,item)
                    ImagePosition = self.ImageDist[beam_ind][Energies[1]][crl]
                    item = QTableWidgetItem(f"{ImagePosition:.4g} m")
                    self.CRLOut.setItem(ind+Ncrls,beam_ind,item)

                    ImSize = self.DiffLim[beam_ind][Energies[1]][crl]
                    item = QTableWidgetItem(f"{ImSize:.3g} " + mu + "m")
                    self.CRLOut.setItem(ind+2*Ncrls,beam_ind,item)
                if ImagePosition < max(self.SamplePositions[beam_ind]):
                    ChromaticWidth = np.interp(ImagePosition,self.SamplePositions[beam_ind],self.BeamPositions[beam_ind][:,2])
                    Text += f"Beam {beam_ind+1} focus at {ImagePosition:.1f} m = {ImSize:.3g} + {ChromaticWidth-ImSize:.3g} {mu}m chromaticity\n"
                    TCC_targets = {k: v for k, v in ComponentPositions.items() if 'TCC' in k}
                    if crl in [3, 4]:
                        if crl == 3:
                            CRL_shift = float(self.CRL3_shift.text())
                            TCC_targets = {k: v for k, v in TCC_targets.items() if abs((v-ImagePosition)*1e3 + CRL_shift) < 500}
                        if crl == 4:
                            CRL_shift = float(self.CRL4_shift.text())
                            TCC_targets = {k: v for k, v in TCC_targets.items() if abs((v-ImagePosition)*1e3) < 500}
                        for tcc, pos in TCC_targets.items():
                            Text += f"Move CRL{crl} {(pos-ImagePosition)*1e3:+.0f}mm to {(pos-ImagePosition)*1e3 + CRL_shift:.0f}mm to align beam {beam_ind+1} to {tcc}\n"
        TestPosition = float(self.TestPosition.text())
        WarningSize = 200
        for beam_ind in self.CRL_f:
            TestSize = np.interp(TestPosition, self.SamplePositions[beam_ind],self.BeamPositions[beam_ind][:,1])
            ChromaticWidth = np.interp(TestPosition, self.SamplePositions[beam_ind],self.BeamPositions[beam_ind][:,2])
            Text += f"Beam {beam_ind+1} size at {TestPosition:.1f} m = {TestSize:.3g} + {ChromaticWidth-TestSize:.3g} {mu}m chromaticity\n"
        
        BeamPars = EstimateLensTransmission(self.BeamPars, CRLs_in, self.GetCRLPos())
        Text += '\n'
        for ind, beam in BeamPars.iterrows():
            Text += f'Estimated lens transmission beam {ind+1} = {beam.Transmission*1e2:.3g}%\n'
        Text += '\n'
        CompPos = ComponentPositions.copy()
        CRL_pos = self.GetCRLPos()
        for crl, lenses in CRLs_in.items():
            if len(lenses):
                CompPos[f'CRL{crl}'] = CRL_pos[crl]
        for beam_ind in self.CRL_f:
            for component, position in CompPos.items():
                if 'TCC' not in component:
                    Size = np.interp(position, self.SamplePositions[beam_ind],self.BeamPositions[beam_ind][:,1])
                    if Size < WarningSize:
                        Text += f"WARNING: Beam {beam_ind+1} size at {component} is {Size:.0f} {mu}m\n"
        self.Warnings.setText(Text)

    def calculatebd_click(self):
        
        self.BeamPars = self.GetBeamPars()
        CRLs_in = self.GetCRLStatus()
        FELBeamSize = float(self.FEL_beamsize_um.text())
        HEDBeamSize = float(self.HED_beamsize_um.text())
        SPText, BDText = [], []
        for ind, beam in self.BeamPars.iterrows():
            SourcePos, BeamDiv, SPerr, BDerr = calc_div(beam.Energy, CRLs_in[1], FELBeamSize, HEDBeamSize)
            SPText.append(f'Beam {ind+1}: {SourcePos:.4g} {chr(0x00b1)} {SPerr:.2g} m')
            BDText.append(f'Beam {ind+1}: {BeamDiv:.4g} {chr(0x00b1)} {BDerr:.2g} {mu}rad')
        self.SourcePositionOutput.setText('\n'.join(SPText))
        self.BeamDivergenceOutput.setText('\n'.join(BDText))
        
        
"""Plotting"""
class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        fig = Figure()#figsize=(width, height), dpi=dpi)
        self.axes = fig.subplots(2,1)
        super(Canvas, self).__init__(fig)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())
