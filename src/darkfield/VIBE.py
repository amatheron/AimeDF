#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------- VIBE (Vacuum Induced Birefringence Explorer) code written by Aimé MATHERON and Michal Smid, Pooyan Khademi and Felix Karbstein--------------------
#----------------------------------------- Latest updated : September 2025. All rights reserved. -------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

import h5py, json
from LightPipes import *
import numpy as np
import sys
import os
import re
import time
from LightPipes import Field

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from astropy.io import ascii
from PIL import Image
from scipy import signal
from scipy.ndimage import map_coordinates

from pathlib import Path
from typing import Optional

import argparse, yaml, warnings

from skimage.transform import resize
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy.constants import e, m_e, epsilon_0, hbar, c, h, pi
from scipy.special import j1
from scipy.interpolate import RegularGridInterpolator

import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils_v2 as mu
import darkfield.regularized_propagation_v2 as rp

from dataclasses import dataclass
from typing import Dict   # only for type hints

from numpy.polynomial.hermite import hermgauss
from scipy.special import wofz  # Faddeeva function (robust for complex)
from scipy.interpolate import RegularGridInterpolator
import darkfield.wavefront_fitting as wft  # your local module

def _select_backend():
    os.environ.pop("MPLBACKEND", None)
    if os.environ.get("DISPLAY", "") == "":
        print("No display detected: using non-interactive 'Agg' backend.")
        matplotlib.use("Agg")
    else:
        print("Display detected: using 'TkAgg' backend.")
        matplotlib.use("TkAgg")


# --------------- Load the input file ---------------
def load_cfg(yaml_path: str) -> dict:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ------------ Create the list of optical elements from the input file ---------
def elements_from_cfg(cfg: dict) -> list:
    elems = []
    for name, obj in cfg.items():
        if name in ("Xbeam", "IRLaser", "simulation", "meta"):
            continue
        obj = dict(obj)
        obj["element_name"] = name
        elems.append([obj["position"], name, obj])
    elems.sort(key=lambda e: e[0])  # sort by z-position
    return elems


# ------------ Read the yaml file values with fallback possibility ---------
def yamlval(k, d, default):
    return d[k] if k in d and d[k] is not None else default

# ------------ Build the input parameters list from the yaml file ---------
def build_input_params(cfg: dict, *, N: int, projectdir: str, filename: str) -> dict:
    X  = cfg.get("Xbeam", {})
    IR = cfg.get("IRLaser", {})
    S  = cfg.get("simulation", {})

    return {
        # ---- identifiers & paths ----
        "N": int(N),
        "filename": filename,
        "projectdir": projectdir,

        # ---- core simulation inputs (X-ray & grid) ----
        "photon_energy": float(yamlval("photonenergy", X, 8766)),   # fallback 8766 eV
        "beamsize":      float(yamlval("size", X, 0.002)),          # fallback 2 mm
        "gauss_x_shift": float(yamlval("offset", X, 0.0)),
        "gauss_x_tilt":  float(yamlval("tilt", X, 0.0)),
        "propsize":      float(yamlval("propsize", S, 0.0)),
        "simulation_type": 0,

        # ---- intensity units & zoom ----
        "intensity_units": yamlval("intensity_units", S, "relative"),
        "Zoom_global":     float(yamlval("Zoom_global", S, 1)),

        # ---- plotting knobs used throughout main_VIBE ----
        "fig_rows": 4, "fig_cols": 5,
        "remove_ticks": 1,
        "fig_start": 3,
        "profiles_subfig": 1,
        "ax_apertures": None,

        # ---- optional flow/figs controls ----
        "figs_to_save":   yamlval("figs_to_save", S, []),
        "figs_to_export": yamlval("figs_to_export", S, []),
        "figs_log":       yamlval("figs_log", S, 1),
        "flow_auto_save": yamlval("flow_auto_save", S, 0),
        "flow_plot_gyax": yamlval("flow_plot_gyax", S, None),
        "flow_plot_clim": yamlval("flow_plot_clim", S, None),
        "profiles_xlim":  yamlval("profiles_xlim", S, [0, 200]),
        "intensity_ylim": yamlval("intensity_ylim", S, [1e-10, 2.0]),
        "apertures_ylim": yamlval("apertures_ylim", S, [1e-10, 2.0]),
        "edge_damping":        yamlval("edge_damping", S, None),
        "edge_damping_shape":  yamlval("edge_damping_shape", S, None),
        "method":              yamlval("method", S, "FFT"),
        "subfigure_size_px":   yamlval("subfigure_size_px", S, 300),
        "flow":                yamlval("flow", S, None),

        # ---- X-ray pulse scaling ----
        "photons_total":   float(yamlval("photons_total", X, 2e11)),
        "X_FWHM_duration": float(yamlval("X_FWHM_duration", X, 25e-15)),

        # ---- IR block (VB mask @ TCC) ----
        "IR_Energy_J":       float(yamlval("IR_Energy_J", IR, 4.8)),
        "IR_FWHM_duration":  float(yamlval("IR_FWHM_duration", IR, 30e-15)),
        "IR_wavelength":     float(yamlval("IR_wavelength", IR, 800e-9)),
        "IR_x_offset_m":     float(yamlval("IR_x_offset_m", IR, 0.0)),
        "IR_y_offset_m":     float(yamlval("IR_y_offset_m", IR, 0.0)),
        "Timing_jitter":     float(yamlval("Timing_jitter", IR, 0.0)),
        "IR_FWHM_gaussian":  float(yamlval("IR_FWHM_gaussian", IR, 1.3e-6)),
        "IR_2Dmap":          yamlval("IR_2Dmap", IR,
                                     ["gaussian", "match_integral", None, None]),
    }





# ------------------------------------------------------------------
# --- DABAM2D surface loader ---------------------------------------
# ------------------------------------------------------------------



# ---- DABAM2D loader (robust) ------------------------------------


def _read_dabam_meta(txt_path: Path) -> dict:
    if not txt_path.exists():
        return {}
    raw = txt_path.read_text()
    # 1) try JSON
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 2) try Python-literal dict (DABAM2D README shows this style)
    try:
        meta = ast.literal_eval(raw)
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass
    # 3) very simple "key: value" fallback
    meta = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta

def load_dabam2d(index: int, dabam_root: str | Path = None):
    """
    Load a DABAM2D surface by index.
    Returns: X (Nx,), Y (Ny,), Z (Ny,Nx), meta (dict)
    """
    if dabam_root is None:
        dabam_root = Path(__file__).parent / "Dabam2D" / "data"
    else:
        dabam_root = Path(dabam_root)

    stem = f"dabam2d-{index:04d}"
    h5_path  = dabam_root / f"{stem}.h5"
    txt_path = dabam_root / f"{stem}.txt"

    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found. Clone the DABAM2D repo into src/darkfield/Dabam2D.")

    with h5py.File(h5_path, "r") as f:
        X = f["surface_file/X"][()]  # horizontal
        Y = f["surface_file/Y"][()]  # vertical
        Z = f["surface_file/Z"][()]  # shape (Y.size, X.size)

    meta = _read_dabam_meta(txt_path)
    return X, Y, Z, meta


def build_dabam_defect_thickness_map(el_dict, params, xm, ym, A_m):
    """
    Returns a thickness *delta* [m] on the simulation grid (same shape as xm/ym)
    to be ADDED to the ideal CRL thickness. Behavior controlled by YAML keys:

      defects: 1/0
      experimental_data: "dabam2d-0021" or int index 21
      defect_type: 1 (Zernike fit only) | 2 (residues only) | 3 (fit + residues)
      remove_avg_profile: 1/0
      Custom_zernike: [flag, noll_1, val_1_m, noll_2, val_2_m, ...]
      nmodes: int (default 37)
      startmode: int (default 1)

    Notes:
      • The DABAM map is rescaled to fill a circle of radius A/2 (aperture).
      • The result is masked outside the aperture.
    """
    import re
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    import darkfield.wavefront_fitting as wft  # local module

    # ---- YAML options -----------------------------------------------
    defects_flag   = int(el_dict.get("defects", 0))
    expdata_raw    = el_dict.get("experimental_data", None)
    defect_type    = int(el_dict.get("defect_type", 3))
    remove_avg     = int(el_dict.get("remove_avg_profile", 0))
    custom_list    = el_dict.get("Custom_zernike", [])
    nmodes         = int(el_dict.get("nmodes", 37))
    startmode      = int(el_dict.get("startmode", 1))

    # try to grab a human-friendly element name
    elem_name = el_dict.get("element_name", el_dict.get("name", el_dict.get("label", "Custom_CRL")))

    custom_active  = (
        isinstance(custom_list, (list, tuple)) and len(custom_list) >= 1 and int(custom_list[0]) == 1
    )

    # Parse DABAM index (prefer 'dabam2d-XXXX', else last digits in the string)
    idx = None
    if isinstance(expdata_raw, (int, np.integer)):
        idx = int(expdata_raw)
    elif isinstance(expdata_raw, str):
        s = expdata_raw.strip()
        m = re.search(r"dabam2d-(\d+)$", s, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
        else:
            nums = re.findall(r"(\d+)", s)
            if nums:
                idx = int(nums[-1])


    # concise config line
    print(
        "[DABAM] "
        f"defects={'on' if defects_flag==1 else 'off'} | "
        f"experimental_data={expdata_raw if expdata_raw is not None else 'n/a'}"
        f"{f' (idx={idx})' if idx is not None else ''} | "
        f"defect_type={defect_type} ({_defect_type_label(defect_type)}) | "
        f"remove_avg_profile={'on' if remove_avg==1 else 'off'} | "
        f"Custom_zernike={'on' if custom_active else 'off'} | "
        f"nmodes={nmodes} | startmode={startmode}"
    )

    # Bail out if not requested
    if defects_flag != 1:
        return np.zeros_like(xm)

    if idx is None:
        print("[DABAM] Could not parse a valid index from 'experimental_data' — no defects applied.")
        return np.zeros_like(xm)

    # --- Load DABAM map (meters) + metadata
    X, Y, Z, meta = load_dabam2d(idx)  # Z in meters, shape (Ny, Nx)
    Ny, Nx = Z.shape

    # ---- Pretty-print metadata block --------------------------------
    def _coerce_to_float(s):
        try:
            return float(str(s))
        except Exception:
            return None

    # order a few important keys first, then the rest (non-null only)
    ordered_keys = [
        "USER_REFERENCE", "MATERIAL", "FACILITY", "INSTRUMENT",
        "RS", "RM", "WIDTH", "LENGTH", "FOCUS_DIR", "SCAN_DATE",
        "YEAR_FABRICATION", "SURFACE_SHAPE", "FUNCTION", "THICK",
        "SUBSTRATE", "COATING", "POLISHING", "ENVIRONMENT",
    ]
    remaining = [k for k in sorted(meta.keys()) if k not in ordered_keys]

    print(f"[DABAM] {elem_name} simulated via {expdata_raw} (idx={idx}) with properties:")
    for k in ordered_keys + remaining:
        if k not in meta:
            continue
        v = meta[k]
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        # pretty units for common metric fields that arrive as strings
        if k in ("RS", "RM", "WIDTH", "LENGTH", "THICK"):
            vf = _coerce_to_float(v)
            if vf is not None:
                print(f"        {k:<14}: {vf:.6g} m")
                continue
        print(f"        {k:<14}: {v}")

    # --- Zernike decomposition --------------------------------------
    Zcoeffs, Zfit, Zres = wft.fit_zernike_circ(
        Z, nmodes=nmodes, startmode=startmode, rec_zern=True
    )

    el_dict["Z_coeffs_native"]  = np.asarray(Zcoeffs).copy()
    el_dict["nmodes_used"]      = int(nmodes)
    el_dict["startmode_used"]   = int(startmode)

    Zres = -Zres # The way Zres is defined in fit_zernike_circ is wrong in sign. We refine it such that Z = Zres+Zfit

    Zres_unproc = Zres.copy()                    # save before any optional processing
    el_dict["Z_residues_native_unproc"] = Zres_unproc

    # --- Optional azimuthal-average removal on residues --------------
    residues_map = Zres
    if remove_avg == 1:
        I_thick_res, R = wft.average_azimuthal(residues_map, X, Y)
        _, residues_map = wft.remove_avg_profile(residues_map, None, X, Y, I_thick_res, R, 'b')
    el_dict["remove_avg_applied"] = int(remove_avg == 1)
    el_dict["Z_residues_native"] = residues_map


    # --- Zernike choice logic (custom vs fitted) ---------------------
    if custom_active:
        pairs = list(custom_list[1:])
        if len(pairs) % 2 != 0:
            raise ValueError("Custom_zernike must contain pairs: [1, n1, v1, n2, v2, ...]")
        nolls = [int(pairs[i])   for i in range(0, len(pairs), 2)]
        vals  = [float(pairs[i]) for i in range(1, len(pairs), 2)]
        max_n = max(nolls) if nolls else 0

        zvec = np.zeros(max(1, max_n), dtype=float)
        for n, v in zip(nolls, vals):
            zvec[n-1] = v  # Noll indices start at 1

        # prefer odd->even safety to match typical DABAM even sizes
        rad_px   = (min(Nx, Ny) - 1) // 2
        Zfit_map = wft.calc_zernike_circ(zvec, rad=rad_px, mask=True)
    else:
        Zfit_map = Zfit  # LSQ fitted Zernike map

    # --- FORCE Zfit_map to match native DABAM size Z.shape (center crop/pad) ---
    ty, tx = Z.shape
    ay, ax = Zfit_map.shape

    # center-crop if Zfit_map is larger
    if ay > ty:
        y0 = (ay - ty) // 2
        Zfit_map = Zfit_map[y0:y0 + ty, :]
    if ax > tx:
        x0 = (ax - tx) // 2
        Zfit_map = Zfit_map[:, x0:x0 + tx]

    # recompute after potential crop
    ay, ax = Zfit_map.shape
    py_top  = (ty - ay) // 2
    py_bot  = (ty - ay) - py_top
    px_left = (tx - ax) // 2
    px_right= (tx - ax) - px_left

    if py_top or py_bot or px_left or px_right:
        Zfit_map = np.pad(
            Zfit_map,
            ((py_top, py_bot), (px_left, px_right)),
            mode='constant',
            constant_values=0.0
        )
    # ---------------------------------------------------------------------------
    # --- after Zfit_map has the same shape as Z (native DABAM grid) ---
    el_dict["Z_fit_panel_map"]   = Zfit_map.copy()     # what the "Zernike fit" panel should show
    el_dict["Zfit_is_custom"]    = 1 if custom_active else 0
    el_dict["dabam_X"]           = X                   # keep coordinates for the plot
    el_dict["dabam_Y"]           = Y
    el_dict["Z_raw_native"]      = Z                   # optional: raw map for the panel
    el_dict["Z_residues_native"] = residues_map        # optional: residues (post-processing if any)


    # --- Compose the defect thickness BEFORE resampling --------------
    if defect_type == 1:
        Z_def_src = Zfit_map
    elif defect_type == 2:
        Z_def_src = residues_map
    elif defect_type == 3:
        Z_def_src = Zfit_map + residues_map
    else:
        print(f"[DABAM] Unknown defect_type={defect_type}, defaulting to Zernike+residues.")
        Z_def_src = Zfit_map + residues_map

    # --- Map DABAM -> simulation grid --------------------------------
    Rx = 0.5 * (X.max() - X.min())
    Ry = 0.5 * (Y.max() - Y.min())
    Rsrc  = min(Rx, Ry) if (Rx > 0 and Ry > 0) else 1.0         # DABAM half-size (min axis)
    Rdst  = A_m / 2.0                                            # simulation lens radius
    rescale_flag = int(el_dict.get("rescale_dabam", 1))

    # build interpolator on native (Y, X) coordinates
    interp = RegularGridInterpolator((Y, X), Z_def_src, bounds_error=False, fill_value=0.0)

    if rescale_flag == 1:
        # --- RESCALE: stretch/shrink to fill the aperture -------------
        scale = Rdst / Rsrc if Rsrc > 0 else 1.0
        pts = np.column_stack([(ym / scale).ravel(), (xm / scale).ravel()])
        Z_def_resampled = interp(pts).reshape(xm.shape)

        policy = "rescale: fill aperture"
        note   = "stretch to fill" if scale > 1 else ("shrink to fill" if scale < 1 else "1:1")

    else:
        # --- NO RESCALE: sample in native metric coordinates ----------
        # points are directly the simulation coords (same unit: meters)
        scale = 1.0
        pts = np.column_stack([ym.ravel(), xm.ravel()])
        Z_def_resampled = interp(pts).reshape(xm.shape)

        policy = "no-rescale: keep native size"
        note   = "DABAM smaller than lens ⇒ zero defects outside" if Rsrc < Rdst else \
                 ("DABAM larger than lens ⇒ masked at lens edge" if Rsrc > Rdst else "1:1 size")

        # Loud warning if DABAM disk is smaller than lens (you’ll have a “clean” ring)
        if Rsrc < Rdst:
            print("="*72)
            print("[DABAM][WARNING] DABAM radius is SMALLER than the lens radius (no-rescale mode).")
            print(f"           Rsrc = {Rsrc*1e6:.1f} µm < Rdst = {Rdst*1e6:.1f} µm")
            print("           Outside the DABAM disk there will be NO defects (zeros).")
            print("           Set 'rescale_dabam: 1' to fill the whole lens with scaled defects.")
            print("="*72)

    # Mask outside aperture (always)
    r = np.sqrt(xm**2 + ym**2)
    Z_def_resampled[r > Rdst] = 0.0

    # Store for diagnostics (incl. plotting extent and resample meta)
    el_dict["Z_def_resampled"]    = Z_def_resampled
    el_dict["Z_interp_extent_um"] = [xm.min()*1e6, xm.max()*1e6, ym.min()*1e6, ym.max()*1e6]
    el_dict["Z_resample_info"]    = {
        "policy": policy,
        "note": note,
        "rescale_flag": int(rescale_flag),
        "Rsrc_um": float(Rsrc*1e6),
        "Rdst_um": float(Rdst*1e6),
        "scale": float(scale),
    }

    # Log what we did
    print(f"[DABAM] Resample policy: {policy} | {note} "
          f"| Rsrc={Rsrc*1e6:.1f} µm → Rdst={Rdst*1e6:.1f} µm | scale={scale:.3f}")
    
    print(
        "[DABAM] Radii: "
        f"Rsrc={Rsrc*1e6:.1f} µm (from DABAM WIDTH/LENGTH), "
        f"Rdst={Rdst*1e6:.1f} µm (from A/2; A treated as DIAMETER), "
        f"scale={scale:.3f}, rescale_dabam={rescale_flag}"
    )

    return Z_def_resampled





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
    rules that the monolithic `main_VIBE` loop currently applies to `F`.
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
    If table_dir is not specified, it is assumed to be next to VIBE.py
    """
    global _index_cache

    # Special case for Hafnium
    if elem == 'Hf':
        return 3.2887e-6 , 1.988e-5
    
    if elem=='W':  
        return 2.85704482E-06 , 3.86332977E-05
    
    if elem=='Au':   #Gold (79)
        return 3.55916109E-06 , 3.95017742E-05
    
    # Locate optical_constants folder relative to the file location of VIBE.py
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


#---------------- ADDS THE DEFECTS FOR CRLS FROM CELESTRE AND SEIBOTH -----------------

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
    if positive:
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
    if np.size(edge_damping_pixels)==1: #doing sine damping #first number is fraction of N where the damping starts
        N_edge=int(N*edge_damping_pixels[0])
        x=np.arange(N_edge)/N_edge*(np.pi/2)
        y=np.sin(x)
        if edge_damping_shape=='square':
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
                x2=N_through+np.flip(np.arange(N_edge))
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
    plt.imshow(imgC,norm=norm,cmap=rofl.cmap(),extent=extent)
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
    



def _sanitize_intensity_map(img: np.ndarray) -> np.ndarray:
    """
    Build a map with no negative values, no NaNs, no infinites, 
    to be able to be used as an external map for laser and xray profiles
    """
    # Convert to float, collapse RGB if needed
    arr = img.astype(float)
    if arr.ndim == 3:  # e.g., RGB
        # luminance or simple mean; choose what you prefer
        arr = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]

    # Replace NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Shift if negative baseline (e.g. high-pass artifacts)
    minv = arr.min()
    if minv < 0:
        arr = arr - minv  # makes min 0

    # Clip negative crumbs
    arr = np.clip(arr, 0.0, None)

    return arr





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



# --------------------------------------------------------------------------
# ----------- GAZJET: plasma density → phase and transmission maps ---------
# --------------------------------------------------------------------------
def build_gazjet_maps(el_dict, params, F=None):
    """
    Build the gas-jet phase map (and ~unity transmission).
    Treats the plasma as n = 1 - δ (β≈0), so:
        phase(x,y) = -k * δ(x,y) * L
        trans(x,y) ≈ 1
    """
    # --- read gazjet params ---
    dens_cc = float(el_dict.get("density", 1e18))            # [cm^-3], electron density
    profile = str(el_dict.get("profile", "gaussian")).lower()
    size_t  = float(el_dict.get("Size_transv", 500e-6))      # [m] (FWHM if gaussian, full width if rectangle)
    size_l  = float(el_dict.get("Size_long",  500e-6))       # [m]
    lam     = float(params.get("wavelength", 1e-10))         # [m] (default 1 Å)
    k0      = 2.0*np.pi/lam

    # --- grid (N, dx) from the current field if available ---
    if F is not None:
        N = int(getattr(F, "N"))
        if hasattr(F, "ps"):
            dx = float(F.ps)
        elif hasattr(F, "grid_size") and hasattr(F, "N"):
            dx = float(F.grid_size) / float(F.N)
        else:
            propsize = float(params["propsize"])
            dx = propsize / N
    else:
        N = int(params["N"])
        propsize = float(params["propsize"])
        dx = propsize / N

    # --- electron density in m^-3 ---
    n_e = dens_cc * 1e6

    # --- plasma index decrement: δ = n_e e² / (2 ε0 m_e ω²) ---
    omega  = 2.0*np.pi*c/lam
    delta0 = n_e * e**2 / (2.0*epsilon_0*m_e*omega**2)

    # --- transverse profile ---
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")

    if profile == "gaussian":
        sigma = size_t / (2.0*np.sqrt(2.0*np.log(2.0)))
        density_map = np.exp(-(X**2 + Y**2) / (2.0*sigma**2))
    else:  # rectangle
        half = size_t / 2.0
        density_map = np.where((np.abs(X) < half) & (np.abs(Y) < half), 1.0, 0.0)

    # --- phase & transmission ---
    delta_map = delta0 * density_map
    phase_map = -k0 * delta_map * size_l         # [radians]
    trans_map = np.ones_like(phase_map, float)   # ≈1 (β≈0)

    return phase_map, trans_map





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
    P_peak, x_offset=0.0, y_offset=0.0,
    return_grid=False, debug=False,
    debug_outdir=None, sim_label=None
):
    """
    2-D circular Gaussian intensity map on an LxL window, NxN samples.
    Normalized to integrate to 1, then scaled by P_peak. Returns I in W/cm^2.
    Uses I(r) ∝ exp(-2 r^2 / w0^2) with w0 = FWHM / sqrt(2 ln 2).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Grid
    dx = L / N
    x  = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')

    if fwhm_diameter <= 0:
        raise ValueError("fwhm_diameter must be > 0.")
    w0 = fwhm_diameter / np.sqrt(2*np.log(2))  # 1/e^2 radius

    # Center shift
    Xs = X - float(x_offset)
    Ys = Y - float(y_offset)
    r  = np.hypot(Xs, Ys)

    # Properly normalized 2D Gaussian that integrates to 1
    G = (2.0 / (np.pi * w0**2)) * np.exp(-2.0 * (r**2) / (w0**2))

    # Intensity in W/cm^2
    I_W_cm2 = (P_peak * G) / 1e4

    if debug:
        # --------- Debug figure (2D + lineout) ----------
        lambda_IR = 800e-9  # only for setting a reasonable zoom
        x_um = x * 1e6
        extent_um = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(10, 4.2))
        im = ax2d.imshow(I_W_cm2, extent=extent_um, origin='lower')
        ax2d.set_xlabel("x [µm]"); ax2d.set_ylabel("y [µm]")
        ax2d.set_title("Gaussian: I [W/cm²]")
        cb = plt.colorbar(im, ax=ax2d); cb.set_label("W/cm²")

        # Plot requested center
        ax2d.plot([x_offset*1e6], [y_offset*1e6], 'wo', ms=5, mec='k', mew=0.8)

        # Zoom window (±4 µm)
        lim_um = 4.0
        ax2d.set_xlim(-lim_um, lim_um)
        ax2d.set_ylim(-lim_um, lim_um)

        # 1D lineout and FWHM
        j = N//2
        x_line = x.copy()
        I_line = I_W_cm2[:, j]
        Imax   = float(I_line.max())
        half   = 0.5 * Imax
        i0     = int(np.argmax(I_line))

        # Left crossing
        il = np.where(I_line[:i0] < half)[0]
        if il.size:
            k  = il[-1]
            xL = x_line[k] + (half - I_line[k]) * (x_line[k+1]-x_line[k]) / (I_line[k+1]-I_line[k])
        else:
            xL = x_line[0]
        # Right crossing
        ir = np.where(I_line[i0:] < half)[0]
        if ir.size:
            k2 = i0 + ir[0] - 1
            xR = x_line[k2] + (half - I_line[k2]) * (x_line[k2+1]-x_line[k2]) / (I_line[k2+1]-I_line[k2])
        else:
            xR = x_line[-1]

        fwhm_meas = (xR - xL)
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

        # Title + save path
        if sim_label:
            fig.suptitle(f"{sim_label} — Gaussian spot debug", y=0.98)

        # Where to save: VB_figures by default, unless debug_outdir is provided
        outdir = Path(debug_outdir) if debug_outdir else Path("/home/yu79deg/darkfield_p5438/Aime/VB_figures")
        outdir.mkdir(parents=True, exist_ok=True)

        label = sim_label or "gaussian"
        fig.tight_layout()
        plt.savefig(outdir / f"{label}_gaussian_debug.png", dpi=300)
        plt.close(fig)

        # Console checks
        print(f"[Gauss] ∫G dxdy (numeric) ≈ {np.sum(G)*dx*dx:.6f}")
        print(f"[Gauss] Peak intensity: {float(I_W_cm2.max()):.3e} W/cm²")
        print(f"[Gauss] FWHM (meas) = {fwhm_meas*1e6:.3f} µm ; FWHM (theo) = {fwhm_theo*1e6:.3f} µm")
        print(f"[Gauss] Saved: {outdir / f'{label}_gaussian_debug.png'}")

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
    print(f"delta = {delta}, beta={beta}")

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
    lens_half_thickness[core_mask] = (xm[core_mask]**2 + ym[core_mask]**2) / (2 * ROC) + t_wall / 2
    lens_thickness = 2 * lens_half_thickness

    # ────────────────────────────────────────────────
    # 4) Build aperture (optional) -------------------
    # ────────────────────────────────────────────────
    if add_aperture == 1:
        custom_ap = {'elem': 'Hf', 'thickness': 0.0001, 'shape': 'circle',
                    'size': A, 'invert': 1}
        Aperture_transmission, _ = doap(custom_ap, params)
        F = MultIntensity(Aperture_transmission, F)  # aperture first

    # ────────────────────────────────────────────────
    # 5) Ideal transmission & phase ------------------
    # ────────────────────────────────────────────────
    transmission_map = np.exp(-nb_lenses * absorption_factor * lens_thickness)   # intensity attenuation
    ideal_phase_map  = nb_lenses * phase_per_m * lens_thickness                  # phase delay [rad]

    # ────────────────────────────────────────────────
    # 6) Defect phase (DABAM), then combine ----------
    # ────────────────────────────────────────────────
    total_phase_map = ideal_phase_map  # start with ideal
    if int(el_dict.get("defects", 0)) == 1:
        Z_def = build_dabam_defect_thickness_map(
            el_dict=el_dict, params=params, xm=xm, ym=ym, A_m=A
        )  # thickness delta [m] on (xm, ym) within aperture

        defect_phase_map = nb_lenses * phase_per_m * Z_def  # convert to phase [rad]
        total_phase_map  = ideal_phase_map + defect_phase_map

    # ────────────────────────────────────────────────
    # 7) Apply exactly once --------------------------
    # ────────────────────────────────────────────────
    F = MultIntensity(transmission_map, F)   # keep ideal absorption only
    F = MultPhase(total_phase_map, F)        # ideal + defect phase

    # ------------ PLOTS ------------
    # ---- build filename stem -------
    lens_name = el_dict.get("element_name") or el_dict.get("name") or el_dict.get("label") or "Lens"
    sim_name  = params.get("filename") or params.get("sim_name") or params.get("name") or Path(projectdir).stem
    stem = f"{sim_name}_{lens_name}"   # e.g. LP_610_CRL4a
    save_dir = Path(projectdir)        # plotting functions save under /Lens_diags



    # Image 1: ideal CRL (no defects)
    plot_custom_crl_ideal(
        projectdir=projectdir,
        Na=Na,
        nb_lenses=nb_lenses,
        lens_thickness=lens_thickness,
        transmission_map=transmission_map,
        total_phase_map_ideal=ideal_phase_map,  # note: ideal only
        add_aperture=add_aperture,
        Aperture_transmission=(Aperture_transmission if add_aperture else None),
        lens_material=lens_material,
        E_eV=E_eV,
        wavelength_m=wavelength_m,
        filename_stem=stem,
        save_dir=save_dir,
        sim_name=sim_name,
        lens_name=lens_name,
    )


    # Image 2: DABAM diagnostics (only if defects requested)
    if int(el_dict.get("defects", 0)) == 1:
        import darkfield.wavefront_fitting as wft

        # Try to use the cached arrays produced by build_dabam_defect_thickness_map
        X      = el_dict.get("dabam_X", None)
        Y      = el_dict.get("dabam_Y", None)
        Zraw   = el_dict.get("Z_raw_native", None)
        Zfit_p = el_dict.get("Z_fit_panel_map", None)    # custom when Custom_zernike=on
        Zres   = el_dict.get("Z_residues_native", None)

        # If any are missing, fall back to recomputing (rare)
        if X is None or Y is None or Zraw is None or Zfit_p is None or Zres is None:
            # Parse index (same as before)
            dabam_sel = el_dict.get("experimental_data", None)
            idx = None
            if isinstance(dabam_sel, (int, np.integer)):
                idx = int(dabam_sel)
            elif isinstance(dabam_sel, str):
                s = dabam_sel.strip()
                m = re.search(r"dabam2d-(\d+)$", s, flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1))
                else:
                    nums = re.findall(r"(\d+)", s)
                    if nums:
                        idx = int(nums[-1])

            if idx is not None:
                X, Y, Zraw, _ = load_dabam2d(idx)
                nmodes    = int(el_dict.get("nmodes", 37))
                startmode = int(el_dict.get("startmode", 1))

                # Baseline LSQ fit (only used if we truly missed the cache)
                Zcoeffs, Zfit_LSQ, Zres = wft.fit_zernike_circ(
                    Zraw, nmodes=nmodes, startmode=startmode, rec_zern=True
                )
                Zres = -Zres

                # Optional removal (for display)
                remove_avg_flag = int(el_dict.get("remove_avg_profile", 0))
                if remove_avg_flag == 1:
                    I_thick_res, R = wft.average_azimuthal(Zres, X, Y)
                    _, Zres = wft.remove_avg_profile(Zres, None, X, Y, I_thick_res, R, 'b')
                el_dict["remove_avg_applied"] = remove_avg_flag

                # The “fit” panel falls back to LSQ fit when we don’t have the cache
                Zfit_p = Zfit_LSQ
            else:
                # No index → nothing to plot
                return F

        # Build label state for the bar plot
        custom_active = (el_dict.get("Zfit_is_custom", 0) == 1)
        custom_pairs  = list(el_dict.get("Custom_zernike", [])[1:]) if custom_active else []

        # Zernike coeffs for the decomposition bars (LSQ of DABAM map)
        Zcoeffs = el_dict.get("Z_coeffs_native", None)
        # If somehow missing, fall back to the recompute path above (keeps behavior robust)
        if Zcoeffs is None:
            nmodes    = int(el_dict.get("nmodes", 37))
            startmode = int(el_dict.get("startmode", 1))
            Zcoeffs, _, _ = wft.fit_zernike_circ(Zraw, nmodes=nmodes, startmode=startmode, rec_zern=True)


        plot_dabam_diagnostics(
            projectdir=projectdir,
            el_dict=el_dict,
            X=X, Y=Y,
            Z_raw=Zraw,
            Zfit=Zfit_p,            # custom or LSQ (panel)
            Zres=Zres,              # post-processed residues (panel)
            Zcoeffs=np.asarray(Zcoeffs),
            Zres_unproc=el_dict.get("Z_residues_native_unproc"),
            custom_active=custom_active,
            custom_pairs=custom_pairs,
            nmodes=int(el_dict.get("nmodes", 37)),
            startmode=int(el_dict.get("startmode", 1)),
            filename_stem=stem,
            save_dir=save_dir,
            sim_name=sim_name,    
            lens_name=lens_name      
        )



    return F


# ----------------- Zernike name map (Noll) -----------------
# Noll → (n, m, canonical name)
ZERN_NOLL = {
    1:  (0, 0,  "Piston"),
    2:  (1,-1,  "Tilt X (horizontal)"),
    3:  (1, 1,  "Tilt Y (vertical)"),
    4:  (2, 0,  "Defocus"),
    5:  (2,-2,  "Primary astigmatism (oblique)"),
    6:  (2, 2,  "Primary astigmatism (vertical)"),
    7:  (3,-1,  "Primary coma X (horizontal)"),
    8:  (3, 1,  "Primary coma Y (vertical)"),
    9:  (3,-3,  "Trefoil (vertical)"),
    10: (3, 3,  "Trefoil (oblique)"),
    11: (4, 0,  "Primary spherical"),
    12: (4,-2,  "Secondary astigmatism (vertical)"),
    13: (4, 2,  "Secondary astigmatism (oblique)"),
    14: (4,-4,  "Quadrafoil (vertical)"),
    15: (4, 4,  "Quadrafoil (oblique)"),
    16: (5,-1,  "Secondary coma X (horizontal)"),
    17: (5, 1,  "Secondary coma Y (vertical)"),
    18: (5,-3,  "Secondary trefoil (oblique)"),
    19: (5, 3,  "Secondary trefoil (vertical)"),
    20: (5,-5,  "Pentafoil (oblique)"),
    21: (5, 5,  "Pentafoil (vertical)"),
    22: (6, 0,  "Secondary spherical"),
    23: (6,-2,  "Tertiary astigmatism (vertical)"),
    24: (6, 2,  "Tertiary astigmatism (oblique)"),
    25: (6,-4,  "Secondary quadrafoil (vertical)"),
    26: (6, 4,  "Secondary quadrafoil (oblique)"),
    27: (6,-6,  "Hexafoil (vertical)"),
    28: (6, 6,  "Hexafoil (oblique)"),
    29: (7,-1,  "Tertiary coma X (horizontal)"),
    30: (7, 1,  "Tertiary coma Y (vertical)"),
    31: (7,-3,  "Tertiary trefoil (oblique)"),
    32: (7, 3,  "Tertiary trefoil (vertical)"),
    33: (7,-5,  "Secondary pentafoil (oblique)"),
    34: (7, 5,  "Secondary pentafoil (vertical)"),
    35: (7,-7,  "Heptafoil (oblique)"),
    36: (7, 7,  "Heptafoil (vertical)"),
    37: (8, 0,  "Tertiary spherical"),
}



def _defect_type_label(v: int) -> str:
    if v == 1: return "Zernike only"
    if v == 2: return "Residues only"
    if v == 3: return "Zernike + residues"
    return f"Unknown ({v})"


# ---------- Image 1: ideal CRL (no defects) ----------
def plot_custom_crl_ideal(projectdir: Path,
                          Na: np.ndarray,
                          nb_lenses: float,
                          lens_thickness: np.ndarray,
                          transmission_map: np.ndarray,
                          total_phase_map_ideal: np.ndarray,
                          add_aperture: int,
                          Aperture_transmission: np.ndarray | None,
                          lens_material: str,
                          E_eV: float,
                          wavelength_m: float,
                          filename_stem: str | None = None,
                          save_dir: Path | None = None,
                          sim_name: str | None = None,
                          lens_name: str | None = None):
    import matplotlib.pyplot as plt
    N = lens_thickness.shape[0]
    center_idx = N // 2
    x_um = Na * 1e6
    extent_um = [Na[0]*1e6, Na[-1]*1e6, Na[0]*1e6, Na[-1]*1e6]

    thickness_1d     = lens_thickness[center_idx, :] * 1e6
    transmission_1d  = transmission_map[center_idx, :]
    phase_1d         = total_phase_map_ideal[center_idx, :]

    ncols = 4 if add_aperture else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5.2*ncols, 7), dpi=120)

    im0 = axes[0,0].imshow(nb_lenses * lens_thickness * 1e6, cmap='inferno',
                           extent=extent_um, origin='lower')
    axes[0,0].set(title='2-D Thickness [µm]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[0,1].imshow(transmission_map, cmap='viridis',
                           extent=extent_um, origin='lower')
    axes[0,1].set(title='2-D Transmission', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046)

    im2 = axes[0,2].imshow(total_phase_map_ideal, cmap='twilight',
                           extent=extent_um, origin='lower')
    axes[0,2].set(title='2-D Phase (ideal) [rad]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im2, ax=axes[0,2], fraction=0.046)

    if add_aperture:
        im3 = axes[0,3].imshow(Aperture_transmission, cmap='gray',
                               extent=extent_um, origin='lower')
        axes[0,3].set(title='2-D Aperture (T)', xlabel='x [µm]', ylabel='y [µm]')
        plt.colorbar(im3, ax=axes[0,3], fraction=0.046)

    axes[1,0].plot(x_um, nb_lenses * thickness_1d); axes[1,0].grid()
    axes[1,0].set(ylabel='Thickness [µm]', xlabel='x [µm]', title='1-D Thickness (centre cut)')

    axes[1,1].plot(x_um, transmission_1d, color='green'); axes[1,1].grid()
    axes[1,1].set(ylabel='Transmission', xlabel='x [µm]', title='1-D Transmission (centre cut)')

    axes[1,2].plot(x_um, phase_1d, color='purple'); axes[1,2].grid()
    axes[1,2].set(ylabel='Phase [rad]', xlabel='x [µm]', title='1-D Phase (centre cut)')

    if add_aperture:
        axes[1,3].plot(x_um, Aperture_transmission[center_idx,:], color='black'); axes[1,3].grid()
        axes[1,3].set(ylabel='Aperture T', xlabel='x [µm]', title='1-D Aperture (centre cut)')

    # --- prepend simulation and lens names ---
    sim_name  = sim_name  or Path(projectdir).stem
    lens_name = lens_name or "Lens"

    fig.suptitle(f"{sim_name}, {lens_name}, Custom CRL (ideal) — {lens_material}, E={E_eV:.0f} eV",
                fontsize=13)

    plt.tight_layout(rect=[0,0,1,0.95])

    if save_dir is None:
        save_dir = Path(projectdir)
    save_dir = save_dir / "Lens_diags"

    stem = (filename_stem or "Lens_CRL_cut")
    out = save_dir / f"{stem}_ideal.png"
    plt.savefig(out, dpi=300); plt.close(fig)


# ---------- Image 2: DABAM diagnostics ----------



def plot_dabam_diagnostics(projectdir: Path,
                           el_dict: dict,
                           X: np.ndarray, Y: np.ndarray,
                           Z_raw: np.ndarray, Zfit: np.ndarray, Zres: np.ndarray,
                           Zcoeffs: np.ndarray,
                           Zres_unproc: Optional[np.ndarray] = None,
                           custom_active: bool = False,
                           custom_pairs: list[int | float] = None,
                           nmodes: int = 37, startmode: int = 1,
                           filename_stem: str | None = None,
                           save_dir: Path | None = None,
                           sim_name: str | None = None,
                           lens_name: str | None = None):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle


    # what to show in the bar plot
    if custom_active:
        nolls = [int(custom_pairs[i]) for i in range(0, len(custom_pairs), 2)]
        vals  = [float(custom_pairs[i]) for i in range(1, len(custom_pairs), 2)]
        show_pairs = list(zip(nolls, vals))
        title_bar = "Custom Zernike amplitudes"
    else:
        # display up to the 37th Zernike (or fewer if nmodes < 37)
        j0 = startmode
        jmax = min(j0 + 37, len(Zcoeffs) + 1)
        show_pairs = [(j, Zcoeffs[j - 1]) for j in range(j0, jmax)]
        title_bar = f"Zernike decomposition (up to mode {jmax - 1})"

    # convert to µm for images
    Zraw_um = Z_raw * 1e6
    Zfit_um = Zfit * 1e6
    Zres_um = Zres * 1e6
    Zunproc_um = (Zres_unproc if Zres_unproc is not None else (Z_raw - Zfit)) * 1e6

    XX, YY = np.meshgrid(X * 1e6, Y * 1e6)
    extent_um = [X.min() * 1e6, X.max() * 1e6, Y.min() * 1e6, Y.max() * 1e6]

    # optional interpolation map (resampled to same grid as used in simulation)
    # NOTE: only available if we had to interpolate previously
    Zinterp = None
    if "Z_def_resampled" in el_dict:
        Zinterp = el_dict["Z_def_resampled"] * 1e6  # µm

    interp_extent = el_dict.get("Z_interp_extent_um", None)
    resample_info = el_dict.get("Z_resample_info", None)

    # build parameter summary
    defects_flag = int(el_dict.get("defects", 0))
    defect_type  = int(el_dict.get("defect_type", 3))
    remove_avg   = int(el_dict.get("remove_avg_applied",
                                el_dict.get("remove_avg_profile", 0)))
    expdata      = el_dict.get("experimental_data", "n/a")

    summary = (f"defects={defects_flag} | data={expdata} | "
            f"type={_defect_type_label(defect_type)} | "
            f"remove_avg_profile={remove_avg} | "
            f"custom_zernike={'on' if custom_active else 'off'}")

    # --- Figure layout: 2x2 + 1 wide bar chart, or 2x3 if interpolation map available
    if Zinterp is not None:
        fig = plt.figure(figsize=(16, 10), dpi=120)
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.25)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :])  # full-width residues (post-processed)
        ax5 = fig.add_subplot(gs[2, :])  # full-width Zernike bars
    else:
        fig = plt.figure(figsize=(14, 9), dpi=120)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.25)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])
        ax4 = fig.add_subplot(gs[2, :])
        ax5 = None

    # ----------- PLOTS (using colormap='GnBu') --------------------
    cmap = "GnBu"

    im0 = ax0.imshow(Zraw_um, extent=extent_um, origin='lower', cmap=cmap)
    ax0.set(title="Raw DABAM surface [µm]", xlabel="x [µm]", ylabel="y [µm]")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    im1 = ax1.imshow(Zfit_um, extent=extent_um, origin='lower', cmap=cmap)
    ax1.set(title="Zernike fit [µm]", xlabel="x [µm]", ylabel="y [µm]")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # --- CHANGED TITLE HERE ---
    im2 = ax2.imshow(Zunproc_um, extent=extent_um, origin='lower', cmap=cmap,vmin=-5, vmax=5)
    ax2.set(title="Residues (not processed) [µm]", xlabel="x [µm]", ylabel="y [µm]")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    # --- RESIDUES POST-PROCESSING ---
    im3 = ax3.imshow(Zres_um, extent=extent_um, origin='lower', cmap=cmap,vmin=-5, vmax=5)
    ax3.set(title="Residues (post processing) [µm]", xlabel="x [µm]", ylabel="y [µm]")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    # --- OPTIONAL INTERPOLATED MAP (new subplot) ---
    if Zinterp is not None and ax5 is not None:
        im4 = ax4.imshow(
            Zinterp,
            extent=(interp_extent if interp_extent is not None else extent_um),
            origin='lower',
            cmap=cmap
        )
        ax4.set(title="Resampled (interpolated) map [µm]", xlabel="x [µm]", ylabel="y [µm]")
        fig.colorbar(im4, ax=ax4, fraction=0.046)

        # === NEW: draw a circle for the lens DIAMETER (i.e., radius = A/2) ===
        # Prefer the radius computed by the builder; otherwise derive from YAML A
        Rdst_um = None
        if resample_info is not None:
            Rdst_um = resample_info.get("Rdst_um", None)
        if Rdst_um is None:
            A = el_dict.get("A", None)        # A is DIAMETER in your code
            if A is not None:
                Rdst_um = float(A * 1e6 / 2.0)
        if Rdst_um is not None:
            from matplotlib.patches import Circle
            lens_circle = Circle((0.0, 0.0), Rdst_um,
                                edgecolor='white', facecolor='none',
                                linewidth=1.4, alpha=0.95,linestyle='--')
            ax4.add_patch(lens_circle)
            # optional: label
            ax4.text(0.02, 0.02, f"Lens Ø ≈ {2*Rdst_um:.1f} µm",
                    transform=ax4.transAxes, ha='left', va='bottom',
                    fontsize=9, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.35, linewidth=0))

        # annotate resampling policy & sizes (you already had this)
        if resample_info is not None:
            ax4.text(
                0.02, 0.98,
                (f"{resample_info.get('policy','')}\n"
                f"Rsrc={resample_info.get('Rsrc_um',0):.1f} µm → "
                f"Rdst={resample_info.get('Rdst_um',0):.1f} µm\n"
                f"{resample_info.get('note','')}"),
                transform=ax4.transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, linewidth=0)
            )

        bar_ax = ax5
    else:
        # <<< add this so bar_ax is always defined >>>
        bar_ax = ax4



    # ----------- BAR CHART (Zernike decomposition) ----------------
    js   = [j for j, _ in show_pairs]
    vals = [v * 1e6 for _, v in show_pairs]  # µm

    labels = []
    for j in js:
        nm = ZERN_NOLL.get(j)  # -> (n, m, name) or None
        if nm is None:
            labels.append(f"$Z_{{{j}}}$")
        else:
            n, m, name = nm
            # two lines: top "Z_j  Z_n^m" (MathText), bottom the name
            labels.append(f"$Z_{{{j}}}$ $Z_{{{n}}}^{{{m}}}$\n{name}")


    # --- vertical lines behind bars ---
    for x in range(len(js)):
        bar_ax.axvline(x, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)

    # --- bar plot (zorder>0 so it stays on top of the grid lines) ---
    bar_ax.bar(range(len(js)), vals, color="#2c7fb8", zorder=2)

    bar_ax.bar(range(len(js)), vals, color="#2c7fb8")
    bar_ax.set_xticks(range(len(js)), labels, rotation=90, ha='center', va='top')
    bar_ax.set_ylabel("Thickness defect [µm]")
    bar_ax.set_title(title_bar)
    bar_ax.tick_params(axis='x', labelsize=8)


    # ----------- TITLE & SAVE ------------------------
    lens_name = (lens_name
                 or el_dict.get("element_name")
                 or el_dict.get("name")
                 or el_dict.get("label")
                 or "Lens")
    sim_name  = (sim_name
                 or el_dict.get("sim_name")
                 or el_dict.get("filename")
                 or el_dict.get("simulation_name")
                 or Path(projectdir).stem)

    fig.suptitle(f"{sim_name}, {lens_name}, {summary}", fontsize=13)

    fig.subplots_adjust(bottom=0.25, top=0.93, hspace=0.5)

    if save_dir is None:
        save_dir = Path(projectdir)
    save_dir = save_dir / "Lens_diags"
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = (filename_stem or "Lens_DABAM")
    out = save_dir / f"{stem}_defects.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)





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





def _save_center_crop_debug(I, params, fname_tag="debug", title_tag="", outdir=None, crop_um=4.0):
    """
    Save a 2x2 debug figure (2D linear/log + central 1D linear/log)
    for a ±crop_um (µm) window around the center. Expects I to be an intensity map (relative units).
    Scales to photons/m² if params['scale_phot'] is present.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Optional scaling to photons / m²
    scale_ph = params.get("scale_phot", None)
    if scale_ph is None:
        print(f"[TCC] scale_phot not available → skipping {fname_tag} debug figure.")
        return
    I_ph = I * scale_ph  # photons / m²

    # Window half-size in pixels
    pxsize = float(params["pxsize"])            # [m/px]
    half_win_m  = float(crop_um) * 1e-6         # [m]
    half_win_px = max(1, int(round(half_win_m / pxsize)))

    # Centered crop
    Ny, Nx = I_ph.shape
    cy, cx = Ny // 2, Nx // 2
    y0, y1 = max(0, cy - half_win_px), min(Ny, cy + half_win_px)
    x0, x1 = max(0, cx - half_win_px), min(Nx, cx + half_win_px)
    Iw = I_ph[y0:y1, x0:x1]

    # Axes extent in µm (centered)
    win_x_um = Iw.shape[1] * pxsize * 1e6
    win_y_um = Iw.shape[0] * pxsize * 1e6
    ext = (-0.5 * win_x_um, 0.5 * win_x_um, -0.5 * win_y_um, 0.5 * win_y_um)
    x_um = np.linspace(ext[0], ext[1], Iw.shape[1])
    profile = Iw[Iw.shape[0] // 2, :]

    # Colormap (fallback if rofl.cmap() not available)
    try:
        cmap = rofl.cmap()
    except Exception:
        cmap = None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    use_log = bool(params.get("simulation", {}).get("figs_log", 0))
    clim    = params.get("simulation", {}).get("flow_plot_clim", [None, None])

    if title_tag:
        fig.suptitle(title_tag, y=0.98)

    # 2D linear
    im0 = axes[0, 0].imshow(
        Iw, origin="lower", extent=ext, interpolation="nearest",
        aspect="equal", cmap=(cmap or "viridis")
    )
    axes[0, 0].set_title("2D (linear) — photons/m²")
    axes[0, 0].set_xlabel("x [µm]"); axes[0, 0].set_ylabel("y [µm]")
    c0 = plt.colorbar(im0, ax=axes[0, 0], shrink=0.9); c0.set_label("photons / m²")

    # 2D log
    A = np.asarray(Iw, float)
    if use_log:
        pos = A[np.isfinite(A) & (A > 0)]
        if pos.size:
            vmin = float(pos.min())
            vmax = float(pos.max())
            # Apply YAML clim if provided
            if clim:
                lo, hi = clim
                if lo is not None:
                    vmin = max(vmin, float(lo))
                if hi is not None:
                    vmax = min(vmax, float(hi))
            # Enforce strictly positive + ordered bounds
            vmin = max(vmin, np.finfo(float).tiny)
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = np.nextafter(vmin, np.inf)
            im1 = axes[0, 1].imshow(
                np.where(A > 0, A, np.nan),
                origin="lower", extent=ext, interpolation="nearest",
                aspect="equal", cmap=(cmap or "viridis"),
                norm=LogNorm(vmin=vmin, vmax=vmax)
            )
            axes[0, 1].set_title("2D (log) — photons/m²")
        else:
            # no positive data → linear fallback
            im1 = axes[0, 1].imshow(
                A, origin="lower", extent=ext, interpolation="nearest",
                aspect="equal", cmap=(cmap or "viridis")
            )
            axes[0, 1].set_title("2D (linear fallback) — photons/m²")
    else:
        # YAML requested linear plotting
        im1 = axes[0, 1].imshow(
            A, origin="lower", extent=ext, interpolation="nearest",
            aspect="equal", cmap=(cmap or "viridis")
        )
        axes[0, 1].set_title("2D (linear) — photons/m²")

    axes[0, 1].set_xlabel("x [µm]"); axes[0, 1].set_ylabel("y [µm]")
    c1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.9); c1.set_label("photons / m²")


    # 1D central cuts
    axes[1, 0].plot(x_um, profile)
    axes[1, 0].set_title("Central cut (linear)")
    axes[1, 0].set_xlabel("x [µm]"); axes[1, 0].set_ylabel("photons / m²")

    axes[1, 1].semilogy(x_um, np.clip(profile, 1e-300, None))
    axes[1, 1].set_title("Central cut (log)")
    axes[1, 1].set_xlabel("x [µm]"); axes[1, 1].set_ylabel("photons / m²")

    # Save
    save_dir = Path(outdir) if outdir is not None else Path(params["projectdir"])
    sim = params.get("filename", "sim")
    outfile = save_dir / f"{sim}_{fname_tag}_at_TCC.png"

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[TCC] Saved {title_tag} 2D map + central cut to {outfile}")



def F_of(chi: float, chi0: float, rho: float, n_nodes: int = 160) -> float:
    """
    Universal overflow-safe evaluator for the pulse-shape correction integrand.
    Analytically cancels the large exponentials; integrates a benign kernel:
        ∫ e^{-K^2} |S|^2 dK = ∫ e^{-2*rho^2*K^2} | w(i(z+chi)) + w(i(-z+chi)) |^2 dK
    with z = rho*K - i*chi0.  Works for any (chi, chi0, rho).
    """
    K, W = hermgauss(n_nodes)

    # Skip extreme tails where exp(-2*rho^2*K^2) underflows (no contribution).
    # 2*rho^2*K^2 ≳ 700 → exp(-...) ~ 5e-305 (below double precision relevance)
    mask = (2.0*(rho*rho)*(K*K) < 700.0)
    if not np.any(mask):
        return 0.0

    K = K[mask]; W = W[mask]
    z = rho*K - 1j*chi0

    a = 1j*( z + chi)   # i(z+chi)
    b = 1j*(-z + chi)   # i(-z+chi)
    Wsum = wofz(a) + wofz(b)

    integrand = np.exp(-2.0*(rho*rho)*(K*K)) * (Wsum.real**2 + Wsum.imag**2)
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

    I = np.sum(W * integrand)
    pref = np.sqrt((1.0 + 2.0*rho*rho)/3.0) * (chi*chi)
    return pref * I


# --------- Building the 2D map of IR laser at focus -------
def build_ir_focus_map(IR_spatial_params: list,
                       F,
                       params: dict,
                       P_peak: float,
                       x_off: float = 0.0,
                       y_off: float = 0.0):
    """
    Build IR focus intensity map on the simulation grid, from either a
    Gaussian descriptor or an external image.

    IR_spatial_params (params['IR_2Dmap']):
        ["gaussian" | "external", norm_policy, path, calib_um_per_px]
        - gaussian: ignores path/calib, uses params['IR_FWHM_gaussian'].
        - external: 'norm_policy' controls scaling. Currently supported:
            * "match_integral" → ∫I dA = P_peak (spatial PDF × P_peak)
              using provided calibration (μm/px) or the image extent
              inferred from calib.
    Returns:
        I_W_cm2 : (N,N) array, intensity at focus in W/cm^2
        x       : (N,) array, physical x-grid [m]
        fwhm_diam_used : float, effective FWHM diameter [m]
    """
    mode = str(IR_spatial_params[0]).lower() if IR_spatial_params else "gaussian"

    # Target grid (meters), centered
    x = np.linspace(-F.grid_size/2, F.grid_size/2, F.N)
    Xg, Yg = np.meshgrid(x, x, indexing="xy")

    def _fwhm_diameter(A: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """
        Estimate effective FWHM *diameter* of a spot in A on a rect. grid (x_coords, y_coords),
        using a peak-centered radial profile with a 1D fallback.
        Returns diameter in the same units as x_coords/y_coords.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.size == 0 or not np.isfinite(A).any() or A.max() <= 0:
            return float("nan")

        # Peak location and center coordinates
        iy, ix = np.unravel_index(np.argmax(A), A.shape)
        xc, yc = float(x_coords[ix]), float(y_coords[iy])

        # Radial profile (vectorized binning)
        Xg, Yg = np.meshgrid(x_coords, y_coords, indexing="xy")
        r = np.hypot(Xg - xc, Yg - yc)
        In = A / (A.max() + 1e-300)

        n_bins = 200
        rb = np.linspace(0.0, float(r.max()), n_bins + 1)
        bins = np.digitize(r.ravel(), rb) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        sums = np.bincount(bins, weights=In.ravel(), minlength=n_bins)
        cnts = np.bincount(bins, minlength=n_bins)
        prof = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
        rmid = 0.5 * (rb[:-1] + rb[1:])

        hit = np.where((prof[:-1] >= 0.5) & (prof[1:] < 0.5))[0]
        if hit.size:
            k = int(hit[0])
            y1, y2 = prof[k], prof[k+1]
            x1, x2 = rmid[k], rmid[k+1]
            r_half = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
            if np.isfinite(r_half) and r_half > 0:
                return float(2.0 * r_half)

        # Fallback: 1D FWHM along row/col through the peak → geometric mean
        def _fwhm_1d(y, coord):
            y = np.asarray(y, float)
            if y.max() <= 0: return float("nan")
            y = y / (y.max() + 1e-300)
            i0 = int(np.argmax(y)); half = 0.5
            L = np.where(y[:i0] < half)[0]
            if L.size:
                kL = L[-1]
                xL = coord[kL] + (half - y[kL]) * (coord[kL+1] - coord[kL]) / (y[kL+1] - y[kL] + 1e-300)
            else:
                xL = coord[0]
            R = np.where(y[i0:] < half)[0]
            if R.size:
                kR = i0 + R[0] - 1
                xR = coord[kR] + (half - y[kR]) * (coord[kR+1] - coord[kR]) / (y[kR+1] - y[kR] + 1e-300)
            else:
                xR = coord[-1]
            return float(xR - xL)

        fx = _fwhm_1d(A[iy, :], x_coords)
        fy = _fwhm_1d(A[:, ix], y_coords)
        if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
            return float(np.sqrt(fx * fy))
        return float("nan")

    # ----------------- GAUSSIAN -----------------
    if mode == "gaussian":
        FWHM_diam = float(params.get("IR_FWHM_gaussian", 1.3e-6))

        I_W_cm2, _ = gaussian_spot_map(
            F.grid_size, F.N,
            fwhm_diameter=FWHM_diam,
            P_peak=P_peak,
            return_grid=True,
            debug=True,
            debug_outdir=Path(params["projectdir"]) / "VB_figures",
            sim_label=params.get("filename", "")
        )
        # optional offsets via resampling (kept as before)
        if abs(x_off) > 0.0 or abs(y_off) > 0.0:
            interp = RegularGridInterpolator((x, x), I_W_cm2, bounds_error=False, fill_value=0.0)
            pts = np.column_stack([(Yg - y_off).ravel(), (Xg - x_off).ravel()])
            I_W_cm2 = interp(pts).reshape(F.N, F.N)

        return I_W_cm2, x, float(FWHM_diam)

    # ----------------- EXTERNAL IMAGE -----------------
    # IR_spatial_params = ["external", norm_policy, path, calib_um_per_px]
    norm_policy    = str(IR_spatial_params[1]).lower() if len(IR_spatial_params) > 1 and IR_spatial_params[1] else "match_integral"
    path_img       = IR_spatial_params[2] if len(IR_spatial_params) > 2 else None
    calib_um_per_px = IR_spatial_params[3] if len(IR_spatial_params) > 3 else None

    if not path_img:
        raise ValueError("IR_2Dmap 'external' requires a valid image path at IR_2Dmap[2].")

    # --- Load grayscale ---
    try:
        from imageio.v2 import imread
        img = imread(path_img)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path_img).convert("L"))

    # --- Sanitize external image (no NaN/Inf/negatives) ---
    img = _sanitize_intensity_map(img)

    Ny, Nx = img.shape

    # --- Pixel size from calibration (μm/px) OR scale-to-Gaussian-FWHM when == 0 ---
    if calib_um_per_px is None:
        raise ValueError("External IR map: provide calibration (IR_2Dmap[3] in μm/px) or 0 to scale to IR_FWHM_gaussian.")

    calib_val = float(calib_um_per_px)
    if calib_val > 0.0:
        dpx = calib_val * 1e-6  # [m/px]
        dx_m = dy_m = dpx
    else:
        # Measure FWHM on the *raw image* in pixel units and choose dpx to match IR_FWHM_gaussian
        fwhm_target_m = float(params.get("IR_FWHM_gaussian", 1.3e-6))
        # pixel coordinate axes (units: px)
        xs_px = np.arange(Nx, dtype=float)
        ys_px = np.arange(Ny, dtype=float)
        fwhm_img_px = _fwhm_diameter(img, xs_px, ys_px)  # [px]
        if not np.isfinite(fwhm_img_px) or fwhm_img_px <= 0:
            raise ValueError("Could not measure FWHM on the external IR image to scale-to-Gaussian.")
        dpx = fwhm_target_m / fwhm_img_px  # [m/px] so that FWHM_px * dpx = target
        dx_m = dy_m = dpx
        print(f"[IR-map] calib=0 → scale-to-Gaussian-FWHM: FWHM_img≈{fwhm_img_px:.2f} px → "
            f"dpx={dpx*1e6:.3f} µm/px (target {fwhm_target_m*1e6:.3f} µm)")

    # Build source physical axes for interpolation (meters), centered on image center
    xs = (np.arange(Nx) - (Nx - 1)/2.0) * dx_m
    ys = (np.arange(Ny) - (Ny - 1)/2.0) * dy_m
    
    # Interpolate onto simulation grid with requested offsets
    interp = RegularGridInterpolator((ys, xs), img, bounds_error=False, fill_value=0.0, method="linear")
    pts = np.column_stack([(Yg - y_off).ravel(), (Xg - x_off).ravel()])
    img_resampled = interp(pts).reshape(F.N, F.N)

    # ---- Recenter peak to (x_off, y_off) (kept as you fixed) ----
    center_peak = bool(params.get("IR_center_peak", True))
    if center_peak:
        iy, ix = np.unravel_index(np.argmax(img_resampled), img_resampled.shape)
        x_peak, y_peak = float(x[ix]), float(x[iy])
        dx_shift, dy_shift = (x_peak - x_off), (y_peak - y_off)
        pts = np.column_stack([(Yg - y_off + dy_shift).ravel(),
                               (Xg - x_off + dx_shift).ravel()])
        img_resampled = interp(pts).reshape(F.N, F.N)

    # ---- Normalization policy ----
    if norm_policy == "match_integral":
        # Spatial density s(x,y) with ∫ s dA = 1 using ORIGINAL pixel area
        total_counts = img.sum()
        if total_counts <= 0:
            raise ValueError(f"External IR map '{path_img}' has zero/negative sum.")
        px_area = dx_m * dy_m
        s_density = img_resampled / (total_counts * px_area)   # [1/m^2]
        I_W_m2  = P_peak * s_density
        I_W_cm2 = I_W_m2 / 1e4

    elif norm_policy == "match_peak":
        # Peak-match to the Gaussian spot with IR_FWHM_gaussian
        fwhm_gauss = float(params.get("IR_FWHM_gaussian", 1.3e-6))
        w0 = fwhm_gauss / np.sqrt(2.0*np.log(2.0))              # [m]
        I_target_peak_Wcm2 = (P_peak * (2.0 / (np.pi*w0*w0))) / 1e4  # [W/cm^2]

        vmax = float(np.max(img_resampled))
        if vmax <= 0.0 or not np.isfinite(vmax):
            raise ValueError("External IR map peak is zero/invalid after resampling; cannot match peak.")

        shape_peak_norm = img_resampled / vmax                  # unitless, peak=1
        I_W_cm2 = I_target_peak_Wcm2 * shape_peak_norm          # enforce desired peak

        # (optional sanity print)
        dx_sim = F.grid_size / F.N
        P_on_grid = np.nansum(I_W_cm2) * dx_sim * dx_sim * 1e4  # back to W
        print(f"[IR-map] match_peak: I_peak={I_target_peak_Wcm2:.3e} W/cm²; "
            f"∫I dA on grid = {P_on_grid:.3e} W (P_peak={P_peak:.3e} W)")


    else:
        # Future options land here (e.g., "match_fwhm", "match_energy_and_peak" via compromise, etc.)
        raise NotImplementedError(f"Unknown IR external norm_policy: '{norm_policy}'")


    # --- Effective FWHM diameter (on the simulation grid) ---
    try:
        fwhm_diam_used = _fwhm_diameter(I_W_cm2, x, x)
        if not np.isfinite(fwhm_diam_used) or fwhm_diam_used <= 0:
            # Fallback: 1/4 of the smallest full span covered by mapping
            full_span_x = Nx * dx_m
            full_span_y = Ny * dy_m
            fwhm_diam_used = float(0.25 * min(full_span_x, full_span_y))
    except Exception:
        full_span_x = Nx * dx_m
        full_span_y = Ny * dy_m
        fwhm_diam_used = float(0.25 * min(full_span_x, full_span_y))


    # ---------- Debug figure (2D + lineout) ----------
    try:
        dbg_dir = Path(params["projectdir"]) / "VB_figures"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        # Axes in microns for display
        x_um = x * 1e6
        extent_um = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

        # Find peak (after resampling + offsets)
        iy, ix = np.unravel_index(np.argmax(I_W_cm2), I_W_cm2.shape)
        x_peak, y_peak = x[ix], x[iy]

        # Prepare figure
        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(10, 4.2))

        # --- 2D map
        im = ax2d.imshow(I_W_cm2, extent=extent_um, origin="lower")
        ax2d.set_xlabel("x [µm]"); ax2d.set_ylabel("y [µm]")
        ax2d.set_title("External IR focus: I [W/cm²]")
        cb = plt.colorbar(im, ax=ax2d); cb.set_label("W/cm²")

        # Mark requested offsets and measured peak
        ax2d.plot([x_off*1e6], [y_off*1e6], 'wo', ms=5, mec='k', mew=0.8, label="requested offset")
        ax2d.plot([x_peak*1e6], [y_peak*1e6], 'rx', ms=6, mew=1.2, label="peak")
        ax2d.legend(loc="upper right", frameon=True)

        # Zoom to a ±4 µm window around the **peak** (robust for off-center beams)
        lim_um = 4.0
        ax2d.set_xlim(x_peak*1e6 - lim_um, x_peak*1e6 + lim_um)
        ax2d.set_ylim(y_peak*1e6 - lim_um, y_peak*1e6 + lim_um)

        # --- 1D lineout through the peak row (horizontal profile)
        x_line = x.copy()
        I_line = I_W_cm2[iy, :]  # horizontal line at the peak y
        Imax   = float(I_line.max())
        half   = 0.5 * Imax

        # Locate half-maximum crossings around the peak index
        i0 = int(np.argmax(I_line))

        # Left crossing
        il = np.where(I_line[:i0] < half)[0]
        if il.size:
            kL  = il[-1]
            xL  = x_line[kL] + (half - I_line[kL]) * (x_line[kL+1] - x_line[kL]) / (I_line[kL+1] - I_line[kL] + 1e-300)
        else:
            xL  = x_line[0]

        # Right crossing
        ir = np.where(I_line[i0:] < half)[0]
        if ir.size:
            kR  = i0 + ir[0] - 1
            xR  = x_line[kR] + (half - I_line[kR]) * (x_line[kR+1] - x_line[kR]) / (I_line[kR+1] - I_line[kR] + 1e-300)
        else:
            xR  = x_line[-1]

        fwhm_meas = (xR - xL)  # [m]

        # Plot the lineout
        ax1d.plot(x_line*1e6, I_line, lw=1.6)
        ax1d.axhline(half, color='gray', ls='--', lw=1)
        ax1d.axvline(xL*1e6, color='gray', ls=':', lw=1)
        ax1d.axvline(xR*1e6, color='gray', ls=':', lw=1)
        ax1d.set_xlabel("x [µm]"); ax1d.set_ylabel("I [W/cm²]")
        ax1d.set_title("Lineout through peak (y = y_peak)")
        ax1d.grid(True)

        # Make the x-limits match the 2D zoom window (centered on peak)
        ax1d.set_xlim(x_peak*1e6 - lim_um, x_peak*1e6 + lim_um)

        # Annotate
        ax1d.text(
            0.05, 0.95,
            f"Peak = {Imax:.3e} W/cm²\n"
            f"FWHM (meas) = {fwhm_meas*1e6:.3f} µm",
            transform=ax1d.transAxes, ha='left', va='top'
        )

        # Title + save
        label = params.get("filename", "run")
        fig.suptitle(f"{label} — External IR spot debug", y=0.98)
        fig.tight_layout()
        out_png = dbg_dir / f"{label}_IRmap_resampled.png"
        plt.savefig(out_png, dpi=300)
        plt.close(fig)

        # Console checks (use the original pixel-area normalization)
        print(f"[IR-map] Peak intensity: {Imax:.3e} W/cm²")
        print(f"[IR-map] FWHM (meas along x@peak) = {fwhm_meas*1e6:.3f} µm")
        print(f"[IR-map] Saved: {out_png}")

    except Exception as e:
        # keep silent in batch but don’t block the run
        print(f"[IR-map] Debug plotting failed: {e}")


    return I_W_cm2, x, fwhm_diam_used





def _fwhm_diameter_generic(A: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    """
    Effective FWHM *diameter* of a spot in A on rect. grid (x_coords, y_coords).
    Peak-centered radial profile with 1D fallback. Returns same units as coords.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0 or not np.isfinite(A).any() or A.max() <= 0:
        return float("nan")
    iy, ix = np.unravel_index(np.argmax(A), A.shape)
    xc, yc = float(x_coords[ix]), float(y_coords[iy])

    Xg, Yg = np.meshgrid(x_coords, y_coords, indexing="xy")
    r = np.hypot(Xg - xc, Yg - yc)
    In = A / (A.max() + 1e-300)

    n_bins = 200
    rb = np.linspace(0.0, float(r.max()), n_bins + 1)
    bins = np.digitize(r.ravel(), rb) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    sums = np.bincount(bins, weights=In.ravel(), minlength=n_bins)
    cnts = np.bincount(bins, minlength=n_bins)
    prof = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    rmid = 0.5 * (rb[:-1] + rb[1:])

    hit = np.where((prof[:-1] >= 0.5) & (prof[1:] < 0.5))[0]
    if hit.size:
        k = int(hit[0])
        y1, y2 = prof[k], prof[k+1]
        x1, x2 = rmid[k], rmid[k+1]
        r_half = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        if np.isfinite(r_half) and r_half > 0:
            return float(2.0 * r_half)

    # Fallback: geometric mean of horizontal / vertical FWHM through the peak
    def _fwhm_1d(y, coord):
        y = np.asarray(y, float)
        if y.max() <= 0: return float("nan")
        y = y / (y.max() + 1e-300)
        i0 = int(np.argmax(y)); half = 0.5
        L = np.where(y[:i0] < half)[0]
        if L.size:
            kL = L[-1]
            xL = coord[kL] + (half - y[kL]) * (coord[kL+1] - coord[kL]) / (y[kL+1] - y[kL] + 1e-300)
        else:
            xL = coord[0]
        R = np.where(y[i0:] < half)[0]
        if R.size:
            kR = i0 + R[0] - 1
            xR = coord[kR] + (half - y[kR]) * (coord[kR+1] - coord[kR]) / (y[kR+1] - y[kR] + 1e-300)
        else:
            xR = coord[-1]
        return float(xR - xL)

    fx = _fwhm_1d(A[iy, :], x_coords)
    fy = _fwhm_1d(A[:, ix], y_coords)
    if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
        return float(np.sqrt(fx * fy))
    return float("nan")




def build_external_shaper_mask(el_dict: dict, F, params: dict) -> np.ndarray:
    """
    Build a transmission mask (0..1) from an external image to shape the
    *intensity* after the mask to match the image (up to a global factor).

    YAML (under a 'type: aperture' element):
      shape: external_map
      path: "/abs/path/to/Xray_initial_beam.png"
      calibration: <μm/px>      # if 0, scale the image to FWHM='size'
      size: <FWHM_diameter_m>   # used only when calibration == 0
      center: "barycenter" | "max" | "no"
      center_sigma_px: 2.0       # (only for center="max") smoothing in px
      x_offset_m: 0.0            # desired final beam center (meters)
      y_offset_m: 0.0
      # or in microns:
      # x_offset_um: 0.0
      # y_offset_um: 0.0
    """
    path = el_dict.get("path", None)
    calib_um_per_px = el_dict.get("calibration", None)
    if not path:
        raise ValueError("beam_shaper.external_map needs a 'path' to an image.")
    if calib_um_per_px is None:
        raise ValueError("beam_shaper.external_map needs 'calibration' (μm/px), or 0 to use 'size' as FWHM.")

    # -- load + sanitize (make grayscale, drop NaN/Inf, clamp negatives)
    try:
        from imageio.v2 import imread
        img = imread(path)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))  # color or L, both fine

    img = _sanitize_intensity_map(img)    # ← your sanitizer
    Ny, Nx = img.shape

    # -- pixel scale
    calib_val = float(calib_um_per_px)
    if calib_val > 0.0:
        dpx = calib_val * 1e-6  # [m/px]
    else:
        target_fwhm = float(el_dict.get("size", 0.0))
        if target_fwhm <= 0:
            raise ValueError("With calibration=0 you must provide a positive 'size' (FWHM diameter in meters).")
        xs_px = np.arange(Nx, dtype=float)
        ys_px = np.arange(Ny, dtype=float)
        fwhm_px = _fwhm_diameter_generic(img, xs_px, ys_px)
        if not np.isfinite(fwhm_px) or fwhm_px <= 0:
            raise ValueError("Failed to measure FWHM on external map for calibration=0 scaling.")
        dpx = target_fwhm / fwhm_px
        print(f"[beam_shaper] calib=0 → scale-to-FWHM: img≈{fwhm_px:.2f}px → dpx={dpx*1e6:.3f} µm/px (target {target_fwhm*1e6:.3f} µm)")

    # -- source axes (meters) centered on image center
    xs = (np.arange(Nx) - (Nx - 1) / 2.0) * dpx
    ys = (np.arange(Ny) - (Ny - 1) / 2.0) * dpx

    # -- destination (simulation) grid
    dx = F.siz / F.N
    x  = (np.arange(F.N) - F.N / 2) * dx
    Xg, Yg = np.meshgrid(x, x, indexing="xy")

    # -------- centering & offsets ---------------------------------
    center_mode = str(el_dict.get("center", "no")).lower()

    # desired final center (default 0)
    x_off_m = float(el_dict.get("x_offset_m", 0.0))
    y_off_m = float(el_dict.get("y_offset_m", 0.0))
    if "x_offset_um" in el_dict: x_off_m = float(el_dict["x_offset_um"]) * 1e-6
    if "y_offset_um" in el_dict: y_off_m = float(el_dict["y_offset_um"]) * 1e-6

    center_x = 0.0
    center_y = 0.0

    if center_mode == "barycenter":
        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        w = img
        tot = float(w.sum())
        if tot > 0.0:
            center_x = float((w * Xs).sum() / tot)
            center_y = float((w * Ys).sum() / tot)
    elif center_mode == "max":
        try:
            from scipy.ndimage import gaussian_filter
            sigma_px = float(el_dict.get("center_sigma_px", 2.0))
            G = gaussian_filter(img, sigma=sigma_px) if sigma_px > 0 else img
        except Exception:
            G = img
        iy, ix = np.unravel_index(np.argmax(G), G.shape)
        center_x = float(xs[ix]); center_y = float(ys[iy])
    elif center_mode in ("no", "none", "false", "0"):
        pass
    else:
        raise ValueError("beam_shaper.center must be 'barycenter', 'max' or 'no'.")

    # Shift needed in source coordinates so that final center = (x_off_m, y_off_m)
    dx_center = x_off_m - center_x
    dy_center = y_off_m - center_y
    # --------------------------------------------------------------

    # -- resample the *shifted* map to the simulation grid
    interp = RegularGridInterpolator((ys, xs), img, bounds_error=False, fill_value=0.0)
    T = interp(np.column_stack([(Yg - dy_center).ravel(), (Xg - dx_center).ravel()])).reshape(F.N, F.N)
    T = np.maximum(T, 0.0)

    # -- build intensity transmission so that (M * I_in) ∝ T (shape match)
    I_in = Intensity(0, F)
    eps = 1e-300
    M0 = T / (I_in + eps)                 # desired ratio in intensity units
    mmax = float(np.nanmax(M0))
    if not np.isfinite(mmax) or mmax <= 0:
        return np.zeros_like(I_in)
    M = np.clip(M0 / mmax, 0.0, 1.0)      # cap so it's a *transmission* (≤1)

    return M





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

        # ############# ELEMENT : GAZ JET (pure phase) ###############
        if el_type.lower() == "gazjet":
            if ch_name == "main":
                # Build plasma phase (ignore transmission ~ 1 for these densities)
                phase_map, _ = build_gazjet_maps(el_dict, params, F)

                # Apply phase to the main channel and persist
                F = MultPhase(phase_map, F)
                bundle.fields[ch_name] = F
                
            else:
                # Skip Gazjet for non-main channels

                pass


        ############# ELEMENT : PURE APERTURE ###########
        if 'aperture' in el_type:
            # -----External_map shaping at the beam entrance ---
            if el_dict.get('shape', '') == 'external_map':
                M = build_external_shaper_mask(el_dict, F, params)
                F = MultIntensity(M, F)
                bundle.fields[ch_name] = F
            else:
                # --- Build a regular aperture ---
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

        # X and Y offset of the laser 
        x_off = float(params.get("IR_x_offset_m", 0.0))      # x offset of the laser at TCC [m]
        y_off = float(params.get("IR_y_offset_m", 0.0))    # y offset of the laser at TCC [m]

        # Calculate the intensity of IR laser :
        wavelength_IR = float(params.get("IR_wavelength", 800e-9))            # [m]
        phi_VB = np.pi / 4 #45 degrees angle between the IR and X beams.
        phi_VB       = np.deg2rad(float(params.get("phi_VB_deg", 45.0))) #Polarisation angle between the IR laser and the Xray beam [rad]

        # Define E_pulse (OR) P_peak
        tau_FWHM = float(params.get("IR_FWHM_duration", 30e-15))             # Pump IR Laser FWHM duration in second
        E_pulse  = float(params.get("IR_Energy_J", 4.8))                      # [J]
        #P_peak_direct = 200e12 # Peak power of the laser, in Watt

        P_peak = peak_power(E=E_pulse, tau_FWHM=tau_FWHM) # In [Watt]
        #P_peak = peak_power(P_peak=P_peak_direct)

        # 1) ------- Option 1 = Airy Disk -------
        #I_W_cm2, x = airy_disk_map(F.grid_size, F.N, P_peak, lam=wavelength_IR, f=f_IR, D=D_IR,return_grid=True)

        # 2) ------- Option 2 = Gaussian --------
        IR_spatial_params = yamlval("IR_2Dmap", params, ["gaussian", "match_integral", None, None])

        I_W_cm2, x, FWHM_diam = build_ir_focus_map(
            IR_spatial_params=IR_spatial_params,
            F=F,
            params=params,
            P_peak=P_peak,
            x_off=x_off,
            y_off=y_off
        )


        #I_W_cm2, x = gaussian_spot_map(
        #    F.grid_size, F.N,
        #    fwhm_diameter=FWHM_diam,
        #    P_peak=P_peak,
        #    return_grid=True,
        #    debug=True,
        #    debug_outdir=Path(params["projectdir"]) / "VB_figures",
        #    sim_label=params.get("filename", "")
        #)

        # ---------------- Pulse Shape factor ---------------
        # The following factor is made to account for: 1) The finite pump Rayleigh length. 2) The probe and pump finite durations. 3) The time offset of the pump/probe focus

        T_FWHM = float(params.get("X_FWHM_duration", 25e-15))               # probe duration FWHM [s]
        t0  = float(params.get("Timing_jitter", 0.0))               # [s] pump–probe focus timing offset

        tau_Felix = tau_FWHM * np.sqrt(2 / np.log(2)) #The tau defined by Felix : tau_Felix = sqrt(2/ln(2)) * tau_FWHM
        w0 = FWHM_diam / (np.sqrt(2 * np.log(2)))       # 1/e^2 waist radius [m]
        zR  = np.pi * w0**2 / wavelength_IR             # Rayleigh length [m]
                      
        T   = T_FWHM * 2 / (np.sqrt(2 * np.log(2)))          # probe duration (1/e^2) [s]
        tau = tau_FWHM * 2 / (np.sqrt(2 * np.log(2)))        # pump duration (1/e^2) [s]

        chi   = 4.0 * zR / (np.sqrt((c*T)**2 + (c*tau)**2 / 2.0))
        chi0  = 2.0 * (c*t0) / (np.sqrt((c*T)**2 + (c*tau)**2 / 2.0))
        rho   = T / tau

        Pulse_shape_factor  = (np.sqrt(3.0*np.pi) / 4.0) * F_of(chi, chi0, rho, n_nodes=180) # Pulse shaped factor (unitless)
        print(f"Pulse shape factor = {Pulse_shape_factor:.12g}")
        print(f"t0={t0}, FWHM_diam={FWHM_diam}, T_FWHM={T_FWHM}, tau_FWHM = {tau_FWHM}")
        # --------------- Creating the masks -------------------

        prefactor = (c * tau_Felix / wavelength) * (alpha_cst / 90) * np.sqrt(np.pi / 2) * np.sqrt(Pulse_shape_factor)
        VB_mask_parr = (I_W_cm2 / I_cr) * prefactor * (11 - 3 * np.cos(2 * phi_VB))  #mask of the intensity of IR laser at TCC (unitless)
        VB_mask_perp = (I_W_cm2 / I_cr) * prefactor * ( 3 * np.sin(2 * phi_VB))      #mask of the intensity of IR laser at TCC (unitless)

        print(f"Speed of light = {c} m/s")
        print(f"max Intensity IR = {np.max(I_W_cm2)} W/cm^2")
        print(f"alpha fine structure =  {alpha_cst}")
        print(f"phi VB = {phi_VB}")
        print(f"wavelenght x ={wavelength} m")
        print(f"tau_Felix=tau_FWHM*sqrt(2/ln(2)) ={tau_Felix} s")
        print(f"prefactor VB = {prefactor}")
        print(f"I_W_cm2 / I_cr = {I_W_cm2 / I_cr}")
        print(f"11 - 3 * np.cos(2 * phi_VB) = {11 - 3 * np.cos(2 * phi_VB)}")
        print(f"3 * np.sin(2 * phi_VB) = {3 * np.sin(2 * phi_VB)}")



        bundle = maybe_spawn_VB_channels(bundle,
                                        params,
                                        VB_mask_parr,
                                        VB_mask_perp)

    return bundle, bundle.fields["main"], def_do_plot





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

    bundle.fields["VB_parr"] = MultIntensity(VB_mask_parr**2, F_main)
    bundle.fields["VB_perp"] = MultIntensity(VB_mask_perp**2, F_main)

    return bundle






def main_VIBE(params,elements):
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
                tau         = params.get('X_FWHM_duration')

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
            if el_name == "TCC" and ch_name in ("main", "VB_perp"):
                vb_dir = Path(params["projectdir"]) / "VB_figures"
                sim    = params.get("filename", "")
                tag    = "Xray" if ch_name == "main" else "VB_perp"
                title  = "Main @ TCC" if ch_name == "main" else "VB_perp @ TCC"
                _save_center_crop_debug(
                    I_ch, params,
                    fname_tag=tag,
                    title_tag=f"{sim} — {title}",
                    outdir=vb_dir,
                )
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
    
    flow_list = params.get("flow", [])
    if not (isinstance(flow_list, (list, tuple)) and len(flow_list) > 0):
        print("[FLOW] Disabled (no windows provided).")
        return params, trans, figs
        
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

    # ─── Total photon count at beam shaper plane ───
    elements_dict = res[0]
    beam_shaper_pos = None
    if "beam_shaper" in elements_dict:
        if yamlval("in", elements_dict["beam_shaper"], 1):
            beam_shaper_pos = elements_dict["beam_shaper"]["position"]

    # Perform only for MAIN channel
    if (beam_shaper_pos is not None) and (channel == "main") and (len(ffigs) > 0):
        # find the closest slice to the beam shaper position
        idx = int(np.argmin(np.abs(zax - beam_shaper_pos)))
        idx = max(0, min(idx, len(ffigs) - 1))  # clamp to valid range

        fig_key = ffigs[idx]
        print(f"\n✅ Beam shaper at z = {beam_shaper_pos:.2f} m → closest slice z = {zax[idx]:.2f} m")

        # get the data from that slice
        raw_img, _, propsize, _ = pic[fig_key]
        img_N = raw_img.shape[0]
        dx = dy = propsize / img_N

        img_scaled = {
            "photons": raw_img * scale_ph,
            "Wcm2": raw_img * scale_Wcm,
            "relative": raw_img
        }.get(unit_sel, raw_img)

        total_photons = np.nansum(img_scaled) * dx * dy
        photons_target = params.get("photons_total", None)

        if unit_sel == "photons":
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





def run_from_yaml(yaml_path: str, N: int):
    cfg      = load_cfg(yaml_path)
    elements = elements_from_cfg(cfg)
    yamlname = Path(yaml_path).stem

    # Reproduce your cluster/local fallback behavior
    cluster_path = Path("/home/yu79deg/darkfield_p5438")
    local_path   = Path("/Users/aimematheron/Dropbox/AimeDF")
    basepath     = cluster_path if cluster_path.exists() else local_path
    projectdir   = str(basepath / "Aime")

    input_params = build_input_params(cfg, N=N, projectdir=projectdir, filename=yamlname)

    # ---- Run the engine exactly as before ----
    out_params, trans, figs = main_VIBE(input_params, elements)

    # ---- Optional: keep Launch's extra res pickle (cfg + params) ----
    try:
        pickles_dir = Path(projectdir) / "pickles"
        pickles_dir.mkdir(parents=True, exist_ok=True)
        mu.dumpPickle([cfg, out_params], str(pickles_dir / f"{yamlname}_res"))
    except Exception as e:
        print(f"[warn] Could not write extra res pickle: {e}")

    print("Simulation finished.")
    return out_params, trans, figs




if __name__ == "__main__":
    _select_backend()
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="Run VIBE directly from a YAML.")
    ap.add_argument("--yaml", required=True, help="Path to YAML config")
    ap.add_argument("-N", type=int, default=1000, help="Number of simulation points")
    args = ap.parse_args()

    run_from_yaml(args.yaml, args.N)


