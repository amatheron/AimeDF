"""
Gerchberg–Saxton phase retrieval from multiple transverse intensity planes.

Goal
-----
Given measured intensity images I_k(x, y) at several z-planes around focus,
retrieve a consistent complex field E(x, y, z) by iterating constraints:
    1) measured amplitude in each measurement plane
    2) free-space propagation between planes (angular spectrum)

This script provides a reference implementation with:
- CPU (NumPy/FFT) or optional GPU (CuPy) backends
- Angular-spectrum propagator with anti-aliasing padding
- Error-Reduction (ER) and optional Hybrid-Input–Output (HIO)
- Support/ROI masks and per-plane weights
- Logging of per-plane amplitude error

Inputs
------
- A stack of intensity images (2D arrays) calibrated to the same pixel pitch
- Wavelength (meters), refractive index (default 1.0), plane z-positions (meters)
- Reference plane index (default the central plane)

Outputs
-------
- Complex field at the reference plane
- Error history and last per-plane fields

"""

from __future__ import annotations
import numpy as _np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # noqa: BLE001
    _HAS_CUPY = False


Array = _np.ndarray


# --------------------------- Utility: backend selection ---------------------------

def _xp(use_gpu: bool):
    return _cp if (use_gpu and _HAS_CUPY) else _np


def _fftpack(xp):
    # Unified FFT interface for numpy/cupy
    if xp is _np:
        return _np.fft
    else:
        return _cp.fft  # type: ignore[attr-defined]


# --------------------------- Angular spectrum propagator ---------------------------

def angular_spectrum_propagate(
    E0: Array,
    z: float,
    wavelength: float,
    dx: float,
    dy: float,
    n: float = 1.0,
    use_gpu: bool = False,
) -> Array:
    """
    Propagate a complex field E0 by distance z in a homogeneous medium (index n)
    using the angular-spectrum method with exact propagator.

    Parameters
    ----------
    E0 : complex array [Ny, Nx]
    z : float (m)
    wavelength : float (m)
    dx, dy : sampling pitches (m)
    n : refractive index
    use_gpu : use CuPy if available
    """
    xp = _xp(use_gpu)
    fft = _fftpack(xp)

    Ny, Nx = E0.shape
    k0 = 2.0 * _np.pi / wavelength
    k = n * k0

    fx = xp.fft.fftfreq(Nx, d=dx)  # cycles/m
    fy = xp.fft.fftfreq(Ny, d=dy)
    FX, FY = xp.meshgrid(fx, fy, indexing="xy")
    # Spatial frequencies in rad/m
    kx = 2.0 * _np.pi * FX
    ky = 2.0 * _np.pi * FY

    kz_sq = (k**2) - (kx**2 + ky**2)
    # Avoid negative due to sampling beyond NA: evanescent components get decaying factor
    kz = xp.sqrt(xp.maximum(kz_sq, 0.0)) - 1j * xp.sqrt(xp.maximum(-kz_sq, 0.0))

    H = xp.exp(1j * kz * z)

    E0_f = fft.fft2(E0)
    Ez_f = E0_f * H
    Ez = fft.ifft2(Ez_f)
    return Ez


# --------------------------- Constraints ---------------------------

def enforce_measured_amplitude(
    E: Array,
    target_intensity: Array,
    weight: float = 1.0,
    eps: float = 1e-12,
    mask: Optional[Array] = None,
) -> Array:
    """
    Replace amplitude with measured amplitude (Error-Reduction) inside mask.

    E_new = ( (1-weight) * |E| + weight * sqrt(I_meas) ) * exp(i*arg(E)) in mask,
             |E| * exp(i*arg(E)) outside mask.
    """
    A = _np.abs(E)
    phase = _np.exp(1j * _np.angle(E))
    target_A = _np.sqrt(_np.maximum(target_intensity, 0.0))

    if mask is None:
        A_new = (1 - weight) * A + weight * target_A
        return A_new * phase
    else:
        A_new = A.copy()
        A_new[mask] = (1 - weight) * A[mask] + weight * target_A[mask]
        return A_new * phase


def hybrid_input_output(
    E: Array,
    target_intensity: Array,
    beta: float = 0.9,
    support: Optional[Array] = None,
) -> Array:
    """
    HIO amplitude enforcement variant inside support; outside, apply feedback.
    If `support` is None, behaves like ER.
    """
    if support is None:
        return enforce_measured_amplitude(E, target_intensity, weight=1.0)

    A = _np.abs(E)
    phase = _np.exp(1j * _np.angle(E))
    target_A = _np.sqrt(_np.maximum(target_intensity, 0.0))

    # Inside support: set amplitude
    E_new = E.copy()
    E_new[support] = target_A[support] * phase[support]

    # Outside support: feedback
    E_new[~support] = E[~support] - beta * E_new[~support]
    return E_new


# --------------------------- Error metric ---------------------------

def amplitude_rmse(E: Array, I_meas: Array, eps: float = 1e-12) -> float:
    A = _np.abs(E)
    target_A = _np.sqrt(_np.maximum(I_meas, 0.0))
    num = _np.mean((A - target_A) ** 2)
    return float(_np.sqrt(num + eps))


# --------------------------- Core solver ---------------------------

@dataclass
class PlaneData:
    z: float  # position of plane relative to reference plane (m)
    intensity: Array  # measured intensity (Ny, Nx)
    weight: float = 1.0
    mask: Optional[Array] = None  # boolean mask where to enforce amplitude


@dataclass
class GSConfig:
    wavelength: float  # m
    dx: float  # m/pixel
    dy: float  # m/pixel
    n: float = 1.0
    n_iterations: int = 300
    method: str = "ER"  # "ER" or "HIO"
    hio_beta: float = 0.9
    center_strategy: str = "barycenter"  # "barycenter", "max", "no"
    pad_factor: int = 2  # zero-padding factor for propagation (~anti-alias)
    use_gpu: bool = False
    random_seed: Optional[int] = 123
    store_all_planes: bool = False  # memory heavy if True


@dataclass
class GSResult:
    E_ref: Array  # complex field at reference plane (unpadded, original size)
    errors: List[Dict[str, float]]  # per-iteration summary
    last_planes: Optional[List[Array]]  # complex fields at all planes at last iter


def _zero_pad(arr: Array, pad_factor: int) -> Tuple[Array, Tuple[slice, slice]]:
    Ny, Nx = arr.shape
    Ny2 = pad_factor * Ny
    Nx2 = pad_factor * Nx
    out = _np.zeros((Ny2, Nx2), dtype=arr.dtype)
    sy = slice((Ny2 - Ny) // 2, (Ny2 - Ny) // 2 + Ny)
    sx = slice((Nx2 - Nx) // 2, (Nx2 - Nx) // 2 + Nx)
    out[sy, sx] = arr
    return out, (sy, sx)


def _crop(arr: Array, crop_slices: Tuple[slice, slice]) -> Array:
    sy, sx = crop_slices
    return arr[sy, sx]


def _recentre_amplitude(I: Array, strategy: str) -> Array:
    if strategy == "no":
        return I
    from scipy.ndimage import gaussian_filter, center_of_mass  # lazy import

    F = gaussian_filter(I, sigma=1.0)
    if strategy == "max":
        # Move global maximum to the array center
        idx = _np.unravel_index(_np.argmax(F), F.shape)
        cy, cx = (s // 2 for s in I.shape)
        dy = cy - idx[0]
        dx = cx - idx[1]
    elif strategy == "barycenter":
        cy, cx = (s // 2 for s in I.shape)
        y = _np.arange(I.shape[0])
        x = _np.arange(I.shape[1])
        X, Y = _np.meshgrid(x, y)
        m = _np.maximum(F, 0)
        m_sum = m.sum() + 1e-16
        x_c = (X * m).sum() / m_sum
        y_c = (Y * m).sum() / m_sum
        dx = int(round(cx - x_c))
        dy = int(round(cy - y_c))
    else:
        return I
    return _np.roll(_np.roll(I, shift=dy, axis=0), shift=dx, axis=1)


def gerchberg_saxton_multiplane(
    planes: List[PlaneData],
    cfg: GSConfig,
    init_phase: Optional[Array] = None,
    reference_plane_index: Optional[int] = None,
) -> GSResult:
    """
    Multi-plane GS: cycle over planes forward and backward each iteration.

    Steps per iteration:
      - Start at reference plane field E_ref
      - For k in forward order: propagate to plane k, enforce amplitude, continue
      - For k in reverse order back to ref: propagate, enforce, return to ref
    """
    assert len(planes) >= 2, "Need at least two planes"

    # Sort planes by z and define reference
    if reference_plane_index is None:
        # choose plane with z closest to zero or middle plane if not containing zero
        zs = _np.array([p.z for p in planes])
        ref_idx = int(_np.argmin(_np.abs(zs)))
        print(f"Chosen reference plane = {ref_idx}")
    else:
        ref_idx = int(reference_plane_index)

    # Ensure consistent shapes
    Ny, Nx = planes[0].intensity.shape
    for p in planes:
        assert p.intensity.shape == (Ny, Nx), "All planes must share the same shape"

    # Optional recentring per plane
    planes_I = []
    for p in planes:
        I = _np.asarray(p.intensity, dtype=_np.float64)
        I = _np.maximum(I, 0.0)
        I = _recentre_amplitude(I, cfg.center_strategy)
        planes_I.append(I)

    # Zero-padding for propagation
    pad_factor = max(1, int(cfg.pad_factor))
    I0_pad, crop_slices = _zero_pad(planes_I[ref_idx], pad_factor)
    Ny2, Nx2 = I0_pad.shape

    # Initial field at reference plane
    rng = _np.random.default_rng(cfg.random_seed)
    if init_phase is None:
        phase0 = rng.uniform(-_np.pi, _np.pi, size=(Ny2, Nx2))
    else:
        # If provided, pad to padded size
        ph = init_phase
        if ph.shape != (Ny, Nx):
            raise ValueError("init_phase must match original image size")
        ph_pad, _ = _zero_pad(_np.angle(_np.exp(1j * ph)), pad_factor)
        phase0 = ph_pad

    A0 = _np.sqrt(_np.maximum(I0_pad, 0.0))
    E_ref = A0 * _np.exp(1j * phase0)

    # Precompute padded intensities and masks
    planes_pad: List[Tuple[float, Array, Optional[Array], float]] = []
    for k, p in enumerate(planes):
        I_pad, _ = _zero_pad(planes_I[k], pad_factor)
        mask_pad = None
        if p.mask is not None:
            mask = _np.asarray(p.mask, dtype=bool)
            assert mask.shape == (Ny, Nx)
            mask_pad, _ = _zero_pad(mask.astype(_np.uint8), pad_factor)
            mask_pad = mask_pad.astype(bool)
        planes_pad.append((p.z - planes[ref_idx].z, I_pad, mask_pad, float(p.weight)))

    # Iteration
    errors: List[Dict[str, float]] = []
    E_planes_last: Optional[List[Array]] = None

    for it in range(cfg.n_iterations):
        # Forward pass: reference -> ... -> last
        E = E_ref
        err_forward: Dict[str, float] = {}
        for k in range(ref_idx, len(planes_pad)):
            z, I_pad, mask_pad, w = planes_pad[k]
            if k != ref_idx:
                E = angular_spectrum_propagate(
                    E, z=z, wavelength=cfg.wavelength, dx=cfg.dx, dy=cfg.dy, n=cfg.n, use_gpu=cfg.use_gpu
                )
            # Enforce amplitude
            if cfg.method.upper() == "HIO":
                E = hybrid_input_output(E, I_pad, beta=cfg.hio_beta, support=mask_pad)
            else:
                E = enforce_measured_amplitude(E, I_pad, weight=w, mask=mask_pad)
            # Track error
            err_forward[f"rmse_k{k}"] = amplitude_rmse(E, I_pad)

        # Backward pass: last -> ... -> reference
        for k in range(len(planes_pad) - 2, -1, -1):  # go back to 0
            z_next = planes_pad[k][0]
            z_curr = planes_pad[k + 1][0]
            dz = z_next - z_curr  # propagate backwards
            E = angular_spectrum_propagate(
                E, z=dz, wavelength=cfg.wavelength, dx=cfg.dx, dy=cfg.dy, n=cfg.n, use_gpu=cfg.use_gpu
            )
            I_pad, mask_pad, w = planes_pad[k][1], planes_pad[k][2], planes_pad[k][3]
            if cfg.method.upper() == "HIO":
                E = hybrid_input_output(E, I_pad, beta=cfg.hio_beta, support=mask_pad)
            else:
                E = enforce_measured_amplitude(E, I_pad, weight=w, mask=mask_pad)

            err_forward[f"rmse_k{k}"] = amplitude_rmse(E, I_pad)

        # Return to exact reference plane
        # current k==ref_idx-1, need to move to ref_idx (dz from k to ref_idx)
        z_back = planes_pad[ref_idx][0] - planes_pad[0][0]
        # But because of the loop above, we are now at k=0; propagate to ref plane:
        E = angular_spectrum_propagate(
            E,
            z=z_back,
            wavelength=cfg.wavelength,
            dx=cfg.dx,
            dy=cfg.dy,
            n=cfg.n,
            use_gpu=cfg.use_gpu,
        )

        E_ref = E  # update reference plane field

        # Save errors
        mean_rmse = float(_np.mean(list(err_forward.values()))) if err_forward else _np.nan
        errors.append({"iter": it + 1, "mean_rmse": mean_rmse, **err_forward})

    if cfg.store_all_planes:
        # Propagate once to store all planes from the final reference field
        E_planes_last = []
        for k in range(len(planes_pad)):
            z = planes_pad[k][0]
            if k == ref_idx:
                E_k = E_ref.copy()
            else:
                # propagate from ref to k
                E_k = angular_spectrum_propagate(
                    E_ref, z=z, wavelength=cfg.wavelength, dx=cfg.dx, dy=cfg.dy, n=cfg.n, use_gpu=cfg.use_gpu
                )
            E_planes_last.append(_crop(E_k, crop_slices))
    else:
        E_planes_last = None

    # Return cropped reference field
    E_ref_out = _crop(E_ref, crop_slices)
    return GSResult(E_ref=E_ref_out, errors=errors, last_planes=E_planes_last)


# --------------------------- LightPipes/"flow" benchmarking helpers ---------------------------

def _load_lightpipes_planes(
    figs_pickle_path: str,
    plane_keys: List[str],
    intensity_scale: float = 1.0,
) -> Tuple[List[PlaneData], float, float, _np.ndarray]:
    """
    Load 10 (or N) planes from your LightPipes-style `*_figs.pickle`.

    Expect each entry to be a tuple/list like: [image, ei, prop_size_m, z_m].
    Returns (planes, dx, dy, zs_array).
    """
    import pickle, os
    assert os.path.exists(figs_pickle_path), f"Missing: {figs_pickle_path}"
    with open(figs_pickle_path, "rb") as f:
        figs = pickle.load(f)

    planes: List[PlaneData] = []
    dx = dy = None  # type: ignore
    zs: List[float] = []

    for key in plane_keys:
        if key not in figs:
            raise KeyError(f"Key '{key}' not found in figs.pickle. Available keys include e.g. {list(figs.keys())[:10]} ...")
        data = figs[key]
        # Robust unpacking: data may be list/tuple. We expect indices [0]=image, [2]=prop_size, [3]=z
        img = _np.asarray(data[0], dtype=float)
        prop_size_m = float(data[2])
        z_m = float(data[3])
        if dx is None:
            N = img.shape[0]
            dx = dy = prop_size_m / N
        I = _np.maximum(img * float(intensity_scale), 0.0)
        planes.append(PlaneData(z=z_m, intensity=I, weight=1.0, mask=None))
        zs.append(z_m)

    # Sort planes by z
    order = _np.argsort(_np.asarray(zs))
    planes = [planes[i] for i in order]
    zs_sorted = _np.sort(_np.asarray(zs))
    assert dx is not None and dy is not None
    return planes, float(dx), float(dy), zs_sorted


def benchmark_gs_with_lightpipes(
    figs_pickle_path: str,
    res_pickle_path: Optional[str],
    plane_keys: List[str],
    wavelength: float,
    n_medium: float = 1.0,
    n_iterations: int = 250,
    pad_factor: int = 2,
    method: str = "ER",
    hio_beta: float = 0.9,
    center_strategy: str = "barycenter",
    use_gpu: bool = False,
    compare_against_complex_fields: Optional[str] = None,
) -> Dict[str, _np.ndarray]:
    """
    Run GS on LightPipes intensity planes and compare the reconstructed field against
    simulated data.

    Parameters
    ----------
    figs_pickle_path : path to `<yaml_tag>_figs.pickle` containing images
    res_pickle_path  : path to `<yaml_tag>_res.pickle` to retrieve photon/W scaling (optional)
    plane_keys       : list of N plane keys (e.g., 10) such as 'flow_042_1.02_VB_perp'
    wavelength       : laser central wavelength [m]
    n_medium         : refractive index
    n_iterations     : GS iterations
    pad_factor       : zero-padding factor for propagation
    method           : 'ER' or 'HIO'
    hio_beta         : HIO beta
    center_strategy  : 'barycenter'|'max'|'no'
    use_gpu          : CuPy acceleration if available
    compare_against_complex_fields : optional path to a pickle holding complex fields
        in a dict keyed by the same plane names (values complex 2D arrays). If given,
        we will align global phase and compute complex-field errors.

    Returns
    -------
    dict with keys: 'E_ref', 'errors_table', 'per_plane_metrics', 'dx', 'dy', 'zs',
    and optionally 'per_plane_complex_metrics'.
    """
    import pickle, os

    # Optional scaling from _res.pickle
    intensity_scale = 1.0
    if res_pickle_path and os.path.exists(res_pickle_path):
        with open(res_pickle_path, "rb") as fr:
            res = pickle.load(fr)
        # Expected structure: (figs_dict?, params_dict, ...)
        try:
            params = res[1]
            intensity_scale = float(params.get("scale_phot", 1.0))
        except Exception:
            pass

    planes, dx, dy, zs = _load_lightpipes_planes(figs_pickle_path, plane_keys, intensity_scale)

    cfg = GSConfig(
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        n=n_medium,
        n_iterations=int(n_iterations),
        method=method,
        hio_beta=hio_beta,
        center_strategy=center_strategy,
        pad_factor=pad_factor,
        use_gpu=use_gpu,
        random_seed=123,
        store_all_planes=False,
    )

    # Run GS
    result = gerchberg_saxton_multiplane(planes, cfg)
    E_ref = result.E_ref

    # Compare by re-propagating to each plane
    per_plane_metrics: List[Dict[str, float]] = []
    for key, P in zip(plane_keys, planes):
        E_k = angular_spectrum_propagate(E_ref, z=(P.z - planes[_np.argmin(_np.abs([pp.z for pp in planes]))].z),
                                         wavelength=wavelength, dx=dx, dy=dy, n=n_medium, use_gpu=use_gpu)
        Ik_rec = _np.abs(E_k) ** 2
        Ik_meas = _np.maximum(_np.asarray(P.intensity, dtype=float), 0.0)
        # Metrics
        mse = float(_np.mean((Ik_rec - Ik_meas) ** 2))
        rmse = float(_np.sqrt(mse))
        nrmse = float(rmse / (_np.max(Ik_meas) + 1e-16))
        corr = float(_np.corrcoef(Ik_rec.ravel(), Ik_meas.ravel())[0, 1]) if Ik_meas.std() > 0 else _np.nan
        per_plane_metrics.append({"plane": key, "z": float(P.z), "rmse": rmse, "nrmse": nrmse, "corr": corr})

    out: Dict[str, _np.ndarray] = {
        "E_ref": E_ref,
        "errors_table": _np.array(result.errors, dtype=object),
        "per_plane_metrics": _np.array(per_plane_metrics, dtype=object),
        "dx": dx,
        "dy": dy,
        "zs": zs,
    }

    # Optional: compare complex fields if available
    if compare_against_complex_fields and os.path.exists(compare_against_complex_fields):
        with open(compare_against_complex_fields, "rb") as fc:
            gt_fields = pickle.load(fc)  # dict: key -> complex field (Ny, Nx)
        complex_metrics: List[Dict[str, float]] = []
        # Determine reference plane index (closest to z=0)
        ref_idx = int(_np.argmin(_np.abs(_np.array([p.z for p in planes]))))
        for key, P in zip(plane_keys, planes):
            if key not in gt_fields:
                continue
            # Propagate our reference to this plane
            E_k = angular_spectrum_propagate(E_ref, z=(P.z - planes[ref_idx].z),
                                             wavelength=wavelength, dx=dx, dy=dy, n=n_medium, use_gpu=use_gpu)
            E_gt = _np.asarray(gt_fields[key])
            # Align global phase using inner product
            alpha = _np.vdot(E_gt.ravel(), E_k.ravel())
            phase_correction = _np.exp(-1j * _np.angle(alpha))
            E_k_aligned = E_k * phase_correction
            # Complex error metrics
            l2 = float(_np.linalg.norm(E_k_aligned - E_gt) / (_np.linalg.norm(E_gt) + 1e-16))
            i_rmse = float(_np.sqrt(_np.mean((_np.abs(E_k_aligned)**2 - _np.abs(E_gt)**2) ** 2)))
            complex_metrics.append({"plane": key, "z": float(P.z), "rel_L2_complex": l2, "I_RMSE": i_rmse})
        out["per_plane_complex_metrics"] = _np.array(complex_metrics, dtype=object)

    return out


# --------------------------- Example benchmarking script ---------------------------
if __name__ == "__main__":
    # Example: fill your real keys from the simulation
    example_plane_keys = [
        "flow_042_1.02_VB_perp",
        "flow_055_0.80_VB_perp",
        "flow_060_0.60_VB_perp",
        "flow_065_0.40_VB_perp",
        "flow_070_0.20_VB_perp",
        "flow_075_0.00_VB_perp",
        "flow_080_-0.20_VB_perp",
        "flow_085_-0.40_VB_perp",
        "flow_090_-0.60_VB_perp",
        "flow_095_-0.80_VB_perp",
    ]

    figs_pickle = "/home/yu79deg/darkfield_p5438/Aime/pickles/VB_37_figs.pickle"
    res_pickle  = "/home/yu79deg/darkfield_p5438/Aime/pickles/VB_37_res.pickle"

    out = benchmark_gs_with_lightpipes(
        figs_pickle_path=figs_pickle,
        res_pickle_path=res_pickle,
        plane_keys=example_plane_keys,
        wavelength=800e-9,
        n_medium=1.0,
        n_iterations=250,
        pad_factor=2,
        method="ER",
        center_strategy="barycenter",
        use_gpu=False,
        compare_against_complex_fields=None,  # or path to optional complex-fields pickle
    )

    # Print a compact metrics table
    print("Per-plane intensity metrics (LightPipes vs GS):")  # noqa: T201
    for row in out["per_plane_metrics"]:
        print(f"{row['plane']:>24s}  z={row['z']:+.4e} m  corr={row['corr']:.4f}  RMSE={row['rmse']:.3e}  NRMSE={row['nrmse']:.3e}")  # noqa: T201

    print("Final GS avg RMSE across planes:", _np.mean([r['rmse'] for r in out["per_plane_metrics"]]))  # noqa: T201

# --------------------------- Example usage ---------------------------

if __name__ == "__main__":
    # Minimal demo with synthetic data (replace with real images)
    import matplotlib.pyplot as plt
    Ny, Nx = 256, 256
    dx = dy = 2.5e-6  # 2.5 µm/pixel
    wavelength = 800e-9

    # Build a synthetic focused Gaussian beam at z=0 as ground truth
    x = _np.arange(Nx) - Nx // 2
    y = _np.arange(Ny) - Ny // 2
    X, Y = _np.meshgrid(x, y)
    X *= dx
    Y *= dy
    w0 = 15e-6
    phase_curv = _np.zeros_like(X)
    E_true_0 = _np.exp(-(X**2 + Y**2) / (w0**2)) * _np.exp(1j * phase_curv)

    # Create planes around focus
    zs = _np.linspace(-1.0e-3, 1.0e-3, 10)  # 10 planes within ±1 mm

    planes = []
    for z in zs:
        E_z = angular_spectrum_propagate(E_true_0, z, wavelength, dx, dy, n=1.0)
        I = _np.abs(E_z) ** 2
        planes.append(PlaneData(z=float(z), intensity=I, weight=1.0, mask=None))

    cfg = GSConfig(
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        n=1.0,
        n_iterations=150,
        method="ER",
        hio_beta=0.9,
        center_strategy="barycenter",
        pad_factor=2,
        use_gpu=False,
        random_seed=123,
        store_all_planes=False,
    )

    result = gerchberg_saxton_multiplane(planes, cfg)

    E_rec = result.E_ref
    print("Recovered field shape:", E_rec.shape)
    print("Final mean RMSE:", result.errors[-1]["mean_rmse"])  # noqa: T201

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(_np.abs(E_true_0) ** 2, cmap="magma"); axs[0].set_title("True I @ ref")
    axs[1].imshow(_np.abs(E_rec) ** 2, cmap="magma"); axs[1].set_title("Recovered I @ ref")
    axs[2].imshow(_np.angle(E_rec), cmap="twilight"); axs[2].set_title("Recovered phase @ ref")
    for a in axs: a.axis("off")
    plt.tight_layout(); plt.show()
