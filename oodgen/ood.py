"""Out-of-distribution (OoD) generators for the FAIR-Universe Weak Lensing Phase-2 task.

The organizers state, verbatim (pages/Phase2/data.md):

    "participants are not provided with the ground-truth labels, ANY EXPLICIT OoD
     EXAMPLES, or details regarding how the OoD data are generated."

and (white paper, section 5.2.3) that the baselines fail "across some OoD
CATEGORIES" whose identity is never disclosed.

This module fills that gap with two families of *controllable, per-sample* OoD:

  A. PHYSICAL  -- baryonic-model reweighting in Fourier space.
                  kappa_new(k) = kappa(k) * T_new(k) / T_old(k)
                  T(k) = P_baryon(k) / P_DMO(k) from HMcode-2020 (CAMB).
                  Physics basis: the organizers applied exactly such a transfer
                  function to the density field (white paper section 2), so the
                  inverse-application on the resulting convergence map is
                  well founded.  Moving logT_AGN outside the training prior
                  [7.2, 8.5] yields a physically motivated parametric OoD.

  B. SPATIAL   -- the three "strong spatial biases" of the organizers' own
                  companion paper (Diao, Dai & Seljak 2025, arXiv:2505.00632,
                  section 3.6): smoothing / scaling / masking.

Every generator is PER-SAMPLE: the output for one map never depends on any
other map.  This complies with the competition's Rules for OoD Detection Methods.

Reference for the transfer-function route:
  Mead et al. 2021 (HMcode-2020), arXiv:2009.01858
"""
from __future__ import annotations

import os

import numpy as np

_GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tk_grid.npz")


# --------------------------------------------------------------------------
# B. spatial biases  (organizers' companion paper, section 3.6)
# --------------------------------------------------------------------------
def spatial_bias(flat, mask, kind, strength=1.0, rng=None):
    """flat: (n, n_masked) float array; mask: bool 2-D.  Returns same shape."""
    from scipy.ndimage import gaussian_filter
    if kind not in ("smooth", "scale", "mask"):
        raise ValueError("kind must be smooth | scale | mask")
    rng = np.random.default_rng(0) if rng is None else rng
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        img = np.zeros(mask.shape, dtype=np.float32)
        img[mask] = row
        if kind == "smooth":
            img = gaussian_filter(img, 1.5 * strength)
        elif kind == "scale":
            img = img * (1.0 + 0.15 * strength)
        else:
            y0 = mask.shape[0] // 3
            img[y0:y0 + mask.shape[0] // 3, :] = 0.0
        out[i] = img[mask]
    return out


# --------------------------------------------------------------------------
# A. physical bias -- baryonic transfer function
# --------------------------------------------------------------------------
def comoving_hmpc(z, Om=0.315):
    """Comoving distance in h^-1 Mpc (flat LCDM, c/(100 km/s) = 2998)."""
    from scipy.integrate import quad
    E = lambda zz: np.sqrt(Om * (1 + zz) ** 3 + (1 - Om))
    val, _ = quad(lambda zz: 1.0 / E(zz), 0, z)
    return 2998.0 * val


def ell_grid(shape, pix_arcmin=2.0):
    """|ell| for every rfft2 mode: |ell| = 2*pi*freq / theta_pix."""
    ny, nx = shape
    th = np.deg2rad(pix_arcmin / 60.0)
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    return 2 * np.pi * np.sqrt(fy ** 2 + fx ** 2) / th


def transfer_camb(logT_AGN, A_baryon=3.13, eta_baryon=0.603,
                  kmax=30.0, npts=120, z=0.0):
    """T(k) = P_baryon / P_DMO from CAMB's HMcode-2020 feedback.

    !! CAMB 2.x gotcha (verified): the HMcode parameters MUST be assigned on the
    NonLinearModel object.  Assigning them on CAMBparams is SILENTLY IGNORED
    (no error, T(k) comes back identically 1).  Both forms are shown below.

        p.NonLinearModel.halofit_version = "mead2020_feedback"   # works
        p.NonLinearModel.HMCode_logT_AGN = 7.8                   # works
        p.halofit_version = "mead2020_feedback"                  # silently ignored
        p.HMCode_logT_AGN = 7.8                                  # silently ignored
    """
    import camb

    def pk(feedback):
        p = camb.CAMBparams()
        p.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.0, omk=0)
        p.NonLinear = camb.model.NonLinear_both
        p.set_matter_power(redshifts=[z], kmax=kmax, nonlinear=True)
        if feedback:
            p.NonLinearModel.halofit_version = "mead2020_feedback"
            p.NonLinearModel.HMCode_logT_AGN = float(logT_AGN)
            p.NonLinearModel.HMCode_A_baryon = float(A_baryon)
            p.NonLinearModel.HMCode_eta_baryon = float(eta_baryon)
        else:
            p.NonLinearModel.halofit_version = "mead2020"
        r = camb.get_results(p)
        kh, _, P = r.get_matter_power_spectrum(minkh=0.02, maxkh=kmax, npoints=npts)
        return kh, P[0]

    kd, pd = pk(False)
    kb, pb = pk(True)
    k = np.logspace(np.log10(0.02), np.log10(kmax), npts)
    return k, np.interp(k, kb, pb) / np.interp(k, kd, pd)


def load_transfer_grid(path=None):
    """Pre-computed T(k) grid (logT_AGN = 6.0 ... 9.5).  Regenerate with transfer_camb."""
    path = path or _GRID
    z = np.load(path)
    return {k: z[k] for k in z.files}


def baryon_bias(flat, mask, logT_old, logT_new, tk=None, chi_eff=None, pix_arcmin=2.0):
    """Apply a DIFFERENT baryonic model to maps already baryonified with logT_old.

    logT_old : scalar or per-sample array (use the competition labels!)
    logT_new : target logT_AGN, e.g. outside the training prior [7.2, 8.5]

    Approximation: single effective comoving distance (Limber), k = ell / chi_eff.
    """
    tk = load_transfer_grid() if tk is None else tk
    if chi_eff is None:
        chi_eff = comoving_hmpc(0.4)               # near the lensing-kernel peak
    kk = np.asarray(tk["k"])
    keys = sorted((float(k[2:]), k) for k in tk if str(k).startswith("lT"))
    if not keys:
        raise ValueError("transfer grid has no lT* entries")

    def T_at(lt):
        return np.asarray(tk[min(keys, key=lambda p: abs(p[0] - float(lt)))[1]])

    k_of_ell = ell_grid(mask.shape, pix_arcmin) / chi_eff
    Tn = np.interp(k_of_ell, kk, T_at(logT_new))
    old = np.full(flat.shape[0], float(logT_old)) if np.isscalar(logT_old) else np.asarray(logT_old)

    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        img = np.zeros(mask.shape, dtype=np.float64)
        img[mask] = flat[i]
        mean = img[mask].mean()
        K = np.fft.rfft2(img - mean)
        K *= Tn / np.interp(k_of_ell, kk, T_at(old[i]))
        # preserve the map mean: T(k) is defined as P_baryon/P_DMO, i.e. it acts on
        # fluctuations only.  Dropping the mean would inject a spurious offset.
        out[i] = np.fft.irfft2(K, s=mask.shape)[mask] + mean
    return out.astype(np.float32)
