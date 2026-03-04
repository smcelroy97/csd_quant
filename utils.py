import numpy as np
import ot  # pip install POT
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, sosfiltfilt


def interp_csd_to_grid(
    csd: np.ndarray,
    sp_len: int = 30,
    t_len: int = 100,
) -> np.ndarray:
    """
    Interpolate a CSD array to (sp_len, t_len) on a normalized [0,1]x[0,1] grid.
    Assumes csd shape = (depth, time)
    """
    z = np.asarray(csd, dtype=float)
    if z.ndim != 2:
        raise ValueError("csd must be 2D (depth, time)")

    d_len, t_orig = z.shape

    # Original grid (normalized, as in Rimehaug et al.)
    depth = np.linspace(0.0, 1.0, d_len)
    time = np.linspace(0.0, 1.0, t_orig)

    interp = RegularGridInterpolator(
        (depth, time),
        z,
        method="linear",      # cubic not supported; linear is what POT expects anyway
        bounds_error=False,
        fill_value=0.0,
    )

    # New grid
    # New grid
    depth_new = np.linspace(0.0, 1.0, sp_len)
    time_new = np.linspace(0.0, 1.0, t_len)
    dd, tt = np.meshgrid(depth_new, time_new, indexing="ij")

    pts = np.column_stack([dd.ravel(), tt.ravel()])
    z_new = interp(pts).reshape(sp_len, t_len)

    return z_new


def _to_probability_mass(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip to nonnegative and normalize to sum=1 (required by balanced EMD)."""
    a = np.asarray(arr, dtype=float)
    a[a < 0] = 0.0
    s = a.sum()
    if s <= eps:
        # no mass: return all zeros (caller can treat as distance 0)
        return np.zeros_like(a)
    return a / s


def wasserstein_2d_mass(
    a2d: np.ndarray,
    b2d: np.ndarray,
    depth_weight: float = 1.0,
    time_weight: float = 1.0,
) -> float:
    """
    First Wasserstein distance between two 2D nonnegative "mass" arrays using POT EMD.
    Arrays must have the same shape (depth, time). Returns scalar distance.
    """
    if a2d.shape != b2d.shape:
        raise ValueError("a2d and b2d must have the same shape. Interpolate first.")

    a = _to_probability_mass(a2d).reshape(-1)
    b = _to_probability_mass(b2d).reshape(-1)

    if a.sum() == 0 and b.sum() == 0:
        return 0.0

    d_len, t_len = a2d.shape

    # Coordinates for each pixel/bin in (depth,time), each scaled to [0,1]
    depth = np.linspace(0.0, 1.0, d_len)
    time = np.linspace(0.0, 1.0, t_len)
    dd, tt = np.meshgrid(depth, time, indexing="ij")  # (depth,time)

    coords = np.column_stack([depth_weight * dd.reshape(-1), time_weight * tt.reshape(-1)])

    # Cost matrix (normalize like their code)
    M = ot.dist(coords, coords)
    M /= M.max()

    # Transport plan + cost
    G = ot.emd(a, b, M)
    return float((M * G).sum())


def wasserstein_csd(
    csd_a: np.ndarray,
    csd_b: np.ndarray,
    interpolate: bool = True,
    sp_len: int = 30,
    t_len: int = 100,
    depth_weight: float = 1.0,
    time_weight: float = 1.0,
) -> float:
    """
    Total WD between two CSD patterns = WD(sinks) + WD(sources).
    Sinks are abs(negative CSD); sources are positive CSD. :contentReference[oaicite:1]{index=1}
    """
    A = interp_csd_to_grid(csd_a, sp_len, t_len) if interpolate else np.asarray(csd_a, float)
    B = interp_csd_to_grid(csd_b, sp_len, t_len) if interpolate else np.asarray(csd_b, float)

    if A.shape != B.shape:
        raise ValueError("After interpolation, shapes still differ. Check inputs.")

    sinks_A = np.maximum(-A, 0.0)
    sinks_B = np.maximum(-B, 0.0)
    src_A = np.maximum(A, 0.0)
    src_B = np.maximum(B, 0.0)

    wd_sinks = wasserstein_2d_mass(sinks_A, sinks_B, depth_weight, time_weight)
    wd_src = wasserstein_2d_mass(src_A, src_B, depth_weight, time_weight)
    return wd_sinks + wd_src


def pairwise_wd_csd(csds: list[np.ndarray], **kwargs) -> np.ndarray:
    """Pairwise total WD matrix for a list of CSD arrays."""
    n = len(csds)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            d = wasserstein_csd(csds[i], csds[j], **kwargs)
            D[i, j] = D[j, i] = d
    return D


def getbandpass(lfps, sampr, minf=0.05, maxf=300):
    datband = []
    sos = butter(4, [minf, maxf], btype="bandpass", fs=sampr, output="sos")
    for i in range(len(lfps[0])):
        datband.append(sosfiltfilt(sos, lfps[:, i]))
    datband = np.array(datband)
    return datband


def get_csd(lfps, sampr, spacing_um=100.0, minf=0.05, maxf=300, norm=True):
    datband = getbandpass(lfps, sampr, minf, maxf)
    if datband.shape[0] > datband.shape[1]:  # take CSD along smaller dimension
        ax = 1
    else:
        ax = 0

    # when drawing CSD make sure that negative values (depolarizing intracellular current) drawn in red,
    # and positive values (hyperpolarizing intracellular current) drawn in blue
    spacing_mm = spacing_um/1000  # spacing in mm
    csd = -np.diff(datband, n=2, axis=ax)/spacing_mm**2  # now each column (or row) is an electrode -- CSD along electrodes

    return csd
