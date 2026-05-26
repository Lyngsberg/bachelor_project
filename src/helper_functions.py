import numpy as np
from skimage.filters import threshold_otsu



def otsu_threshold(image: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Apply Otsu thresholding to an image, returning a new binary image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D grayscale array).
    log_scale : bool, optional
        If True (default), apply log scaling to the image before computing
        the Otsu threshold. This helps when pixel intensities are heavily
        skewed (e.g. CT/microscopy data with bright sparse features).

    Returns
    -------
    np.ndarray
        Binary image of the same shape as `image`, with True where pixel
        intensity is above the Otsu threshold and False elsewhere.
    """
    img = image.astype(np.float64)

    scale_factor = 0.85

    if log_scale:
        # Shift so the minimum value is 1 before taking the log (avoids log(0))
        img = np.log1p(img - img.min())

    threshold = threshold_otsu(img)
    threshold *= scale_factor
    binary = img > threshold

    return binary


def uniform_sphere_points(n: int, radius: float = 1.0) -> np.ndarray:
    """
    Generate `n` uniformly distributed points on the surface of a sphere.

    Uses the normalized-Gaussian method: draw (X, Y, Z) from independent
    standard normals, normalize the resulting vector to unit length, then
    scale by `radius`.  This yields a provably uniform distribution on the
    sphere because the 3-D standard normal is spherically symmetric.

    Parameters
    ----------
    n : int
        Number of points to generate.
    radius : float, optional
        Radius of the sphere (default 1.0).

    Returns
    -------
    np.ndarray, shape (n, 3)
        Array of (x, y, z) coordinates, one point per row.
    """
    # Draw n vectors from a 3-D standard normal distribution
    vecs = np.random.randn(n, 3)

    # Normalise each vector to unit length (ignore near-zero vectors, extremely
    # unlikely but handled by re-drawing those rows)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    too_small = (norms < 1e-10).ravel()
    while too_small.any():
        vecs[too_small] = np.random.randn(too_small.sum(), 3)
        norms[too_small] = np.linalg.norm(vecs[too_small], axis=1, keepdims=True)
        too_small = (norms < 1e-10).ravel()

    return radius * vecs / norms