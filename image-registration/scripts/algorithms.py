"""Standalone numerical routines for translational image registration."""
from typing import Any, Literal, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from scipy import optimize
from scipy.special import erf
from skimage import feature
from skimage.metrics import normalized_mutual_information
from skimage.registration import phase_cross_correlation as skimage_phase_cross_correlation

def _gaussian_rect_window(shape: tuple[int, int], decay_fraction: float = 0.2) -> np.ndarray:
    """2D Gaussian-softened rectangle window.

    The mask is 1 in the center and decays smoothly to 0 at each edge.
    The decay is the convolution of a step function with a Gaussian, implemented
    via the error function. The transition from 1 to 0 spans ``decay_fraction``
    of the image size on each side.
    """
    def _win_1d(size: int) -> np.ndarray:
        decay_len = decay_fraction * size
        sigma = decay_len / 4.0
        s2 = sigma * np.sqrt(2.0)
        x = np.arange(size, dtype=float)
        left = 0.5 * (1.0 + erf((x - decay_len / 2.0) / s2))
        right = 0.5 * (1.0 - erf((x - (size - 1.0 - decay_len / 2.0)) / s2))
        return np.minimum(left, right)

    return np.outer(_win_1d(shape[0]), _win_1d(shape[1]))


def phase_cross_correlation(
    moving: np.ndarray,
    ref: np.ndarray,
    filtering_method: Optional[Literal["hanning", "gaussian"]] = "hanning",
    upsample_factor: int = 1,
) -> np.ndarray | Tuple[np.ndarray, float]:
    """Phase correlation with windowing. The result gives
    the translation offset to apply to the moving image so that it aligns
    with the reference image.

    Parameters
    ----------
    moving : np.ndarray
        A 2D image.
    ref : np.ndarray
        A 2D image.
    filtering_method : {"hanning", "gaussian"} or None, optional
        Window function applied to both images before phase correlation to
        reduce spectral leakage.  ``"hanning"`` uses a standard Hanning window.
        ``"gaussian"`` uses a Gaussian-softened rectangle that is 1 in the
        centre and decays to 0 at each edge over 20 % of the image size.
        Pass ``None`` to disable windowing.
    upsample_factor : int, optional
        Upsampling factor for subpixel accuracy in phase correlation.
        A value of 1 yields pixel-level precision.

    Returns
    -------
    np.ndarray
        The translation offset (dy, dx) to apply to the moving image.
    """
    assert np.all(np.array(moving.shape) == np.array(ref.shape)), (
        "The shapes of the moving and reference images must be the same."
    )
    moving = moving - moving.mean()
    ref = ref - ref.mean()
    if filtering_method == "hanning":
        win_y = np.hanning(moving.shape[0])
        win_x = np.hanning(moving.shape[1])
        win = np.outer(win_y, win_x)
        moving_for_registration = moving * win
        ref_for_registration = ref * win
    elif filtering_method == "gaussian":
        win = _gaussian_rect_window(moving.shape, decay_fraction=0.2)
        moving_for_registration = moving * win
        ref_for_registration = ref * win
    elif filtering_method is None:
        moving_for_registration = moving
        ref_for_registration = ref
    else:
        raise ValueError(
            f"Unknown filtering_method {filtering_method!r}. "
            "Use 'hanning', 'gaussian', or None."
        )

    shift, _, _ = skimage_phase_cross_correlation(
        ref_for_registration,
        moving_for_registration,
        upsample_factor=upsample_factor,
    )
    return shift


def error_minimization_registration(
    moving: np.ndarray,
    ref: np.ndarray,
    y_valid_fraction: float = 0.8,
    x_valid_fraction: float = 0.8,
    subpixel: bool = True,
) -> np.ndarray:
    """Image registration by exhaustive integer-shift MSE search with quadratic
    subpixel refinement.

    A central window of size ``(y_valid_fraction * h, x_valid_fraction * w)``
    is fixed in the reference image.  The moving image is sampled at the same
    window position for every integer shift (dy, dx) within the margins
    ``[-max_dy, max_dy] × [-max_dx, max_dx]``, where the margins are the pixel
    gaps between the valid window and the image boundary.  No wrap-around pixels
    are ever included: the valid window is identical for all shifts.

    The resulting 2-D MSE map is fitted with a 2-D quadratic polynomial.  The
    analytic minimum of that polynomial is returned as the sub-pixel shift.

    Parameters
    ----------
    moving : np.ndarray
        2-D image to register.
    ref : np.ndarray
        2-D reference image with the same shape as *moving*.
    y_valid_fraction : float
        Fraction of the image height occupied by the comparison window.
        Values close to 1 leave little margin and therefore a small search range.
    x_valid_fraction : float
        Same as *y_valid_fraction* along the x (column) axis.
    subpixel : bool
        If True, perform subpixel refinement using a 2D quadratic fit.

    Returns
    -------
    np.ndarray
        Estimated (dy, dx) shift to apply to *moving* so that it aligns with
        *ref*.
    """
    assert moving.shape == ref.shape, (
        "The shapes of the moving and reference images must be the same."
    )
    h, w = ref.shape

    vh = int(round(y_valid_fraction * h))
    vw = int(round(x_valid_fraction * w))

    # Centre the valid window; margin on each side = max search range
    r0 = (h - vh) // 2
    c0 = (w - vw) // 2
    r1, c1 = r0 + vh, c0 + vw
    max_dy, max_dx = r0, c0

    if max_dy == 0 and max_dx == 0:
        return np.zeros(2)

    dy_vals = np.arange(-max_dy, max_dy + 1)
    dx_vals = np.arange(-max_dx, max_dx + 1)

    ref_crop = ref[r0:r1, c0:c1].astype(float)
    moving_f = moving.astype(float)

    # Exhaustive integer-shift MSE map
    error_map = np.empty((len(dy_vals), len(dx_vals)))
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            diff = moving_f[r0 + dy : r1 + dy, c0 + dx : c1 + dx] - ref_crop
            error_map[i, j] = np.mean(diff * diff)

    # Fit quadratic in a local neighbourhood around the integer minimum.
    # Neighbourhood half-width: 10% of image size / 2, at least 1 (→ 3×3 minimum).
    min_i, min_j = np.unravel_index(np.argmin(error_map), error_map.shape)
    if not subpixel:
        return -np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])

    half_y = max(1, int(round(0.05 * h)))
    half_x = max(1, int(round(0.05 * w)))
    i_lo = max(0, min_i - half_y)
    i_hi = min(len(dy_vals) - 1, min_i + half_y)
    j_lo = max(0, min_j - half_x)
    j_hi = min(len(dx_vals) - 1, min_j + half_x)
    local_dy = dy_vals[i_lo : i_hi + 1]
    local_dx = dx_vals[j_lo : j_hi + 1]
    local_err = error_map[i_lo : i_hi + 1, j_lo : j_hi + 1]

    # The 2-D quadratic has 6 parameters; require ≥3 points in each dimension so
    # the design matrix is well-determined and the Hessian is not rank-deficient.
    if len(local_dy) >= 3 and len(local_dx) >= 3:
        # Fit: f(dy, dx) = a*dy² + b*dx² + c*dy*dx + d*dy + e*dx + g
        dy_mesh, dx_mesh = np.meshgrid(local_dy, local_dx, indexing="ij")
        dy_f = dy_mesh.ravel()
        dx_f = dx_mesh.ravel()
        design = np.column_stack(
            [dy_f**2, dx_f**2, dy_f * dx_f, dy_f, dx_f, np.ones(len(dy_f))]
        )
        coeffs, _, _, _ = np.linalg.lstsq(design, local_err.ravel(), rcond=None)
        a, b, c, d, e, _ = coeffs

        # Analytic minimum: solve Hessian @ [dy_min, dx_min]ᵀ = -gradient
        # Hessian = [[2a, c], [c, 2b]]; gradient at origin = [d, e]
        hess = np.array([[2.0 * a, c], [c, 2.0 * b]])
        try:
            if np.all(np.linalg.eigvalsh(hess) > 0):
                shift = np.linalg.solve(hess, np.array([-d, -e]))
            else:
                raise np.linalg.LinAlgError("Hessian not positive definite")
        except np.linalg.LinAlgError:
            shift = np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])
    else:
        shift = np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])

    # Negate: the MSE is minimised at the offset where moving[r0+dy:] matches
    # ref[r0:], but the caller wants the shift to apply to moving so that
    # roll(moving, shift) ≈ ref, which is the opposite direction.
    return -shift


def normalize_image_01(image: np.ndarray) -> np.ndarray:
    """Normalize image intensities to [0, 1]."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return np.zeros_like(image, dtype=np.float32)


def warp_translation(
    image: np.ndarray,
    shift: np.ndarray | tuple[float, float] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image by translation and return warped image with valid mask.

    Parameters
    ----------
    image : np.ndarray
        A 2D image to warp.
    shift : np.ndarray | tuple[float, float] | list[float]
        Internal translation in (dy, dx) used by the optimizer.
    """
    dy, dx = map(float, shift)
    rows, cols = image.shape
    y, x = np.indices((rows, cols), dtype=np.float32)
    sample_y = y - dy
    sample_x = x - dx
    warped = ndi.map_coordinates(
        image,
        [sample_y, sample_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    valid = (
        (sample_y >= 0)
        & (sample_y <= rows - 1)
        & (sample_x >= 0)
        & (sample_x <= cols - 1)
    )
    return warped, valid


def translation_nmi_registration(
    moving: np.ndarray,
    ref: np.ndarray,
    pyramid_levels: tuple[int, ...] = (4, 2, 1),
    bins: int = 64,
    sample_frac: float = 0.2,
    smooth_sigmas: Optional[dict[int, float]] = None,
    optimizer: Literal["powell", "nelder-mead"] = "nelder-mead",
    max_iter: int = 60,
    tol: float = 1e-4,
) -> np.ndarray:
    """Estimate translation (dy, dx) to apply to moving image by maximizing NMI."""
    if moving.ndim != 2 or ref.ndim != 2:
        raise ValueError("`moving` and `ref` must both be 2D images.")
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError("`sample_frac` must be in (0, 1].")
    if smooth_sigmas is None:
        smooth_sigmas = {4: 1.5, 2: 1.0, 1: 0.0}

    shift_full = np.zeros(2, dtype=np.float64)

    for level in pyramid_levels:
        zoom_factor = 1.0 / float(level)
        moving_l = ndi.zoom(moving, zoom_factor, order=1)
        ref_l = ndi.zoom(ref, zoom_factor, order=1)

        sigma = float(smooth_sigmas.get(level, 0.0))
        if sigma > 0:
            moving_l = ndi.gaussian_filter(moving_l, sigma=sigma)
            ref_l = ndi.gaussian_filter(ref_l, sigma=sigma)

        moving_l = normalize_image_01(moving_l)
        ref_l = normalize_image_01(ref_l)

        shift_level_0 = shift_full / float(level)

        def objective(shift_level: np.ndarray) -> float:
            warped, valid_warp = warp_translation(moving_l, shift_level)
            overlap_indices = np.flatnonzero(valid_warp.ravel())
            if overlap_indices.size < 8:
                return 0.0

            n_samples = max(8, int(overlap_indices.size * sample_frac))
            if n_samples < overlap_indices.size:
                rng = np.random.default_rng(13)
                chosen = rng.choice(overlap_indices, size=n_samples, replace=False)
            else:
                chosen = overlap_indices

            ref_samples = ref_l.ravel()[chosen]
            warped_samples = warped.ravel()[chosen]
            nmi = normalized_mutual_information(ref_samples, warped_samples, bins=bins)
            if not np.isfinite(nmi):
                return 1e6
            return -float(nmi)

        if optimizer == "powell":
            result = optimize.minimize(
                objective,
                shift_level_0,
                method="Powell",
                options={
                    "xtol": tol,
                    "ftol": tol,
                    "maxiter": max_iter,
                    "direc": np.diag([2.0, 2.0]),
                },
            )
        elif optimizer == "nelder-mead":
            simplex = np.vstack(
                [
                    shift_level_0,
                    shift_level_0 - np.array([2.0, 0.0]),
                    shift_level_0 - np.array([0.0, 2.0]),
                ]
            )
            result = optimize.minimize(
                objective,
                shift_level_0,
                method="Nelder-Mead",
                options={
                    "xatol": tol,
                    "fatol": tol,
                    "maxiter": max_iter,
                    "initial_simplex": simplex,
                },
            )
        else:
            raise ValueError(
                f"Unsupported optimizer '{optimizer}'. Use 'powell' or 'nelder-mead'."
            )

        shift_opt = result.x if result.success else shift_level_0
        shift_full = np.array(shift_opt, dtype=np.float64) * float(level)

    return shift_full


class ImageRegistration:
    """Translation registration with image preprocessing and subpixel offsets."""

    def __init__(
        self,
        image_coordinates_origin: Literal["top_left", "center"] = "top_left",
        registration_method: str = "ncc",
        zoom: float = 1.0,
        log_scale: bool = False,
    ) -> None:
        self.image_coordinates_origin = image_coordinates_origin
        self.registration_method = registration_method
        self.zoom = zoom
        self.log_scale = log_scale


    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to prepare it for registration.
        If the input image is a 3D array, the last dimension is assumed to
        be the channel dimension and will be averaged over.
        
        Parameters
        ----------
        image : np.ndarray
            A 2D or 3D array representing an image.

        Returns
        -------
        image : np.ndarray
            A 2D array representing an image.
        """
        if image.ndim == 3:
            image = np.mean(image, axis=-1)
        image[np.isnan(image)] = np.mean(image)
        if self.log_scale:
            image = np.log10(image + 1)
        return image

    def zoom_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured registration zoom factor to an image."""
        if self.zoom <= 0:
            raise ValueError("zoom must be positive.")
        if self.zoom == 1.0:
            return image
        return ndi.zoom(image, zoom=self.zoom, order=1, mode="nearest")

    @staticmethod
    def _ncc_preprocess(img: np.ndarray, name: str, sigma: float) -> np.ndarray:
        """Apply one of the tuned NCC preprocessing transforms."""
        if name == "log1p":
            out = np.log1p(np.clip(img, 0, None))
        elif name == "sqrt":
            out = np.sqrt(np.clip(img, 0, None))
        elif name == "rank":
            flat = img.ravel()
            ranks = np.empty_like(flat, dtype=np.float64)
            ranks[np.argsort(flat)] = np.arange(len(flat), dtype=np.float64)
            out = ranks.reshape(img.shape)
        elif name == "power":
            out = np.power(np.clip(img, 0, None) + 1e-12, 0.3)
        elif name == "asinh":
            out = np.arcsinh(img)
        else:
            raise ValueError(f"Unknown NCC preprocessing method: {name}")
        if sigma > 0:
            out = ndi.gaussian_filter(out, sigma=sigma)
        return out

    @staticmethod
    def _overlap_ncc(ref: np.ndarray, moving: np.ndarray, dy: int, dx: int) -> float:
        h, w = ref.shape
        if dy >= 0:
            ry, my = slice(dy, h), slice(0, h - dy)
        else:
            ry, my = slice(0, h + dy), slice(-dy, h)
        if dx >= 0:
            rx, mx = slice(dx, w), slice(0, w - dx)
        else:
            rx, mx = slice(0, w + dx), slice(-dx, w)
        ref_overlap = ref[ry, rx].ravel()
        moving_overlap = moving[my, mx].ravel()
        if len(ref_overlap) < 3:
            return -2.0
        ref_overlap = ref_overlap - np.mean(ref_overlap)
        moving_overlap = moving_overlap - np.mean(moving_overlap)
        denom = np.sqrt(np.sum(ref_overlap**2) * np.sum(moving_overlap**2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(ref_overlap * moving_overlap) / denom)

    def _compute_ncc_surface(
        self,
        ref: np.ndarray,
        moving: np.ndarray,
        max_shift: int,
        min_overlap: int = 3,
    ) -> tuple[int, int, float, dict[tuple[int, int], float]]:
        h, w = ref.shape
        sy = min(max_shift, h - min_overlap)
        sx = min(max_shift, w - min_overlap)
        best_ncc, best_dy, best_dx = -2.0, 0, 0
        ncc_map: dict[tuple[int, int], float] = {}
        for dy in range(-sy, sy + 1):
            for dx in range(-sx, sx + 1):
                ncc = self._overlap_ncc(ref, moving, dy, dx)
                ncc_map[(dy, dx)] = ncc
                if ncc > best_ncc:
                    best_ncc, best_dy, best_dx = ncc, dy, dx
        return best_dy, best_dx, best_ncc, ncc_map

    @staticmethod
    def _extract_ncc_surface_patch(
        ncc_map: dict[tuple[int, int], float],
        peak_dy: int,
        peak_dx: int,
        radius: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        coords = []
        vals = []
        for ody in range(-radius, radius + 1):
            for odx in range(-radius, radius + 1):
                key = (peak_dy + ody, peak_dx + odx)
                if key in ncc_map and ncc_map[key] > -1.5:
                    coords.append([float(ody), float(odx)])
                    vals.append(ncc_map[key])
        if not coords:
            return None, None
        return np.array(coords), np.array(vals)

    @staticmethod
    def _gp_subpixel(
        coords: np.ndarray | None,
        vals: np.ndarray | None,
        step: float = 0.02,
        clamp: float = 0.5,
        lengthscale: float = 1.2,
        noise_var: float = 1e-4,
    ) -> tuple[float, float, float]:
        if coords is None or vals is None or len(coords) < 5:
            return 0.0, 0.0, -1.0
        dist_sq = np.sum(
            (coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        kernel = np.exp(-0.5 * dist_sq / (lengthscale**2))
        kernel += noise_var * np.eye(len(coords))
        try:
            chol = np.linalg.cholesky(kernel)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, vals))
        except np.linalg.LinAlgError:
            kernel += 1e-3 * np.eye(len(coords))
            try:
                alpha = np.linalg.solve(kernel, vals)
            except np.linalg.LinAlgError:
                return 0.0, 0.0, -1.0

        grid_1d = np.arange(-clamp, clamp + step / 2, step)
        gy, gx = np.meshgrid(grid_1d, grid_1d, indexing="ij")
        test_pts = np.column_stack([gy.ravel(), gx.ravel()])
        dist_sq_test = np.sum(
            (test_pts[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        pred = np.exp(-0.5 * dist_sq_test / (lengthscale**2)) @ alpha
        best_idx = np.argmax(pred)
        return float(test_pts[best_idx, 0]), float(test_pts[best_idx, 1]), float(pred[best_idx])

    @staticmethod
    def _quadratic_2d_subpixel(
        coords: np.ndarray | None,
        vals: np.ndarray | None,
    ) -> tuple[float, float]:
        if coords is None or vals is None or len(coords) < 6:
            return 0.0, 0.0
        mask = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= 2.0
        if mask.sum() < 6:
            mask = np.ones(len(coords), dtype=bool)
        c = coords[mask]
        v = vals[mask]
        design = np.column_stack(
            [c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1], c[:, 0], c[:, 1], np.ones(len(c))]
        )
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, v, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        a, b, cross, d, e, _ = coeffs
        denom = 4 * a * b - cross**2
        if abs(denom) < 1e-14 or a > 0 or b > 0:
            return 0.0, 0.0
        peak_dy = float(np.clip((cross * e - 2 * b * d) / denom, -0.5, 0.5))
        peak_dx = float(np.clip((cross * d - 2 * a * e) / denom, -0.5, 0.5))
        return peak_dy, peak_dx

    def ncc_registration(
        self,
        moving: np.ndarray,
        ref: np.ndarray,
        max_shift: int = 7,
        surface_radius: int = 3,
        gp_lengthscale: float = 1.2,
        ncc_power: float = 48.0,
        configs: Optional[list[tuple[str, float]]] = None,
    ) -> np.ndarray:
        """Register images with the tuned NCC ensemble from CVEvolve.

        Parameters
        ----------
        moving : np.ndarray
            Moving image to shift.
        ref : np.ndarray
            Reference image.
        max_shift : int, optional
            Maximum integer shift, in pixels, searched along each axis.
        surface_radius : int, optional
            Radius around the integer peak used for subpixel surface fitting.
        gp_lengthscale : float, optional
            RBF-kernel length scale for Gaussian-process subpixel refinement.
        ncc_power : float, optional
            Power applied to NCC scores when weighting preprocessing pipelines.
        configs : list[tuple[str, float]], optional
            Preprocessing pipeline names and Gaussian smoothing sigmas.

        Returns
        -------
        np.ndarray
            Estimated ``[dy, dx]`` shift to apply to ``moving``.
        """
        if configs is None:
            configs = [
                ("log1p", 0.3),
                ("log1p", 0.5),
                ("log1p", 0.7),
                ("sqrt", 0.3),
                ("sqrt", 0.5),
                ("rank", 0.3),
                ("rank", 0.5),
                ("power", 0.5),
                ("asinh", 0.5),
            ]

        votes: dict[tuple[int, int], int] = {}
        pipelines: list[dict[str, Any]] = []
        for preprocess_name, sigma in configs:
            ref_processed = self._ncc_preprocess(ref.copy(), preprocess_name, sigma)
            moving_processed = self._ncc_preprocess(moving.copy(), preprocess_name, sigma)
            peak_dy, peak_dx, peak_ncc, ncc_map = self._compute_ncc_surface(
                ref_processed,
                moving_processed,
                max_shift=max_shift,
            )
            key = (peak_dy, peak_dx)
            votes[key] = votes.get(key, 0) + 1
            pipelines.append(
                {
                    "int_dy": peak_dy,
                    "int_dx": peak_dx,
                    "peak_ncc": peak_ncc,
                    "ncc_map": ncc_map,
                }
            )

        if not pipelines:
            return np.zeros(2)

        sorted_keys = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)
        best_key = sorted_keys[0]
        if len(sorted_keys) > 1 and votes[sorted_keys[1]] >= votes[best_key] - 1:
            key2 = sorted_keys[1]
            ncc1 = np.mean(
                [
                    p["peak_ncc"]
                    for p in pipelines
                    if (p["int_dy"], p["int_dx"]) == best_key
                ]
            )
            ncc2 = np.mean(
                [
                    p["peak_ncc"]
                    for p in pipelines
                    if (p["int_dy"], p["int_dx"]) == key2
                ]
            )
            if ncc2 > ncc1:
                best_key = key2
        center_dy, center_dx = best_key
        agreeing = [
            p
            for p in pipelines
            if (p["int_dy"], p["int_dx"]) == (center_dy, center_dx)
        ] or pipelines

        pipe_weights = [max(0.01, p["peak_ncc"]) ** ncc_power for p in agreeing]
        weight_total = sum(pipe_weights)
        if weight_total <= 0:
            pipe_weights = [1.0 / len(agreeing)] * len(agreeing)
        else:
            pipe_weights = [w / weight_total for w in pipe_weights]

        coord_values: dict[tuple[int, int], list[float]] = {}
        for pipeline, weight in zip(agreeing, pipe_weights):
            for ody in range(-surface_radius, surface_radius + 1):
                for odx in range(-surface_radius, surface_radius + 1):
                    key = (center_dy + ody, center_dx + odx)
                    if key in pipeline["ncc_map"] and pipeline["ncc_map"][key] > -1.5:
                        point = (ody, odx)
                        if point not in coord_values:
                            coord_values[point] = [0.0, 0.0]
                        coord_values[point][0] += weight * pipeline["ncc_map"][key]
                        coord_values[point][1] += weight

        avg_coords = []
        avg_vals = []
        for (ody, odx), (weighted_sum, total_weight) in coord_values.items():
            avg_coords.append([float(ody), float(odx)])
            avg_vals.append(float(weighted_sum / total_weight))
        avg_coords_arr = np.array(avg_coords) if avg_coords else None
        avg_vals_arr = np.array(avg_vals) if avg_vals else None

        avg_gp_dy, avg_gp_dx, _ = self._gp_subpixel(
            avg_coords_arr,
            avg_vals_arr,
            step=0.01,
            lengthscale=gp_lengthscale,
        )
        quad_dy, quad_dx = self._quadratic_2d_subpixel(avg_coords_arr, avg_vals_arr)

        per_pipe_dys = []
        per_pipe_dxs = []
        for pipeline in agreeing:
            coords, vals = self._extract_ncc_surface_patch(
                pipeline["ncc_map"],
                center_dy,
                center_dx,
                surface_radius,
            )
            if coords is None or len(coords) < 5:
                continue
            pipe_dy, pipe_dx, _ = self._gp_subpixel(
                coords,
                vals,
                step=0.02,
                lengthscale=gp_lengthscale,
            )
            if abs(pipe_dy) <= 0.55 and abs(pipe_dx) <= 0.55:
                per_pipe_dys.append(pipe_dy)
                per_pipe_dxs.append(pipe_dx)

        estimates_dy = []
        estimates_dx = []
        if abs(avg_gp_dy) <= 0.55 and abs(avg_gp_dx) <= 0.55:
            estimates_dy.append(avg_gp_dy)
            estimates_dx.append(avg_gp_dx)
        if per_pipe_dys:
            estimates_dy.append(float(np.median(per_pipe_dys)))
            estimates_dx.append(float(np.median(per_pipe_dxs)))
        if not estimates_dy:
            if abs(quad_dy) <= 0.55 and abs(quad_dx) <= 0.55:
                return np.array([center_dy + quad_dy, center_dx + quad_dx])
            return np.array([float(center_dy), float(center_dx)])

        return np.array(
            [
                center_dy + float(np.mean(estimates_dy)),
                center_dx + float(np.mean(estimates_dx)),
            ]
        )

    def register_images(
        self,
        image_t: np.ndarray,
        image_r: np.ndarray,
        psize_t: float,
        psize_r: float,
        registration_method: Optional[Literal["phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"]] = None,
        registration_algorithm_kwargs: Optional[dict[str, Any]] = None,
    ) -> np.ndarray | Tuple[np.ndarray, float] | str:
        """
        Register the target image with the reference image.
        
        Parameters
        ----------
        image_t : np.ndarray
            The target image.
        image_r : np.ndarray
            The reference image.
        psize_t : float
            The pixel size of the target image.
        psize_r : float
            The pixel size of the reference image.
        registration_method : Optional[Literal["phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"]], optional
            Overrides the default registration method for this call. 
        registration_algorithm_kwargs : Optional[dict[str, Any]], optional
            Optional keyword arguments forwarded to the selected registration
            algorithm. Supported keys and defaults depend on `registration_method`:

            - `registration_method="phase_correlation"`:
              - `filtering_method` ("hanning", "gaussian", or None; default: "hanning")
              - `upsample_factor` (int, default: `1`)

            - `registration_method="mutual_information"`:
              - `pyramid_levels` (tuple[int, ...], default: `(4, 2, 1)`)
              - `bins` (int, default: `64`)
              - `sample_frac` (float, default: `0.2`)
              - `smooth_sigmas` (Optional[dict[int, float]], default: `None`)
              - `optimizer` (Literal["powell", "nelder-mead"], default: `"nelder-mead"`)
              - `max_iter` (int, default: `60`)
              - `tol` (float, default: `1e-4`)

            - `registration_method="error_minimization"`:
              - `y_valid_fraction` (float, default: `0.8`)
              - `x_valid_fraction` (float, default: `0.8`)
              - `subpixel` (bool, default: `True`)

            - `registration_method="ncc"`:
              - `max_shift` (int, default: `7`)
              - `surface_radius` (int, default: `3`)
              - `gp_lengthscale` (float, default: `1.2`)
              - `ncc_power` (float, default: `48.0`)
              - `configs` (list[tuple[str, float]], optional)

            - `registration_method="sift"`:
              - No algorithm kwargs are currently supported; pass `None` or `{}`.

        Returns
        -------
        np.ndarray | str
            The translation offset (dy, dx) to apply to the target image so it aligns
            with the reference image. Positive y means shifting the test image
            downward; positive x means shifting the target image rightward. Returned
            values are in reference-image pixels, with registration zoom undone.
        """
        method = registration_method or self.registration_method
        algorithm_kwargs = dict(registration_algorithm_kwargs or {})
        image_t = self.process_image(np.array(image_t, copy=True))
        image_r = self.process_image(np.array(image_r, copy=True))

        # Handle pixel size and image size differences
        if psize_t != psize_r:
            # Resize the target image to have the same pixel size as the reference image
            image_t = ndi.zoom(image_t, psize_t / psize_r)

        image_t = self.zoom_image(image_t)
        image_r = self.zoom_image(image_r)

        if method in {"phase_correlation", "mutual_information", "error_minimization", "ncc"}:
            image_t = self.reconcile_image_shape(image_t, image_r.shape)

        if method == "phase_correlation":
            phase_kwargs = {"filtering_method": "hanning"}
            phase_kwargs.update(algorithm_kwargs)
            offset = phase_cross_correlation(
                image_t,
                image_r,
                **phase_kwargs,
            )
        elif method == "mutual_information":
            mi_kwargs = {
                "pyramid_levels": (4, 2, 1),
                "bins": 64,
                "sample_frac": 0.2,
                "optimizer": "nelder-mead",
                "max_iter": 60,
                "tol": 1e-4,
            }
            mi_kwargs.update(algorithm_kwargs)
            offset = translation_nmi_registration(
                moving=image_t,
                ref=image_r,
                **mi_kwargs,
            )
        elif method == "error_minimization":
            em_kwargs = {"y_valid_fraction": 0.8, "x_valid_fraction": 0.8, "subpixel": True}
            em_kwargs.update(algorithm_kwargs)
            offset = error_minimization_registration(image_t, image_r, **em_kwargs)
        elif method == "ncc":
            ncc_kwargs = {
                "max_shift": 7,
                "surface_radius": 3,
                "gp_lengthscale": 1.2,
                "ncc_power": 48.0,
            }
            ncc_kwargs.update(algorithm_kwargs)
            offset = self.ncc_registration(image_t, image_r, **ncc_kwargs)
        elif method == "sift":
            if len(algorithm_kwargs) > 0:
                raise ValueError(
                    "`registration_algorithm_kwargs` is not supported for "
                    "registration_method='sift'."
                )
            offset = self.feature_based_registration(image_t, image_r)
        else:
            raise ValueError(f"Invalid registration method: {method}")
        return np.array(offset, dtype=float) / self.zoom

    def reconcile_image_shape(
        self,
        image_t: np.ndarray,
        reference_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Crop or pad target image to match reference shape."""
        if image_t.shape == reference_shape:
            return image_t

        output = image_t
        for i in range(2):
            if self.image_coordinates_origin == "top_left":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    pad_len[i] = (0, reference_shape[i] - output.shape[i])
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    slicer[i] = slice(0, reference_shape[i])
                    output = output[tuple(slicer)]
            elif self.image_coordinates_origin == "center":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    delta = reference_shape[i] - output.shape[i]
                    pad_len[i] = (delta // 2, delta - delta // 2)
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    delta = output.shape[i] - reference_shape[i]
                    slicer[i] = slice(delta // 2, delta // 2 + reference_shape[i])
                    output = output[tuple(slicer)]
            else:
                raise ValueError(
                    f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
                )
        return output

    def prepare_image_for_feature_matching(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return image

    def adjust_points_for_origin(
        self,
        points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        if self.image_coordinates_origin == "center":
            center = np.array([(image_shape[1] - 1) / 2, (image_shape[0] - 1) / 2])
            return points - center
        if self.image_coordinates_origin == "top_left":
            return points
        raise ValueError(
            f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
        )

    def feature_based_registration(
        self,
        image_t: np.ndarray,
        image_r: np.ndarray,
    ) -> np.ndarray:
        image_t_float = self.prepare_image_for_feature_matching(image_t)
        image_r_float = self.prepare_image_for_feature_matching(image_r)
        sift_t = feature.SIFT()
        sift_r = feature.SIFT()
        sift_t.detect_and_extract(image_t_float)
        sift_r.detect_and_extract(image_r_float)

        descriptors_t = sift_t.descriptors
        descriptors_r = sift_r.descriptors
        keypoints_t = sift_t.keypoints
        keypoints_r = sift_r.keypoints

        if (
            descriptors_t is None
            or descriptors_r is None
            or descriptors_t.size == 0
            or descriptors_r.size == 0
        ):
            raise RuntimeError("SIFT feature detection failed to find descriptors.")

        matches = feature.match_descriptors(
            descriptors_t,
            descriptors_r,
            metric="euclidean",
            cross_check=True,
            max_ratio=0.75,
        )

        if matches.shape[0] < 3:
            raise RuntimeError("Not enough SIFT matches to estimate translation.")

        pts_t = keypoints_t[matches[:, 0]][:, ::-1]
        pts_r = keypoints_r[matches[:, 1]][:, ::-1]
        pts_t = self.adjust_points_for_origin(pts_t, image_t.shape)
        pts_r = self.adjust_points_for_origin(pts_r, image_r.shape)
        deltas = pts_r - pts_t
        offset_x = np.median(deltas[:, 0])
        offset_y = np.median(deltas[:, 1])
        return np.array([offset_y, offset_x])
