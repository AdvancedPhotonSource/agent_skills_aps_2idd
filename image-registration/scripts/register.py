"""Register two NumPy images and print the translation offset as JSON."""

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms import ImageRegistration


METHODS = (
    "ncc", "phase_correlation", "sift", "mutual_information", "error_minimization",
)


def positive_float(value: str) -> float:
    """Parse a finite positive scale for argparse."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def load_image(path: Path, log_scale: bool) -> np.ndarray:
    """Load a finite real image from a non-pickled .npy file.

    Parameters
    ----------
    path : Path
        Input image, either (height, width) or (height, width, channels).
    log_scale : bool
        Whether registration will apply log10(image + 1).

    Returns
    -------
    np.ndarray
        Validated input, preserving the stored dtype.
    """
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{path}: expected a .npy image")
    image = np.load(path, allow_pickle=False)
    if image.ndim not in (2, 3) or min(image.shape) < 1:
        raise ValueError(f"{path}: expected a nonempty 2D or channel-last 3D image")
    if image.dtype.kind not in "buif" or not np.isfinite(image).all():
        raise ValueError(f"{path}: image must contain finite real numeric values")
    if min(image.shape[:2]) < 3:
        raise ValueError(f"{path}: height and width must each be at least 3")
    if log_scale and np.min(image) <= -1:
        raise ValueError(f"{path}: --log-scale requires values greater than -1")
    return image


def main() -> None:
    """Run translation registration using command-line settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="Test/moving .npy image")
    parser.add_argument("--reference", type=Path, required=True, help="Reference .npy image")
    parser.add_argument("--method", choices=METHODS, default="ncc")
    parser.add_argument("--test-pixel-size", type=positive_float, default=1.0)
    parser.add_argument("--reference-pixel-size", type=positive_float, default=1.0)
    parser.add_argument("--origin", choices=("top_left", "center"), default="top_left")
    parser.add_argument("--zoom", type=positive_float, default=1.0)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument(
        "--algorithm-kwargs", default="{}", metavar="JSON",
        help="JSON object of method parameters; see SKILL.md for keys and defaults",
    )
    args = parser.parse_args()
    try:
        kwargs = json.loads(args.algorithm_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--algorithm-kwargs must be a JSON object")
        if args.method == "mutual_information" and kwargs.get("smooth_sigmas") is not None:
            sigmas = kwargs["smooth_sigmas"]
            if not isinstance(sigmas, dict):
                raise ValueError("smooth_sigmas must be a JSON object or null")
            kwargs["smooth_sigmas"] = {int(key): value for key, value in sigmas.items()}
        test = load_image(args.test, args.log_scale)
        reference = load_image(args.reference, args.log_scale)
        registration = ImageRegistration(
            image_coordinates_origin=args.origin,
            registration_method=args.method,
            zoom=args.zoom,
            log_scale=args.log_scale,
        )
        offset = registration.register_images(
            test, reference,
            psize_t=args.test_pixel_size,
            psize_r=args.reference_pixel_size,
            registration_algorithm_kwargs=kwargs,
        )
        print(json.dumps(offset.tolist(), allow_nan=False))
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(2, f"registration failed: {error}\n")


if __name__ == "__main__":
    main()
