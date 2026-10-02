"""Exercise registration through the CLI using temporary NumPy images."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi
from skimage import data


CLI = Path(__file__).with_name("register.py")


class RegistrationTests(unittest.TestCase):
    """Check signs, units, preprocessing, algorithm parameters, and CLI errors."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = data.camera()[180:244, 180:244].astype(float)
        self.test = np.roll(self.reference, (3, -4), axis=(0, 1))

    def run_cli(
        self,
        test: np.ndarray,
        reference: np.ndarray,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        """Run the installed interpreter on actual .npy files from another cwd."""
        np.save(self.root / "test.npy", test)
        np.save(self.root / "reference.npy", reference)
        return subprocess.run(
            [sys.executable, str(CLI), "--test", str(self.root / "test.npy"),
             "--reference", str(self.root / "reference.npy"), *args],
            cwd=self.root, capture_output=True, text=True, check=False,
        )

    def assert_offset(
        self,
        result: subprocess.CompletedProcess[str],
        expected: tuple[float, float],
        tolerance: float = 0.3,
    ) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)
        offset = json.loads(result.stdout)
        self.assertEqual(len(offset), 2)
        np.testing.assert_allclose(offset, expected, atol=tolerance, rtol=0)

    def test_algorithms(self) -> None:
        """Every algorithm recovers a known shift with the same sign and ordering."""
        for method in ("ncc", "phase_correlation", "sift", "mutual_information", "error_minimization"):
            with self.subTest(method=method):
                reference, test = self.reference, self.test
                kwargs = {}
                if method == "sift":
                    reference = data.camera().astype(float)
                    test = np.roll(reference, (3, -4), axis=(0, 1))
                elif method == "mutual_information":
                    kwargs = {"pyramid_levels": [2, 1], "max_iter": 200,
                              "sample_frac": 1.0, "bins": 32,
                              "smooth_sigmas": {"2": 1.0, "1": 0.0}}
                result = self.run_cli(test, reference, "--method", method,
                                      "--algorithm-kwargs", json.dumps(kwargs))
                self.assert_offset(result, (-3, 4))

    def test_small_noisy_different_sharpness(self) -> None:
        rng = np.random.default_rng(7)
        reference = self.reference[8:40, 8:40]
        test = ndi.gaussian_filter(np.roll(reference, (2, -2), axis=(0, 1)), 0.7)
        test += rng.normal(0, 1.0, test.shape)
        self.assert_offset(self.run_cli(test, reference), (-2, 2), 0.4)

    def test_integer_error_minimization(self) -> None:
        self.assert_offset(self.run_cli(
            self.test, self.reference, "--method", "error_minimization",
            "--algorithm-kwargs", '{"subpixel": false, "y_valid_fraction": 0.5, "x_valid_fraction": 0.5}',
        ), (-3, 4), 0)

    def test_phase_windows(self) -> None:
        for window in ("hanning", "gaussian", None):
            with self.subTest(window=window):
                self.assert_offset(self.run_cli(
                    self.test, self.reference, "--method", "phase_correlation",
                    "--algorithm-kwargs", json.dumps({"filtering_method": window}),
                ), (-3, 4), 0)

    def test_subpixel(self) -> None:
        reference = ndi.gaussian_filter(self.reference, 1.0)
        test = ndi.shift(reference, (0.4, -0.7), mode="nearest")
        self.assert_offset(self.run_cli(
            test, reference, "--method", "phase_correlation",
            "--algorithm-kwargs", '{"upsample_factor": 100}',
        ), (-0.4, 0.7), 0.2)

    def test_pixel_size_and_zoom(self) -> None:
        """Pixel sizes resample inputs; output retains reference-pixel units."""
        test = np.roll(self.reference, (4, -4), axis=(0, 1))
        self.assert_offset(self.run_cli(
            test, self.reference, "--method", "phase_correlation", "--zoom", "0.5",
            "--test-pixel-size", "2.5", "--reference-pixel-size", "2.5",
        ), (-4, 4), 0)
        test = self.reference[::2, ::2]
        reference = np.roll(ndi.zoom(test, 2), (3, -4), axis=(0, 1))
        self.assert_offset(self.run_cli(
            test, reference, "--method", "phase_correlation",
            "--test-pixel-size", "2", "--reference-pixel-size", "1",
        ), (3, -4), 0)

    def test_origins(self) -> None:
        for origin, padding in (("top_left", ((0, 8), (0, 10))),
                                ("center", ((4, 4), (5, 5)))):
            with self.subTest(origin=origin):
                reference = np.pad(self.reference, padding)
                self.assert_offset(self.run_cli(
                    self.reference, reference, "--origin", origin,
                    "--method", "phase_correlation",
                ), (0, 0), 0)
                self.assert_offset(self.run_cli(
                    reference, self.reference, "--origin", origin,
                    "--method", "phase_correlation",
                ), (0, 0), 0)

    def test_channels_and_log(self) -> None:
        self.assert_offset(self.run_cli(
            np.stack([self.test] * 3, axis=-1),
            np.stack([self.reference] * 3, axis=-1),
            "--method", "phase_correlation", "--log-scale",
        ), (-3, 4), 0)

    def test_invalid_input(self) -> None:
        for image, args in (
            (np.ones(8), []),
            (np.full((8, 8), np.nan), []),
            (np.full((8, 8), -2), ["--log-scale"]),
            (self.test, ["--zoom", "0"]),
            (self.test, ["--algorithm-kwargs", "[]"]),
            (self.test, ["--algorithm-kwargs", '{"misspelled_parameter": 1}']),
            (self.test, ["--method", "sift", "--algorithm-kwargs", '{"max_shift": 3}']),
        ):
            with self.subTest(args=args, shape=image.shape):
                result = self.run_cli(image, self.reference, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
