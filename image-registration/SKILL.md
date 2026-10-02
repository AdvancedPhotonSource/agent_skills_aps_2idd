---
name: image-registration
description: Estimate the translational offset between test and reference NumPy images using a standalone CLI with NCC, phase correlation, SIFT, mutual information, or error minimization. Use for numerical registration of .npy images, including noisy, small, or differently blurred images.
---

# Image registration

Run the bundled CLI to estimate the translation to apply to a test/moving image
so it aligns with a reference image. **NCC is the best choice for images that are
noisy, small in size, or have different sharpness.** It is the default method.
These algorithms estimate translation only; they do not fit rotation or distortion.

## Install, configure, and test

Keep this entire directory together when copying or installing the skill.
`scripts/algorithms.py` bundles the registration routines and image-processing
helpers. Runtime dependencies are NumPy, SciPy, and scikit-image.
`pyproject.toml`, `.python-version`, and `uv.lock` define the environment, with
Python managed by `uv`.

On Linux/macOS, first check `command -v uv`. If unavailable, install the standalone
binary using the [uv installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Set `SKILL_DIR` to the absolute directory containing this `SKILL.md`, then install
and test. The following location is an example; adjust it after moving the skill.
[uv manages Python itself](https://docs.astral.sh/uv/guides/install-python/).

```bash
SKILL_DIR=/absolute/path/to/image-registration
uv python install 3.11
uv sync --project "$SKILL_DIR" --locked --all-extras --managed-python
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" --help
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/test_registration.py"
```

The tests generate temporary `.npy` files, exercise all five methods, and check
offset signs, subpixel shifts, noise/blur, pixel sizes, zoom, image origins,
channel averaging, log scaling, and invalid inputs. Success ends with `OK`.

Initial setup requires network access to download Python and locked packages.
The default environment lives in `$SKILL_DIR/.venv`. If `UV_PROJECT_ENVIRONMENT`
is already set for another project, unset it first to use this local environment.
For a read-only skill directory, set it to an absolute writable environment path
and use the same value for setup and subsequent runs. `UV_CACHE_DIR` can likewise
select a writable cache. Use `uv run --project ... --locked` for every invocation;
Python and dependencies are resolved from the skill environment. After setup,
`uv run --offline --project ... --locked` works without network access.

## Inputs and result

Both inputs must be real, finite numeric `.npy` arrays, either `(height, width)` or
`(height, width, channels)`. Channel-last arrays are averaged across channels.
Height and width must each be at least 3. Supply a plane explicitly for a volume
or time series. Object arrays, NaNs, and infinities are rejected.

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test /absolute/path/test.npy --reference /absolute/path/reference.npy \
  --method ncc --algorithm-kwargs '{"max_shift": 12}'
```

Stdout contains exactly one JSON array, `[dy, dx]`. Positive `dy` shifts the test
image down; positive `dx` shifts it right. A test image displaced 3 pixels down
and 4 pixels left should return approximately `[-3, 4]`. Errors go to stderr and
exit with a nonzero status.

**Offsets are in reference-image pixels, with `zoom` undone.** For physical
displacement, multiply the output by `--reference-pixel-size`; for displacement
in original test-image pixels, multiply by
`reference_pixel_size / test_pixel_size`.

Global settings:

| Option | Default | Effect |
| --- | --- | --- |
| `--method` | `ncc` | Select one of the five methods below. |
| `--test-pixel-size` | `1.0` | Isotropic test pixel size, in the same units as the reference pixel size. |
| `--reference-pixel-size` | `1.0` | Isotropic reference pixel size; the test image is resampled by their ratio. |
| `--origin` | `top_left` | `top_left` crops/pads the bottom and right; `center` crops/pads centrally. SIFT instead adjusts feature coordinates to the chosen origin. |
| `--zoom` | `1.0` | Positive resampling factor for both images before registration; output is divided by this factor. |
| `--log-scale` | off | Apply `log10(image + 1)` before resampling; input values must exceed `-1`. |
| `--algorithm-kwargs` | `{}` | JSON object with the selected method's keyword parameters. Unknown keys fail. |

After pixel-size resampling and zoom, all methods except SIFT crop/pad the test
image to the reference shape. Search distances and smoothing sigmas below refer
to this processed grid. Large downsampling factors can remove useful structure;
ensure processed images still contain enough pixels and overlap.

## Algorithms

### NCC (`ncc`)

Use first for noisy, small, or differently sharp images. It computes normalized
cross-correlation on overlapping pixels across several intensity transforms and
Gaussian smoothing levels. The pipelines vote on the integer peak; weighted
surfaces and Gaussian-process refinement estimate the subpixel shift, with a
quadratic fallback.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_shift` | `7` | Nonnegative integer search radius per axis, capped by image size to preserve at least three rows/columns of overlap. Increase for larger translations. |
| `surface_radius` | `3` | Integer neighborhood radius for subpixel surface fitting. |
| `gp_lengthscale` | `1.2` | Positive RBF-kernel length scale for subpixel refinement. |
| `ncc_power` | `48.0` | Exponent controlling preference for pipelines with higher peak NCC. |
| `configs` | see below | Nonempty list of `[transform, sigma]` pairs; sigma is a nonnegative Gaussian smoothing width. |

Default pipelines: `[["log1p",0.3],["log1p",0.5],["log1p",0.7],["sqrt",0.3],
["sqrt",0.5],["rank",0.3],["rank",0.5],["power",0.5],["asinh",0.5]]`.
The `power` transform uses exponent 0.3. `log1p`, `sqrt`, and `power` clip negative
intensities to zero; `rank` and `asinh` can be useful for signed data.

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test test.npy --reference reference.npy --method ncc \
  --algorithm-kwargs '{"max_shift": 15, "surface_radius": 3, "gp_lengthscale": 1.2, "ncc_power": 48, "configs": [["rank", 0.3], ["asinh", 0.5]]}'
```

Runtime grows with image area, search area, and pipeline count. Set the search
radius above the expected displacement. A result at the search boundary warrants
checking overlap and increasing the radius if appropriate.

### Phase correlation (`phase_correlation`)

Use for fast Fourier-domain translation estimation on images with similar
structure and sharpness. Mean subtraction and windowing suppress edge artifacts.
Periodic ambiguity or very weak signal can give incorrect peaks.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `filtering_method` | `"hanning"` | `"hanning"`, `"gaussian"` (soft rectangle), or JSON `null` (no window). |
| `upsample_factor` | `1` | Positive integer; nominal subpixel grid spacing is `1 / upsample_factor` before undoing zoom. |

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test test.npy --reference reference.npy --method phase_correlation \
  --algorithm-kwargs '{"filtering_method": "gaussian", "upsample_factor": 20}'
```

### SIFT (`sift`)

Use when images have numerous distinctive local features, potentially with
different fields of view. Images are normalized independently; SIFT descriptors
are matched using a 0.75 ratio test and cross-checking. The returned translation
is the median displacement of matched keypoints. At least three matches are
required. Small, smooth, or heavily blurred images may yield too few features.
Although SIFT descriptors tolerate scale/rotation differences, this registration
routine fits only translation.

There are no algorithm parameters; pass `{}` or omit `--algorithm-kwargs`.

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test test.npy --reference reference.npy --method sift --origin center
```

### Mutual information (`mutual_information`)

Use when corresponding structures have different intensity mappings or contrast.
It maximizes normalized mutual information from coarse to fine image scales,
starting at zero translation. It is slower and can settle at a local optimum;
test sufficient sampling and iteration budgets on representative images.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pyramid_levels` | `[4,2,1]` | Positive integer downsampling factors, usually coarse to fine and ending in 1. |
| `bins` | `64` | Histogram bins for normalized mutual information. |
| `sample_frac` | `0.2` | Fraction of overlapping pixels sampled, in `(0,1]`; sampling uses a fixed seed. |
| `smooth_sigmas` | `null` | Defaults to `{"4":1.5,"2":1.0,"1":0.0}`; custom JSON object maps levels to smoothing sigmas in level pixels, with omitted levels unsmoothed. |
| `optimizer` | `"nelder-mead"` | `"nelder-mead"` or `"powell"`. |
| `max_iter` | `60` | Maximum optimization iterations per level. |
| `tol` | `0.0001` | Optimizer convergence tolerance. |

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test test.npy --reference reference.npy --method mutual_information \
  --algorithm-kwargs '{"pyramid_levels": [4,2,1], "bins": 32, "sample_frac": 1.0, "smooth_sigmas": {"4": 1.5, "2": 1.0, "1": 0.0}, "optimizer": "powell", "max_iter": 200, "tol": 0.0001}'
```

The optimizer retains the incoming shift at a pyramid level if
optimization does not report success. A returned zero is therefore not evidence
that alignment has been confirmed.

### Error minimization (`error_minimization`)

Use for images with comparable intensities when a bounded exhaustive MSE search
is suitable. A central reference window stays fixed while candidate test windows
are compared, avoiding wrap-around. A local quadratic fit refines the integer
minimum when possible. Brightness or contrast changes can bias the MSE.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `y_valid_fraction` | `0.8` | Fraction of image height used by the central comparison window, in `(0,1]`. |
| `x_valid_fraction` | `0.8` | Fraction of image width used by the window, in `(0,1]`. |
| `subpixel` | `true` | Enable local quadratic refinement; `false` returns the integer minimum. |

For processed height `H`, the integer y search radius is
`(H - round(y_valid_fraction * H)) // 2`, with the analogous expression for x.
Smaller fractions enlarge the search while retaining less comparison area. Keep
both rounded window dimensions nonzero. A fraction of 1 leaves no search margin
on that axis; both fractions at 1 return `[0,0]`.

```bash
uv run --project "$SKILL_DIR" --locked python "$SKILL_DIR/scripts/register.py" \
  --test test.npy --reference reference.npy --method error_minimization \
  --algorithm-kwargs '{"y_valid_fraction": 0.6, "x_valid_fraction": 0.8, "subpixel": true}'
```

## Interpreting results

Report the method, significant settings, and `[dy, dx]` with its units. The CLI
returns an estimate without a confidence score. Featureless images, repeated
patterns, insufficient overlap, or a violated translation-only model can produce
misleading offsets even when execution succeeds. For uncertain results, compare
the shifted test image with the reference or cross-check another suitable method.
