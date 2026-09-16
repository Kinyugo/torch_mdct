# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-09-16

### Fixed
- `kaiser_bessel_derived` now builds its base Kaiser window with `periodic=False`.
  The previous asymmetric base window violated the Princen-Bradley condition, which
  broke MDCT/IMDCT perfect reconstruction (round-trip error up to ~0.3 for a
  unit-scale signal at `win_length=64`). Fixes #3, by @tombackstrom in #4.

### Added
- Test suite under `tests/` covering the Princen-Bradley condition and MDCT/IMDCT
  round-trip reconstruction for both windows (`pip install -e .[test]`, then `pytest`).
- GitHub Actions CI running the tests on Python 3.9 through 3.13 and checking that
  the sdist and wheel build cleanly.
- `torch_mdct.__version__`.

### Changed
- The package version is now derived from git tags via `hatch-vcs` instead of
  being hard-coded in `pyproject.toml`. Releases are cut by publishing a GitHub
  release tagged `vX.Y.Z`; the publish workflow runs the tests, verifies the built
  version matches the tag, and uploads to PyPI.
- Minimum supported Python is now 3.9 (current `torch` wheels no longer support 3.8).

## [0.4.1] - 2024-12-17

Last release with a hard-coded version. See the GitHub releases page for earlier history.

[Unreleased]: https://github.com/Kinyugo/torch_mdct/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Kinyugo/torch_mdct/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Kinyugo/torch_mdct/releases/tag/v0.4.1
