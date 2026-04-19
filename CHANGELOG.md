# Changelog

## 1.0.0 - 2026-04-13

### Added
- Formalized software metadata for a release-oriented repository layout.
- Added machine-readable citation metadata and an MIT license.
- Added documentation notes for software availability and benchmark reporting.
- Added minimal quickstart examples and pytest-based regression tests.
- Added compatibility wrappers for `premise.binaryio`, `premise.geotiff`, `premise.grib`, and `premise.hdf`.

### Changed
- Updated package metadata from a placeholder development state to a release-style `1.0.0` record.
- Simplified the top-level import behavior to avoid hard failures from optional dependencies.
- Clarified the repository scope around harmonization, index computation, and comparative evaluation.
- Marked correction and fusion modules as optional or experimental extensions in documentation.

### Fixed
- Resolved reader import inconsistency by restoring a dedicated `premise.reader.grib` entry point.
- Added a missing NSE metric implementation and standardized the public metric exports.
- Removed duplicate imports in workflow helpers.
