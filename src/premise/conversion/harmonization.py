from __future__ import annotations


def summarize_harmonization_scope() -> dict:
    """Describe the main harmonization responsibilities of the conversion layer.

    This helper is intentionally lightweight and documentation-oriented. It can
    be used by examples, CLIs, or documentation builders to present the scope
    of the conversion module.
    """
    return {
        "formats": ["binary raster", "GeoTIFF", "GRIB", "HDF", "NetCDF"],
        "standardization": [
            "variable-name mapping",
            "unit normalization",
            "time decoding and alignment",
            "coordinate normalization",
            "NetCDF export",
        ],
    }
