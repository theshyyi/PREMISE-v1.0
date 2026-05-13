# CATALOG_GUIDE.md

This document explains how to define and maintain `catalog.py` in PREMISЕ. It is intended for developers who add new datasets, revise existing source definitions, or troubleshoot why a source cannot be downloaded correctly.

---

## 1. Purpose of `catalog.py`

`catalog.py` is the central registry of data-source metadata and download instructions. It does **not** perform downloading by itself. Instead, it tells the acquisition layer:

- what a dataset is,
- which downloader method should be used,
- what parameters that downloader needs,
- whether authentication is required,
- what file formats are expected,
- and how the source should be searched and identified.

A good catalog entry should be:

- **machine-usable** for download automation,
- **human-readable** for maintenance,
- **specific enough** to avoid ambiguous file matching,
- and **stable enough** to survive small upstream website changes.

---

## 2. Core structure

A source is defined as a `DataSource` object.

Typical fields:

```python
DataSource(
    key="chirps_daily_tif",
    title="CHIRPS daily GeoTIFF",
    variables=("precipitation",),
    temporal_resolution="daily",
    spatial_resolution="0.05 degree",
    provider="Climate Hazards Center",
    method="http_direct",
    params={...},
    aliases=("chirps",),
    format_hints=("GeoTIFF",),
    notes="CHIRPS v3 daily final satellite GeoTIFF files.",
    official_url="https://data.chc.ucsb.edu/products/CHIRPS/v3.0/",
    access_mode="direct_http",
    auth_required=False,
    product_notes="Downloader targets daily/final/sat/YYYY/chirps-v3.0.sat.YYYY.MM.DD.tif.",
)
```

---

## 3. Field-by-field explanation

### `key`

A unique machine-readable identifier.

Guidelines:

- lowercase only,
- use underscores,
- keep stable once published,
- do not reuse a key for a different dataset,
- include version/resolution/access mode when needed.

Good examples:

- `chirps_daily_tif`
- `gpcp_daily_cdr_v1_3_nc`
- `cmorph_hourly_025deg_http`

Avoid:

- `dataset1`
- `CHIRPS`
- `cmfd v2`

If a dataset has two different download routes, prefer separate keys instead of overloading one key:

- `cmorph_daily_025deg_ftp`
- `cmorph_daily_025deg_http`

---

### `title`

Human-readable display name.

This should be clear and descriptive, but it does not need to be used programmatically.

---

### `variables`

Tuple of canonical variable names represented by the source.

Use consistent names across the project. For example:

- `precipitation`
- `temperature`
- `air_temperature`
- `soil_moisture`
- `runoff`
- `potential_evaporation`

Recommendations:

- prefer semantic names over dataset-native abbreviations,
- include both broad and specific terms when useful,
- do not add too many vague aliases here.

Example:

```python
variables=("precipitation", "prate")
```

Use this only when both are genuinely helpful in search and interpretation.

---

### `temporal_resolution`

Free-text description such as:

- `daily`
- `monthly`
- `hourly`
- `3-hourly`
- `half-hourly to daily`
- `model-dependent`

Keep wording consistent across similar products.

---

### `spatial_resolution`

Free-text description such as:

- `0.1 degree`
- `0.25 degree`
- `0.5 x 0.625 degree`
- `8 km`
- `model-dependent`

---

### `provider`

The organization or platform serving the data.

Examples:

- `NOAA PSL`
- `NOAA NCEI`
- `NASA GES DISC / Earthdata`
- `Copernicus Climate Data Store`
- `CHRS / UCI`

---

### `method`

This is the most important field. It selects which downloader implementation should be used.

Typical supported values in PREMISЕ:

- `http_direct`
- `http_listing`
- `erddap`
- `cds_api`
- `ftp`
- `sftp`
- `rclone`
- `portal_export`
- `esgf`
- `earthdata_cmr`
- `repository_api` (recommended for Zenodo/Figshare-type repositories)

Rule: **choose method by access mechanism, not by dataset family**.

For example:

- CHIRPS daily file template -> `http_direct`
- CHRS public directory listing -> `http_listing`
- PERSIANN-CDR NOAA subset service -> `erddap`
- ERA5 -> `cds_api`
- CMORPH CPC FTP -> `ftp`
- GLEAM registered SSH file access -> `sftp`
- MSWEP shared Google Drive via rclone -> `rclone`
- CMIP via ESGF search -> `esgf`
- MERRA2/GLDAS/FLDAS/TRMM via Earthdata CMR -> `earthdata_cmr`

---

### `params`

Downloader-specific parameters. This is where the download logic is configured.

This field must match the contract expected by the selected `method`.

Details are given in Section 4.

---

### `aliases`

Alternative search names for the same dataset.

Examples:

```python
aliases=("gpcc", "gpcc_daily")
```

Use aliases for:

- common abbreviations,
- old project-internal names,
- names users are likely to search.

Do not add excessive aliases that could create ambiguous searches.

---

### `format_hints`

Human-readable hints about the raw file format.

Examples:

- `("NetCDF",)`
- `("GeoTIFF",)`
- `("HDF5",)`
- `("Z-compressed binary",)`
- `("GRIB", "NetCDF")`

This field helps users understand what the downloader retrieves before conversion.

---

### `notes`

Concise technical description of the source.

Good note style:

- one or two sentences,
- objective,
- focused on structure and access.

---

### `official_url`

Primary official landing page or provider reference page.

This should preferably point to:

- the dataset page,
- the provider data page,
- or the root public directory if that is the official access route.

---

### `access_mode`

Human-readable label describing access style.

Examples:

- `direct_http`
- `http_listing`
- `api`
- `ftp`
- `sftp`
- `earthdata_cmr_https`
- `esgf_search_and_wget`
- `rclone_shared_drive`

This field is descriptive and useful for reporting.

---

### `auth_required`

Boolean indicating whether authentication is needed.

Use `True` for:

- CDS API,
- Earthdata,
- SFTP,
- Rclone shared access,
- ESGF token or login workflows,
- private repository records,
- authenticated FTP.

Use `False` for:

- public HTTP directory listings,
- public ERDDAP endpoints,
- public anonymous FTP,
- public NCEI/PSL file downloads.

---

### `product_notes`

Extra detail about the exact access implementation.

Use this to capture:

- file naming assumptions,
- why a particular endpoint was chosen,
- warnings about dataset discontinuation,
- or practical caveats.

---

## 4. How to write `params` by method

### 4.1 `http_direct`

Use when the file URL can be constructed directly from a stable template.

Typical keys:

- `base_url`
- `filename_pattern`
- `time_expansion`
- optional `directory_pattern`

#### Example: annual files

```python
params={
    "base_url": "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/",
    "filename_pattern": "precip.{year}.nc",
    "time_expansion": "year",
}
```

#### Example: daily files in year subdirectories

```python
params={
    "base_url": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/",
    "filename_pattern": "chirps-v3.0.sat.{year}.{month}.{day}.tif",
    "directory_pattern": "{year}/",
    "time_expansion": "date",
}
```

#### Example: one static file

```python
params={
    "base_url": "https://downloads.psl.noaa.gov/Datasets/cmap/std/",
    "filename_pattern": "precip.mon.mean.nc",
    "time_expansion": "none",
}
```

Supported `time_expansion` values should ideally include:

- `none`
- `year`
- `date`
- `decade`

Important rule: if `time_expansion="none"`, the downloader should directly request `base_url + filename_pattern`.

---

### 4.2 `http_listing`

Use when the dataset is exposed as a web directory and files must be discovered by parsing links.

Typical keys:

- `base_url`
- `directory_pattern`
- `file_regex`
- `time_expansion`
- optional `contains`
- optional `contains_template`
- optional `filename_date_regex`
- optional `file_limit_default`

#### Example: annual listing

```python
params={
    "base_url": "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/access/",
    "directory_pattern": "{year}/",
    "time_expansion": "year",
    "file_regex": r'href="([^"]+\.nc)"',
    "filename_date_regex": r"d(?P<yearmonth>\d{6})",
}
```

#### Example: multi-level listing

```python
params={
    "base_url": "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg/",
    "directory_pattern": "{year}/{month}/{day}/",
    "time_expansion": "date",
    "file_regex": r'href="([^"]+)"',
}
```

#### Example: CHRS PERSIANN family listing

```python
params={
    "base_url": "https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CDR/daily/",
    "time_expansion": "date",
    "file_regex": r'href="([^"]+\.bin\.gz)"',
    "date_token_mode": "yyddd",
}
```

When using `http_listing`, make sure you have checked:

- whether the page is a real directory index,
- whether files are in the same directory or nested by year/month/day,
- whether filenames contain date tokens,
- and whether there are readme files that clarify the naming convention.

---

### 4.3 `erddap`

Use when the provider offers a subsettable ERDDAP service.

Typical keys:

- `base_url`
- `query_variable`
- `default_bbox`
- `time_step`
- `lat_step`
- `lon_step`

Example:

```python
params={
    "base_url": "https://www.ncei.noaa.gov/erddap/griddap/cdr_persiann_by_time_lon_lat.nc",
    "query_variable": "precipitation",
    "default_bbox": (0.125, -59.875, 359.875, 59.875),
    "time_step": 1,
    "lat_step": 1,
    "lon_step": 1,
}
```

Use `erddap` only when the service is truly subset-capable.

---

### 4.4 `cds_api`

Use for Copernicus CDS datasets.

Typical keys:

- `dataset_name`
- `default_data_format`
- `default_download_format`
- `time_mode`

Example:

```python
params={
    "dataset_name": "reanalysis-era5-land",
    "default_data_format": "grib",
    "default_download_format": "unarchived",
    "time_mode": "hourly_all",
}
```

Keep these entries minimal. Variable names and date ranges are usually passed at request time, not hard-coded in catalog.

---

### 4.5 `ftp`

Use when the source is a real FTP service.

Typical keys:

- `host`
- `port` (optional; default 21)
- `remote_base_dir`
- `time_expansion`
- optional `filename_pattern`
- optional `file_regex`
- optional `contains_template`
- optional `filename_date_regex`

#### Example: rule-based daily FTP

```python
params={
    "host": "ftp.cpc.ncep.noaa.gov",
    "remote_base_dir": "/precip/global_CMORPH/daily_025deg",
    "filename_pattern": "CMORPH+MWCOMB_DAILY-025DEG_{date}.Z",
    "time_expansion": "date",
}
```

#### Example: flat directory with many sub-daily files

```python
params={
    "host": "ftp.cpc.ncep.noaa.gov",
    "remote_base_dir": "/precip/global_CMORPH/30min_025deg",
    "time_expansion": "date",
    "file_regex": r"CMORPH_025DEG-30MIN_.*\.Z$",
    "contains_template": "{date}",
    "filename_date_regex": r"(?P<date>\d{10})",
}
```

Authenticated FTP credentials should not be stored in catalog. Use environment variables instead.

---

### 4.6 `sftp`

Use for SSH-based file transfer, not plain FTP.

Typical keys:

- `default_port`
- product-specific directory maps or file naming hints

Example:

```python
params={
    "default_port": 2225,
    "variable_directory_map": {
        "potential_evaporation": "Ep",
        "evapotranspiration": "E",
    },
}
```

Again, credentials belong in environment variables, not in catalog.

---

### 4.7 `rclone`

Use when the official distribution is via a shared cloud drive accessed through rclone.

Typical keys:

- `default_subdir`
- `file_limit_default`

Example:

```python
params={
    "default_subdir": "Daily",
    "file_limit_default": None,
}
```

---

### 4.8 `portal_export`

Use only when the site requires a user to generate an export task and then download from an export URL.

If the source is actually a public directory listing, prefer `http_listing`.

Typical keys:

- often empty in catalog,
- actual export URL supplied at runtime.

This method should be reserved for true portal-based export workflows.

---

### 4.9 `esgf`

Use for CMIP and related data distributed through ESGF search and wget mechanisms.

Typical keys:

- `project`
- `default_search_node`
- `default_wget_node`
- `default_type`
- `latest`
- `replica`
- `distrib`

Example:

```python
params={
    "project": "CMIP6",
    "default_search_node": "https://esgf-node.llnl.gov/esg-search/search",
    "default_wget_node": "https://esgf-node.llnl.gov/esg-search/wget",
    "default_type": "File",
    "latest": "true",
    "replica": "false",
    "distrib": "true",
}
```

---

### 4.10 `earthdata_cmr`

Use for NASA datasets discovered through CMR and downloaded through Earthdata-authenticated HTTPS.

Typical keys:

- `short_name`
- `version`
- `provider`
- `preferred_link_substrings`
- optional `filename_regex`

Example:

```python
params={
    "short_name": "GLDAS_NOAH025_3H",
    "version": "2.1",
    "provider": "GES_DISC",
    "preferred_link_substrings": ["gesdisc", "disc"],
    "filename_regex": r"\.(nc4|nc)$",
}
```

Use `product_notes` to describe important caveats such as discontinued product coverage.

---

### 4.11 `repository_api`

Recommended for Zenodo, Figshare, Dataverse-like repository records.

Typical keys:

- `provider`
- `identifier`
- `identifier_type`
- optional `include`
- optional `exclude`

Example:

```python
params={
    "provider": "zenodo",
    "identifier": "10.5281/zenodo.1234567",
    "identifier_type": "doi",
    "include": [r".*2020.*\.nc$"],
}
```

---

## 5. Choosing the correct method

Use this decision logic.

If the file URL is predictable from a template, use `http_direct`.

If you must parse a public directory page, use `http_listing`.

If the site offers a subset API with time and space query parameters, use `erddap` or another API-specific method.

If the source is a real FTP server, use `ftp`.

If it is SSH file access, use `sftp`.

If it requires the user to generate an export job in a portal, use `portal_export`.

If it is Earthdata CMR + authenticated HTTPS, use `earthdata_cmr`.

If it is ESGF search and wget script generation, use `esgf`.

If it is a repository record like Zenodo or Figshare, use `repository_api`.

Do not choose method by dataset branding. Choose method by access pattern.

---

## 6. Common catalog mistakes

### Mistake 1: using the landing page instead of the download directory

Bad:

```python
"base_url": "https://psl.noaa.gov/data/gridded/data.cmap.html"
```

Good:

```python
"base_url": "https://downloads.psl.noaa.gov/Datasets/cmap/std/"
```

---

### Mistake 2: using `portal_export` for a public directory listing

If the website openly lists files or subdirectories, it should usually be `http_listing`, not `portal_export`.

This was the case for the CHRS PERSIANN family directory.

---

### Mistake 3: putting credentials in catalog

Never put:

- usernames,
- passwords,
- tokens,
- API secrets

inside `catalog.py`.

Use environment variables.

---

### Mistake 4: duplicate keys

Every `key` must be unique. A duplicate key causes confusion in reporting, download routing, and testing.

---

### Mistake 5: vague aliases

Avoid aliases that are too generic, such as:

- `daily`
- `satellite`
- `climate`

Use only meaningful aliases.

---

### Mistake 6: not checking filename rules

Before writing `filename_pattern`, always inspect the provider directory or readme. Many datasets use date tokens like:

- `YYYY`
- `YYYYMM`
- `YYYYMMDD`
- `YYDDD`
- `YYMMDD`

If you guess the rule, the source may silently return no files.

---

## 7. Recommended maintenance workflow for new sources

When adding a new source:

1. Determine the real access mechanism.
2. Decide the correct `method`.
3. Check directory structure or API docs.
4. Confirm file naming convention.
5. Add a precise catalog entry.
6. Add or update a downloader if needed.
7. Add a small source-specific test.
8. Run unified acquisition tests.
9. Write any auth/setup notes into `USE.md` and `AUTH_GUIDE.md`.

---

## 8. Example patterns for common source families

### NOAA PSL static files

Use `http_direct` + `time_expansion="none"`.

### NOAA NCEI date-organized public directories

Use `http_listing` with year/month/day directory patterns.

### CHRS PERSIANN family public directories

Use `http_listing`, not `portal_export`.

### NASA GES DISC gridded archives

Use `earthdata_cmr`.

### CMIP via ESGF

Use `esgf`.

### TPDC or institutional authenticated file servers

Use `ftp` or `sftp` depending on the actual protocol.

---

## 9. Suggested future enhancements

You may want to extend the catalog system to support:

- `status` fields such as `implemented`, `partial`, `deprecated`, `auth_required`,
- source family tags such as `gauge`, `satellite`, `reanalysis`, `repository`,
- dataset version metadata,
- preferred raw variable names,
- conversion hints for known file formats,
- and downloader capability flags such as `supports_bbox`, `supports_time_subset`, `supports_streaming`.

---

## 10. Minimal checklist before committing catalog changes

Before you commit a new or modified source, check all of the following:

- key is unique,
- method is correct,
- params match the downloader contract,
- base URL is a real data endpoint,
- filename rule is verified,
- auth_required is correct,
- official_url is valid,
- aliases are helpful but not excessive,
- notes and product_notes explain any caveat,
- and unified tests still pass.

---

## 11. Related documents

This guide should be used together with:

- `USE.md`
- `AUTH_GUIDE.md`
- downloader-specific implementation files under `acquisition/downloaders/`
- unified download test scripts
- source-specific smoke tests

