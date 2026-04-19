from __future__ import annotations

import glob
import gc
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
from shapely.geometry import Point

try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover
    make_valid = None


def ensure_float32(da: xr.DataArray) -> xr.DataArray:
    try:
        return da.astype('float32')
    except Exception:
        return da


def optional_load_vector(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')
    return gdf


def load_vector(path: str | Path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')
    return gdf


def clean_geometry_list(geoms):
    cleaned = []
    for geom in geoms:
        if geom is None:
            continue
        try:
            if geom.is_empty:
                continue
            if not geom.is_valid:
                if make_valid is not None:
                    geom = make_valid(geom)
                else:
                    geom = geom.buffer(0)
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'GeometryCollection':
                parts = list(geom.geoms)
            elif geom.geom_type.startswith('Multi'):
                parts = list(geom.geoms)
            else:
                parts = [geom]
            for g in parts:
                if g is None or g.is_empty:
                    continue
                if not g.is_valid:
                    try:
                        g = make_valid(g) if make_valid is not None else g.buffer(0)
                    except Exception:
                        pass
                if g is not None and (not g.is_empty):
                    cleaned.append(g)
        except Exception:
            continue
    return cleaned


def load_china_geoms(path: str | Path | None):
    if path is None:
        return []
    gdf = optional_load_vector(path)
    if gdf is None or gdf.empty:
        return []
    return clean_geometry_list(list(gdf.geometry))


def add_china_outline(ax, china_geoms, **kwargs):
    if not china_geoms:
        return
    ax.add_geometries(list(china_geoms), crs=ccrs.PlateCarree(), **kwargs)


def add_gdf(ax, gdf, **kwargs):
    if gdf is None or len(gdf) == 0:
        return
    ax.add_geometries(list(gdf.geometry), crs=ccrs.PlateCarree(), **kwargs)


def plot_nine_dash(ax, gdf, color='red', lw=0.7):
    if gdf is None or len(gdf) == 0:
        return
    geom_types = set(gdf.geom_type.astype(str).tolist())
    if any(gt in geom_types for gt in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']):
        for geom in gdf.geometry:
            if geom is None:
                continue
            try:
                ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=color, linewidth=lw)
            except Exception:
                pass
    else:
        for geom in gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == 'Point':
                ax.plot(geom.x, geom.y, marker='_', markersize=8, color=color, markeredgewidth=1.0,
                        transform=ccrs.PlateCarree(), linestyle='None')
            elif geom.geom_type == 'MultiPoint':
                for pt in geom.geoms:
                    ax.plot(pt.x, pt.y, marker='_', markersize=8, color=color, markeredgewidth=1.0,
                            transform=ccrs.PlateCarree(), linestyle='None')


def map_common_decor(ax, china_geoms, islands_gdf, nine_dash_gdf, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_china_outline(ax, china_geoms, facecolor='none', edgecolor='black', linewidth=0.6)
    add_gdf(ax, islands_gdf, facecolor='none', edgecolor='black', linewidth=0.45)
    plot_nine_dash(ax, nine_dash_gdf, color='red', lw=0.7)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}


def add_scs_inset(fig, parent_ax, data2d, lon, lat, china_geoms, islands_gdf, nine_dash_gdf,
                  cmap, vmin, vmax, extent):
    bbox = parent_ax.get_position()
    ax = fig.add_axes([bbox.x1 - 0.06, bbox.y0 + 0.01, 0.06, 0.09], projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, data2d, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    add_gdf(ax, islands_gdf, facecolor='none', edgecolor='black', linewidth=0.35)
    plot_nine_dash(ax, nine_dash_gdf, color='red', lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def draw_simple_heatmap(
    ax, df, title, cmap='viridis', vmin=None, vmax=None,
    fmt='.2f', cbar=True, fig=None,
    title_fs=13, tick_fs=11, cell_fs=9,
):
    data = df.values.astype(float)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#E6E6E6')
    im = ax.imshow(data, aspect='auto', cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=title_fs)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=30, ha='right', fontsize=tick_fs)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=tick_fs)
    valid_vals = data[np.isfinite(data)]
    threshold = np.nanmean(valid_vals) if valid_vals.size else 0.5
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha='center', va='center', fontsize=cell_fs,
                        color='white' if val < threshold else 'black')
    if cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=tick_fs)
    return im


def detect_lon_lat(ds):
    lon_candidates = ['lon', 'longitude', 'x']
    lat_candidates = ['lat', 'latitude', 'y']
    lon_name = next((n for n in lon_candidates if n in ds.coords), None)
    lat_name = next((n for n in lat_candidates if n in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError('Could not detect longitude/latitude coordinate names.')
    return lon_name, lat_name


def detect_data_var(ds, lon_name, lat_name):
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if lon_name in dims and lat_name in dims:
            return v
    raise ValueError('Could not detect a gridded data variable.')


def detect_time_name(ds):
    return 'time' if 'time' in ds.coords else None


def wrap_longitude(ds, lon_name):
    lon = ds[lon_name].values
    if np.nanmax(lon) > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon})
        ds = ds.sortby(lon_name)
    return ds


def sort_latitude(da, lat_name='lat'):
    if da[lat_name][0] > da[lat_name][-1]:
        da = da.sortby(lat_name)
    return da


def standardize_da(da, lon_name, lat_name, time_name):
    rename_dict = {}
    if lon_name != 'lon':
        rename_dict[lon_name] = 'lon'
    if lat_name != 'lat':
        rename_dict[lat_name] = 'lat'
    if time_name and time_name != 'time':
        rename_dict[time_name] = 'time'
    if rename_dict:
        da = da.rename(rename_dict)
    return da


def convert_precip_to_mmday(da, target_unit='mm/day'):
    units = str(da.attrs.get('units', '')).strip().lower()
    if units in ['mm/day', 'mm d-1', 'mm d^-1', 'mm day-1', 'mm day^-1', 'mm']:
        return da
    if units in ['kg m-2 s-1', 'kg m^-2 s^-1', 'kg/m2/s', 'kg m-2 sec-1', 'mm/s']:
        out = da * 86400.0
        out.attrs.update(da.attrs)
        out.attrs['units'] = target_unit
        return out
    if units in ['mm/hr', 'mm h-1', 'mm h^-1', 'mm/hour']:
        out = da * 24.0
        out.attrs.update(da.attrs)
        out.attrs['units'] = target_unit
        return out
    return da


def parse_product_name(filepath: str | Path, suffix='.TIMEFIX.daily.CHINA.nc'):
    base = os.path.basename(str(filepath))
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]


def get_file_list(data_dir: str | Path, file_glob: str, explicit_files: list[str] | None = None):
    if explicit_files:
        files = [str(Path(f)) for f in explicit_files if Path(f).exists()]
    else:
        files = sorted(glob.glob(str(Path(data_dir) / file_glob)))
    return files


def find_reference_file(files, keywords):
    for fp in files:
        name = parse_product_name(fp).upper()
        if any(k.upper() in name for k in keywords):
            return fp
    raise FileNotFoundError('Could not identify reference file from filenames.')


def open_product(filepath, *, time_start=None, time_end=None, target_unit='mm/day', chunks=None):
    ds = xr.open_dataset(filepath, chunks=chunks)
    lon_name, lat_name = detect_lon_lat(ds)
    ds = wrap_longitude(ds, lon_name)
    lon_name, lat_name = detect_lon_lat(ds)
    time_name = detect_time_name(ds)
    var_name = detect_data_var(ds, lon_name, lat_name)
    da = ds[var_name]
    da = standardize_da(da, lon_name, lat_name, time_name)
    da = sort_latitude(da, 'lat')
    da = convert_precip_to_mmday(da, target_unit=target_unit)
    if 'time' not in da.dims:
        raise ValueError(f'No time dimension in {filepath}')
    da = da.sortby('time')
    if time_start is not None or time_end is not None:
        da = da.sel(time=slice(time_start, time_end))
    da = ensure_float32(da)
    da.name = parse_product_name(filepath)
    return da


def align_to_reference(prod, ref):
    common_time = np.intersect1d(prod['time'].values, ref['time'].values)
    if common_time.size == 0:
        return None, None
    prod = prod.sel(time=common_time)
    ref2 = ref.sel(time=common_time)
    same_lon = prod.sizes.get('lon', -1) == ref2.sizes.get('lon', -2) and np.allclose(prod['lon'].values, ref2['lon'].values)
    same_lat = prod.sizes.get('lat', -1) == ref2.sizes.get('lat', -2) and np.allclose(prod['lat'].values, ref2['lat'].values)
    if not (same_lon and same_lat):
        prod = prod.interp(lon=ref2['lon'], lat=ref2['lat'], method='linear')
    return ensure_float32(prod), ensure_float32(ref2)


def geometry_mask_from_xy(lon2d, lat2d, geom):
    try:
        from shapely import contains_xy
        return np.asarray(contains_xy(geom, lon2d, lat2d))
    except Exception:
        try:
            from shapely.vectorized import contains
            return np.asarray(contains(geom, lon2d, lat2d))
        except Exception:
            mask = np.zeros(lon2d.shape, dtype=bool)
            for i in range(lon2d.shape[0]):
                for j in range(lon2d.shape[1]):
                    mask[i, j] = geom.contains(Point(float(lon2d[i, j]), float(lat2d[i, j])))
            return mask


def make_domain_masks(ref_da, basin_gdf, basin_field, basins_to_use):
    lon2d, lat2d = np.meshgrid(ref_da['lon'].values, ref_da['lat'].values)
    valid_ref = np.isfinite(ref_da.mean('time', skipna=True).values)
    masks = OrderedDict()
    masks['China'] = valid_ref
    for basin_name in basins_to_use:
        sub = basin_gdf[basin_gdf[basin_field] == basin_name]
        if sub.empty:
            continue
        geom = sub.unary_union
        mask = geometry_mask_from_xy(lon2d, lat2d, geom) & valid_ref
        masks[basin_name] = mask
    return masks


def mask_to_da(mask, ref_da):
    return xr.DataArray(mask, coords={'lat': ref_da['lat'], 'lon': ref_da['lon']}, dims=('lat', 'lon'))


def basin_mean_series(da, mask):
    mask_da = mask_to_da(mask, da)
    out = da.where(mask_da).mean(dim=('lat', 'lon'), skipna=True)
    try:
        out = out.compute()
    except Exception:
        pass
    return out.to_series().dropna()


def continuous_metrics_series(sim, obs):
    df = pd.concat([sim.rename('sim'), obs.rename('obs')], axis=1).dropna()
    if len(df) < 3:
        return {'Bias': np.nan, 'RMSE': np.nan, 'CC': np.nan, 'KGE': np.nan}
    s = df['sim'].values.astype(float)
    o = df['obs'].values.astype(float)
    bias = np.mean(s - o)
    rmse = np.sqrt(np.mean((s - o) ** 2))
    cc = np.corrcoef(s, o)[0, 1] if np.nanstd(s) > 0 and np.nanstd(o) > 0 else np.nan
    alpha = (np.nanstd(s, ddof=1) / np.nanstd(o, ddof=1)) if np.nanstd(o, ddof=1) > 0 else np.nan
    beta = (np.nanmean(s) / np.nanmean(o)) if np.nanmean(o) != 0 else np.nan
    if np.isfinite(cc) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    else:
        kge = np.nan
    return {'Bias': bias, 'RMSE': rmse, 'CC': cc, 'KGE': kge}


def event_metrics_series(sim, obs, threshold):
    df = pd.concat([sim.rename('sim'), obs.rename('obs')], axis=1).dropna()
    if len(df) < 3:
        return {'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'EventScore': np.nan}
    s = df['sim'].values >= threshold
    o = df['obs'].values >= threshold
    hits = np.sum(s & o)
    misses = np.sum((~s) & o)
    false_alarms = np.sum(s & (~o))
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    vals = []
    if np.isfinite(pod):
        vals.append(pod)
    if np.isfinite(far):
        vals.append(1.0 - far)
    if np.isfinite(csi):
        vals.append(csi)
    event_score = np.nanmean(vals) if len(vals) > 0 else np.nan
    return {'POD': pod, 'FAR': far, 'CSI': csi, 'EventScore': event_score}


def max_consecutive_true(arr_bool):
    max_run = 0
    current = 0
    for val in arr_bool:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def compute_extreme_indices_from_series(series, wet_day_threshold=1.0):
    s = series.dropna().copy()
    if len(s) < 30:
        return {'Rx1day': np.nan, 'Rx5day': np.nan, 'SDII': np.nan, 'CDD': np.nan}
    rx1 = s.resample('YS').max().mean()
    roll5 = s.rolling(5, min_periods=5).sum()
    rx5 = roll5.resample('YS').max().mean()
    def annual_sdii(x):
        wet = x[x >= wet_day_threshold]
        if len(wet) == 0:
            return np.nan
        return wet.sum() / len(wet)
    sdii = s.resample('YS').apply(annual_sdii).mean()
    def annual_cdd(x):
        dry = (x < wet_day_threshold).values
        return max_consecutive_true(dry)
    cdd = s.resample('YS').apply(annual_cdd).mean()
    return {'Rx1day': rx1, 'Rx5day': rx5, 'SDII': sdii, 'CDD': cdd}


def extreme_error_scores(prod_series, ref_series, wet_day_threshold=1.0):
    prod_idx = compute_extreme_indices_from_series(prod_series, wet_day_threshold=wet_day_threshold)
    ref_idx = compute_extreme_indices_from_series(ref_series, wet_day_threshold=wet_day_threshold)
    out = {}
    per_index_scores = []
    for k in ['Rx1day', 'Rx5day', 'SDII', 'CDD']:
        p = prod_idx.get(k, np.nan)
        r = ref_idx.get(k, np.nan)
        if not np.isfinite(p) or not np.isfinite(r):
            out[k] = np.nan
            continue
        abs_rel_err = abs(p - r) / (abs(r) + 1e-12)
        score = 1.0 / (1.0 + abs_rel_err)
        out[k] = score
        per_index_scores.append(score)
    out['ExtremeScore'] = np.nanmean(per_index_scores) if per_index_scores else np.nan
    return out


def robust_vrange(arr):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    return np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)


def grid_metrics(prod, ref, min_valid_days=365):
    both = xr.where(np.isfinite(prod) & np.isfinite(ref), 1, 0)
    valid_count = both.sum('time')
    mask_valid = valid_count >= min_valid_days
    diff = prod - ref
    bias = diff.mean('time', skipna=True).where(mask_valid)
    rmse = np.sqrt((diff ** 2).mean('time', skipna=True)).where(mask_valid)
    cc = xr.corr(prod, ref, dim='time').where(mask_valid)
    mean_p = prod.mean('time', skipna=True)
    mean_r = ref.mean('time', skipna=True)
    std_p = prod.std('time', skipna=True)
    std_r = ref.std('time', skipna=True)
    alpha = std_p / (std_r + 1e-12)
    beta = mean_p / (mean_r + 1e-12)
    kge = (1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)).where(mask_valid)
    return bias, rmse, cc, kge


def normalize_metric_for_topsis(values, higher_better=True):
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    good = np.isfinite(arr)
    if good.sum() == 0:
        return out
    v = arr[good]
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if np.isclose(vmax, vmin):
        out[good] = 1.0
        return out
    out[good] = (v - vmin) / (vmax - vmin) if higher_better else (vmax - v) / (vmax - vmin)
    return out


def entropy_weight_topsis(df_benefit: pd.DataFrame):
    X = df_benefit.values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(~np.isfinite(col)):
            X[:, j] = 0.5
        else:
            m = np.nanmean(col)
            col[~np.isfinite(col)] = m
            X[:, j] = col
    colsum = X.sum(axis=0)
    colsum[colsum == 0] = 1e-12
    P = X / colsum
    P = np.where(P <= 0, 1e-12, P)
    m = X.shape[0]
    k = 1.0 / np.log(m) if m > 1 else 1.0
    e = -k * np.sum(P * np.log(P), axis=0)
    d = 1.0 - e
    w = np.full(X.shape[1], 1.0 / X.shape[1]) if np.allclose(d.sum(), 0) else d / d.sum()
    V = X * w
    ideal_best = np.nanmax(V, axis=0)
    ideal_worst = np.nanmin(V, axis=0)
    s_plus = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))
    c = s_minus / (s_plus + s_minus + 1e-12)
    out = pd.Series(c, index=df_benefit.index, name='TOPSIS_Closeness')
    weights = pd.Series(w, index=df_benefit.columns, name='EntropyWeight')
    return out, weights


def taylor_stats(sim, obs):
    df = pd.concat([sim.rename('sim'), obs.rename('obs')], axis=1).dropna()
    if len(df) < 3:
        return np.nan, np.nan
    s = df['sim'].values.astype(float)
    o = df['obs'].values.astype(float)
    std_ratio = np.std(s, ddof=1) / (np.std(o, ddof=1) + 1e-12)
    corr = np.corrcoef(s, o)[0, 1] if np.std(s, ddof=1) > 0 and np.std(o, ddof=1) > 0 else np.nan
    return std_ratio, corr


def plot_taylor(ax, stats_df, colors):
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('E')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    corrs = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    for c in corrs:
        theta = np.arccos(c)
        ax.plot([theta, theta], [0, 2.0], color='lightgray', linewidth=0.6)
        ax.text(theta, 2.05, f'{c:.2f}', fontsize=7, ha='center', va='bottom')
    for r in [0.5, 1.0, 1.5, 2.0]:
        ang = np.linspace(0, np.pi / 2, 180)
        ax.plot(ang, np.full_like(ang, r), color='lightgray', linewidth=0.6)
        ax.text(np.deg2rad(88), r, f'{r:.1f}', fontsize=7, ha='right', va='bottom')
    ax.scatter([0], [1], c='black', s=40, marker='*', zorder=5, label='Reference')
    handles = []
    for name, row in stats_df.iterrows():
        std_ratio = row['StdRatio']
        corr = row['Correlation']
        if not np.isfinite(std_ratio) or not np.isfinite(corr):
            continue
        theta = np.arccos(np.clip(corr, -1, 1))
        color = colors.get(name, 'tab:blue')
        ax.scatter(theta, std_ratio, s=35, color=color, edgecolor='black', linewidth=0.4, zorder=5)
        handles.append(Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color,
                              markeredgecolor='black', markersize=6, label=name))
    ax.set_rlim(0, 2.1)
    ax.set_title('Taylor diagram (China basin-mean daily series)', fontsize=11, pad=18)
    return handles


def build_target_grid(extent, res):
    minlon, maxlon, minlat, maxlat = extent
    lon_new = np.arange(np.floor(minlon / res) * res, np.ceil(maxlon / res) * res + res / 2, res)
    lat_new = np.arange(np.floor(minlat / res) * res, np.ceil(maxlat / res) * res + res / 2, res)
    lon_new = lon_new[(lon_new >= minlon - res) & (lon_new <= maxlon + res)]
    lat_new = lat_new[(lat_new >= minlat - res) & (lat_new <= maxlat + res)]
    return lon_new, lat_new


def draw_target_grid(ax, extent, res, step=1, color='k', lw=0.2, alpha=0.35):
    minlon, maxlon, minlat, maxlat = extent
    lon_lines = np.arange(np.floor(minlon / res) * res, np.ceil(maxlon / res) * res + res / 2, res)
    lat_lines = np.arange(np.floor(minlat / res) * res, np.ceil(maxlat / res) * res + res / 2, res)
    lon_lines = lon_lines[::step]
    lat_lines = lat_lines[::step]
    for x in lon_lines:
        ax.plot([x, x], [minlat, maxlat], transform=ccrs.PlateCarree(), color=color, linewidth=lw, alpha=alpha, zorder=4)
    for y in lat_lines:
        ax.plot([minlon, maxlon], [y, y], transform=ccrs.PlateCarree(), color=color, linewidth=lw, alpha=alpha, zorder=4)


def cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
