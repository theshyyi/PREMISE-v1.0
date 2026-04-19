# USE.md

## PREMISЕ 数据获取模块使用说明

这个文件面向 **acquisition / conversion / application** 层用户，目标是把不同类型数据源的获取方式、账号要求、批量下载方法、单文件下载方法和常见问题写清楚。

本文按 **下载类型（method）** 组织，而不是按单个数据集组织。这样后续新增数据源时，只要补充 catalog 条目，通常不需要重写使用说明。

---

## 1. 总体原则

PREMISЕ 当前建议把数据源分成以下几类：

- `http_direct`：已知稳定文件 URL 模板，适合 CHIRPS、GPCC、CPC、UDEL、PREC/L、CMAP、20CR 等
- `http_listing`：公开目录索引，先列目录再筛选文件，适合 IMERG、GPCP、CMORPH（NCEI）、PERSIANN 家族（CHRSdata）等
- `erddap`：ERDDAP 子集服务，适合按时间和空间裁剪下载，如 PERSIANN-CDR NOAA ERDDAP
- `ftp`：标准 FTP 下载，适合 CMORPH CPC FTP、CMFD FTP 等
- `sftp`：SFTP 下载，适合 GLEAM
- `cds_api`：Copernicus Climate Data Store API，适合 ERA5 / ERA5-Land
- `earthdata_cmr`：NASA Earthdata + CMR 检索 + HTTPS 下载，适合 TRMM、MERRA-2、GLDAS、FLDAS
- `rclone`：共享网盘访问，适合 MSWEP
- `esgf`：ESGF 搜索与 wget 脚本，适合 CMIP5 / CMIP6
- `portal_export`：需要人工先在网页端生成导出任务的情形；当前已不推荐用于 PERSIANN 家族
- `repository_api`：Zenodo / Figshare / Dataverse 等仓储型下载（建议后续新增）

推荐顺序：

1. 能用 `http_direct` 的优先用 `http_direct`
2. 其次用 `http_listing`
3. 需要认证的源再走 `ftp / sftp / cds_api / earthdata_cmr / rclone / esgf`
4. `portal_export` 仅用于没有公开目录、没有 API、也没有稳定 FTP/HTTP 入口的情况

---

## 2. 目录与环境变量约定

建议统一把认证信息写进环境变量，不要硬编码到 `catalog.py`、测试脚本或 Git 仓库里。

建议命名风格：

```bash
ACQ_<METHOD>_<FIELD>__<SOURCE_KEY>
```

例如：

```bash
ACQ_FTP_USERNAME__CMFD=your_username
ACQ_FTP_PASSWORD__CMFD=your_password
ACQ_FTP_HOST__CMFD=ftp2.tpdc.ac.cn
ACQ_FTP_PORT__CMFD=6201

ACQ_SFTP_USERNAME__GLEAM_MONTHLY_NC=your_username
ACQ_SFTP_PASSWORD__GLEAM_MONTHLY_NC=your_password

ACQ_CDSAPI_URL=https://cds.climate.copernicus.eu/api
ACQ_CDSAPI_KEY=xxxxxxxxxxxxxxxx

ACQ_EARTHDATA_USERNAME=your_username
ACQ_EARTHDATA_PASSWORD=your_password
ACQ_EARTHDATA_TOKEN=optional_token_if_supported

ACQ_RCLONE_REMOTE__MSWEP=GoogleDrive
ACQ_RCLONE_PATH__MSWEP=/MSWEP
```

建议输出目录统一分层：

```text
output/
  raw_download/
  converted_nc/
  final_ready/
  reports/
  logs/
```

---

## 3. 单数据源下载与批量下载

### 3.1 单数据源下载

适合：
- 先验证某个数据源能否正常下载
- 调试路径、账号、token、日期范围
- 做 smoke test

推荐步骤：

1. 在 `catalog.py` 中确认 `source_key`
2. 准备认证信息（如需要）
3. 先下载一个最小时间范围，例如 1 天或 1 个月
4. 检查输出目录是否有文件
5. 确认文件格式、命名规则、时间覆盖是否正确

### 3.2 批量下载

适合：
- 批量测试 catalog 全部数据源
- 一次性获取某一年的数据
- 为后续 conversion / evaluation 做数据准备

建议策略：

- 先跑公开源：`http_direct / http_listing / erddap`
- 再跑认证源：`ftp / sftp / cds_api / earthdata_cmr / rclone / esgf`
- 最后单独处理 `portal_export`

批量测试输出建议至少包括：

- `downloaded`
- `skipped`
- `failed`
- `no_files_returned`

并写明原因，例如：

- `auth_missing`
- `unsupported_method`
- `year_not_available`
- `downloader_returned_no_files`
- `current_downloader_does_not_enforce_time_filter`

---

## 4. 各 method 的使用说明

## 4.1 `http_direct`

### 适用场景

文件 URL 模板固定，可通过年、月、日或静态文件名直接拼接。

### 典型数据源

- CHIRPS
- GPCC
- CPC
- UDEL
- PREC/L
- CMAP
- 20CR

### catalog 关键参数

```python
params={
    "base_url": "https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/",
    "filename_pattern": "precip.mon.total.v501.nc",
    "time_expansion": "none",
}
```

或：

```python
params={
    "base_url": "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/",
    "filename_pattern": "precip.{year}.nc",
    "time_expansion": "year",
}
```

### 注意事项

- `time_expansion="none"` 表示静态单文件下载，downloader 必须直接拼 `base_url + filename_pattern`
- 这类源最容易因为 downloader 只实现了 `year/date` 展开而返回空文件列表
- NOAA PSL 建议优先使用 `https://downloads.psl.noaa.gov/Datasets/` 目录而不是 THREDDS `fileServer` 目录作为 catalog 基础路径

### 适合批量下载吗

适合，且通常最稳定。

---

## 4.2 `http_listing`

### 适用场景

网页提供公开目录索引，需要先列目录，再筛选文件或递归进入子目录。

### 典型数据源

- IMERG PPS 目录
- GPCP monthly / daily
- CMORPH NCEI 目录
- PERSIANN 家族 CHRSdata

### catalog 关键参数

```python
params={
    "base_url": "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/",
    "directory_pattern": "{year}/",
    "file_regex": r'href="([^"]+\.nc)"',
    "contains_template": ["d{date}"],
    "filename_date_regex": r'd(?P<date>\d{8})',
    "time_expansion": "date",
}
```

### 对 PERSIANN 家族的建议

PERSIANN 家族现在建议走 CHRSdata 公开目录，而不是 `portal_export`：

- `https://persiann.eng.uci.edu/CHRSdata/PERSIANN/daily/`
- `https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CCS/daily/`
- `https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CDR/daily/`
- `https://persiann.eng.uci.edu/CHRSdata/PDIRNow/PDIRNowdaily/`
- `https://persiann.eng.uci.edu/CHRSdata/PUnet/PUnetdaily/`
- `https://persiann.eng.uci.edu/CHRSdata/PUnetCDR/PUnetCDR1d/`
- `https://persiann.eng.uci.edu/CHRSdata/PCCSCDR/daily/`

这类源通常需要 downloader 支持：

- 多级 `directory_pattern`
- `yyddd` 或 `yymmdd` 日期 token
- `file_regex` 和 `contains_template` 组合筛选

### 适合批量下载吗

适合，但前提是 `http_listing` downloader 足够通用。

---

## 4.3 `erddap`

### 适用场景

通过 ERDDAP 按时间和空间直接子集下载。

### 典型数据源

- PERSIANN-CDR NOAA ERDDAP

### 示例

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

### 注意事项

- ERDDAP 适合可裁剪的源，不适合静态文件集合
- 下载前应确认变量名与维度顺序
- 需要对 `bbox`、时间步长和变量名做严格检查

---

## 4.4 `ftp`

### 适用场景

标准 FTP 服务器，可能匿名，也可能需要账号密码。

### 典型数据源

- CMORPH CPC FTP
- CMFD TPDC FTP

### 典型参数

```python
params={
    "host": "ftp.cpc.ncep.noaa.gov",
    "remote_base_dir": "/precip/global_CMORPH/daily_025deg",
    "filename_pattern": "CMORPH+MWCOMB_DAILY-025DEG_{date}.Z",
    "time_expansion": "date",
}
```

### CMFD 的建议

CMFD 建议按 FTP 管理，而不是 `portal_export`。如果你已从 TPDC 获得 FTP 账号，应先确认：

- host：如 `ftp2.tpdc.ac.cn` 或 `ftp3.tpdc.ac.cn`
- port：如 `6201`
- 真实目录结构
- 文件命名规则
- 是否按变量、年、月或时间尺度分目录

### 认证方式

建议使用环境变量：

```bash
ACQ_FTP_HOST__CMFD=ftp2.tpdc.ac.cn
ACQ_FTP_PORT__CMFD=6201
ACQ_FTP_USERNAME__CMFD=...
ACQ_FTP_PASSWORD__CMFD=...
```

### 适合批量下载吗

适合，但前提是 catalog 中已经明确文件命名规则或目录层级。

---

## 4.5 `sftp`

### 适用场景

SFTP（不是 FTP），通常走 SSH 协议，需要用户名密码。

### 典型数据源

- GLEAM

### 官方访问说明

GLEAM 官方页面提供 SFTP 访问申请，并说明需要使用 **SFTP**、端口 **2225**。收到登录信息后再下载。官方还提醒要检查防火墙，确保端口 2225 未被阻塞。  

### 认证方式

```bash
ACQ_SFTP_HOST__GLEAM_MONTHLY_NC=ftp.gleam.eu
ACQ_SFTP_PORT__GLEAM_MONTHLY_NC=2225
ACQ_SFTP_USERNAME__GLEAM_MONTHLY_NC=...
ACQ_SFTP_PASSWORD__GLEAM_MONTHLY_NC=...
```

### 注意事项

- `ftplib` 不能用于 SFTP
- Python 一般用 `paramiko` 或等效 SFTP 客户端
- 下载前先确认变量目录映射，如 `Ep / E / SMs / SMrz`

---

## 4.6 `cds_api`

### 适用场景

Copernicus Climate Data Store API 下载 ERA5、ERA5-Land 等数据。

### 典型数据源

- ERA5 hourly data on single levels
- ERA5-Land hourly data

### 账号准备

1. 注册 CDS 账号
2. 登录 CDS
3. 在目标数据集页面接受许可证
4. 配置 API 凭据

### 当前官方说明

CDS 官方仍提供 API 下载说明，并要求在使用 API 前先接受数据集许可；ECMWF 说明也指出，在新 CDS 系统中，用户需要重新接受数据集条款后才能通过 API 下载。  

### 推荐配置方式

用环境变量或 `.cdsapirc`：

```bash
ACQ_CDSAPI_URL=https://cds.climate.copernicus.eu/api
ACQ_CDSAPI_KEY=<uid>:<token>
```

或标准 `.cdsapirc`：

```text
url: https://cds.climate.copernicus.eu/api
key: <uid>:<token>
```

### 单次下载建议

初次测试建议使用：

- 单变量
- 单天或单月
- 输出 GRIB

这样最容易定位问题。

---

## 4.7 `earthdata_cmr`

### 适用场景

NASA Earthdata Login + CMR 检索 granule + HTTPS 下载。

### 典型数据源

- TRMM TMPA
- MERRA-2
- GLDAS
- FLDAS

### 账号准备

1. 注册 Earthdata Login 账号
2. 完成登录
3. 根据 downloader 能力准备：
   - 用户名/密码
   - 或 token
   - 或 `.netrc` / `earthaccess`

### 当前官方说明

Earthdata Login 是 NASA EOSDIS 数据的统一注册与身份管理入口。Earthdata 官方开发文档说明，Earthdata Login API 可用于访问大量 EOSDIS 数据产品。  

### 注意事项

- `earthdata_cmr` 不是简单直链，它包括：搜索 collection / granule、筛选链接、带认证下载
- 并不是所有产品在 2020 年都可用，例如 TRMM 3B42 已停更，不应期望 2020 年成功
- 如果 downloader 支持 token，建议优先使用 token；否则用用户名/密码或 `earthaccess`

### 认证方式建议

```bash
ACQ_EARTHDATA_USERNAME=...
ACQ_EARTHDATA_PASSWORD=...
ACQ_EARTHDATA_TOKEN=optional
```

---

## 4.8 `rclone`

### 适用场景

共享 Google Drive / 云盘形式提供的数据。

### 典型数据源

- MSWEP

### 当前官方说明

MSWEP 官方提供了使用 **Rclone 从 Google Drive 下载** 的说明：先下载 Rclone，运行 `rclone config`，创建名为 `GoogleDrive` 的 remote，存储类型选择 `drive`。  

### 典型流程

1. 向数据提供方申请访问权限
2. 收到访问邮件或共享文件夹说明
3. 安装 Rclone
4. 运行 `rclone config`
5. 建立 remote，例如 `GoogleDrive`
6. 测试 `rclone ls GoogleDrive:`
7. 在 PREMISЕ 中配置 remote 名和路径

### 建议环境变量

```bash
ACQ_RCLONE_REMOTE__MSWEP=GoogleDrive
ACQ_RCLONE_PATH__MSWEP=/MSWEP
```

---

## 4.9 `esgf`

### 适用场景

CMIP5 / CMIP6 通过 ESGF 搜索和 wget 脚本方式下载。

### 典型数据源

- CMIP5
- CMIP6

### 使用建议

`esgf` 在当前 PREMISЕ 中更适合作为：

- 搜索
- 生成 wget 链接或脚本
- 人工/外部命令执行下载

而不是一条简单的 HTTP 文件下载。

### 推荐流程

1. 明确搜索 facet：变量、实验、模型、member、时间频率
2. 调用 ESGF 搜索
3. 生成 wget 脚本
4. 保存脚本并执行
5. 将结果纳入 PREMISЕ 后续流程

### 注意事项

- ESGF 登录、OpenID、节点差异都可能影响自动化
- 大文件下载更适合外部 shell / wget / aria2 处理

---

## 4.10 `portal_export`

### 当前建议

仅用于必须在网页中先生成导出任务、且没有公开目录、没有稳定 API、没有 FTP/SFTP 的场景。

### 不推荐继续用于

- PERSIANN 家族（已建议改为 `http_listing`）
- CMFD（已建议改为 `ftp`）

### 使用方式

如果某一源仍必须用 `portal_export`，则环境变量里应提供 **最终导出下载链接**，不是首页、不是 API 文档页，也不是登录页。

例如：

```bash
ACQ_EXPORT_URL__SOME_PORTAL_SOURCE=https://.../download?job=...
```

---

## 4.11 `repository_api`（建议新增）

### 适用场景

公开或半公开科研仓储：

- Zenodo
- Figshare
- Dataverse
- 其他 DOI/record/article 型仓储

### 推荐理由

这些站点的访问方式与 `http_direct`、`portal_export` 都不同，更适合单独抽象成：

- 解析 DOI / record_id / article_id
- 获取 record metadata
- 枚举文件列表
- 按规则选择文件下载

### 当前建议

后续新增一个统一 `repository_api` downloader，再按 provider 区分 `zenodo`、`figshare` adapter。

---

## 5. 各类数据源的注册与官方入口

### CDS / ERA5 / ERA5-Land

- 注册与 API 说明：<https://cds.climate.copernicus.eu/how-to-api>
- ERA5 single levels：<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>
- ERA5-Land：<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land>

### Earthdata / TRMM / MERRA-2 / GLDAS / FLDAS

- Earthdata Login：<https://www.earthdata.nasa.gov/data/earthdata-login>
- 账号入口：<https://urs.earthdata.nasa.gov/>

### GLEAM

- 主页与访问申请：<https://www.gleam.eu/>

### MSWEP

- 主页与 Rclone 下载说明：<https://www.gloh2o.org/>

### PERSIANN 家族 CHRSdata

- 根目录：<https://persiann.eng.uci.edu/CHRSdata/>

### NOAA PSL 公共文件目录

- 总目录：<https://downloads.psl.noaa.gov/Datasets/>

### NCEI CMORPH 公共目录

- 根目录：<https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/>

---

## 6. 批量下载实践建议

### 6.1 先做四层测试

1. **catalog 测试**：source 是否定义完整
2. **router 测试**：method 是否能正确路由到 downloader
3. **smoke test**：最小时间范围下载 1 个文件
4. **real download test**：指定年份批量下载

### 6.2 推荐顺序

先跑：

- `http_direct`
- `http_listing`
- `erddap`

再跑：

- `ftp`
- `sftp`
- `cds_api`
- `earthdata_cmr`
- `rclone`
- `esgf`

最后才跑：

- `portal_export`

### 6.3 对时间筛选的要求

批量下载 2020 年时，建议只有真正支持时间筛选的方法才强制跑；否则应标记为：

- `current_downloader_does_not_enforce_2020_time_filter`

这样结果更可信。

---

## 7. 一个一个下载 vs 批量下载

### 一次只下一个数据源，适合：

- 初次配置账号
- 检查 token / 用户名密码是否可用
- 检查目录结构与文件名规则
- 调试 downloader 参数

### 批量下载，适合：

- 已经验证过单源逻辑
- 想做年度数据清单测试
- 想批量构建后续 conversion / evaluation 工作流

### 建议做法

**永远先单源，再批量。**

---

## 8. 常见问题

### Q1. catalog 里已经有数据源，为什么还是下载不下来？

常见原因：

- `method` 已定义，但测试脚本未支持该 method
- `base_url` 或目录层级写错
- downloader 不支持 `time_expansion="none"`
- 需要认证，但没配环境变量
- 数据集在目标年份不存在

### Q2. 为什么 `http_direct` 的静态单文件总是返回空？

通常是 downloader 没有在 `time_expansion="none"` 时直接拼出唯一文件 URL，而是错误地尝试做时间展开。

### Q3. 为什么 `ftp` 可以，`sftp` 不行？

因为它们不是同一种协议：

- `ftp` 一般用 `ftplib`
- `sftp` 一般用 `paramiko`

### Q4. PERSIANN 家族还要不要继续用 portal_export？

当前不建议。优先用 CHRSdata 公开目录。

### Q5. CMFD 该算什么类型？

如果你拿到的是 TPDC FTP 账号和端口，应归到 `ftp`。

---

## 9. 推荐的后续补充文档

建议再配套维护：

- `CATALOG_GUIDE.md`：说明 catalog 各字段含义
- `AUTH_GUIDE.md`：专门说明各类账号和环境变量
- `TROUBLESHOOTING.md`：集中记录常见报错和修复办法
- `APPLICATIONS.md`：说明 data builder / evaluator / ranker 的配置方式

---

## 10. 维护建议

每当新增一个数据源，建议同步更新：

1. `catalog.py`
2. `USE.md`
3. 至少一个 smoke test
4. 一条示例配置

这样你的 PREMISЕ 才会真正从“代码集合”变成“可用工具”。
