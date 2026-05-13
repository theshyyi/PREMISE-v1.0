# AUTH_GUIDE.md

# PREMISЕ 下载认证与账号配置指南

本文件专门说明 PREMISЕ 中各类需要账号、口令、Token、许可证接受、Rclone 配置或 FTP/SFTP 凭据的数据源应如何准备认证信息、如何配置环境变量，以及如何排查常见认证失败问题。

本指南与 `USE.md` 配套使用：

- `USE.md`：说明不同下载类型应该如何使用。
- `AUTH_GUIDE.md`：说明不同下载类型需要什么认证，以及这些认证如何准备。

---

## 1. 认证配置总原则

PREMISЕ 中凡是涉及用户名、密码、API Key、Access Token、Earthdata Bearer Token、FTP/SFTP 凭据、Rclone 远程盘配置的信息，都**不要写死在代码里**，也不要直接写进 `catalog.py`。

推荐只通过以下三种方式提供认证信息：

1. 环境变量
2. 本地不提交到 Git 的 `.env` 文件
3. 已存在的外部配置文件，如 `~/.cdsapirc`、Rclone config、Netrc 等

建议项目根目录准备一个本地模板文件，例如：

```text
.env.example
```

用户复制为：

```text
.env
```

然后填写自己的信息。

---

## 2. 推荐的环境变量命名规则

建议全部使用下面这种命名风格：

```text
ACQ_<AUTH_TYPE>_<FIELD>__<SOURCE_KEY>
```

例如：

```text
ACQ_HTTP_USERNAME__IMERG_FINAL_HDF5
ACQ_HTTP_PASSWORD__IMERG_FINAL_HDF5
ACQ_EARTHDATA_TOKEN__GLDAS_NOAH025_3H_V21
ACQ_FTP_USERNAME__CMFD
ACQ_FTP_PASSWORD__CMFD
ACQ_SFTP_USERNAME__GLEAM_MONTHLY_NC
ACQ_SFTP_PASSWORD__GLEAM_MONTHLY_NC
ACQ_RCLONE_REMOTE__MSWEP_MONTHLY_NC
```

如果多个数据源共用同一套认证，也可以定义通用环境变量，例如：

```text
ACQ_EARTHDATA_TOKEN
ACQ_CDS_API_KEY
ACQ_ESGF_OPENID
```

程序读取时建议优先级为：

1. source-specific 环境变量
2. 通用环境变量
3. 本地配置文件
4. catalog 默认值

---

## 3. 各类下载方式的认证要求

### 3.1 `http_direct`

这类源多数是公开直链，一般**不需要认证**。

典型例子：

- CHIRPS
- GPCC
- CPC
- UDEL
- PREC/L
- CMAP
- 20CR
- 部分 NOAA/NCEI 公开文件目录

通常不需要任何账号设置。

如果某个直链站点要求 Basic Auth，可以补充：

```text
ACQ_HTTP_USERNAME__SOURCE_KEY
ACQ_HTTP_PASSWORD__SOURCE_KEY
```

但目前 PREMISЕ 中大多数 `http_direct` 不需要。

---

### 3.2 `http_listing`

这类源一般也是公开目录索引，通常**不需要认证**。

典型例子：

- PERSIANN 家族公开目录
- NCEI 的 GPCP 目录
- NCEI 的 CMORPH 网页目录

如果站点目录页需要 Cookie、Session 或 Basic Auth，才需要额外支持 Header/Cookie，但目前你项目里的这批通常属于公开可浏览目录。

可选环境变量预留：

```text
ACQ_HTTP_HEADERS_JSON__SOURCE_KEY
ACQ_HTTP_COOKIES_JSON__SOURCE_KEY
```

如无特殊需要，可忽略。

---

### 3.3 `erddap`

ERDDAP 大多数公开数据**不需要认证**。

典型例子：

- NOAA NCEI 的 PERSIANN-CDR ERDDAP

一般只要请求 URL 正确即可。

如果某些私有 ERDDAP 实例需要认证，可预留：

```text
ACQ_ERDDAP_USERNAME__SOURCE_KEY
ACQ_ERDDAP_PASSWORD__SOURCE_KEY
```

目前通常不用。

---

### 3.4 `ftp`

FTP 可能是匿名，也可能需要账号密码。

#### 匿名 FTP

例如部分 NOAA FTP：

- CMORPH 的部分 CPC FTP 路径

这种通常不用设置用户名密码，程序默认匿名登录即可。

#### 认证 FTP

例如你现在已经确认的：

- CMFD：TPDC FTP 下载

推荐环境变量：

```text
ACQ_FTP_HOST__CMFD=ftp2.tpdc.ac.cn
ACQ_FTP_PORT__CMFD=6201
ACQ_FTP_USERNAME__CMFD=你的用户名
ACQ_FTP_PASSWORD__CMFD=你的密码
```

如果同一源有多个镜像主机，也可以再加：

```text
ACQ_FTP_HOST_BACKUP__CMFD=ftp3.tpdc.ac.cn
```

FTP 使用时除了账号密码，还需要进一步确认：

- 真实 `remote_base_dir`
- 目录结构
- 文件命名规则
- 是否按变量/年份/月份拆分

所以 FTP 源接入一般要先做一次“目录探查”。

---

### 3.5 `sftp`

SFTP 不是 FTP，而是基于 SSH 的文件传输。

你当前最典型的是：

- GLEAM

推荐环境变量：

```text
ACQ_SFTP_HOST__GLEAM_MONTHLY_NC=sftp.gleam.eu
ACQ_SFTP_PORT__GLEAM_MONTHLY_NC=2225
ACQ_SFTP_USERNAME__GLEAM_MONTHLY_NC=你的用户名
ACQ_SFTP_PASSWORD__GLEAM_MONTHLY_NC=你的密码
```

如果以后需要支持私钥认证，也可以扩展：

```text
ACQ_SFTP_KEYFILE__SOURCE_KEY=C:/Users/.../.ssh/id_rsa
ACQ_SFTP_PASSPHRASE__SOURCE_KEY=可选口令
```

注意：

- `ftplib` 不能用于 SFTP。
- SFTP 需要使用 `paramiko` 或等价方案。

---

### 3.6 `cds_api`

典型例子：

- ERA5
- ERA5-Land

这类源不是简单的用户名密码下载，而是需要：

1. 注册 Copernicus / CDS 账号
2. 在网页上接受对应数据集的 licence/terms
3. 配置本地 API 凭据

推荐两种方式：

#### 方式 A：使用官方 `~/.cdsapirc`

这是最标准的做法。用户在家目录放好：

```text
~/.cdsapirc
```

内容类似：

```text
url: https://cds.climate.copernicus.eu/api
key: <uid>:<api-key>
```

#### 方式 B：通过环境变量覆盖

```text
ACQ_CDS_URL=https://cds.climate.copernicus.eu/api
ACQ_CDS_KEY=<uid>:<api-key>
```

如果你的 downloader 支持 source-specific，也可以：

```text
ACQ_CDS_KEY__ERA5_SINGLE_LEVELS_HOURLY=<uid>:<api-key>
ACQ_CDS_KEY__ERA5LAND_HOURLY=<uid>:<api-key>
```

常见失败原因：

- 没接受数据集 licence
- `.cdsapirc` 放错位置
- API key 填写错误
- 数据集变量名称不符合 CDS 要求

---

### 3.7 `earthdata_cmr`

典型例子：

- TRMM
- MERRA-2
- GLDAS
- FLDAS

这类源通常依赖 NASA Earthdata Login。

推荐优先使用 Bearer Token：

```text
ACQ_EARTHDATA_TOKEN=你的 Earthdata token
```

也可以支持 source-specific：

```text
ACQ_EARTHDATA_TOKEN__GLDAS_NOAH025_3H_V21=...
ACQ_EARTHDATA_TOKEN__MERRA2_TAVGM_2D_FLX_NX_MONTHLY=...
```

如果未来你要兼容 username/password 模式，也可预留：

```text
ACQ_EARTHDATA_USERNAME=...
ACQ_EARTHDATA_PASSWORD=...
```

但更推荐 Token。

注意事项：

- Earthdata token 可能过期，需要定期更新。
- 即使认证正确，某些产品也可能因为年份不覆盖而返回空。例如 TRMM 到 2019 年停止。

---

### 3.8 `rclone`

典型例子：

- MSWEP

这类源本质上依赖用户本地已经配置好的 Rclone remote，而不是 PREMISЕ 直接管理账号密码。

推荐环境变量：

```text
ACQ_RCLONE_REMOTE__MSWEP_MONTHLY_NC=mswep_remote
ACQ_RCLONE_REMOTE_PATH__MSWEP_MONTHLY_NC=Monthly
ACQ_RCLONE_REMOTE__MSWEP_DAILY_NC=mswep_remote
ACQ_RCLONE_REMOTE_PATH__MSWEP_DAILY_NC=Daily
```

如果你想支持全局设置：

```text
ACQ_RCLONE_REMOTE=mswep_remote
```

用户需要先自己在命令行完成：

```bash
rclone config
```

并保证下面命令可以正常工作：

```bash
rclone lsd mswep_remote:
rclone ls mswep_remote:
```

如果这两条都不通，PREMISЕ 就无法继续。

---

### 3.9 `esgf`

典型例子：

- CMIP5
- CMIP6

ESGF 通常不是简单直链下载，而是：

1. 搜索文件
2. 生成 wget script
3. 通过 OpenID/证书或登录态完成下载

建议认证信息通过环境变量提供：

```text
ACQ_ESGF_OPENID=你的 OpenID
ACQ_ESGF_USERNAME=你的用户名
ACQ_ESGF_PASSWORD=你的密码
```

如果你后面做得更规范，也可以进一步支持：

```text
ACQ_ESGF_WGET_SCRIPT_DIR=...
ACQ_ESGF_CERT_PATH=...
```

注意：

- ESGF 的“可搜索”不等于“可直接下载”。
- 你当前脚本如果只是生成 wget URL 或搜索结果，也不应该假装成已经完成真实下载。

---

### 3.10 `portal_export`

这类最容易让用户误解。

它不是 FTP，也不是标准 API。

它的逻辑通常是：

1. 用户先去网页端选产品、区域、时间、格式
2. 网页端生成一个导出任务
3. 返回一个临时下载链接
4. 程序再用这个最终链接下载

推荐环境变量：

```text
ACQ_EXPORT_URL__SOURCE_KEY=最终导出的真实文件链接
```

例如：

```text
ACQ_EXPORT_URL__PERSIANN_PORTAL=...
```

如果当前数据源已经确认其实是公开目录索引，就不应继续用 `portal_export`，而应改成 `http_listing`。

也就是说：

- 真正的导出任务型网页：`portal_export`
- 公开目录页：`http_listing`

---

### 3.11 `repository_api`

如果后面你把 Zenodo / Figshare / Dryad 这类仓储站点统一做成 `repository_api`，建议认证也拆分成“公开下载”和“私有/上传”。

#### 公开记录下载

一般不需要认证。

#### 私有或上传

推荐：

```text
ACQ_REPO_TOKEN__SOURCE_KEY=...
```

如果再细分 provider：

```text
ACQ_ZENODO_TOKEN=...
ACQ_FIGSHARE_TOKEN=...
```

---

## 4. 推荐的 .env 模板

你可以在项目根目录放一个 `.env.example`：

```text
# CDS
ACQ_CDS_URL=https://cds.climate.copernicus.eu/api
ACQ_CDS_KEY=

# Earthdata
ACQ_EARTHDATA_TOKEN=

# GLEAM SFTP
ACQ_SFTP_HOST__GLEAM_MONTHLY_NC=
ACQ_SFTP_PORT__GLEAM_MONTHLY_NC=2225
ACQ_SFTP_USERNAME__GLEAM_MONTHLY_NC=
ACQ_SFTP_PASSWORD__GLEAM_MONTHLY_NC=

# CMFD FTP
ACQ_FTP_HOST__CMFD=ftp2.tpdc.ac.cn
ACQ_FTP_PORT__CMFD=6201
ACQ_FTP_USERNAME__CMFD=
ACQ_FTP_PASSWORD__CMFD=

# MSWEP Rclone
ACQ_RCLONE_REMOTE__MSWEP_MONTHLY_NC=
ACQ_RCLONE_REMOTE_PATH__MSWEP_MONTHLY_NC=Monthly
ACQ_RCLONE_REMOTE__MSWEP_DAILY_NC=
ACQ_RCLONE_REMOTE_PATH__MSWEP_DAILY_NC=Daily

# ESGF
ACQ_ESGF_OPENID=
ACQ_ESGF_USERNAME=
ACQ_ESGF_PASSWORD=

# Portal export fallback
ACQ_EXPORT_URL__PERSIANN_PORTAL=
```

---

## 5. 用户应如何逐类完成认证

### 5.1 ERA5 / ERA5-Land

用户操作建议：

1. 注册 CDS 账号
2. 网页端登录并接受相关数据集协议
3. 生成 API key
4. 配置 `.cdsapirc` 或 `ACQ_CDS_KEY`
5. 先下载一个变量、一天的小测试任务

### 5.2 Earthdata 数据

用户操作建议：

1. 注册 Earthdata 账号
2. 生成 token
3. 设置 `ACQ_EARTHDATA_TOKEN`
4. 先对一个时间很短的小任务做测试

### 5.3 GLEAM

用户操作建议：

1. 官网申请访问
2. 收到 SFTP 主机、端口、用户名、密码
3. 写入环境变量
4. 先测试列目录，再测试下载单文件

### 5.4 CMFD

用户操作建议：

1. 在 TPDC 页面申请 FTP 下载权限
2. 获得 FTP host、port、username、password
3. 写入环境变量
4. 先列 FTP 根目录，确认真实目录结构
5. 再完善 `catalog.py` 中的 `remote_base_dir`

### 5.5 MSWEP

用户操作建议：

1. 获得共享盘或授权说明
2. 本地配置好 Rclone remote
3. 用命令行确认 remote 可访问
4. 再把 remote 名称提供给 PREMISЕ

### 5.6 ESGF

用户操作建议：

1. 注册 ESGF 账号
2. 准备 OpenID / 登录信息
3. 先执行一次搜索测试
4. 再执行 wget script 或文件下载测试

---

## 6. 批量下载时的认证建议

批量下载前，建议先按类别做 smoke test。

### 推荐顺序

先测公开源：

- `http_direct`
- `http_listing`
- `erddap`

再测轻度认证源：

- `ftp`
- `sftp`
- `cds_api`

再测重度认证或复杂流程源：

- `earthdata_cmr`
- `rclone`
- `esgf`
- `portal_export`

不要在凭据没验证之前直接跑全量批量下载。

---

## 7. 常见认证失败与排查

### 7.1 401 Unauthorized

说明：

- 用户名密码错误
- Token 过期
- Header 没正确传

优先检查：

- 环境变量是否生效
- token 是否复制完整
- 是否存在多余空格或换行

### 7.2 403 Forbidden

说明：

- 账号虽存在，但没有访问权限
- 还没接受 licence
- Earthdata / CDS / portal 会话已失效

优先检查：

- 是否接受了数据集协议
- 当前 token 是否仍有效
- 是否是临时下载链接已过期

### 7.3 登录成功但列不到文件

说明：

- 目录路径错了
- 账号有权限，但目录结构和 catalog 不一致
- 文件名规则写错

优先检查：

- FTP/SFTP 的真实目录树
- `remote_base_dir`
- `filename_pattern` / `file_regex`

### 7.4 批量时部分源正常、部分源全部失败

说明：

- 通常不是网络整体问题，而是某一类 method 的认证没配好

优先按 method 分组排查，而不是按单个产品逐个乱试。

---

## 8. 开发者建议

对于 PREMISЕ 后续实现，建议所有 downloader 都提供统一的认证读取函数，例如：

```python
get_auth_value(source_key, field, scope='source_then_global')
```

这样不同 downloader 就不用各自重复写环境变量逻辑。

同时建议在日志里明确输出：

- 当前 source_key
- 当前 method
- 认证来源：env / file / default
- 是否检测到必需凭据

但不要在日志里打印真正的密码和 token。

---

## 9. 后续建议配套文件

建议项目文档最终分成：

- `USE.md`：下载方法总说明
- `AUTH_GUIDE.md`：认证、注册、环境变量、token、账号说明
- `CATALOG_GUIDE.md`：catalog 条目如何编写
- `TROUBLESHOOTING.md`：常见报错排查

这样结构会很清晰。

