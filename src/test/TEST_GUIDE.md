# TEST_GUIDE.md

# PREMISЕ 测试指南

本文档说明 PREMISЕ 项目中各类测试应如何组织、运行、解释结果，以及出现问题时应该先检查 `catalog`、`downloader` 还是认证配置。

---

## 1. 测试目标

PREMISЕ 的测试建议分成四层，不要把所有问题都混在一次“真实下载”里排查。

第一层是**结构测试**。  
目标是确认 `catalog.py` 中的数据源定义是否完整、字段是否规范、`method` 是否已在 registry 中注册。

第二层是**接口测试**。  
目标是确认每个 downloader 是否可以被正确实例化，路由器能否把请求分发到正确的 downloader。

第三层是**模拟下载测试**。  
目标是确认请求构造、URL 拼接、时间展开、目录遍历、文件名过滤逻辑是否正确，但不真正联网下载大文件。

第四层是**真实下载测试**。  
目标是确认目标数据源在当前网络、认证、时间范围和参数设定下，确实可以下载到文件。

建议始终按这个顺序排查：  
**结构测试 → 接口测试 → 模拟测试 → 真实下载测试**

---

## 2. 推荐的测试文件组织

建议在项目中至少保留以下几类测试文件：

```text
tests/
  test_catalog.py
  test_registry.py
  test_downloaders.py
  test_conversion_formats.py
  test_acquisition_unified.py
  test_real_download_smoke.py
```

也可以按模块拆分：

```text
tests/
  acquisition/
  conversion/
  evaluation/
  ranking/
  apps/
```

如果你已经有统一测试脚本和报告脚本，也建议保留：

```text
test_acquisition_unified.py
run_acquisition_unified_report.py
download_all_catalog_2020_real.py
```

---

## 3. acquisition 模块测试

### 3.1 catalog 结构测试

这一层不联网，只检查定义是否合理。

建议检查以下内容：

- `key` 是否唯一
- `title` 是否非空
- `variables` 是否为元组
- `method` 是否非空
- `official_url` 是否非空
- `params` 是否为字典
- `auth_required` 是否和实际下载方式一致
- `access_mode` 是否与 `method` 大体一致
- `aliases` 中是否存在重复或与其他 key 冲突

最重要的一条是：

**一个数据源是否已经写进 `DEFAULT_SOURCES`，要和“用户以为支持了”区分开。**

也就是说，文档里列了不等于已经接入；  
只有写进 `catalog.py` 才算项目真正支持。

---

### 3.2 registry 测试

这一层检查 downloader 是否已注册。

建议检查：

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
- `repository_api`

如果 `catalog.py` 里有某个 `method`，但 registry 里没有，对应数据源应该在测试里被标记为：

- `skipped`
- 原因：`method_not_registered`

而不是误判为下载失败。

---

### 3.3 模拟下载测试

这一层建议使用 monkeypatch 或 mock。

测试内容主要包括：

- URL 是否被正确拼接
- 年/月/日时间展开是否正确
- `http_listing` 是否正确进入目录并筛选文件
- `ftp` 是否正确生成远程路径
- `cds_api` 是否正确构造 request body
- `earthdata_cmr` 是否正确构造 CMR 查询参数
- `repository_api` 是否正确解析 record/file metadata

这一层的目标不是下载真实文件，而是测试“请求构造逻辑”。

---

### 3.4 真实下载测试

这一层才真正联网。

建议有一个统一脚本，例如：

```bash
python download_all_catalog_2020_real.py
```

或者：

```bash
python download_acquisition_2020_testdata_v3.py
```

真实下载测试建议只做：

- 一个很短的时间范围
- 一个很小的空间范围（仅对支持 subset 的数据源）
- 一个变量
- 一个小的输出目录

不要一开始就下载全年、全区域、多变量。

---

## 4. conversion 模块测试

conversion 模块建议至少覆盖以下四类：

- HDF → NetCDF
- GeoTIFF → NetCDF
- GRIB → NetCDF
- Binary → NetCDF

建议测试点包括：

- 输入文件格式识别是否正确
- `open_dataset()` 是否能自动路由到正确 reader
- `to_netcdf()` 是否能成功输出
- 输出文件变量名、维度名是否符合预期
- 时间坐标是否正确
- 缺测值和 `nodata` 是否被正确处理

如果当前环境没有 `cfgrib` 或 `eccodes`，GRIB 测试可以拆成两层：

- 单元测试：mock `xr.open_dataset(engine="cfgrib")`
- 集成测试：本地具备依赖时再跑真实 GRIB

---

## 5. evaluation 模块测试

评价模块建议分为：

- grid-to-grid
- grid-to-station

建议测试以下能力：

- 文件读取与变量识别
- 时间对齐
- 空间重采样
- 缺测值处理
- 指标计算
- 输出表格
- 出图函数是否正常生成图片文件

关键指标至少建议覆盖：

- Bias
- RMSE
- MAE
- CC
- KGE
- NSE

如果有事件类评价，还建议覆盖：

- POD
- FAR
- CSI
- HSS

---

## 6. ranking 模块测试

排序模块建议覆盖：

- 指标输入表读取
- 正向指标 / 逆向指标识别
- 权重计算
- 排序方法执行
- 输出结果表
- 排名图生成

如果支持多方法排序，建议至少测：

- TOPSIS
- 加权求和
- 平均名次
- Borda 或其他共识排名

---

## 7. apps 层测试

如果已经新增了：

- `hydro_data_builder.py`
- `dataset_evaluator.py`
- `product_ranker.py`

建议为这三类应用各写一个最小任务配置测试。

测试重点不是“算法正确性”，而是：

- 配置文件能否被读取
- 应用能否调用到底层模块
- 输出目录是否被正确组织
- 最终 summary/report 是否生成

---

## 8. 真实下载测试结果如何解释

建议把真实下载结果统一分成以下四类：

### 8.1 downloaded

说明：

- downloader 已成功执行
- 返回了文件
- 文件已写入目标路径

这表示该数据源在当前配置下可用。

---

### 8.2 skipped

说明：

- 脚本主动跳过该数据源
- 不表示下载器坏了

常见原因：

- `auth_required` 但未提供凭据
- downloader 未实现严格时间筛选
- `portal_export` 需要人工先生成导出链接
- `esgf` 当前只支持搜索或 wget 脚本，不直接下载文件
- 请求年份超出产品时间范围
- 对应 `method` 尚未注册

---

### 8.3 failed

说明：

- 脚本尝试执行了下载
- 但中间抛出异常

常见原因：

- 401/403：认证失败
- 404：URL 或路径错误
- timeout：网络或服务器响应过慢
- SSL/连接错误：远程站点握手异常
- 文件写入失败：本地磁盘或权限问题

---

### 8.4 no_files_returned

说明：

- downloader 执行了
- 但最终没找到文件

这是最容易误判的一类。

常见原因包括：

- `filename_pattern` 不对
- `file_regex` 不对
- `directory_pattern` 不对
- `time_expansion` 与真实目录结构不匹配
- `http_direct` 的静态单文件分支没有正确实现
- 年/月/日 token 规则和真实文件命名不一致
- 该产品在请求年份没有数据

---

## 9. 出问题时先查哪里

### 9.1 先查 catalog 的情况

如果表现为：

- `no_files_returned`
- URL 拼错
- 时间范围不对
- 目录层级不对
- 文件名规则不对
- 把 `portal_export` 当成 `http_listing`
- 把 `ftp` 当成 `http_direct`

这类优先查 `catalog.py`。

重点看：

- `method`
- `base_url`
- `directory_pattern`
- `filename_pattern`
- `file_regex`
- `time_expansion`
- `official_url`
- `product_notes`

---

### 9.2 先查 downloader 的情况

如果表现为：

- 同一类数据源大面积返回空
- `time_expansion="none"` 全部失败
- 多级目录 listing 下不来
- `ftp` 能连接但不会遍历目录
- `earthdata_cmr` 会构造请求但不会下载 granule

这类优先查 downloader 实现。

---

### 9.3 先查认证配置的情况

如果表现为：

- 401
- 403
- 返回登录页 HTML
- 目录为空但网页端能访问
- API 提示 unauthorized
- Earthdata/Copernicus/Rclone 无法访问

这类优先查：

- 环境变量
- token
- 用户名密码
- API key
- license/terms 是否已接受
- 本地 `~/.cdsapirc`
- 本地 `rclone.conf`

---

## 10. 推荐的排查顺序

建议按下面顺序排查，不要跳步：

1. 这个源是否已经在 `DEFAULT_SOURCES` 中  
2. 这个源的 `method` 是否已经注册  
3. `catalog.py` 中的 `base_url / pattern / regex` 是否真实可用  
4. downloader 是否支持这种目录结构  
5. 是否需要认证  
6. 是否请求了产品不支持的年份  
7. 是否是输出路径或本地权限问题

---

## 11. 哪些数据源默认可能被 skip

下面这些类型很常见会被默认跳过：

### 11.1 认证型但未提供凭据

例如：

- CDS API
- Earthdata
- GLEAM SFTP
- MSWEP Rclone
- 受限 FTP

---

### 11.2 不能严格按时间切片的源

例如：

- 一次性整包下载
- 门户导出链接
- Rclone 共享盘整目录
- ESGF 搜索结果

如果测试目标是“只下载 2020 年”，这类很可能被脚本主动 skip。

---

### 11.3 portal 型源

如果 downloader 只是“接收现成 export_url 并下载”，而不是自动生成导出任务，那么在没有人工生成导出链接时，这类通常要 skip。

---

## 12. 建议的测试命令

### 12.1 结构和接口测试

```bash
pytest -q tests/
```

或者：

```bash
pytest -q test_acquisition_unified.py
```

---

### 12.2 生成统一报告

```bash
python run_acquisition_unified_report.py
```

---

### 12.3 真实下载测试

```bash
python download_all_catalog_2020_real.py
```

如果脚本支持环境变量，也可以先设置：

```bash
RUN_ACQUISITION_LIVE=1
```

---

## 13. 建议输出哪些测试报告

建议至少导出：

- `summary.json`
- `datasets.csv`
- `datasets.json`
- `tests.csv`

推荐字段包括：

- `source_key`
- `method`
- `status`
- `message`
- `auth_required`
- `start_date`
- `end_date`
- `output_dir`
- `n_files`
- `downloaded_paths`

这样后面复盘非常方便。

---

## 14. 推荐的回归测试策略

每次修改下面这些文件后，建议立即回归测试：

- `catalog.py`
- `registry.py`
- 任一 downloader
- `io_api.py`
- `conversion/readers/*`
- `apps/*`

最少要跑：

1. catalog/registry 结构测试  
2. acquisition 统一测试  
3. conversion 最小测试  
4. 至少 1 个真实下载 smoke test

---

## 15. 一套推荐工作流

每新增一个数据源，建议按这个流程走：

1. 先确认官方访问方式  
2. 决定 `method`
3. 在 `catalog.py` 中新增条目
4. 写最小结构测试
5. 写 mock 下载测试
6. 做单源真实下载 smoke test
7. 再纳入全量统一测试
8. 最后补文档到 `USE.md` / `AUTH_GUIDE.md` / `CATALOG_GUIDE.md`

---

## 16. 最后建议

测试时最忌讳两件事：

第一，catalog 还没核对真实目录，就直接怀疑 downloader。  
第二，认证没配好，就直接把结果当成“数据源不可用”。

你这个项目现在最需要坚持的原则是：

**先把数据源定义正确，再验证 downloader；先把 mock 测试跑通，再做真实下载测试。**

这样维护成本会低很多。
