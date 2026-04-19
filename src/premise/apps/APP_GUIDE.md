# APP_GUIDE.md

# PREMISЕ 应用层指南

本文档说明 PREMISЕ 中三个应用层任务如何组织、配置、运行和输出结果：

- `HydroDataBuilder`
- `DatasetEvaluator`
- `ProductRanker`

这三个应用不是替代底层模块，而是把你已经实现的下载、转换、裁剪、评价、排序能力串联成用户可直接执行的任务。

---

## 1. 应用层设计目标

PREMISЕ 底层已经具备多种能力：

- acquisition：下载数据
- conversion：转换为 NetCDF
- basin：裁剪、重采样、区域处理
- product_evaluation：指标评价
- product_ranking：排序
- visualization：出图

应用层的目的不是重复实现这些能力，而是提供一个统一入口，让用户只需要填写任务配置，就可以完成完整工作流。

建议所有应用层都遵循一个共同原则：

**用户描述任务，应用层组织流程，底层模块完成实际计算。**

---

## 2. 三个应用的关系

建议把三个应用理解成一条连续工作流，而不是三个互相孤立的脚本。

### 2.1 HydroDataBuilder

目标：  
为水文模型、气候分析或指数计算准备标准化输入数据。

核心流程：

- 下载原始数据
- 转换为 NetCDF
- 裁剪研究区域
- 筛选时间
- 变量标准化
- 重采样到目标分辨率
- 输出 ready-to-use 数据

---

### 2.2 DatasetEvaluator

目标：  
评价一个或多个数据集与参考数据的一致性，并自动输出统计表和图件。

核心流程：

- 读取产品数据
- 读取参考数据
- 对齐时间和空间
- 计算评价指标
- 输出表格和图件
- 汇总报告

---

### 2.3 ProductRanker

目标：  
基于评价结果表，对多个产品进行综合排序和推荐。

核心流程：

- 读取指标表
- 识别正向/逆向指标
- 计算权重
- 运行排序方法
- 输出排名结果和图件

---

## 3. 目录建议

建议在项目中采用如下目录：

```text
premise/
  apps/
    common.py
    runner.py
    hydro_data_builder.py
    dataset_evaluator.py
    product_ranker.py
    templates/
      hydro_data_builder_template.json
      dataset_evaluator_template.json
      product_ranker_template.json
```

如果你已经使用了应用层骨架，这个目录可以直接沿用。

---

## 4. 通用配置原则

所有应用建议采用配置驱动，而不是命令行参数写死。

推荐格式：

- JSON
- YAML

建议一个任务文件只描述一个任务。  
批量任务可以使用多个配置文件，或一个清单文件引用多个子任务。

每个配置文件建议至少包含：

- `app`
- `task_name`
- `output_dir`
- `log_level`
- `overwrite`
- 应用特有参数

例如：

```json
{
  "app": "hydro_data_builder",
  "task_name": "era5land_yangtze_2020",
  "output_dir": "I:/PREMISE_runs/era5land_yangtze_2020",
  "log_level": "INFO",
  "overwrite": false
}
```

---

## 5. HydroDataBuilder

## 5.1 适用场景

适用于以下任务：

- 为 VIC、SWAT、HBV 等模型准备输入
- 为 SPEI、SRI、SSMI 等指数计算准备输入
- 统一多个来源数据的变量、区域和时间范围
- 生成后续评价或排序需要的标准化数据集

---

## 5.2 最小输入内容

HydroDataBuilder 至少需要这些信息：

- 下载源名称 `source_key`
- 变量列表 `variables`
- 时间范围 `start_date / end_date`
- 区域范围：`bbox` 或 `shp_path`
- 输出目录
- 最终变量名映射

---

## 5.3 推荐配置模板

```json
{
  "app": "hydro_data_builder",
  "task_name": "era5land_basin_daily_2020",
  "output_dir": "I:/PREMISE_runs/era5land_basin_daily_2020",
  "overwrite": false,

  "source": {
    "source_key": "era5land_hourly",
    "variables": [
      "total_precipitation",
      "2m_temperature"
    ]
  },

  "time_range": {
    "start": "2020-01-01",
    "end": "2020-12-31"
  },

  "region": {
    "type": "shp",
    "path": "I:/basins/yangtze.shp"
  },

  "target": {
    "temporal_resolution": "daily",
    "spatial_resolution": 0.1
  },

  "variable_mapping": {
    "total_precipitation": "pr",
    "2m_temperature": "tas"
  },

  "merge_final": true
}
```

---

## 5.4 HydroDataBuilder 内部建议流程

建议固定成下面六步：

### 第一步：任务校验

检查：

- source 是否存在于 `catalog.py`
- 变量名是否存在于 source 定义中
- 时间范围是否合法
- 区域输入是否存在
- 输出目录是否可写

---

### 第二步：原始下载

调用 acquisition。

输出到：

```text
output/raw_download/
```

如果 source 不需要下载，只是读取本地文件，也可以跳过这一层。

---

### 第三步：转换为 NetCDF

调用 conversion。

输出到：

```text
output/converted_nc/
```

如果下载后的原始格式本来就是 `.nc`，这一层可以直接复制或标准化元数据。

---

### 第四步：区域和时间裁剪

调用 basin 或相关处理工具。

输出到：

```text
output/clipped/
```

这一步建议完成：

- 区域裁剪
- 时间截取
- 维度标准化

---

### 第五步：重采样和变量标准化

输出到：

```text
output/final_ready/
```

建议在这一步完成：

- 时间聚合，如 hourly → daily
- 空间分辨率统一
- 变量改名
- 单位规范化

---

### 第六步：最终合并与报告

可选输出：

```text
output/final_ready/hydro_model_input.nc
output/reports/summary.json
output/reports/summary.md
```

---

## 5.5 推荐输出目录

```text
output/
  raw_download/
  converted_nc/
  clipped/
  final_ready/
  metadata/
  reports/
  logs/
```

---

## 5.6 典型输出

建议输出：

- 单变量 ready 文件
- 合并后的综合输入文件
- 元数据说明
- 变量映射表
- 下载清单
- 简要任务报告

---

## 6. DatasetEvaluator

## 6.1 适用场景

适用于：

- 多个降水产品与参考产品比较
- 多个气象变量数据集比较
- 格点对格点评价
- 格点对站点评价
- 输出表格和常用图件

---

## 6.2 支持的两种主要模式

### 模式 A：grid_to_grid

例如：

- CHIRPS vs MSWEP
- ERA5-Land vs CMFD
- PERSIANN vs IMERG

### 模式 B：grid_to_station

例如：

- gridded product vs station CSV
- gridded product vs gauge observations

---

## 6.3 推荐配置模板

```json
{
  "app": "dataset_evaluator",
  "task_name": "precip_products_eval",
  "output_dir": "I:/PREMISE_runs/precip_products_eval",
  "overwrite": false,

  "mode": "grid_to_grid",

  "products": [
    {
      "name": "CHIRPS",
      "path": "I:/data/chirps.nc",
      "var": "pr"
    },
    {
      "name": "IMERG",
      "path": "I:/data/imerg.nc",
      "var": "precipitation"
    }
  ],

  "reference": {
    "type": "grid",
    "path": "I:/data/mswep.nc",
    "var": "precipitation"
  },

  "time_range": {
    "start": "2001-01-01",
    "end": "2020-12-31"
  },

  "region": {
    "type": "bbox",
    "bbox": [105.0, 25.0, 122.0, 35.0]
  },

  "metrics": [
    "bias",
    "rmse",
    "mae",
    "cc",
    "kge",
    "nse"
  ],

  "plots": [
    "heatmap",
    "timeseries",
    "boxplot",
    "taylor"
  ]
}
```

---

## 6.4 DatasetEvaluator 建议流程

### 第一步：读取产品和参考数据

检查：

- 文件存在
- 变量存在
- 时间维有效
- 空间维有效

---

### 第二步：统一时间和空间

建议完成：

- 时间交集筛选
- 统一时间步长
- 网格重采样或站点抽样
- 缺测值处理

---

### 第三步：计算指标

建议至少支持：

- Bias
- RMSE
- MAE
- CC
- KGE
- NSE

如果项目中已有事件类评价，还可增加：

- POD
- FAR
- CSI
- HSS

---

### 第四步：输出表格

建议输出：

```text
output/tables/
output/metrics/
```

例如：

- `summary_metrics.csv`
- `per_product_metrics.csv`
- `per_region_metrics.csv`

---

### 第五步：输出图件

建议输出：

```text
output/figures/
output/maps/
```

可包括：

- 热图
- 箱线图
- Taylor 图
- 时间序列图
- 空间分布图

---

### 第六步：输出报告

建议输出：

```text
output/report/summary.md
output/report/summary.json
```

---

## 6.5 grid_to_station 模式建议输入

如果参考数据是站点数据，建议统一格式至少包含：

- `station_id`
- `lon`
- `lat`
- `time`
- `value`

如果站点文件字段名不同，建议在配置中显式写映射：

```json
{
  "reference": {
    "type": "station",
    "path": "I:/station/obs.csv",
    "columns": {
      "station_id": "station",
      "lon": "longitude",
      "lat": "latitude",
      "time": "date",
      "value": "precip"
    }
  }
}
```

---

## 7. ProductRanker

## 7.1 适用场景

适用于：

- 多产品综合排序
- 多指标综合评价
- 不同区域分别排序
- 不同季节分别排序
- 选出适合后续融合的候选产品

---

## 7.2 推荐输入

ProductRanker 通常不直接读原始 nc，而是读 DatasetEvaluator 已经输出的指标表。

推荐配置如下：

```json
{
  "app": "product_ranker",
  "task_name": "precip_products_ranking",
  "output_dir": "I:/PREMISE_runs/precip_products_ranking",
  "overwrite": false,

  "metric_table": "I:/PREMISE_runs/precip_products_eval/tables/summary_metrics.csv",

  "methods": [
    "topsis",
    "weighted_sum",
    "average_rank"
  ],

  "weights": {
    "type": "entropy"
  },

  "benefit_metrics": [
    "cc",
    "kge",
    "nse"
  ],

  "cost_metrics": [
    "bias",
    "rmse",
    "mae"
  ],

  "group_by": [
    "region"
  ],

  "top_n": 3
}
```

---

## 7.3 ProductRanker 建议流程

### 第一步：读取指标表

检查：

- 表格存在
- 指标列存在
- 产品名称列存在
- 分组列存在（如 basin / region / season）

---

### 第二步：区分正向与逆向指标

正向指标例如：

- CC
- KGE
- NSE
- POD
- CSI

逆向指标例如：

- Bias（如果按绝对偏差）
- RMSE
- MAE
- FAR

---

### 第三步：计算权重

建议支持：

- 手动权重
- entropy
- critic
- equal

---

### 第四步：执行排序

建议支持：

- TOPSIS
- weighted_sum
- average_rank
- borda

---

### 第五步：输出结果

建议输出：

```text
output/tables/
output/figures/
output/report/
```

包括：

- 排名表
- 分组排名表
- 综合得分表
- Top-N 推荐表
- 排名柱状图
- 分组热图

---

## 7.4 Top-N 推荐

这一层非常有用。

除了输出排序结果，建议增加一个推荐清单，例如：

- Top 3 precipitation products
- Top 3 temperature products
- Top 3 candidates for fusion

这样 ProductRanker 不只是“做排序”，而是“给决策建议”。

---

## 8. 统一运行方式

建议所有应用都支持统一入口函数，例如：

```python
from premise.apps import run_application_from_file
run_application_from_file("I:/configs/hydro_task.json")
```

或者：

```python
from premise.apps.runner import run_task
run_task("I:/configs/eval_task.json")
```

如果要批量运行多个任务，可以增加：

```python
run_task_directory("I:/configs/batch/")
```

---

## 9. 日志与异常处理建议

所有应用都建议统一处理：

- 任务开始时间
- 配置快照
- 关键输入文件列表
- 每一步执行状态
- 错误信息
- 最终输出文件列表

建议至少写入：

```text
output/logs/run.log
output/report/summary.json
```

不要只把报错打印在终端里。

---

## 10. 配置字段建议

建议三个应用共同支持这些通用字段：

- `app`
- `task_name`
- `output_dir`
- `overwrite`
- `log_level`
- `save_intermediate`
- `report_format`
- `n_workers`

这样不同应用之间会有统一风格。

---

## 11. 推荐输出风格

建议每个应用都输出：

- `summary.json`
- `summary.md`

其中：

- `summary.json` 便于程序读取
- `summary.md` 便于人工查看

如果后面你想扩展网页报告或 GUI，这个结构也方便复用。

---

## 12. 应用层与底层模块的边界

建议始终保持：

### 应用层负责：
- 读取配置
- 校验任务
- 组织流程
- 管理输出目录
- 汇总结果

### 底层模块负责：
- 下载
- 转换
- 裁剪
- 指标计算
- 排序
- 出图

不要把太多业务逻辑塞回底层模块里。

---

## 13. 建议的开发顺序

如果三个应用都要继续完善，建议顺序是：

1. `HydroDataBuilder`
2. `DatasetEvaluator`
3. `ProductRanker`

原因是：

- DataBuilder 最容易先形成可用成果
- Evaluator 可以直接复用你已有模块
- Ranker 最依赖前一步的指标表规范

---

## 14. 建议的最小回归测试

每新增一个应用功能后，建议至少测试：

### HydroDataBuilder
- 一个公开数据源
- 一个变量
- 一个小时间范围
- 一个小区域

### DatasetEvaluator
- 两个产品
- 一个参考数据
- 两个指标
- 一张图

### ProductRanker
- 一张小型指标表
- 两种排序方法
- 一个 Top-N 输出

---

## 15. 常见问题

### 15.1 DataBuilder 下载成功但 final_ready 为空

先查：

- conversion 是否失败
- region 是否为空
- 时间筛选是否越界
- 变量名映射是否错误

---

### 15.2 Evaluator 结果表为空

先查：

- 产品与参考时间是否有交集
- 空间是否已对齐
- 变量名是否正确
- 缺测值是否过多

---

### 15.3 Ranker 无法输出结果

先查：

- 指标表字段名是否符合预期
- 正向/逆向指标是否配置正确
- 指标列是否存在非数值字段
- 权重配置是否有效

---

## 16. 最后建议

应用层最重要的不是“功能多”，而是：

- 配置清晰
- 输出清晰
- 报错清晰
- 用户能跑通一条最小链路

所以你后面继续完善时，建议优先做：

- 配置模板
- 最小样例任务
- 日志
- summary 报告
- 一键运行入口

这样用户体验会提升得非常明显。


---

## 6. Case-study apps for manuscript figures

The app layer now includes three manuscript-oriented applications that package the case-study workflows used in Sections 3.1–3.3:

- `basin_forcing_case`  
  Generates basin-oriented forcing outputs and the main data-driven figures for basin selection, preprocessing, and basin-ready export.
- `comparative_benchmark_case`  
  Runs the comparative benchmarking workflow over China and representative basins and writes Figures 8–11 together with the supporting CSV tables.
- `extremes_drought_case`  
  Runs extremes- and drought-oriented diagnostics and writes Figures 12–14 together with the supporting CSV tables.

All three applications are configuration-driven and can be executed through the unified runner:

```python
from premise.apps import run_application_from_file
run_application_from_file("path/to/config.json")
```

Example configuration templates are provided under `premise/apps/templates/`.
