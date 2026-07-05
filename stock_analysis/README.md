# Stock Analysis — 股票预测使用文档

基于 [Darts](https://unit8co.github.io/darts/) 时序预测框架，接入真实交易数据，完成**数据拉取 → 特征工程 → 训练 → 回测 → 正式预测**的完整流程。

> **方法原理与预测解读**请参阅 [预测方法讲解.md](./预测方法讲解.md)（预测目标、算法原理、回测逻辑、结果解读）。

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境安装](#2-环境安装)
3. [快速开始](#3-快速开始)
4. [命令行用法](#4-命令行用法)
5. [Python API 用法](#5-python-api-用法)
6. [数据源说明](#6-数据源说明)
7. [配置参数](#7-配置参数)
8. [输出文件说明](#8-输出文件说明)
9. [工作流程建议](#9-工作流程建议)
10. [扩展与修改](#10-扩展与修改)
11. [常见问题](#11-常见问题)

**相关文档：** [预测方法讲解.md](./预测方法讲解.md) — 预测原理、模型方法、回测逻辑、结果解读

---

## 1. 项目结构

```
AI static/
├── darts-master/              # Darts 框架源码（需先安装）
├── statsmodels-main/          # Darts 依赖（需先安装）
└── stock_analysis/            # 股票分析业务模块
    ├── __init__.py
    ├── __main__.py            # python -m stock_analysis 入口
    ├── cli.py                 # 命令行接口
    ├── config.py              # 配置类 StockConfig
    ├── forecaster.py          # 核心类 StockForecaster
    ├── features.py            # 技术指标特征工程
    ├── requirements.txt       # 业务层依赖
    ├── data/
    │   └── loader.py          # 真实数据接入（yfinance / akshare / CSV）
    └── output/                # 运行后自动生成
        ├── data/              # 缓存的 OHLCV 数据
        ├── models/            # 训练好的模型
        ├── forecasts/         # 正式预测结果
        └── backtests/         # 回测结果与指标
```

---

## 2. 环境安装

### 2.1 前置要求

- Python **3.10+**（推荐 3.11）
- 已克隆本仓库，且包含 `darts-master`、`statsmodels-main` 目录

### 2.2 安装步骤

在项目根目录 `AI static/` 下执行：

```powershell
# 1. 创建并激活虚拟环境（推荐）
conda create -n stock-darts python=3.11 -y
conda activate stock-darts

# 2. 安装 statsmodels（darts 依赖）
pip install -e "d:\VS\AI static\statsmodels-main"

# 3. 安装 darts（含 LightGBM / XGBoost 等 ML 模型，不含 PyTorch）
pip install -e "d:\VS\AI static\darts-master[notorch]"

# 4. 安装股票分析模块依赖
pip install -r "d:\VS\AI static\stock_analysis\requirements.txt"
```

### 2.3 可选依赖

| 用途 | 安装命令 |
|------|----------|
| A 股数据（akshare） | `pip install akshare` |
| XGBoost 模型 | `pip install xgboost` |
| 深度学习模型（TFT、N-BEATS 等） | `pip install -e "d:\VS\AI static\darts-master[torch,notorch]"` |

### 2.4 验证安装

```powershell
cd "d:\VS\AI static"
python -c "from darts.models import LightGBMModel; from stock_analysis import StockForecaster; print('OK')"
```

---

## 3. 快速开始

### 美股 AAPL：拉数据 → 回测 → 正式预测

```powershell
cd "d:\VS\AI static"

# 1. 下载并缓存数据
python -m stock_analysis fetch --symbol AAPL --market us --start 2020-01-01

# 2. 滚动回测（评估模型效果）
python -m stock_analysis backtest --symbol AAPL --model lightgbm --horizon 5

# 3. 正式预测未来 5 个交易日
python -m stock_analysis predict --symbol AAPL --model lightgbm --horizon 5
```

预测结果保存在 `stock_analysis/output/forecasts/AAPL_YYYYMMDD_HHMMSS.csv`。

### A 股示例（需安装 akshare）

```powershell
python -m stock_analysis predict --symbol 600519 --market cn --start 2020-01-01 --horizon 5
```

### 本地 CSV 数据

CSV 需包含列：`date, open, high, low, close, volume`（也支持中文列名：日期、开盘、收盘 等）。

```powershell
python -m stock_analysis predict --symbol MY_STOCK --market csv --csv-path "D:\data\600519.csv"
```

---

## 4. 命令行用法

### 4.1 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | `AAPL` | 股票代码 |
| `--market` | `us` | 数据源：`us` / `cn` / `csv` |
| `--start` | `2020-01-01` | 起始日期 |
| `--end` | 无 | 结束日期（默认到最新） |
| `--csv-path` | 无 | 本地 CSV 路径（`market=csv` 时必填） |
| `--model` | `lightgbm` | 模型：`lightgbm` / `xgboost` / `linear` / `arima` |
| `--horizon` | `5` | 预测未来 N 个交易日 |
| `--target` | `log_return` | 预测目标：`log_return`（对数收益率）或 `close`（收盘价） |

### 4.2 子命令

#### `fetch` — 下载并缓存数据

```powershell
python -m stock_analysis fetch --symbol TSLA --market us --start 2019-01-01
```

#### `train` — 训练模型（使用 80% 数据，用于开发调试）

```powershell
python -m stock_analysis train --symbol AAPL --model lightgbm
```

模型保存至 `stock_analysis/output/models/`。

#### `backtest` — 滚动回测

```powershell
python -m stock_analysis backtest --symbol AAPL --model lightgbm --horizon 5
```

输出示例：

```
Backtest metrics:
  mae: 0.0123
  rmse: 0.0189
  mape: 85.6
  n_points: 480
  backtest_path: stock_analysis/output/backtests/AAPL_20250705_132000.csv
```

#### `predict` — 正式预测

使用**全部历史数据**训练，并预测未来 N 天：

```powershell
python -m stock_analysis predict --symbol AAPL --model lightgbm --horizon 5
```

加载已保存模型（不再重新训练）：

```powershell
python -m stock_analysis predict --symbol AAPL --load
```

---

## 5. Python API 用法

### 5.1 基本流程

```python
from stock_analysis import StockConfig, StockForecaster

# 配置
config = StockConfig(
    symbol="AAPL",
    market="us",
    start_date="2020-01-01",
    model_type="lightgbm",
    forecast_horizon=5,
    target="log_return",
)

forecaster = StockForecaster(config)

# 拉取数据
df = forecaster.fetch_data()
print(f"共 {len(df)} 条记录")

# 回测
metrics = forecaster.backtest()
print(metrics)

# 正式预测
result = forecaster.predict()
print(result)
print(f"结果已保存: {result.attrs['output_path']}")
```

### 5.2 分步调用

```python
from stock_analysis import StockConfig, StockForecaster

config = StockConfig(symbol="600519", market="cn")
fc = StockForecaster(config)

# 1. 数据 + 特征
fc.fetch_data()
target, past_cov = fc.prepare()

# 2. 训练（开发模式：80% 数据）
fc.train()

# 3. 或：生产模式（100% 数据）
fc.train(use_full_data=True)

# 4. 加载已保存模型
fc.load()
forecast = fc.predict(n=10)
```

### 5.3 使用本地 DataFrame

```python
import pandas as pd
from stock_analysis import StockConfig, StockForecaster

df = pd.read_csv("my_stock.csv")  # 需含 date, open, high, low, close, volume

config = StockConfig(symbol="CUSTOM", market="csv")
fc = StockForecaster(config)
result = fc.predict(df=df)
```

---

## 6. 数据源说明

| market | 数据源 | 代码示例 | 说明 |
|--------|--------|----------|------|
| `us` | [yfinance](https://github.com/ranaroussi/yfinance) | `AAPL`, `TSLA`, `MSFT` | 美股、部分全球标的 |
| `cn` | [akshare](https://github.com/akfamily/akshare) | `600519`, `000001` | A 股日线，前复权 |
| `csv` | 本地文件 | 任意 | 券商导出、Wind、Tushare 等 |

### CSV 格式要求

```csv
date,open,high,low,close,volume
2024-01-02,185.5,186.2,184.1,185.8,52000000
2024-01-03,185.9,187.0,185.2,186.5,48000000
```

也支持中文列名：`日期, 开盘, 最高, 最低, 收盘, 成交量`。

---

## 7. 配置参数

通过 `StockConfig` 类配置（`config.py`）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | str | `"AAPL"` | 股票代码 |
| `market` | str | `"us"` | 数据源 |
| `csv_path` | str | None | CSV 路径 |
| `start_date` | str | `"2020-01-01"` | 起始日期 |
| `end_date` | str | None | 结束日期 |
| `target` | str | `"log_return"` | 预测目标 |
| `train_ratio` | float | `0.8` | 训练集比例（`train` 命令用） |
| `forecast_horizon` | int | `5` | 预测步长 |
| `model_type` | str | `"lightgbm"` | 模型类型 |
| `lags` | int | `30` | 目标序列滞后阶数 |
| `lags_past_covariates` | int | `10` | 协变量滞后阶数 |
| `backtest_start` | float | `0.6` | 回测起始位置（60% 处开始） |
| `backtest_retrain` | bool | `True` | 回测时是否每步重训 |
| `output_dir` | Path | `stock_analysis/output` | 输出目录 |

### 自动生成的特征（`features.py`）

| 特征 | 说明 |
|------|------|
| `log_return` | 对数收益率（默认预测目标） |
| `ma5 / ma20 / ma60` | 移动平均线 |
| `volatility_20` | 20 日波动率 |
| `rsi_14` | 相对强弱指标 |
| `volume_ratio` | 成交量 / 5 日均量 |
| `hl_spread` | 高低价差率 |
| `oc_spread` | 开收价差率 |

---

## 8. 输出文件说明

### 8.1 目录结构

```
stock_analysis/output/
├── data/
│   └── AAPL.csv                 # 缓存的原始 OHLCV
├── models/
│   ├── AAPL_lightgbm.pkl        # 训练好的 Darts 模型
│   ├── AAPL_lightgbm.json       # 模型元信息
│   └── AAPL_pipeline.pkl        # 数据预处理 Pipeline
├── forecasts/
│   └── AAPL_20250705_132000.csv # 正式预测结果
└── backtests/
    ├── AAPL_20250705_131500.csv # 回测预测序列
    └── AAPL_20250705_131500.json # 回测指标 (MAE/RMSE/MAPE)
```

### 8.2 预测结果 CSV 示例

当 `target=log_return` 时：

| date | type | log_return | predicted_close |
|------|------|------------|-----------------|
| 2025-07-08 | forecast | 0.0012 | 195.32 |
| 2025-07-09 | forecast | -0.0008 | 195.16 |
| ... | ... | ... | ... |

- `log_return`：预测的对数收益率  
- `predicted_close`：由最后已知收盘价 + 累计收益率推算的预测价格  

---

## 9. 工作流程建议

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. fetch   │ ──▶ │ 2. backtest │ ──▶ │  3. train   │ ──▶ │ 4. predict  │
│  拉取数据    │     │  评估模型    │     │  保存模型    │     │  正式预测    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **fetch**：确认数据能正常拉取，检查 `output/data/` 缓存  
2. **backtest**：查看 MAE / RMSE / MAPE，对比不同 `--model`  
3. **train** 或 **predict**：`predict` 会自动用全量数据训练  
4. **predict --load**：日常更新时，若模型未过期可直接加载预测  

### 模型选型建议

| 场景 | 推荐模型 |
|------|----------|
| 快速验证 | `lightgbm`（默认） |
| 可解释线性基线 | `linear` |
| 更强非线性 | `xgboost` |
| 经典统计基线 | `arima` |

---

## 10. 扩展与修改

### 10.1 添加新特征

编辑 `stock_analysis/features.py` 的 `build_feature_frame()`：

```python
out["macd"] = ...  # 添加新指标
feature_cols.append("macd")  # 加入协变量列表
```

### 10.2 更换 / 新增模型

编辑 `stock_analysis/forecaster.py` 的 `_build_model()`，参考 Darts 文档接入 TFT、N-BEATS 等。

### 10.3 接入新数据源

在 `stock_analysis/data/loader.py` 中新增 `_load_xxx()` 函数，并在 `load_ohlcv()` 里增加 `market` 分支。

### 10.4 修改输出目录

```python
config = StockConfig(
    symbol="AAPL",
    output_dir="D:/my_output",
)
```

---

## 11. 常见问题

### Q1: `No data returned for symbol`

- 检查代码是否正确（美股大写，A 股 6 位数字）  
- 检查 `--start` / `--end` 日期范围  
- A 股需安装：`pip install akshare`  

### Q2: `Insufficient data (N rows). Need at least 60 trading days.`

- 扩大 `--start` 日期范围，或检查 CSV 是否过短  

### Q3: 回测很慢

- 回测默认 `retrain=True`，每步重新训练  
- 可在代码中设置 `config.backtest_retrain = False` 加速（精度略降）  

### Q4: 预测结果是否可直接用于交易？

**不建议。** 本模块仅提供统计预测，未考虑：

- 交易成本、滑点  
- 涨跌停、停牌  
- 风控与仓位管理  

请仅作研究与辅助参考，不构成投资建议。

### Q5: 如何定时自动预测？

Windows 任务计划程序或 cron，每日收盘后执行：

```powershell
cd "d:\VS\AI static"
python -m stock_analysis predict --symbol AAPL --load
```

---

## 附录：命令速查

```powershell
# 美股预测
python -m stock_analysis predict --symbol AAPL --market us --horizon 5

# A 股预测
python -m stock_analysis predict --symbol 600519 --market cn --horizon 5

# 本地 CSV
python -m stock_analysis predict --symbol TEST --market csv --csv-path "data.csv"

# 回测
python -m stock_analysis backtest --symbol AAPL --model lightgbm

# 仅下载数据
python -m stock_analysis fetch --symbol TSLA --start 2018-01-01
```
