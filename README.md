# MathorCup 2026 Problem A

基于量子计算的智慧物流优化建模与算法设计

## 一 目录与职责对应

- README.md
  - 项目使用说明
- project_report.md
  - 项目报告参考稿
- requirements.txt
  - 依赖清单
- pyproject.toml
  - 打包与项目配置
- configs
  - q1.yaml 对应第一题配置
  - q2.yaml 对应第二题配置
  - q3.yaml 对应第三题配置
  - q4.yaml 对应第四题配置
- data
  - raw 存放原始数据
  - processed 存放预处理缓存
  - platform_feedback 存放平台回填文件
- docs
  - problem_statement.md 题目整理
  - modeling_assumptions.md 建模假设
  - qubo_derivation.md 推导说明
  - experiment_protocol.md 实验流程
- src
  - main.py 命令入口与流程编排
  - io 数据读取与校验
  - core 图模型 时间窗 容量
  - qubo QUBO 构造
  - solvers 求解器实现
  - algorithms 聚类 解码 局部搜索 分配
  - eval 评估与敏感性分析
  - viz 绘图模块
- outputs
  - logs 日志
  - figures 图像结果
  - results 结果表
  - qubo_ising 导出矩阵与清单
  - prescreen 粗筛结果
- tests
  - 核心模块测试

## 二 环境准备

### Windows PowerShell

```powershell
cd "c:\Users\ayaka\Desktop\Mathor Cup\MathorCup_A"

C:\Users\ayaka\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv --system-site-packages
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip install -e .
```

### 数据放置

- 将原始数据放入 data/raw/reference_case.xlsx

### 鉴权变量

```powershell
$env:KAIWU_USER_ID="<your_user_id>"
$env:KAIWU_SDK_CODE="<your_sdk_code>"
```

## 三 运行命令与目录对应

### 第一题求解

```powershell
python -m src.main --config configs/q1.yaml
```

- 结果写入 outputs/results/q1_result_*.csv
- 图像写入 outputs/figures/q1_route_*.png

### 第二题求解

```powershell
python -m src.main --config configs/q2.yaml
```

- 结果写入 outputs/results/q2_result_*.csv
- 图像写入 outputs/figures/q2_route_*.png

### 第三题求解

```powershell
python -m src.main --config configs/q3.yaml
```

- 结果写入 outputs/results/q3_result_*.csv
- 图像写入 outputs/figures/q3_route_*.png

### 第四题求解

```powershell
python -m src.main --config configs/q4.yaml
```

- 结果写入 outputs/results/q4_result_*.csv
- 过程写入 outputs/results/q4_vehicles_*.csv
- 图像写入 outputs/figures/q4_routes_*.png 与 q4_cost_breakdown_*.png

## 四 导出上机矩阵

```powershell
python -m src.main --config configs/q1.yaml --phase export
python -m src.main --config configs/q2.yaml --phase export
python -m src.main --config configs/q3.yaml --phase export
python -m src.main --config configs/q4.yaml --phase export
```

- 第一题与第二题导出到 outputs/qubo_ising
- 第三题导出 q3_export_manifest.json 与分簇矩阵
- 第四题导出 q4_export_manifest_vXX.json 与总清单 q4_export_manifest_vehicle_sweep.json

## 五 平台回填

### 单题回填

```powershell
python -m src.main --config configs/q1.yaml --solution data/platform_feedback/q1_run_01.log
python -m src.main --config configs/q2.yaml --solution data/platform_feedback/q2_run_01.log
python -m src.main --config configs/q3.yaml --solution data/platform_feedback
```

### 第四题批量回填

```powershell
python -m src.main --config configs/q4.yaml --solution data/platform_feedback
```

回填规则如下

- 自动扫描 data/platform_feedback 下 q4_vXX_kYY 子目录
- 每个子目录按编号文件顺序回填
- 每批结果追加写入 outputs/results/q4_vehicles_cpqc550.csv
- 最优批次写入 outputs/results/q4_result_cpqc550.csv
- 若配置目录无清单则自动回退到 outputs/qubo_ising

## 六 测试

```powershell
pytest tests -v
```

## 七 常见问题

- 依赖安装后仍不可用
  - 检查是否在同一个虚拟环境执行安装与运行
- 回填提示清单缺失
  - 先执行第四题导出命令
  - 检查 outputs/qubo_ising 下是否存在 q4_export_manifest.json
- 回填提示子问题文件不足
  - 检查对应 q4_vXX_kYY 目录中文件数量是否与该批车辆数一致

## 八 结果文件速查

- 第一题结果 outputs/results/q1_result_*.csv
- 第二题结果 outputs/results/q2_result_*.csv
- 第三题结果 outputs/results/q3_result_*.csv
- 第四题最优结果 outputs/results/q4_result_*.csv
- 第四题批次过程 outputs/results/q4_vehicles_*.csv
- 导出矩阵与清单 outputs/qubo_ising
- 图像输出 outputs/figures
- 运行日志 outputs/logs
