# MathorCup 2026 — Problem A
**基于量子计算的智慧物流优化建模与算法设计**

Quantum Computing-Based Smart Logistics Optimization Modeling and Algorithm Design

---

## Project Structure

```
.
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ data/
│  ├─ raw/                  # 原始 Excel 数据
│  ├─ processed/            # 预处理缓存（.pkl）
│  └─ platform_feedback/    # 上机平台返回结果（log/json/csv）
├─ configs/
│  ├─ q1.yaml               # 问题1 实验配置
│  ├─ q2.yaml               # 问题2 实验配置
│  ├─ q3.yaml               # 问题3 实验配置
│  └─ q4.yaml               # 问题4 实验配置
├─ docs/
│  ├─ problem_statement.md
│  ├─ modeling_assumptions.md
│  ├─ qubo_derivation.md
│  └─ experiment_protocol.md
├─ src/
│  ├─ main.py               # CLI 入口
│  ├─ io/                   # 数据读取与校验
│  ├─ core/                 # 核心模型（图、时间窗、容量）
│  ├─ qubo/                 # QUBO 构造
│  ├─ solvers/              # 求解后端
│  ├─ algorithms/           # 聚类、解码、局部搜索
│  ├─ eval/                 # 评估指标与敏感性分析
│  └─ viz/                  # 可视化
├─ outputs/
│  ├─ logs/
│  ├─ figures/
│  ├─ results/              # 归档后的结果表（csv）
│  ├─ qubo_ising/           # 归档后的QUBO/Ising矩阵与meta
│  └─ prescreen/            # 归档后的粗筛结果
└─ tests/
```

## Quick Start

### Windows (PowerShell)

```powershell
# 1) 进入项目目录
cd "c:\Users\ayaka\Desktop\Mathor Cup\MathorCup_A"

# 2) 创建并激活虚拟环境（Python 3.10）
# 如果 Kaiwu 只安装在系统 Python，可使用 --system-site-packages 继承它
C:\Users\ayaka\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv --system-site-packages
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3) 安装依赖
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip install -e .

# 4) 放置原始数据
# data/raw/reference_case.xlsx

# 5) 配置 Kaiwu 鉴权信息
$env:KAIWU_USER_ID="<your_user_id>"
$env:KAIWU_SDK_CODE="<your_sdk_code>"

# 6) 运行各题
python -m src.main --config configs/q1.yaml   # 问题1
python -m src.main --config configs/q2.yaml   # 问题2
python -m src.main --config configs/q3.yaml   # 问题3
python -m src.main --config configs/q4.yaml   # 问题4

# 6.1) Q1/Q2/Q3/Q4 导出 QUBO（用于上机平台）
python -m src.main --config configs/q1.yaml --phase export   # outputs/qubo_ising/q1_qubo.csv 或 q1_ising.csv（取决于output_model）
python -m src.main --config configs/q2.yaml --phase export   # outputs/qubo_ising/q2_qubo.csv
python -m src.main --config configs/q3.yaml --phase export   # outputs/qubo_ising/q3_export_manifest.json（分簇子问题矩阵）
python -m src.main --config configs/q4.yaml --phase export   # outputs/qubo_ising/q4_export_manifest.json（分车子问题矩阵）

# 6.2) 回填上机结果并生成评估与图（支持 txt/csv 位向量、平台 JSON 日志）
python -m src.main --config configs/q1.yaml --solution data/platform_feedback/q1_run_01.log
python -m src.main --config configs/q2.yaml --solution data/platform_feedback/q2_run_01.log
python -m src.main --config configs/q3.yaml --solution data/platform_feedback      # Q3 单目录候选池：子问题1用1&6，2用2&7...
python -m src.main --config configs/q4.yaml --solution data/platform_feedback      # 读取 q4_run_*.log 并按 q4 manifest 回填

# 也支持本地位向量文件
python -m src.main --config configs/q1.yaml --solution outputs/qubo_ising/q1_solution.txt
python -m src.main --config configs/q2.yaml --solution outputs/qubo_ising/q2_solution.txt

# 7) 运行测试
pytest tests/ -v
```

### Linux / macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m src.main --config configs/q1.yaml
```

## Runtime Notes

1. 解释器一致性：请确保安装依赖和运行命令使用同一个 Python/venv（建议统一使用 `.venv`）。
2. Kaiwu 依赖：部分环境下 `pip install kaiwu` 可能不可用，若系统 Python 已安装 Kaiwu，建议创建 venv 时使用 `--system-site-packages`。
3. 鉴权配置：优先通过环境变量 `KAIWU_USER_ID` / `KAIWU_SDK_CODE` 注入，避免把密钥写入配置文件。
4. 数据规模对齐：当原始表是 50 客户而 `q1.yaml`/`q2.yaml` 配置 `num_customers=15` 时，程序会自动截取前 15 个客户参与求解，避免口径混乱。
5. 输出位置（默认）：日志、图、结果分别写入 `outputs/logs/`、`outputs/figures/`、`outputs/results/`。
	输出归档（推荐）：
	- 结果表（csv）归档到 `outputs/results/`
	- QUBO/Ising 矩阵与 meta 归档到 `outputs/qubo_ising/`
	- 粗筛相关文件归档到 `outputs/prescreen/`
6. 上机流程：先 `--phase export` 导出 QUBO，再将平台返回解通过 `--solution` 回填解码评估（当前回填支持 Q1/Q2/Q3/Q4）。
	- 建议将平台返回文件统一放在 `data/platform_feedback/` 目录。
	- 平台 `.log` 若为 JSON 列表（含 `quboValue` 和 `solutionVector`），程序可直接解析并自动选择候选解。
	- Q3 统一使用 `data/platform_feedback/` 单目录，不再区分 `q3_run_01/`、`q3_run_02/` 子目录。
	- 候选池按编号自动配对：子问题1使用 `q3_run_01` 和 `q3_run_06`，子问题2使用 `q3_run_02` 和 `q3_run_07`，以此类推。
	- 推荐命名：`q3_run_01.log` ... `q3_run_10.log`（也支持 json/txt/csv）。
	- Q4 回填建议将分车子问题反馈命名为 `q4_run_01.log`、`q4_run_02.log`...，并放在 `data/platform_feedback/` 目录。
7. 结果汇总：`q1_result_*.csv` / `q2_result_*.csv` / `q3_result_*.csv` 文件末尾会追加汇总块（Route、Travel time、TW penalty、Objective、Customers served）。
	- Q4 输出采用来源后缀命名：本地求解为 `q4_result_hybrid.csv` 或 `q4_result_kaiwu.csv`，回填为 `q4_result_cpqc550.csv`。
8. 8bit 适配：导出时会根据 `qubo_export.precision_method` 自动做精度适配，默认 `truncate`。
9. 导出产物：
	- `*_qubo_raw.csv`：原始QUBO矩阵
	- `*_qubo.csv`：8bit适配后的QUBO上机矩阵（整数，QUBO模式）
	- `*_ising.csv`：8bit适配后的Ising上机矩阵（整数，Ising模式）
	- `*_qubo_meta.json` 或 `*_ising_meta.json`：导出元数据（含变量规模、方法；split模式下含恢复信息；Ising模式可含辅助位信息）
	- `q3_export_manifest.json`：Q3 分簇导出清单（列出每个 cluster 的 raw/adapted/meta 文件）
	- `q4_export_manifest.json`：Q4 分车导出清单（列出每个 vehicle 子问题的 raw/adapted/meta 文件）

## Problems

| 题号 | 描述 | 节点数 | 特殊约束 |
|------|------|--------|----------|
| Q1 | 单车无时间窗无容量 | 15 | — |
| Q2 | 单车含时间窗惩罚无容量 | 15 | 时间窗惩罚 |
| Q3 | 单车大规模 | 50 | 分解+混合求解 |
| Q4 | 多车含时间窗+容量 | 50 | 多车、容量、车辆数敏感性 |

## Penalty Formula

```
penalty_i = 10 * max(0, e_i - t_i)^2 + 20 * max(0, t_i - l_i)^2
```

## Solver Backends

1. **Kaiwu SDK**
2. **Simulated Annealing**
3. **Hybrid Large-Scale**（Q3/Q4，聚类分解 + QUBO + 局部修复）

## Dependencies

- Python 3.10
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn, networkx
- pyyaml, click, rich
- kaiwu SDK
