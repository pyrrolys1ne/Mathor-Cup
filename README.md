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
│  └─ processed/            # 预处理缓存（.pkl）
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
│  └─ tables/
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

# 5) （可选）配置 Kaiwu 鉴权信息
$env:KAIWU_USER_ID="<your_user_id>"
$env:KAIWU_SDK_CODE="<your_sdk_code>"

# 6) 运行各题
python -m src.main --config configs/q1.yaml   # 问题1
python -m src.main --config configs/q2.yaml   # 问题2
python -m src.main --config configs/q3.yaml   # 问题3
python -m src.main --config configs/q4.yaml   # 问题4

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
5. 输出位置：日志、图、表分别写入 `outputs/logs/`、`outputs/figures/`、`outputs/tables/`。

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
