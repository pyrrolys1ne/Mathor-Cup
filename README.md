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

```bash
# 1. 建立虚拟环境（Python 3.10）
python3.10 -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .

# 3. 将原始数据放入 data/raw/
cp /path/to/reference_case.xlsx data/raw/

# 4. 运行各题
python -m src.main --config configs/q1.yaml   # 问题1
python -m src.main --config configs/q2.yaml   # 问题2
python -m src.main --config configs/q3.yaml   # 问题3
python -m src.main --config configs/q4.yaml   # 问题4

# 5. 运行测试
pytest tests/ -v
```

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

1. **Kaiwu SDK**（主，量子/量子启发式）
2. **Simulated Annealing**（备，经典）
3. **Hybrid Large-Scale**（Q3/Q4，聚类分解 + QUBO + 局部修复）

## Dependencies

- Python 3.10
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn, networkx
- pyyaml, click, rich
- kaiwu SDK（可选，需单独安装）
