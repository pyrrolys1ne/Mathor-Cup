# 实验方案 / Experiment Protocol

## 环境要求

| 项目 | 规格 |
|------|------|
| Python | 3.10.x |
| 操作系统 | Windows 10/11、Ubuntu 22.04、macOS 13+ |
| 内存 | ≥ 8 GB |
| CPU | ≥ 4 核（混合算法建议 8 核） |
| Kaiwu SDK | 可选，需单独申请（部分环境无法直接 `pip install kaiwu`） |

---

## 可复现性保证

- 所有随机过程（SA 初始解、聚类、扰动）使用固定 `seed`（默认 42）；
- 依赖版本锁定在 `requirements.txt`；
- 预处理结果缓存到 `data/processed/`，保证数据一致性；
- 实验配置文件（`configs/*.yaml`）与代码版本同步提交。

---

## 运行前检查（强烈建议）

- 安装依赖与运行命令使用同一个解释器（建议统一 `.venv`）；
- Windows + PowerShell 若无法激活脚本，可先执行：
	`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`；
- 若 Kaiwu 已安装在系统 Python 而非 venv，可创建 venv 时使用：
	`python -m venv .venv --system-site-packages`；
- Kaiwu 鉴权优先通过环境变量注入：
	`KAIWU_USER_ID`、`KAIWU_SDK_CODE`。

---

## 实验流程

### Phase 1：数据预处理

```bash
python -m src.main --config configs/q1.yaml --phase data
```

输出：
- `data/processed/nodes.pkl`：节点属性 DataFrame
- `data/processed/travel_time.pkl`：旅行时间矩阵 ndarray
- `outputs/logs/q1_run.log`：数据校验与运行日志

说明：
- 当原始数据是 50 客户、但 `q1.yaml`/`q2.yaml` 中 `num_customers=15` 时，程序会自动截取前 15 个客户参与求解，避免口径混乱。

---

### Phase 2：问题1 实验

```bash
python -m src.main --config configs/q1.yaml
```

输出：
- `outputs/tables/q1_result.csv`：路径方案 + 目标值
- `outputs/figures/q1_route.png`：路径可视化
- `outputs/logs/q1_run.log`：求解日志

说明：
- `solver.backend=kaiwu` 且 Kaiwu 不可用时，程序会自动回退到 SA。

---

### Phase 3：问题2 实验

```bash
python -m src.main --config configs/q2.yaml
```

附加输出：
- 每个客户的时间窗违反量与惩罚值
- `outputs/tables/q2_result.csv`（包含每客户到达时刻、早到/晚到违反量、惩罚）

---

### Phase 4：问题3 实验

```bash
python -m src.main --config configs/q3.yaml
```

附加输出：
- 聚类分组可视化 `outputs/figures/q3_clusters.png`
- 全局路径结果 `outputs/tables/q3_result.csv`

---

### Phase 5：问题4 实验

```bash
# 词典序优化
python -m src.main --config configs/q4.yaml

# 敏感性分析（车辆数扫描）
python -m src.main --config configs/q4.yaml --sensitivity
```

附加输出：
- `outputs/figures/q4_sensitivity.png`：车辆数敏感性曲线
- `outputs/tables/q4_sensitivity.csv`：详细数据

---

## 评估指标

| 指标 | 说明 |
|------|------|
| 总旅行时间 | 所有车辆行驶时间之和 |
| 时间窗惩罚总量 | $\sum_i \text{penalty}_i$ |
| 综合目标值 | 旅行时间 + 时间窗惩罚 |
| 车辆数 | 实际使用车辆数（Q4） |
| 每客户服务开始时刻 | $t_i$（当前实现不等待，达到即服务） |
| 每车负载率 | $\sum d_i / Q$（Q4） |

---

## 自检清单

- [ ] `data/raw/reference_case.xlsx` 存在且可读
- [ ] Sheet1 行数 = 51（含配送中心）
- [ ] Sheet2 矩阵维度 = 51×51
- [ ] 对角线元素全为 0
- [ ] 所有旅行时间非负
- [ ] 求解器输出路径覆盖所有客户节点
- [ ] 路径从配送中心出发并返回配送中心
- [ ] 时间窗惩罚计算正确（对照手算验证）
- [ ] 图表文件生成成功
