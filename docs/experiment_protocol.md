# 实验方案 / Experiment Protocol

## 环境要求

| 项目 | 规格 |
|------|------|
| Python | 3.10.x |
| 操作系统 | Ubuntu 22.04 / macOS 13+ |
| 内存 | ≥ 8 GB |
| CPU | ≥ 4 核（混合算法建议 8 核） |
| Kaiwu SDK | 可选，需单独申请 |

---

## 可复现性保证

- 所有随机过程（SA 初始解、聚类、扰动）使用固定 `seed`（默认 42）；
- 依赖版本锁定在 `requirements.txt`；
- 预处理结果缓存到 `data/processed/`，保证数据一致性；
- 实验配置文件（`configs/*.yaml`）与代码版本同步提交。

---

## 实验流程

### Phase 1：数据预处理

```bash
python -m src.main --config configs/q1.yaml --phase data
```

输出：
- `data/processed/nodes.pkl`：节点属性 DataFrame
- `data/processed/travel_time.pkl`：旅行时间矩阵 ndarray
- `outputs/logs/data_validation.log`：数据校验报告

---

### Phase 2：问题1 实验

```bash
python -m src.main --config configs/q1.yaml
```

输出：
- `outputs/tables/q1_result.csv`：路径方案 + 目标值
- `outputs/figures/q1_route.png`：路径可视化
- `outputs/logs/q1_run.log`：求解日志

---

### Phase 3：问题2 实验

```bash
python -m src.main --config configs/q2.yaml
```

附加输出：
- 每个客户的时间窗违反量与惩罚值
- `outputs/tables/q2_tw_violations.csv`

---

### Phase 4：问题3 实验

```bash
python -m src.main --config configs/q3.yaml
```

附加输出：
- 聚类分组可视化 `outputs/figures/q3_clusters.png`
- 子路径与全局路径对比

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
| 每客户服务开始时刻 | $t_i$（含等待） |
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
