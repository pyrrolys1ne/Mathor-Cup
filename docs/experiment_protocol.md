# 实验流程说明

## 运行环境

- Python 3.10
- 依赖见 requirements.txt
- 可选 Kaiwu SDK

## 统一入口

全部实验通过以下入口运行。

```bash
python -m src.main --config configs/q1.yaml
```

可用参数。

- --phase data 只执行数据阶段
- --phase export 只导出 QUBO 或 Ising
- --phase solve 或 all 执行求解
- --sensitivity 仅对 q4 生效
- --solution 用平台回填结果解码

## 数据阶段

```bash
python -m src.main --config configs/q1.yaml --phase data
```

行为与输出。

- 从 data.raw_excel 读取原始数据
- 可缓存到 data.processed_dir
- 当配置 num_customers 小于原始客户数时自动截断
- 记录日志到 output.log_dir

## 常规求解阶段

### 问题一

```bash
python -m src.main --config configs/q1.yaml
```

输出文件。

- results q1_result_sa.csv 或 q1_result_kaiwu.csv
- figures q1_route_sa.png 或 q1_route_kaiwu.png

### 问题二

```bash
python -m src.main --config configs/q2.yaml
```

输出文件。

- results q2_result_sa.csv 或 q2_result_kaiwu.csv
- figures q2_route_sa.png 或 q2_route_kaiwu.png

### 问题三

```bash
python -m src.main --config configs/q3.yaml
```

输出文件。

- results q3_result_kaiwu.csv
- figures q3_route_kaiwu.png
- figures q3_clusters_kaiwu.png

说明。

- q3 内部根据 solver.backend 与 hybrid.sub_solver 选择子求解器
- 结果文件名当前固定使用 kaiwu 后缀

### 问题四

```bash
python -m src.main --config configs/q4.yaml
```

输出文件。

- results q4_vehicles_hybrid.csv 或 q4_vehicles_kaiwu.csv
- results q4_result_hybrid.csv 或 q4_result_kaiwu.csv
- figures q4_routes_hybrid.png 或 q4_routes_kaiwu.png
- figures q4_cost_breakdown_hybrid.png 或 q4_cost_breakdown_kaiwu.png

说明。

- vehicle.optimization_mode 支持 lexicographic 与 weighted
- 容量优先读取 Excel，缺失时回退 vehicle.capacity

## q4 敏感性分析

```bash
python -m src.main --config configs/q4.yaml --sensitivity
```

输出文件。

- prescreen q4_sensitivity.csv
- figures q4_sensitivity.png

## 导出阶段

```bash
python -m src.main --config configs/q4.yaml --phase export
```

统一规则。

- 输出目录由 output.qubo_dir 控制
- 导出原始矩阵与适配后矩阵
- 输出元信息 meta json

q1 q2 输出单文件。

- q1_qubo.csv 或 q1_ising.csv
- q2_qubo.csv 或 q2_ising.csv

q3 q4 输出分解清单。

- q3_export_manifest.json
- q4_export_manifest_kXX.json 或 q4_export_manifest_vXX.json
- q4_export_manifest_vehicle_sweep.json
- q4_export_manifest.json 作为最新别名

## 回填解码阶段

### q1 q2 回填

```bash
python -m src.main --config configs/q1.yaml --solution path_to_log_or_vector
python -m src.main --config configs/q2.yaml --solution path_to_log_or_vector
```

输出文件。

- q1_result_cpqc550.csv
- q2_result_cpqc550.csv

### q3 回填

```bash
python -m src.main --config configs/q3.yaml --solution data/platform_feedback
```

规则。

- 读取 q3_export_manifest.json
- 支持目录内多文件配对
- 支持单文件复用全部子问题

输出文件。

- q3_result_cpqc550.csv
- q3_route_cpqc550.png
- q3_clusters_cpqc550.png

### q4 回填

```bash
python -m src.main --config configs/q4.yaml --solution data/platform_feedback
```

规则。

- 支持目录 q4_vXX_kYY 批量回填
- 支持 q4_run_*.log 与 q4_vXX_kYY_*.log
- 支持 manifest 回退到 outputs/qubo_ising
- 支持 meta 文件按同名重定位
- 批次失败自动跳过
- 成功批次按 optimization_mode 选最优

输出文件。

- q4_vehicles_cpqc550.csv
- q4_result_cpqc550.csv
- q4_routes_cpqc550.png
- q4_cost_breakdown_cpqc550.png

## 复现实验建议

- 固定 sa.seed 与 hybrid.seed
- 保持导出与回填使用同一批 manifest
- 回填目录按 q4_vXX_kYY 命名可减少歧义
