# 题目与工程实现对应说明

## 项目目标

本项目实现 MathorCup 2026 A 题四个问题的统一求解流程，覆盖。

- 数据加载与校验
- QUBO 或 Ising 导出
- 本地求解
- 平台回填解码
- 指标评估与可视化

统一入口为 src.main，统一配置为 configs 下的 yaml 文件。

## 问题一

- 任务：单车路径优化
- 约束：每个客户访问一次并回仓
- 目标：最小总旅行时间
- 支持后端：sa 与 kaiwu

输出结果按后端区分。

- q1_result_sa.csv 或 q1_result_kaiwu.csv

## 问题二

- 任务：单车路径含时间窗惩罚
- 目标：最小旅行时间加时间窗惩罚
- 惩罚函数：二次型早到晚到惩罚

输出结果按后端区分。

- q2_result_sa.csv 或 q2_result_kaiwu.csv

## 问题三

- 任务：50 客户规模的分解混合求解
- 方法：聚类 分解 子问题求解 拼接 局部改进
- 支持从平台分解日志回填重构全局路径

输出。

- q3_result_kaiwu.csv
- q3_result_cpqc550.csv

## 问题四

- 任务：多车辆带容量与时间窗惩罚
- 支持模式：词典序与加权
- 支持敏感性分析扫描车辆数
- 支持平台多批目录回填与自动选优

常规求解输出。

- q4_vehicles_hybrid.csv 或 q4_vehicles_kaiwu.csv
- q4_result_hybrid.csv 或 q4_result_kaiwu.csv

平台回填输出。

- q4_vehicles_cpqc550.csv
- q4_result_cpqc550.csv

## 平台交互流程

### 导出

- q1 q2 导出单个矩阵和 meta
- q3 q4 导出多子问题与 manifest
- q4 支持车辆数驱动导出 sweep

### 回填

- 读取平台日志中的 solutionVector
- 自动处理 Ising 与 split 还原
- q4 支持 q4_vXX_kYY 目录批量处理
- 选优遵循当前 optimization_mode

## 数据与配置

- 原始数据默认来自 data/raw
- 可缓存到 data/processed
- 日志 图表 结果统一输出到 outputs

核心配置分组。

- solver 控制后端
- sa 与 kaiwu 控制参数
- qubo 与 qubo_export 控制导出
- vehicle objective time_window 控制问题四目标
