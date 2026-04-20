# QUBO 推导与工程实现说明

## 变量编码

对单路径子问题采用排列位置编码。

$$
x_{i,p} \in \{0,1\}
$$

其中 $x_{i,p}=1$ 表示节点 $i$ 位于路径位置 $p$。

## 基础目标

基础旅行时间项写为。

$$
H_{travel}=\sum_{i,j}\sum_p T_{ij} x_{i,p}x_{j,p+1}
$$

索引换行按环路处理。

## 硬约束惩罚

### 节点唯一访问

$$
H_{visit}=A\sum_i\left(1-\sum_p x_{i,p}\right)^2
$$

### 位置唯一占用

$$
H_{position}=B\sum_p\left(1-\sum_i x_{i,p}\right)^2
$$

总能量为。

$$
H = H_{travel}+H_{visit}+H_{position}
$$

## 问题二与时间窗

当前工程实现将时间窗惩罚放在解码评估阶段，不直接写入 QUBO。

评估目标为。

$$
Obj = Travel + \sum_i penalty_i
$$

## 问题三与问题四分解

- q3 对客户聚类后逐簇构造问题一形式 QUBO
- q4 对每辆车子问题构造问题一形式 QUBO
- 容量约束主要由分组阶段保证

## 矩阵导出格式

标准导出形式为上三角 QUBO 矩阵，目标函数写为。

$$
\min x^T Q x
$$

工程中同时保存。

- 原始浮点矩阵
- 适配后整型矩阵
- meta 信息

## 精度适配策略

支持四种方法。

- none 不做精度适配
- truncate 直接精度调整
- mutate 动态范围变异
- split 变量拆分

当 output_model 为 ising 时，矩阵会投影为。

- 对称矩阵
- 对角元素为零

## mutate 兼容路径

在部分 SDK 缺失 qubo 侧 mutate 接口时，工程支持以下兼容路径。

1. QUBO 转 Ising 辅助位表示
2. 对零对角 Ising 做 mutate
3. 再转回 QUBO

该路径会在 meta 中记录 ising_aux_index，供回填解码时还原。

## 回填解码还原

回填阶段按以下顺序规范化向量。

1. 处理 Ising 辅助位
2. 必要时处理 split 还原
3. Ising 自旋转二进制
4. 按阈值二值化

该流程确保平台回传向量可与本地 QUBO 解码器对接。
