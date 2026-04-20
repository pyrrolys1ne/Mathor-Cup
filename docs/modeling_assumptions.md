# 建模假设与实现口径

## 通用口径

- 配送中心节点编号固定为 0
- 客户节点编号为 1 到 N
- 路径默认从配送中心出发并回到配送中心
- 旅行时间直接采用输入矩阵，不强制对称
- 到达即服务，不设置等待机制

## 时间窗与惩罚

节点 i 的惩罚定义为。

$$
penalty_i = \alpha \cdot \max(0, e_i - t_i)^2 + \beta \cdot \max(0, t_i - l_i)^2
$$

实现默认值。

- $\alpha = 10$
- $\beta = 20$

相关参数由 time_window.alpha 与 time_window.beta 控制。

## 容量约束口径

问题四中每车负载满足。

$$
\sum_{i \in route_k} d_i \le Q
$$

容量来源优先级。

1. data.raw_excel 中可识别容量列
2. vehicle.capacity

## q4 优化目标口径

### 词典序模式

vehicle.optimization_mode 为 lexicographic 时采用两级比较键。

1. 车辆数最小
2. 在车辆数相同前提下比较总旅行时间
3. 若仍相同再比较总惩罚

实现键为 $(n_vehicles, total_travel, total_penalty)$。

### 加权模式

vehicle.optimization_mode 为 weighted 时，比较分数。

$$
score = w_v \cdot n_vehicles + w_t \cdot total_travel + w_p \cdot total_penalty
$$

权重来源。

- $w_v$ 对应 objective.alpha
- $w_t$ 对应 objective.beta
- $w_p$ 对应 objective.gamma

## q3 q4 分解假设

- 先聚类再分组，子问题分别求解
- 子问题路径可做 two_opt 局部改进
- q3 采用拼接策略合并子路径
- q4 先分配客户到车辆再逐车解码

## 回填解码口径

- 平台日志优先解析 solutionVector
- 支持多候选向量，按子问题目标选择最优候选
- 支持 Ising 自旋向量自动转二进制
- 若为 split 适配，可按 meta 还原变量维度
- q4 支持多批目录，失败批次跳过

## 精度适配口径

precision_method 支持 none truncate mutate split。

- output_model 为 ising 时矩阵会标准化为对称且对角为零
- mutate 在部分 SDK 缺失时可走 QUBO 转 Ising 辅助位再回转路径
- 最终导出矩阵统一量化为整型
