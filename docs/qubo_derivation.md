# QUBO 推导说明 / QUBO Derivation

## 1. TSP → QUBO（问题1/2/3基础）

### 1.1 变量定义

定义二值变量矩阵：

$$
x_{i,p} \in \{0, 1\}, \quad i \in \{0, 1, \ldots, N\}, \quad p \in \{0, 1, \ldots, N\}
$$

其中 $x_{i,p} = 1$ 表示节点 $i$ 在路径中的第 $p$ 个位置被访问。

变量总数：$(N+1)^2$（含配送中心）。

---

### 1.2 约束一：每个节点恰好访问一次

$$
H_A = A \sum_{i=0}^{N} \left(1 - \sum_{p=0}^{N} x_{i,p}\right)^2
$$

展开：

$$
H_A = A \sum_i \left(1 - 2\sum_p x_{i,p} + \sum_p x_{i,p}^2 + 2\sum_{p < q} x_{i,p} x_{i,q}\right)
$$

---

### 1.3 约束二：每个位置恰好一个节点

$$
H_B = B \sum_{p=0}^{N} \left(1 - \sum_{i=0}^{N} x_{i,p}\right)^2
$$

---

### 1.4 目标：最小化总旅行时间

$$
H_C = \sum_{i=0}^{N} \sum_{j=0}^{N} T_{ij} \sum_{p=0}^{N} x_{i,p} \cdot x_{j,(p+1) \bmod (N+1)}
$$

---

### 1.5 完整 QUBO（问题1）

$$
H = H_A + H_B + H_C
$$

**惩罚系数选择原则**：

$$
A, B \ge \max_{i,j} T_{ij} \times (N+1)
$$

即惩罚系数需大于目标函数的最大可能值，保证可行解严格优于任何违约解。

---

## 2. 时间窗惩罚嵌入（问题2）

到达时刻 $t_i$ 依赖路径顺序，是路径变量的非线性函数，难以直接编入 QUBO。

### 处理方案

**方案A（软约束近似）**：将时间窗惩罚作为额外目标项，在解码阶段计算，不嵌入QUBO。QUBO 仅优化纯旅行时间，解码后对路径评分加入时间窗惩罚。

**方案B（线性化近似）**：对时间窗违反量引入辅助变量，但变量数大幅增加，不推荐用于大规模问题。

**本项目采用方案A**：

$$
H_{\text{Q2}} = H_A + H_B + H_C
$$

解码后目标值：

$$
\text{Obj} = \text{TravelTime} + \sum_i \text{penalty}_i
$$

---

## 3. 容量约束嵌入（问题4）

### 3.1 多车变量扩展

引入车辆维度：$x_{k,i,p} = 1$ 表示车辆 $k$ 在位置 $p$ 访问节点 $i$。

变量数：$K \times (N+1)^2$，规模较大，通常采用分解策略。

### 3.2 容量软约束

$$
H_D = D \sum_k \max\left(0, \sum_i d_i \cdot \mathbb{1}[i \in \text{route}_k] - Q\right)^2
$$

实践中容量约束通过聚类阶段（预先保证每簇需求 $\le Q$）来满足，不全部嵌入 QUBO。

---

## 4. QUBO 矩阵构造

QUBO 标准形式：

$$
\min \mathbf{x}^T Q \mathbf{x}, \quad \mathbf{x} \in \{0,1\}^n
$$

变量向量：将 $x_{i,p}$ 拉平为一维向量，索引 $k = i \times (N+1) + p$。

矩阵填充规则：
- 对角元 $Q_{kk}$：来自一次项系数（$x_{i,p}^2 = x_{i,p}$，合并到线性项）；
- 非对角元 $Q_{kl}$（$k < l$）：来自二次交叉项系数（上三角存储）。

---

## 5. 变量数规模估计

| 问题 | 节点数 N | QUBO 变量数 | 备注 |
|------|---------|------------|------|
| Q1/Q2 | 15 | 256 | $(15+1)^2$ |
| Q3 | 50 | 2601 | $(50+1)^2$，需分解 |
| Q4 | 50 × K | ~2601×K | 分解后每子问题≤256 |

典型量子退火设备（D-Wave 等）可处理数千变量，Kaiwu SDK 支持范围以实际 API 为准。

---

## 参考文献

1. Lucas, A. (2014). *Ising formulations of many NP problems*. Frontiers in Physics.
2. Glover, F., Kochenberger, G., & Du, Y. (2019). *Quantum Bridge Analytics I: A Tutorial on Formulating and Using QUBO Models*. 4OR.
