# 理论与代码映射

> 入口整理（2026-10-08）：本文所述旧脚本已移入 `scripts/archive/legacy_20261008/`，
> 历史数值与用途保留；日常运行请看 [三个半导体主程序](entrypoints_zh.md)。

> 2026-10-08校对提示：本文保留早期教程/推导和历史结果；最新修复、可信度与设计评价以
> [本轮总审计](research_design_audit_20261008_zh.md)为准。旧“最佳”、精确阈值、单模及易加工
> 判断未经最新收敛/工艺验证时不得直接引用，能带是近Γ局域片段，远场能量对评估视窗归一化。

## 1. 四个基本波

方形晶格第二阶 Gamma 点附近只显式保留四个慢变包络：

```text
Phi = (Rx, Sx, Ry, Sy)^T
```

它们分别沿 `+x, -x, +y, -y` 传播。高阶波和垂直辐射波被消元后进入复耦合矩阵 `C`。
时域方程采用 Inoue 2019 式 (8) 的形式：

```text
dPhi/dt = vg[-i delta + (g-alpha_in)/2]Phi
           - vg * directional_gradient(Phi)
           - gamma Phi + i vg C Phi + f_sp
```

原版式(8)、(A6)–(A9)漏印耦合项的`i`；这不是看图时字体太小。
[2019官方勘误](https://doi.org/10.1103/PhysRevB.99.169904)明确修正为`+i vg C Phi`。
结合稳态本征式 `(delta+i alpha)V=CV` 与式(9)的能量关系，若 `lambda(C)=delta+i alpha_rad`，其时间演化为
`exp[(i delta-alpha_rad)vg t]`，即 `Im(C)>0` 产生辐射衰减。

`solver.py` 用 Strang splitting：局部 4x4 矩阵指数、二阶 Lax-Wendroff 定向传播、再做半步局部
演化。默认 `CFL=0.8`，配置检查强制要求 `0<CFL<=1`。

折射率的时间项按附录式 (A10) 的有源层均匀扰动近似实现为
`gamma = 2 Gamma_active (dn/dN)(dN/dt)/n_active`；它的量纲为 `s^-1`。
这里分母是发生折射率变化的有源层折射率，不是整模的有效折射率。

## 2. 载流子

载流子方程对应论文式 (10)：

```text
dN/dt = J/(e d_active) - N/tau_c - vg g_active(N) U + D laplacian(N)
```

边界使用零法向扩散通量。电极轮廓是 300 um 方窗与标准差 25 um 的高斯核卷积，再归一化使
二维积分等于总电流。增益使用论文式 (11)：

```text
g_active(N) = gmax (N-Ntr) / [N + (gmax/-g0)Ntr]
```

场方程中的模增益按附录式 (A10) 乘以 `Gamma_active * n_active/n_eff`。

## 3. 折射率、空间烧孔和多模

有效折射率变化近似为：

```text
Delta n_eff = Gamma_active (n_active/n_eff) (dn/dN) (N-Nref)
Delta delta = beta0 Delta n_eff / n_eff
```

局部受激复合使高光强区的 `N` 降低，形成空间烧孔；由 `dn/dN<0` 进一步形成论文图 4(d)
所述的带边频率空间分布。所有横向本征模都包含在四个包络场的二维分布中，不需要预先指定
“基模/高阶模”的独立速率方程。记录的复相干信号做 FFT 后得到频谱。

## 4. 输出量

- 光子密度 `U`：论文式 (10) 下方定义；绘图时乘 `d_active` 转为 `cm^-2`。
- 辐射功率：论文式 (9)，对 `4 eps0 neff c Im(Phi^H C Phi)` 做面积积分。
- 近场：最终载流子和光子面密度。
- 光谱：A 带边投影的时间序列 FFT。

## 5. 三角晶格六波扩展

三角晶格不能沿用方形晶格的 `(Rx,Sx,Ry,Sy)`。新模块按 Liang 第 5.2 节和附录 B 保留六个等长
倒格阶次：

```text
(R1,S1,R2,S2,R3,S3)
= ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1))
```

直接基矢为 `a1=(sqrt(3)a/2,-a/2)`、`a2=(sqrt(3)a/2,+a/2)`，倒格矢满足
`G_mn=beta0((m+n)/2,sqrt(3)(n-m)/2)`、`beta0=4*pi/(sqrt(3)a)`。每个 TE 基本波的偏振向量为
`(rho,eta)=(G_y,-G_x)/|G|`。

几何、纵向 TE0 与 Fourier 分量共同产生

```text
C_6 = Cb + Cr + Ch
```

其中 `Cb` 是六个基本波的直接耦合，`Cr` 是经 `(0,0)` 垂直波的辐射耦合，`Ch` 是消去倏逝
高阶波后的间接耦合。圆孔的高阶截断采用 `m^2-mn+n^2<=D^2` 的 C6 对称圆壳，避免数值截断
破坏 B/D 双简并。

局域能带对每个 `q=k_parallel/(2*pi/a)` 求解

```text
eig[C_6(Gamma) + diag(Delta_1(q),...,Delta_6(q))]
Delta_i = beta0 * (|g_i + Delta k/beta0| - 1)
```

再以相邻波数点的本征矢最大重叠跟踪支路。方形四波新增能带输出也使用同一个支路跟踪器，但失配项
仍是其自身的 `(+kx,-kx,+ky,-ky)`，没有改变原四波物理模型。

当前六波范围包括 Gamma 带边、`M <- Gamma -> X` 局域能带、辐射常数、单晶胞场、有限器件
阈值、整体包络和矢量远场。三角
晶格的这两个物理方向对应 Liang 图 5.4 所称的 `X <- Gamma -> J`，只是按工程统一约定重新命名。
有限六角器件采用式 (5.17) 的一阶特征网格和六方向开放边界，再以 `1/N` 外推阈值；论文式
(5.18) 的原始 staggered generalized mass matrix 尚未独立实现。公式、代码逐项映射及运行方法见
[三角晶格六波说明](triangular_six_wave_zh.md)。

## 6. 从半导体迁移到全固态增益介质

传播与晶格耦合层可以保留，但必须替换半导体载流子状态方程。掺杂晶体通常应使用
能级粒子数、泵浦吸收、受激发射/吸收截面和上能级寿命；此时不再使用半导体的准费米能级增益
和 `dn/dN`。当前 `solid_state.py` 已实现准三能级 Yb:YAG 的透明反转、净增益、阈值反转、
泵浦吸收和速率方程，统一增益介质协议仍是后续重构方向：

```text
SemiconductorQWMedium
RareEarthFourLevelMedium
```

二者向光学求解器提供同样的 `modal_gain(state)`、`index_shift(state)` 和 `rhs(state, photon)` 接口。

`vertical.py` 已提供任意多层结构的标量 TE0 有限差分本征模求解，可计算 `neff` 和各层限制因子。
它是生成新基底参数的工具；高折射率反差、强矢量效应或开放边界辐射模仍应与全矢量 FEM/FDTD
交叉验证。

`scripts/archive/legacy_20261008/run_geometry_derived_ybyag_pcsel.py` 已把方形晶格几何、纵向 TE0、四波耦合、有限器件、
远场与准三能级阈值串成一条可运行原型。默认 `500 um` 器件得到约 `86.74 cm^-1` 所需材料功率
增益和 `31.94` 的阈值反转分数，超过 Yb:YAG 的物理范围，因此被程序明确判为不可达；这是一项
有价值的否定性结果，而不是已经可制造的固态 PCSEL 设计。完整假设、尺寸扫描和下一步验证要求见
[Yb:YAG 模型说明](ybyag_model_zh.md)。

