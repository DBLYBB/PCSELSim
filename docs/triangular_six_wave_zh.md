# 三角晶格六波 3D-CWT：推导、代码和运行说明

> 入口整理（2026-10-08）：本文所述旧脚本已移入 `scripts/archive/legacy_20261008/`，
> 历史数值与用途保留；日常运行请看 [三个半导体主程序](entrypoints_zh.md)。

> 2026-10-08校对提示：本文保留早期教程/推导和历史结果；最新修复、可信度与设计评价以
> [本轮总审计](research_design_audit_20261008_zh.md)为准。旧“最佳”、精确阈值、单模及易加工
> 判断未经最新收敛/工艺验证时不得直接引用，能带是近Γ局域片段，远场能量对评估视窗归一化。

本文对应 `scripts/archive/legacy_20261008/run_custom_triangular_six_wave_pcsel.py` 和
`src/pcselsim/triangular_six_wave.py`。理论主线采用 Liang 论文第 5.2 节及附录 B 的
三角晶格 TE 六波三维耦合波理论；实现是独立的 6×6 模型，没有把方形晶格 4×4 矩阵补零或拟合成
六波矩阵。

## 1. 当前能做什么、还不能做什么

当前已经实现：

- 任意圆/椭圆多原子三角晶胞的解析介电常数 Fourier 系数；
- 与三角晶格 Bragg 条件自洽的纵向标量 TE0 模；
- 六个基本波间的直接耦合 `Cb`、零阶辐射耦合 `Cr` 和高阶波间接耦合 `Ch`；
- 无限周期结构的六个 Gamma 带边态、局域 `M <- Gamma -> X` 能带和辐射常数；
- 基本波、零阶辐射波和截断范围内高阶波共同重构的单晶胞矢量场。
- 圆形、正六边形或方形有限器件上的六方向特征网格、零入射开放边界和 `1/N` 多网格外推；
- 六个有限模式的整个器件包络、冷腔阈值和复矢量远场。

有限求解器直接离散 Liang 式 (5.17) 的六条特征方向，是式 (5.18)–(5.19) 的一阶有限体积对应形式；
它没有照搬论文的广义 staggered mass matrix。阈值必须随网格外推，并与论文表 5.1 或外部 FEM/FDTD
交叉验证。当前仍没有六波与半导体载流子耦合的时域方程，所以不输出 L-I 曲线或动态光谱。

## 2. 从三角晶格到六个基本波

直接晶格基矢取为

```text
a1 = (sqrt(3)a/2, -a/2)
a2 = (sqrt(3)a/2, +a/2)
```

倒格矢写成

```text
G_mn = beta0 * ((m+n)/2, sqrt(3)(n-m)/2)
beta0 = 4*pi/(sqrt(3)a)
```

于是 `|G_mn|^2 / beta0^2 = m^2-mn+n^2`。第二阶 Gamma 点上六个等长的基本阶次为

```text
(R1,S1,R2,S2,R3,S3)
= ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1))
```

对每个非零阶次，TE 面内偏振为

```text
(rho_mn, eta_mn) = (G_y, -G_x)/|G_mn|
```

这些定义分别对应代码中的 `reciprocal_components()`、`reciprocal_metric_squared()`、
`TRIANGULAR_BASIC_ORDERS` 和 `te_polarization()`。

## 3. 晶胞 Fourier 系数

介电常数展开为

```text
epsilon(x,y) = sum_mn xi_mn exp[-i G_mn dot r]
```

对位于分数坐标 `(u0,v0)`、半轴为 `(rx,ry)` 的椭圆，`TriangularLatticeCell.fourier_epsilon()`
使用椭圆解析形状因子

```text
F(G) = f * 2 J1(Q)/Q * exp[i 2*pi(m*u0+n*v0)]
Q = sqrt[(rx*G'x)^2 + (ry*G'y)^2]
f = pi*rx*ry / (sqrt(3)a^2/2)
```

其中 `G'` 是旋转到椭圆主轴后的倒格矢。改变孔半径、椭圆率、旋转角、孔心、孔内介电常数或背景
介电常数都会重新计算全部 `xi_mn`，不是只改变示意图。

`TriangularPolygon` 还支持三角形、正方形和任意简单多边形。顶点用 `(a1,a2)` 分数坐标保存，
Fourier 系数通过多边形边界解析积分计算，不经过像素栅格或图像 FFT。正多边形可用
`TriangularPolygon.regular()` 按真实笛卡尔外接半径、边数和旋转角构造。理想尖角包含较强高阶谐波，
因此多边形必须额外检查 `truncation_order` 收敛，并用实际刻蚀圆角做全波复核。

## 4. 纵向模式与 Bragg 波长

三角晶格的同步条件为

```text
beta = beta0 = 4*pi/(sqrt(3)a)
lambda_B = sqrt(3)*a*n_eff/2
```

`solve_triangular_bragg_vertical_mode()` 在 `lambda_B` 与纵向 TE0 求得的 `n_eff(lambda)` 之间迭代，
直至波长收敛。当前纵向模型是标量有限差分 TE0；高折射率反差、材料色散或强矢量效应仍应使用
全矢量 FEM/FDTD 交叉验证。

## 5. 六波耦合矩阵

Gamma 点本征问题写成

```text
(delta + i*alpha) V = C V
C = Cb + Cr + Ch
V = (R1,S1,R2,S2,R3,S3)^T
```

### 5.1 直接基本波耦合 `Cb`

两个基本波 `i=(p,q)`、`j=(r,s)` 的直接项为

```text
Cb_ij = -k0^2/(2*beta0)
        * integral_PC |Theta0(z)|^2 dz
        * xi_(p-r,q-s)
        * (rho_i*rho_j + eta_i*eta_j)
```

这对应 Liang 式 (5.16)。代码最后作厄米对称化，以去除纵向离散积分的浮点不对称。

### 5.2 零阶辐射耦合 `Cr`

六个基本波都可经晶格 Fourier 分量散射到 `(0,0)` 垂直波，再散射回另一个基本波：

```text
Cr_ij = -k0^4/(2*beta0)
        * xi_(p,q) * xi_(-r,-s)
        * (rho_i*rho_j + eta_i*eta_j)
        * double_integral Theta*(z) g0(z,z') Theta(z') dz dz'
```

`g0` 是零阶辐射 Green 函数。其反厄米部分给出面辐射损耗；程序对数值舍入导致的微小非被动分量
进行半正定投影，并把修正量写入 JSON，便于审计。

### 5.3 高阶波间接耦合 `Ch`

其余 `(m,n)` 阶由附录 B 的 `mu_mn`、`nu_mn`、`zeta_x,mn`、`zeta_y,mn` 消元后汇总到 `Ch`。
程序使用

```text
1 < m^2-mn+n^2 <= D^2
```

的 C6 对称倒空间圆壳，而不是简单的矩形 `|m|,|n|<=D`。这样圆孔晶胞在有限截断时仍严格保留
六重旋转对称性，不会人为劈裂 `B1/B2` 或 `D1/D2`。应增加 `CWT_TRUNCATION_ORDER` 检查能带和
辐射常数是否收敛。

## 6. 局域能带怎样计算

能带图不是把若干抛物线手工画在一起。对每一个归一化波数
`q = k_parallel/(2*pi/a)`，六个基本波的精确径向失配按 Liang 式 (B12) 计算：

```text
Delta_i(q) = beta0 * ( |g_i + Delta k/beta0| - 1 )
C(q) = C(Gamma) + diag(Delta_1,...,Delta_6)
```

图上统一使用用户指定的 `M <- Gamma -> X` 坐标：左半轴为 0 度特征方向，右半轴为 30 度特征
方向。Liang 图 5.4 对同一对三角晶格方向采用 `X <- Gamma -> J` 命名，因此这里是高对称点名称
约定的转换，不是把波矢方向随意替换。每个 `q` 都重新求解复本征值，并通过相邻点本征矢最大重叠
进行全局匹配，避免简并附近按频率排序造成支路互换。归一化频率用冻结的 `n_eff` 转换：

```text
a/lambda = a/lambda_B + Re(delta)*a/(2*pi*n_eff)
alpha_r = 2*max[Im(delta),0]
```

这是二阶 Gamma 点附近的局域 CWT 能带，不是完整布里渊区 PWEM。

方形四波入口的新增 `05_band_structure` 使用相同绘图与支路跟踪器。左侧 Gamma-M 取
`kx=ky=|q|/sqrt(2)`，右侧 Gamma-X 取 `kx=q, ky=0`，传播失配为
`diag(+kx,-kx,+ky,-ky)`；因此四波旧耦合矩阵和旧有限腔/时域模型没有被改写。

## 7. 默认结果怎样解释

默认采用 Liang 第 5 章的 `a=341 nm`、单个居中圆孔 `r/a=0.20`，以及表 3.1 的等效
GaAs/AlGaAs 纵向层。当前计算给出：

| 状态 | 归一化频率约值 | 辐射性质 |
|---|---:|---|
| A | 0.33714 | Gamma 点对称保护，`alpha_r=0` |
| B1/B2 | 0.34058 | 双简并、暗态 |
| C | 0.34170 | 暗态 |
| D1/D2 | 0.34929 | 双简并、亮态 |

这与论文图 5.4 的关键结构一致：频率次序为 `A -> B 双重态 -> C -> D 双重态`，圆孔 C6 对称性
保留两个严格双简并，且只有 D 双重态在法线方向有有限辐射。绝对频率与辐射常数不应声称逐点一致，
因为这里使用标量纵向模式、理想圆柱孔、近似开放 Green 边界、无材料色散，并采用 C6 圆壳而非论文
数值实现的具体截断窗口。

特别注意：无限周期 Gamma 点的 `alpha_r=0` 不等于有限器件阈值为零。有限尺寸会引入横向泄漏，
论文第 5 章中的最低阈值模还随填充率和器件尺寸改变。当前程序没有输出该有限腔结论。

## 8. 怎样运行

在 PyCharm 中打开整个 `PCSELSim`，右键运行：

```text
scripts/archive/legacy_20261008/run_custom_triangular_six_wave_pcsel.py
```

或在项目根目录执行：

```powershell
python scripts\archive\legacy_20261008\run_custom_triangular_six_wave_pcsel.py
```

默认结果写入 `results/custom_semiconductor_triangular_six_wave`，不会覆盖方形晶格结果。步骤开关为：

| 开关 | 输出 | 含义 |
|---|---|---|
| `01_parameters` | JSON、CSV | 参数、`Cb/Cr/Ch/C` 和六态数据 |
| `02_lattice` | 晶格图 | 实空间三角晶格和原胞 |
| `03_k_space` | k 空间图 | 六个保留基本波 |
| `04_vertical_mode` | TE0 图 | 纵向折射率与场 |
| `05_band_structure` | 能带图 | `M <- Gamma -> X` 局域能带 |
| `06_radiation_constants` | 辐射常数图 | 六条支路的 `alpha_r(q)` |
| `07_band_edge_states` | 六态图 | 单原胞 `|E|^2` 和矢量场 |
| `08_finite_modes` | CSV、阈值图、收敛图 | 有限器件本征值和 `1/N` 外推 |
| `09_finite_mode_atlas` | 三列六行模式图 | 整体包络、单晶胞和矢量远场 |
| `10_best_mode_far_field` | 矢量远场诊断 | 最低阈值模的偏振分量、相位和中心强度 |

只跑指定步骤：

```powershell
python scripts\archive\legacy_20261008\run_custom_triangular_six_wave_pcsel.py --only 05_band_structure 07_band_edge_states
```

## 9. 修改参数的安全顺序

1. 先改 `LATTICE_CONSTANT_NM`、背景/孔介电常数和纵向层结构。
2. 再改 `UNIT_CELL_INCLUSIONS` 的孔心、半轴和角度。
3. 用 `CWT_TRUNCATION_ORDER=6,8,10,12` 做收敛扫描。
4. 用更小 `VERTICAL_STEP_NM` 检查纵向离散收敛。
5. 缩小 `BAND_Q_MAX_2PI_OVER_A` 到 `0.01` 可得到与示例图相近的局部视窗。
6. 先检查圆孔时 `B1/B2`、`D1/D2` 是否简并；若不简并，优先检查几何是否真的保留 C6 对称。
7. `DEVICE_SHAPE` 可选 `circle`、`hexagon` 或 `square`；相同 `DEVICE_RADIUS_UM` 分别表示半径、
   六边形外接半径或方形半边长。

当前有限网格默认使用 `N=(7,9,11)`，报告值是复本征值对 `1/N` 线性外推的截距；图片使用最细
网格的实际本征矢。远场由每个网格点的 `radiation_amplitudes()` 复振幅直接作非均匀 Fourier 求和，
不会把六角斜坐标数组当成笛卡尔图像 FFT，也不会人为乘 `(x+iy)` 或高斯光斑。

下一步应实现论文式 (5.18) 的原始 staggered generalized-eigenvalue mass matrix，与当前特征网格结果
互相核对；之后再接半导体载流子方程。

## 10. 主要文件映射

| 理论/功能 | 文件与对象 |
|---|---|
| 三角倒格、六波、TE 偏振 | `triangular_six_wave.py` 顶部函数和常量 |
| 椭圆 Fourier 系数 | `TriangularLatticeCell.fourier_epsilon()` |
| 多边形边界 Fourier 积分 | `TriangularPolygon`、`polygon_fourier_integral()` |
| Bragg/纵向 TE0 | `solve_triangular_bragg_vertical_mode()` |
| `Cb+Cr+Ch` | `build_triangular_coupling()` |
| 六波能带 | `triangular_band_diagram()` |
| 有限六方向本征问题 | `triangular_finite.py` |
| 非均匀复矢量远场 | `triangular_vector_far_field()` |
| 四波/六波通用支路跟踪与绘图 | `band_structure.py` |
| 可编辑主程序 | `run_custom_triangular_six_wave_pcsel.py` |
| 晶格设计扫描 | `design_triangular_bright_mode.py` |
| 三孔孔形—位置联合扫描 | `design_triangular_three_hole_shape_position.py` |
| 300 μm 同层栈四波—六波比较 | `compare_square_triangular_equal_size.py` |
| 回归测试 | `tests/test_triangular_six_wave.py`、`tests/test_triangular_finite.py` |

同尺寸比较的数值结果、阈值电流审计和边界形状权衡见
[300 μm 同尺寸方形—三角对比](equal_size_square_triangular_comparison_zh.md)。

