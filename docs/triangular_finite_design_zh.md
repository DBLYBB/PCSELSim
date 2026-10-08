# 六波有限器件、整体包络、远场与亮心晶格设计

> 入口整理（2026-10-08）：本文所述旧脚本已移入 `scripts/archive/legacy_20261008/`，
> 历史数值与用途保留；日常运行请看 [三个半导体主程序](entrypoints_zh.md)。

> 2026-10-08校对提示：本文保留早期教程/推导和历史结果；最新修复、可信度与设计评价以
> [本轮总审计](research_design_audit_20261008_zh.md)为准。旧“最佳”、精确阈值、单模及易加工
> 判断未经最新收敛/工艺验证时不得直接引用，能带是近Γ局域片段，远场能量对评估视窗归一化。

本文说明 `src/pcselsim/triangular_finite.py`、
`scripts/archive/legacy_20261008/run_custom_triangular_six_wave_pcsel.py` 和
`scripts/archive/legacy_20261008/design_triangular_bright_mode.py`。目标是把三角晶格六波无限周期模型扩展到有限圆形器件，
并尝试让中心暗斑模式不再占据最低阈值位置。

## 1. 有限器件离散

Liang 式 (5.17) 中六个包络分别沿

```text
d1 = (1/2,-sqrt(3)/2),  -d1
d2 = (1/2,+sqrt(3)/2),  -d2
d3 = (1,0),             -d3
```

传播。程序在轴坐标 `(j,k)` 上取

```text
x = h(j+k)/2
y = sqrt(3)h(k-j)/2
j^2-jk+k^2 <= N^2
h = L/N
```

从而得到半径约为 `L` 的圆形六角网格。每个传播方向使用迎风特征差分；入射方向的相邻点落在器件外
时，其 ghost 值设为零，对应吸收/开放边界。稀疏本征问题为

```text
(delta+i*alpha) Phi = C6 Phi + i D_hex Phi
```

这里 `C6` 是同一晶胞几何产生的 `Cb+Cr+Ch`，没有调用四波有限腔矩阵。

该实现是一阶有限体积形式，不是逐项复刻 Liang 式 (5.18) 的 staggered generalized mass matrix。
因此默认计算 `N=7,9,11`，将每个复本征值对 `1/N` 作线性拟合，并把截距作为报告值。场形与远场
仍来自最细的 `N=11` 网格。

## 2. 整个器件包络

每个有限模式包含六个实际本征包络：

```text
Phi = (R1,S1,R2,S2,R3,S3)
I_envelope(x,y) = sum_k (|Rk|^2+|Sk|^2)
```

程序只按最大值归一化用于绘图，不用解析高斯函数替换本征矢。默认圆孔、`L=30 um` 的结果为：

| 模式 | 当前 `N=9` 检查值 `alpha L` | Liang 表 5.1 |
|---|---:|---:|
| A | 0.249 | 0.256 |
| B1/B2 | 0.208/0.222 | 0.176 |
| C | 0.193 | 0.133 |

阈值绝对值仍受一阶边界离散、标量纵向模和 Green 函数近似影响，但最低阈值 C、A 较高以及 D 模
强辐射的排序与论文一致。发布定量数据前应查看 `08_grid_convergence.png`。

## 3. 复矢量远场

每个六角网格点先由三维 CWT 的辐射通道得到

```text
(Ex_rad, Ey_rad) = radiation_amplitudes(Phi)
```

远场在实际笛卡尔坐标 `(xp,yp)` 上直接求和：

```text
Fx(kx,ky) = sum_p Ex_rad,p exp[i(kx*xp+ky*yp)]
Fy(kx,ky) = sum_p Ey_rad,p exp[i(kx*xp+ky*yp)]
I = |Fx|^2+|Fy|^2
```

其中 `kx=k0 tan(theta_x)`、`ky=k0 tan(theta_y)`，并包含 Liang 远场公式的倾斜因子。由于六角数组的
两条索引轴并不互相垂直，程序没有直接对数组做普通 `fft2`。输出包括总强度、两个偏振分量、
`Fx+iFy` 相位、偏振轴、中心/峰值和全 RMS 发散角。

## 4. 默认圆孔为什么仍有中心奇点

单圆孔保持 C6 对称性。有限结果中 C 模最低阈值，而它的复辐射场在法线方向相消，中心/峰值接近
零，形成环形远场。中心亮的 D 模虽然近似单瓣，但圆孔下辐射损耗很大，所以不能靠“选择画 D 模”
解决实际起振竞争。

## 5. 新设计：平衡反向旋转双椭圆

设计脚本实际计算六个候选：单圆孔、两个方向的单椭圆、对称双椭圆、不等圆孔和不等椭圆。粗筛每个
候选都会重新计算 Fourier 系数、6×6 耦合矩阵、有限模式和复矢量远场。评价目标包括：

- 中心亮模式的有限阈值尽量低；
- 中心暗/近奇点模式的阈值高于目标模式；
- 中心/峰值尽量接近 1；
- RMS 发散角不过大。

当前获胜结构是中心位置成对、长轴反向旋转的平衡双椭圆：

```text
ellipse 1: center=(-0.19,0) in (a1,a2), radii=(0.19a,0.11a), angle=+15 deg
ellipse 2: center=(+0.19,0) in (a1,a2), radii=(0.19a,0.11a), angle=-15 deg
total fill fraction = 0.15163
```

完整截断 `D=10`、纵向步长 `3 nm`、有限网格 `N=7,9,11` 外推后的关键结果为：

| 指标 | 结果 |
|---|---:|
| 最低阈值模式 | D2 |
| D2 阈值 | `23.69 cm^-1` |
| D2 `alpha L` | `0.0711` |
| D2 远场中心/峰值 | `0.856` |
| D2 全 RMS 发散角 | `1.91 deg` |
| 最接近暗心的 B2 阈值 | `87.13 cm^-1` |
| B2 与 D2 阈值差 | `63.44 cm^-1` |

因此在当前模型中，原先容易起振的中心暗模式不再是最低阈值；D2 变成中心亮、近单瓣的最低阈值模。
这是一项有希望的数值候选，不是可以直接流片的最终版图。

## 6. 怎样运行

论文圆孔有限器件：

```powershell
python scripts\archive\legacy_20261008\run_custom_triangular_six_wave_pcsel.py --only 08_finite_modes 09_finite_mode_atlas 10_best_mode_far_field
```

重新执行几何筛选：

```powershell
python scripts\archive\legacy_20261008\design_triangular_bright_mode.py
```

设计结果进入：

```text
results/custom_semiconductor_triangular_designed_bright_mode
```

重要输出：

| 文件 | 内容 |
|---|---|
| `01_screening.csv/.png` | 全部候选的粗筛对照 |
| `02_winner.json` | 获胜几何、参数、限制和模式指标 |
| `03_winner_grid_convergence.png` | `N=7,9,11` 网格外推 |
| `05_winner_mode_atlas.png` | 六模式整体包络、晶胞场和远场 |
| `06_winner_bright_mode_far_field.png` | D2 矢量远场详细诊断 |

## 7. 不能忽略的后续验证

1. 将有限网格增加到 `N=13,15`，确认 D2 阈值优势不因外推区间改变。
2. 扫描高阶截断 `D=8,10,12` 与纵向步长。
3. 对椭圆半轴、间距和角度做制造误差 Monte Carlo 扫描。
4. 用 RCWA/FEM/FDTD 验证无限周期复频率、远场偏振和双椭圆的辐射相消。
5. 加入真实材料吸收、载流子增益、热效应和电流扩展后再预测阈值电流。
6. 若追求严格论文离散，另实现式 (5.18) 的 staggered generalized mass matrix，并与当前特征网格
   互相验证。

## 8. 后续三孔优化

在上述双椭圆探索之后，工程新增了固定拓扑的三孔扫描：一个主孔位于原胞中心，两个副孔分别位于
`a1` 与 `a2` 方向。它额外评价峰值偏移、质心偏移、椭圆率和 `1 deg` 包围能量，当前得到法线
近圆单瓣的最低阈值 A 模。几何、15 组扫描和设计经验见
[主孔—双轴副孔三晶格优化](triangular_three_hole_optimization_zh.md)。

