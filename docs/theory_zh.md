# 理论与代码映射

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

论文排版图中的 `i` 很小，但由其稳态本征式 `(delta+i alpha)V=CV` 和式 (9) 的能量守恒
可知耦合生成元必须是 `i vg C`。若 `lambda(C)=delta+i alpha_rad`，其时间演化为
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

## 5. 从半导体迁移到全固态增益介质

传播与晶格耦合层可以保留，必须替换的是 `materials.py` 与载流子状态方程。掺杂晶体通常应使用
能级粒子数（两能级、三能级或四能级）、泵浦吸收、受激发射/吸收截面和上能级寿命；此时不再使用
半导体的准费米能级增益和 `dn/dN`。建议新增 `GainMedium` 协议，并实现：

```text
SemiconductorQWMedium
RareEarthFourLevelMedium
```

二者向光学求解器提供同样的 `modal_gain(state)`、`index_shift(state)` 和 `rhs(state, photon)` 接口。

`vertical.py` 已提供任意多层结构的标量 TE0 有限差分本征模求解，可计算 `neff` 和各层限制因子。
它是生成新基底参数的工具；高折射率反差、强矢量效应或开放边界辐射模仍应与全矢量 FEM/FDTD
交叉验证。

