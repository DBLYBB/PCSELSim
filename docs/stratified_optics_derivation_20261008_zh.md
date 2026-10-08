# 分层光学、底部 DBR 与反射感知 Green 函数

本说明对应独立模块 `src/pcselsim/stratified_optics.py`。它不修改原有半导体
`three_d_cwt.py` 或 `vertical.py`。`reflector_cwt.py` 是单独的四波扩展接口。

## 1. 范围与坐标

层顺序始终为 **底部 → 顶部**，有限层的底面为 `z=0`，坐标单位为米。
外界介质是半无限层；`vertical.py` 的数值 padding 不是物理薄膜，不应带入本模块。
主程序应把旧纵向场在 PC 层内的坐标平移到新物理 stack 的 PC 层。
导波层和透明包层不是“顶镜”，但其 Fresnel 反射仍自然包含在计算中。

本模块只处理均匀、各向同性、非磁性平面层。TE 标量是切向电场，TM 标量是切向
磁场；TM 的场反射相位因而不能不加转换就与切向电场相位比较。
图形化 PC 的散射仍由 CWT 描述，这不是全矢量三维 FEM/FDTD 验证。

采用 `exp(+iωt)` 时间约定，上行出射波为 `exp(-ikz z)`。被动折射率输入
`n-iκ`，不是 `n+iκ`。选择 `Re(kz)≥0, Im(kz)≤0` 的出射/衰减分支：

\[
k_z=\sqrt{\epsilon k_0^2-q^2},\qquad k_0=2\pi/\lambda.
\]

倏逝波取 `kz=-iκz`；恰好 `kz=0` 的临界点需要极限处理，接口显式报错，不任意
加入小常数。折射率在一次调用中固定，真实色散需外部按波长更新。

## 2. 为什么不用普通传输矩阵连乘

界面要求标量场 F 与 `p∂zF` 连续，其中 TE 的 `p=1`，TM 的 `p=1/ε`。
定义导纳 `Y=p kz`，界面振幅为

\[
r_{ij}=(Y_i-Y_j)/(Y_i+Y_j),\qquad t_{ij}=2Y_i/(Y_i+Y_j).
\]

均匀层传播只使用衰减因子 `exp(-ikz d)`。两个散射网络用 Redheffer 星积合并，
避免同时形成 `exp(+κd)` 和 `exp(-κd)`，因此厚层/高阶倏逝波不会因普通传输矩阵
指数增长而溢出。`solve_stack` 返回振幅 r、t、相位与 R、T；真实传播端口中
`T=Re(Yout)/Re(Yin)|t|²`。无损层应满足 `R+T=1`。倏逝入射端口没有入射功率，
此时 R/T/A 返回 `None`，不能把反射幅度平方当作反射功率。

对于吸收的入射半无限介质，`1-R-T` 不是有限层吸收的通常定义，本模块返回 A=None。
透明外界和被动吸收有限层则可用 A 检查吸收。多层光学的分支选择与吸收外界问题可
参照 [Byrnes, Multilayer optical calculations](https://arxiv.org/abs/1603.02720)。

## 3. 内部源 Green 函数的推导

在选定均匀 host 中，令 `x=z-z_host_bottom`、厚度 d，下/上负载的反射幅度为
ρL、ρR。它们从完整上下网络计算，不只用 DBR 的反射率。标量源 Green 解

\[
\left[\partial_z p\partial_z+p k_z^2\right]G(z,z')=-\delta(z-z').
\]

构造朝下/朝上出射的 Jost 解

\[
u_L=e^{ik_z x}+\rho_L e^{-ik_zx},\quad
u_R=e^{-ik_z(x-d)}+\rho_R e^{ik_z(x-d)}.
\]

加权 Wronskian 为 `W=p(uL uR'−uL' uR)`，Green 为
`−uL(x<)uR(x>)/W`，其导数跳跃满足 `p[G'_+−G'_−]=−1`。
展开得到代码使用的、所有路程非负的四路径表达式：

\[
G=\frac{-i}{2pk_z D}\{e^{-ik_z|x-x'|}
+\rho_L e^{-ik_z(x+x')}
+\rho_R e^{-ik_z(2d-x-x')}
+\rho_L\rho_R e^{-ik_z(2d-|x-x'|)}\},
\quad D=1-\rho_L\rho_R e^{-2ik_zd}.
\]

ρL=ρR=0 时，TE 恢复旧 Liang 核 `−i exp(−ikz|z−z'|)/(2kz)`。
若 `kz=−iκz`，它又恢复旧高阶核 `exp(−κz|z−z'|)/(2κz)`。
在真正的导模/腔极点处 D=0，必须加入真实吸收或调整频率，不应截断成假有限值。
分层介质内部源的 Green 与平面多层方法的结合可参照
[Krijn, Optics Letters 17, 163–165 (1992)](https://doi.org/10.1364/OL.17.000163)；
上述四路径式是从此处给出的微分方程直接推导，不是声称复现该论文未读取的全文公式。

## 4. 上下输出与必须通过的守恒检查

令 ttop/tbottom 是 host 边界到各外界的复透射幅度；单位 δ 源逃逸幅度为

\[
a_\uparrow(x')=\frac{-i}{2pk_zD}t_{top}e^{-ik_z(d-x')}
(1+\rho_L e^{-2ik_zx'}),
\]
\[
a_\downarrow(x')=\frac{-i}{2pk_zD}t_{bottom}e^{-ik_zx'}
(1+\rho_R e^{-2ik_z(d-x')}).
\]

对于无损 stack，任意复分布源 f 和相同数值积分权重 w 必须满足

\[
-\operatorname{Im}(f^\dagger WGWf)
=\operatorname{Re}(Y_{top})|\sum a_\uparrow f w|^2
+\operatorname{Re}(Y_{bottom})|\sum a_\downarrow f w|^2.
\]

`kernel.quadrature_flux(z,f,w)` 专门核查该等式；吸收 stack 的差值代表吸收，不是
错误地要求上下输出等于全部损耗。这里的量是 Helmholtz 源归一化，不直接是瓦特。
CWT 中应从这两个通道构造 `Lup/Ldown`，让 `Im(Crad)=Lup+Ldown`。
`side_power_losses` 返回两倍幅度损耗，即功率损耗；转换为时间衰减率还需乘群速度。

**不能把旧 Green 或辐射损耗乘 DBR 反射率 R**：反射复相位会同时改变频移、
干涉、上下逃逸和模式顺序。有限 DBR 的下方透射通常小但不严格为零。非常厚的
极端结构可能因浮点下溢得到数字 0，也不等价于物理上严格完美反射。

## 5. 导波求解排除远处 DBR 的条件

若导波 beta 高于低折射率 spacer 的光线，spacer 中
`κ=sqrt(beta²−n_spacer² k0²)`。远处负载反射返回导波区域的幅度因子为
`ρ_DBR exp(−2κs)`。必须对实际每个 TE family、beta、波长和 DBR 负载检查这个量。
“spacer=2 µm”自身不是隔离证明：靠近包层光线时 κ 很小，或负载接近极点时
|ρDBR|很大，排除 DBR 的导波近似可能失效。

## 6. 已有自动核查

`tests/test_stratified_optics.py`：30 个测试覆盖 TE/TM 正常与斜入射功率守恒、
Fresnel/Brewster、独立单薄膜 Airy 相位、被动吸收、有限 DBR 阻带与非零透射、
均匀介质传播/倏逝 Green 极限、互易性、导数跳跃、分布源光学定理，以及厚倏逝层
稳定性。分布源守恒相对残差测试容许值为 `2e-12`。

`tests/test_reflector_cwt.py`：均匀辐射 stack 恢复旧 CWT 的全部耦合矩阵和晶胞响应；
DBR 情况检查上下损耗闭合、被动性、随机包络的侧向功率损耗，以及实际 beta 下
2 µm spacer 的回程衰减。它们验证算法内部自洽，不证明加工器件一定达到激光阈值。
