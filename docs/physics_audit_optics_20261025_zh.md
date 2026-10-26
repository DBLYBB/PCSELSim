# 光学代码逐式物理审计：2026-10-25

本报告覆盖 `src/pcselsim/` 中几何、纵向波导、四波/六波耦合、分层辐射、局域能带和半导体材料投影共 11 个模块。有限横模搜索、时域离散、Yb:YAG 反转与泵浦速率另由同期独立报告覆盖；不能将这份报告称为全仓库所有功能都已得到实验验证。

**结论：主光学路径与 Liang/Inoue 的耦合波推导有逐式对应，但不是完整三维 Maxwell 求解，也并非所有地方都与原论文无近似地一致。** 本轮发现并修复一个固定频率能带转换错误，以及三类会使无效输入看似有效的算法护栏问题。合法正式输入的冷腔矩阵、本征场与材料阈值没有因此改变；尚存的近 Bragg 归一化、实际孔形、材料色散及完整矢量验证限制在下文单列。

## 1. 原始来源与证据等级

本轮重新读取了原始文献相应公式，而不是把旧审计结论当作验证依据：

1. Liang, *Three-dimensional coupled-wave theory for photonic-crystal surface-emitting lasers*，京都大学学位论文：[官方来源](https://repository.kulib.kyoto-u.ac.jp/bitstream/2433/188596/5/dkogk03875.pdf)。本地 `Zotero/storage/CVYT7MQ6/` 的 PDF 共 138 页；本轮核对 PDF 第 42–46 页的式 (3.12)–(3.35)，第 117–119 页的式 (B1)–(B14)。式 (3.19) 原文明确说明其 Green 近似忽略波导界面反射。
2. Inoue et al., Phys. Rev. B **99**, 035308 (2019)，[原文](https://doi.org/10.1103/PhysRevB.99.035308)，及 [勘误](https://doi.org/10.1103/PhysRevB.99.169904)。本轮核对 PDF 第 2–3 页的四基本波、模式与 SEM 三维孔形说明，以及第 7 页附录 (A10)–(A17)。原文明确称 `C1D`、`C2D` 为 Hermitian，`Crad` 因辐射为非 Hermitian。
3. Byrnes, *Multilayer optical calculations*，[作者原始预印本](https://arxiv.org/abs/1603.02720)，用于核对分层场、吸收端口及复数分支选择的适用边界。程序使用散射网络级联而不是直接连乘含增长指数的传输矩阵。

本地原始 PDF 的 SHA256、源文件 SHA256 和检查数值保存于 [独立检查](../results/current/physics_audit_20261025/optics/independent_checks.json)。

应区分以下证据：

- **逐式映射**：判断某段程序是否实现了文献中的公式，不证明文献近似对任意新结构准确。
- **独立解析/积分基准**：例如解析板波导、Airy 薄膜、形状直接积分、Cartesian 投影组装，能排查实现错误。
- **代数守恒/同源重建**：是必要的内部自洽检查，不独立验证完整器件的电磁场。
- **尚缺的外部验证**：实际三维孔形、全矢量 RCWA/FEM/FDTD、实验传播损耗与材料谱标定，均不能用测试通过代替。

## 2. 完整模块映射

| 模块 | 对应物理与本轮核对 | 可用范围与限制 |
|---|---|---|
| `geometry.py` | 椭圆 Bessel 形状因子；多边形散度定理边界积分；正号 Fourier 系数与负号场展开相配 | 实、正介电常数、非重叠简单孔。多个孔或周期像重叠仍会重复计算；制造检查不可省略。圆角三角是离散 Bezier 边界，不是精确圆弧 |
| `vertical.py` | 标量 TE 方程 `Theta''+k0²n²Theta=beta²Theta`；最高 beta 本征态；归一化与精确层端点积分 | 平面各向同性 TE 波导可使用该标量方程；实际图形层以平均介电常数替代，仍属 CWT 有效波导近似。Dirichlet padding 必须收敛，不能表示开放全波边界 |
| `hybrid_vertical.py` | 同一三对角 TE 算子的多个束缚本征态；按声明的 TE family 取态 | 只数选定层栈的 TE 模；不含 TM、纵向族间耦合及实际图形化矢量模。基底/DBR 与隔离波导的差别仍需逐族检查 |
| `coupling.py` | 固定正交带边基底；从给定本征值反构造矩阵；反厄米部分被动性 | `calibrated_coupling_matrix` 是标定/教学后端，不能拿它对任意孔形作独立预测，也不能用拟合得到原图来宣称几何复现成功 |
| `three_d_cwt.py` | Liang (3.14)、(3.20)、(3.25)–(3.35)：`C1D+Crad+C2D`；保留高阶横向和纵向投影 | 基本波激发辐射/高阶波，忽略高阶彼此完整多重散射；原基线 Green 忽略界面反射。截断、强反差、粗糙孔壁、三维侧壁均限制定量精度 |
| `reflector_cwt.py` | 四波的分层出射 Green；同一源耦合生成 `Crad` 和上/下端口损耗；高阶封闭通道 | 真正包含复反射相位，不是旧损耗乘反射率。当前拒绝吸收膜和高阶开放辐射通道；高阶纵向 `nu` 仍保留原文局域介电近似 |
| `triangular_six_wave.py` | Liang (5.2)、(B1)–(B14)：六方向、TE 偏振、`Cb+Cr+Ch`；倒格度量 `m²-mn+n²` | 独立六波，不是补零四波。C6 完整壳层避免数值截断破坏圆孔双重简并；非对称孔本身不具 C6，A/B1/B2 等只能视作频率标签 |
| `matched300_yag.py` | 六波分层 Green 与上/下端口损耗；与原文相同的横纵向高阶投影 | 是现有六波理论对无损分层结构的扩展，不是 Inoue 对 YAG 的原始验证。`binned_lateral_weights` 保守地投影至 x，未恢复 y 向反转 |
| `stratified_optics.py` | `exp(+i omega t)`、被动 `n-i kappa`；TE 电场/TM 磁场导纳；散射网络；源 Green、互易与通量恒等式 | 对均匀各向同性平面膜层有效；这里的 Green 源通量先是 Helmholtz 归一化量，必须经过 CWT/光子归一化后才是 W。没有图形化全矢量散射 |
| `band_structure.py` | 同一 C 加 Bloch 失配；右本征矢重叠跟踪；群指数与频率原点 | 冻结 C 和纵向场的一阶局域色散，不是全布里渊区 `C(omega,k)` 求解。分支重叠不证明不可约表示或完整模式身份，近简并/异常点仍需子空间处理 |
| `materials.py` | Inoue (11) 饱和材料增益、(A10) 的模增益/折射率投影及时间指数项 | 是半导体经验材料关系，不可移作 Yb:YAG 速率方程。常数来自原量子阱条件；缺少新器件波长/温度/载流子谱标定。非 Bragg 归一化近似见第 5 节 |

输入映射也已检查：日常入口 `run_paper_reproduction.py` 明确声明理想双椭圆，`a=277 nm`、中心相对位移 `(0.25a,0.25a)`；这不是原文未公开的 SEM 逐层体数据。`matched300_*` 优化脚本以参数 JSON 构造孔与层栈；当前正式四腔通过几何和源绑定读回检查。YAG 使用 500 nm 增益层、120 nm 图形层、SiO2 隔离层和有限 DBR 的提出结构，不是从 Inoue 半导体层栈原样复刻。

## 3. 发现并修复的问题

### 3.1 固定频率能带的原点被重复加入

旧的 native 能带转换使用

`a/lambda_ref + Re(mu)*a/(2*pi*ng)`。

这只适合 `mu` 已经是相对参考频率的失谐，或者参考确为未扰动 Bragg 点。`run_ybyag_pcsel.py` 的部分调用传入的是**固定实际波长处的绝对 CWT 本征值**；此时未减去背景失谐会重复计算频移。

当前平方失谐定义为

\[
\delta_{\rm ref}=\frac{\beta_{\rm ref}^2-\beta_0^2}{2\beta_0},\qquad
\frac{d\delta}{d\omega}=\frac{\beta_{\rm ref}}{\beta_0}\frac{n_g}{c}.
\]

本轮将正方/三角 native wrapper 改为

\[
\frac{a}{\lambda_{\rm ref}}+
\frac{a\beta_0}{2\pi n_g\beta_{\rm ref}}
\left[\operatorname{Re}\mu-\delta_{\rm ref}\right].
\]

正方 `beta0=2*pi/a`，三角 `beta0=4*pi/(sqrt(3)*a)`，分别由真实倒格矢给定。低层转换接口不提供 `reference_wavenumber_per_m` 时仍保留“输入为相对失谐”的明确兼容约定。

验证包括：给一个恰好满足物理闭频条件的测试根，绘图必须回到它自己的参考波长；常折射率宿主中独立对平方失谐求频率有限差分；四个正式几何在固定波长下核对 Γ 点原点。**此修复只影响绘图频率转换，不改变 C、本征场、阈值、模式搜索，也不解决文献 AB/BC 间隔比失配。** 相邻间隔比在共同原点/线性比例变换下保持不变。

### 3.2 实质负辐射损耗曾能被 PSD 投影掩盖

旧 `_project_radiation_passive` 会将任意负辐射本征值裁为 0。负对照 `diag(i,-0.2i)` 被接受；这相当于把错误增益通道修成无损通道，不能称物理求解。

现仅允许 `256*eps*||matrix||_2` 量级的负值清理；超出舍入尺度明确报错。保留修正量审计。原始合法四腔在本轮粗审计的修正量为 0 或约 `2e-14 /m`，相对总矩阵远低于浮点阈值。正式参数读回也全部通过。

### 3.3 Hermitian 化必须以原矩阵本来就满足物理条件为前提

本轮在代码的独立副本中，逐处截取 `C1D/C2D/Cb/Ch` **投影前**的矩阵；没有修改生产函数来得到这些读数。四个正式孔形在 `D4/dz5 nm` 下，原始高阶块的相对反厄米范数约 `1e-17–1e-16`，独立 Cartesian 投影组装与代码的相对误差不超过 `3.4e-16`。

这支持当前 Hermitian 化清理的是舍入，而不是凭空重造物理。但接口以前没有拒绝大误差的护栏。新增 `_roundoff_hermitian` 在理论封闭通道中明确检查相对误差；大误差直接失败。对舍入矩阵仍返回与原来逐元素相同的对称化结果。

### 3.4 不支持的复介电常数曾会被默许

旧几何类可接受 `epsilon=1-0.2j`，而后续又取平均介电常数的实部、对封闭项 Hermitian 化。没有吸收算子的 CWT 就可能将吸收物理悄悄删除。

正方/三角背景与孔材料现在要求介电常数为实、正、有限值；复介电常数和金属被明确拒绝。真正加入吸收必须同时扩展纵向模、复耦合/吸收算子及功率预算。`stratified_optics.py` 本身仍可算被动吸收平面膜；反射 CWT 在尚未构造吸收算子时继续拒绝将它接入激光速率方程。

几何同时检查有限坐标、正半径和非零多边形面积。它没有因此获得任意自交多边形或重叠孔的布尔几何能力。

## 4. 独立验证及冻结结果读回

运行入口：

```powershell
E:/Anaconda/envs/env00/python.exe -X utf8 scripts/diagnostics/audit_optical_physics_20261025.py
```

**39 项检查通过**，包括：

- 平移/旋转椭圆的解析 Fourier 系数对独立极坐标直接积分；多边形边界积分对三角重心坐标双积分；零阶面积与复杂相位一并检查。
- 解析对称 TE 板波导色散；四档 `dz` 误差下降；从 `n_eff-lambda*d n_eff/dlambda` 得到的群指数与能量积分式比较。
- 独立单薄膜 Airy 反射与 TE/TM 斜入射功率守恒。
- 非器件模板层栈的分布源 Green 光学定理、互易性。
- 四个正式孔形的原始 Hermitian 误差、Cartesian TE/纵向投影高阶组装、PSD 清理量，以及固定参考频率原点。

光学相关 10 个测试文件共 **85 项通过**。这包含上述修复的反例测试，不仅检查正向结果是否有限。第一次测试的 pytest cache 路径权限告警不影响测试结果；后续使用 `-p no:cacheprovider` 无需写 cache。

正式 `D64/D48/D32、dz1 nm` 四腔又从参数重建，并对冻结有限场检查：

| 正式案例 | C 相对差 | 材料阈值相对差 | 实频闭合残差 /m |
|---|---:|---:|---:|
| 半导体正方 | 0 | 0 | −6.8421e−4 |
| 半导体三角 | 0 | 0 | −1.2677e−6 |
| Yb:YAG 正方 | 0 | 0 | 2.5279e−6 |
| Yb:YAG 三角 | 0 | 0 | 1.7990e−5 |

来源为 [正式光学读回](../results/current/physics_audit_20261025/optics/formal_optical_rebuild_after_guards.json)。所有原始结果文件继续只读，未改写旧来源 SHA、模式数据或阈值。

同源读回证明新护栏没有改变这些合法输入的数值，不是新的 Maxwell 认证。

## 5. 必须公开的近 Bragg 归一化近似

Inoue 附录 (A10) 的原式是

\[
n_g^{\rm paper}=\frac{k_0}{\beta_0}\int n^2|\Theta|^2dz,
\qquad q_g^{\rm paper}=\frac{k_0}{\beta_0}\Gamma n_{\rm active}.
\]

当前纵向接口则使用真实导模的

\[
n_g^{\rm guide}=\frac{\int n^2|\Theta|^2dz}{n_{\rm eff}},
\qquad q_g^{\rm guide}=\frac{\Gamma n_{\rm active}}{n_{\rm eff}},
\quad \beta_{\rm ref}=k_0n_{\rm eff}.
\]

在未扰动 Bragg 条件 `beta_ref=beta0` 下两者完全一致。固定周期、有限腔实际闭频时，两者差因子 `s=beta_ref/beta0`：

| 正式案例 | s | 相对差 |
|---|---:|---:|
| 半导体正方 | 1.00987270 | +0.9873% |
| 半导体三角 | 0.99063256 | −0.9367% |
| Yb:YAG 正方 | 0.99739611 | −0.2604% |
| Yb:YAG 三角 | 0.99879455 | −0.1205% |

见 [近 Bragg 适用范围记录](../results/current/physics_audit_20261025/optics/near_bragg_normalization_scope.json)。因此，不能称当前所有固定频率接口严格逐式等于原附录 A10。现有 CWT 为弱包络近似，这类 `beta≈beta0` 替换仍是模型适用范围的一部分，对绝对阈值和光子寿命有相应量级的归一化影响。

**本轮没有只修改 ng、qg 或某一处损耗。** 若要取消该近似，必须同时统一 CWT 空间衰减、真实传播损耗、时间群速度、能量与增益权重，然后用守恒和独立电磁基准重新验证；只修一个因子会使内部接口更不一致。本轮新 YAG 泵浦/尺寸研究仍沿统一的现有接口作条件分析，不把小于该尺度的差别当成已证实优势。

半导体 `temporal_index_rate_s` 则已与 A10 核对：局域有源层均匀扰动时 `gamma=2*Gamma/n_active*(dn/dN)*(dN/dt)`。这与相位失谐项不同，不应随意删去，也不是辐射损耗修正。

## 6. 对 YAG 微腔可行性结论的约束

光学结果能支持“在提出的实折射率层栈和标量 CWT 内存在条件候选”，不能独自支持“真实 500 nm Yb:YAG 图形微腔已经可实现激光”。至少还需：

1. 实际材料的吸收/发射谱、掺杂与寿命、传播和加工散射损耗；当前低内损耗假设不能由理想无损薄膜计算证明。
2. 对实际孔形/高反差/DBR 的独立周期全矢量求解，检查频率、辐射损耗及模态限制因子；SEM 未公开的文献几何差异仍未消除。
3. 基底、近截止 TE 家族、TM 与其他横向模式的作用。基本 TE0 被隔离层抑制泄漏的证据不能自动扩展给所有模式和尺寸。
4. 可实现的泵耦合和饱和吸收路径。内部侧泵低阈值不是外部泵能量阈值；单程薄层顶泵弱吸收是独立工程限制。
5. 热与应力对光谱、折射率、寿命及损耗的反馈。局域能量闭合通过不等于温升已可承受。

本轮的 39 项独立检查、85 项测试及四腔原值重建，应当作为可复核的实现证据使用；上述物理缺口保留，不能用更多相同代码的测试消除。

## 7. 文件与复算入口

- [机器可读审计汇总](../results/current/physics_audit_20261025/optics/summary.json)：11 个模块完整源 SHA、检查范围、修复和未验证项。
- [独立检查脚本](../scripts/diagnostics/audit_optical_physics_20261025.py)：原始矩阵读回、独立解析/积分/投影检查。
- [修复前负对照](../results/current/physics_audit_20261025/optics/before_guards.json) 与 [修复后检查](../results/current/physics_audit_20261025/optics/independent_checks.json)。
- [正式四腔光学读回](../results/current/physics_audit_20261025/optics/formal_optical_rebuild_after_guards.json)。
- [物理拒绝护栏测试](../tests/test_optical_physics_guards_20261025.py) 与 [频率原点/导数测试](../tests/test_band_structure.py)。

English scope note: These checks establish equation mapping, independent analytic limits and numerical consistency within the stated scalar coupled-wave approximations. They do not establish complete reproduction of the SEM-derived device, a full-vector Maxwell solution, or experimental Yb:YAG microcavity lasing feasibility. Invalid absorption, substantial negative radiation and non-roundoff closed-channel skew now fail explicitly instead of being projected away.
