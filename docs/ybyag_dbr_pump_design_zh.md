# 厚 Yb:YAG 波导 + 底部 DBR 的 PCSEL：空间增益与泵浦闭合

更新：2026-10-08。本文说明新增 `src/pcselsim/ybyag_spatial_rates.py` 的模型接口和可验证范围，不替代旧 `solid_state.py`，也不把器件改成普通上下镜 Fabry–Pérot 激光器。

## 0. 新主程序怎么运行，当前候选是什么

新入口为 `scripts/run_ybyag_pcsel.py`。在 PyCharm 中使用已安装项目依赖的 Python 解释器，打开此文件，修改开头的 `PARAMETER PANEL` 和 `STEPS`，右键选择 Run 即可。也可在工程根目录运行：

```bash
python scripts/run_ybyag_pcsel.py --quick
python scripts/run_ybyag_pcsel.py
```

第一条仅为粗网格接口检查，不能作为最终设计值。第二条使用主程序给定的多个有限器件网格；多网格 CWT 也不等于全矢量 Maxwell 验证。默认输出在 `results/ybyag_crystal_pcsel/时间戳/` 新目录，每次结果单独保存。若指定 `--output`，应给一个不存在的新目录。

可以用 `--only 01_structure 05_spatial_gain` 等选择步骤；速率/场图需要的有限器件本征模仍会自动计算。步骤 01 输出结构、纵向场、多 TE 家族和底 DBR 光谱；02 为近 Γ 的四波 M–Γ–X 图；03 为有限区域模式与数值收敛；04 为包络、单胞和顶部矢量远场；05 为泵耗尽/反转、共享光子库和功率预算；06 为纵向厚度/帽层/反射相位的诊断扫描。

当前默认候选从底到顶是：YAG 基底 / 16 对 Ta₂O₅–SiO₂ 有限 DBR / 2 μm SiO₂ 下间隔波导层 / 8 μm Yb:YAG 增益芯层 / 300 nm 被动 YAG 双圆孔 PC 层（孔由 SiO₂ 填充）/ 245 nm 未图形化 SiN 上导光层 / 300 nm SiO₂ 上包层 / 空气。DBR 和上下透明导光层是不同部件；没有新增上镜。

芯层 `n=1.8166` 和被动 YAG `n=1.8154` 采用已报道陶瓷数值作为候选参考，不是这套薄膜的已测量值。SiN、SiO₂、Ta₂O₅ 为名义薄膜折射率；`YB_DENSITY_CM3=1.38e21` 为名义约 10 at.% 位点密度，需测实际浓度和光谱；`INTERNAL_POWER_LOSS_CM=0.10` 是未实验标定的功率损耗假设。跨材料键合、薄膜应力、散射、PC 孔填充和热管理均未得到实验验证。不能把参数可输入、测试通过称为制造已可行。

主程序以**最细保留网格的原始场衰减常数 × 2 + 假设材料功率损耗**接速率方程，而不是使用可能偏乐观的网格外推截距。上下损耗来自同一有限模式对反射 Green 辐射算子的投影，已经是功率损耗，不再乘 2。

## 1. 先确定我们在计算哪一种激光器

激光反馈仍由面内二维光子晶体产生；向上出光仍是光子晶体衍射产生的面发射。增厚 Yb:YAG、改变上下包层和加入底部 DBR 是纵向波导与辐射边界的改动，不能单独证明面内反馈足够，也不能仅凭出现正输出功率判断起振。

底部 DBR 可以返回向下辐射并改变其与向上辐射的相位。应在光学模型中用复反射系数和传播相位重新构建辐射耦合矩阵、求本征模，再传递该同一个模式的总损耗和上下功率损耗给速率模型。不要在算完冷腔后直接用 `旧辐射损耗 × (1-R)` 代替这一过程。半导体 PCSEL 已有底部 DBR 的实验先例，但不能据此声称 Yb:YAG 实现得到验证。[Hsu 与 Lin，Photonics 2018](https://doi.org/10.3390/photonics5030027)；高功率 PCSEL 理论明确讨论了底部返回波相位对辐射耦合的作用。[Yoshida 等，Nature Communications 2022](https://doi.org/10.1038/s41467-022-30910-7)

## 2. 数据接口：先光学，后空间反转

入口应按以下顺序调用：

1. 为 4/8/15/30/50 μm 厚 Yb:YAG 分别构建真实上下包层和底部 DBR。
2. 求纵向 TE 模及其场、有效折射率、群速度；厚芯层必须记录是否有多个 TE 模，不能默认仍为单纵向模。
3. 对选定纵向模求面内 PCSEL 模式与有限器件包络，求包含辐射、边缘和材料损耗的**功率**损耗。
4. 将增益芯层按 z 分层，传入真实光学权重、同一模式的损耗和内部入口泵浦。
5. 联合求泵耗尽、局域反转、一个模式光子库，检查起振条件和能量闭合。

```python
import numpy as np
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import (
    SpatialPumpConfig, SpatialYbYAGModel, solve_spatial_ybyag_steady,
)

# 下列 optical_*、v_group 和 alpha_* 必须来自同一个新光学本征模，
# 而不是把旧的假设层栈结果复制过来。
model = SpatialYbYAGModel(
    medium=YbYAGMediumConfig(pumped_area_um2=device_area_um2),
    cell_widths_m=np.asarray(active_cell_widths_m),
    optical_energy_fractions=np.asarray(q_energy),
    optical_gain_weights=np.asarray(q_gain),
    group_velocity_m_s=v_group,
    total_power_loss_per_m=alpha_total,
    upward_power_loss_per_m=alpha_up,
    downward_power_loss_per_m=alpha_down,
    lateral_energy_fractions=x_cell_energy_fractions,  # 来自同一有限器件包络，和为1
)
pump = SpatialPumpConfig(
    direction="edge",             # "normal" 为沿 z 泵；"edge" 为沿 x 侧泵
    forward_intensity_W_cm2=I_left,
    backward_intensity_W_cm2=I_right,
    edge_length_um=device_length_um,
    edge_cells=41,
)
result = solve_spatial_ybyag_steady(model, pump, find_threshold=True)
print(result.lasing_gain_condition, result.threshold_pump_scale)
print(result.energy_budget_W)
```

示例是接口示意，不是可直接运行的主程序：变量需要光学入口提供。`cell_widths_m.sum()` 是本次真实增益厚度，优先于旧介质配置的 `gain_thickness_um`；增益体积为 `pumped_area_um2 × 1e-12 × 实际增益厚度(m)`。旧介质的 `confinement_factor` 不会再乘一次，避免重复计算重叠。

### 能量权重不等于增益权重

对于代码使用的非色散标量 TE 纵向模型，令 \(\int |\Theta|^2 dz=1\)：

\[
n_g=\frac{\int n^2|\Theta|^2dz}{n_{\rm eff}},\qquad
q^{E}_i=\frac{\int_i n^2|\Theta|^2dz}{\int n^2|\Theta|^2dz}.
\]

`optical_energy_fractions` 只取增益层中各小层对**整个光学模**能量的份额，\(\sum_i q^E_i=\Gamma_E\le1\)。不能先截取增益层再归一到 1。

材料增益定义为局域行波**强度**满足 \(dI/ds=gI\)。其对光子变化率的贡献为 \((c/n_i)q^E_i g_i\)，因此用共同群速度 \(v_g=c/n_g\) 表示时：

\[
q^g_i=\frac{n_g}{n_i}q^E_i
       =\int_i\frac{n(z)}{n_{\rm eff}}|\Theta(z)|^2dz,
\qquad G_{\rm modal}=\sum_iq^g_i g_i.
\]

`optical_gain_weights` 为 \(q^g\)，一般不等于 \(q^E\)，其和也不强行设为 1。均匀折射率 Yb:YAG 芯层可以不显式传入；模块会用 `c/(n_YAG*v_group)*q_energy` 计算。若增益折射率随 z 变化，入口必须显式积分并传入增益权重。材料色散很强时，应由更完整的电磁能量归一化重新提供这两种权重，不能照搬以上非色散关系。

## 3. 空间准三能级方程

令 \(f_i=N_{2,i}/N_t\)，体积份额 \(w_i=V_i/V_{\rm gain}\)，并定义同一模式光子库 \(S=N_{\rm ph}/V_{\rm gain}\)。侧泵时 i 包括 x 和 z，z 使用真实 TE 场。`lateral_energy_fractions` 可传入同一有限器件四波包络沿 y 积分、再按泵 x 单元积分后的能量份额，和为 1；使用 \(q^g_{xz}=p_xq^g_z\)。体积权重仍是实际均匀 x 单元体积，不能也按光场权重改掉。

不传该参数时才采用均匀面内包络的兼容近似。侧泵要求权重长度等于 `edge_cells`；当前法向泵简化成一列 z 单元，因此只能不传或传长度为 1 的面内权重，并不等价于已经解决完整 x–z 法向泵空间烧孔。侧泵 f(x,z) 记录沿 x 的空间烧孔；y 方向只做积分平均，未求解二维局域反转。

局域激光净增益与泵浦净吸收：

\[
g_i=N_t[\sigma_{el}f_i-\sigma_{al}(1-f_i)],\quad
\alpha_{p,i}=N_t[\sigma_{ap}(1-f_i)-\sigma_{ep}f_i].
\]

其中再吸收和泵浦受激发射都保留，不能简单设所有吸收泵光都能建立完全反转。\(\Phi_{p,i}=(I^+_{p,i}+I^-_{p,i})/(h\nu_p)\)。局域速率与模式光子速率：

\[
\frac{df_i}{dt}=\sigma_{ap}\Phi_{p,i}(1-f_i)-\sigma_{ep}\Phi_{p,i}f_i
 -\frac{f_i}{\tau}-\frac{v_g}{N_t}g_i S\frac{q^g_i}{w_i},
\]

\[
\frac{dS}{dt}=v_g(G_{\rm modal}-\alpha_{\rm total})S
 +\beta\frac{N_t}{\tau}\sum_i w_i f_i.
\]

局域耗尽中的 \(q^g_i/w_i\) 很重要：光场在薄层中的高局域强度不能用等厚平均替代。将第一式乘 \(N_tV_i\) 求和后，受激耗尽恰好等于第二式中的受激光子增加；没有凭空产生或消失的光子。

\(\beta\) 是进入当前模式的自发发射份额，不是实验已标定值。非零 \(\beta\) 会让阈值以下有极小面发光，因此 `lasing_gain_condition` 使用**零光子、泵受限**模态增益是否大于总功率损耗判断，而不是 `output_W > 0`。\(\beta=0\) 起振后有精确增益钳位；\(\beta>0\) 的稳态模态增益略低于总损耗，这是自发源项补偿的结果。

## 4. 泵浦传播与内部/外部功率不能混用

每个小单元内 f 固定，模块采用精确指数传播 \(I_{\rm out}=I_{\rm in}e^{-\alpha_p\ell}\)，以及精确单元平均强度：

\[
\overline I=I_{\rm in}\frac{1-e^{-\alpha_p\ell}}{\alpha_p\ell}.
\]

这样泵入口减出口功率与 \(\sum_iV_i\alpha_{p,i}\overline I_i\) 一致，而不是用粗糙入口强度代替全部单元的吸收。

| 配置 | 传播方向和数组 | 内部入口泵功率 |
|---|---|---|
| `normal` | 从底向顶为 forward，顶向底为 backward；f 为 `(1,Nz)` | \((I^+ + I^-)A\) |
| `edge` | 从左向右为 forward，右向左为 backward；f 为 `(Nx,Nz)` | \((I^+ + I^-)Wt\)，其中 \(W=A/L\) |

两端泵浦入口沿厚度均匀，但传播后的强度和反转可随 x、z 变化。法向双面泵是两个**独立规定的内部入口源**。入口强度必须是穿过界面/泵浦波长 DBR 后进入增益体的值；同一个外部泵光的反射返回不可当成第二束独立入射泵再加一次。模块当前没有联合求解多程反射边界，也不含相干泵驻波：这些须由光学入口显式提供并另行核算外部预算。

公平比较法向/侧面或单端/双端应固定**相同总外部泵功率和耦合效率**，不是固定所有方向相同 W/cm²。底 DBR 应分别计算 940 nm 泵光和约 1030 nm 激光的反射/透射；激光高反不自动意味着泵浦高透。

## 5. 稳态求解与能量预算

`_steady_inversion` 在指定 S 下迭代饱和泵传播和局域反转。`solve_spatial_ybyag_steady` 再用一维 Brent 求根闭合共享光子库；它不是已经验证的完整时域、多模竞争或三维热模型。`find_spatial_threshold_pump_scale` 固定两端泵功率比例，同时缩放二者，求零光子状态的 \(G_{\rm modal}=\alpha_{\rm total}\)。

无限泵也受 \(f_\infty=\sigma_{ap}/(\sigma_{ap}+\sigma_{ep})\) 限制。若该上限仍无法弥补损耗，阈值返回 `None`；`None` 也可表示在指定 `max_scale` 内未括住解，不能写成“泵强度为零”。

结果中的各功率满足：

\[
P_{\rm abs}=
P_{\rm qd}+P_{\rm fl}+P_{\uparrow}+P_{\downarrow}+P_{\rm other}
 +\frac{d(U_{\rm excitation}+U_{\rm photon})}{dt}.
\]

`quantum_defect_heat_W = (1-ν_l/ν_p)*absorbed_pump_W` 仅为模型中的量子亏损热，不是总实验热；实际寄生吸收、非辐射、上转换、热猝灭需要额外模型。`fluorescence_outside_mode_W` 用激光光子能量近似整个荧光能谱，属于有效两库能量核算，不是已测量的荧光光谱积分。

`upward_output_W`、`downward_output_W` 和 `other_cavity_loss_W` 来自同一模式的功率损耗分解，要求 \(0\le\alpha_\uparrow+\alpha_\downarrow\le\alpha_{\rm total}\)。若原光学本征值使用场衰减常数，应先换为功率损耗（通常 \(\alpha_{\rm power}=2\alpha_{\rm field}\)），不能再次混入一个因子 2。真实 DBR 返回波已经包含在新本征模及损耗中，不能在输出功率阶段再次回收该能量。

检查字段包括 `energy_closure_residual_W`、`population_residual_per_s`、`photon_fraction_residual_per_s`。守恒只证明数值和归一化自洽，**不证明输入层栈、截面、DBR 或器件能实验实现**。

## 6. 建议如何优化而不是盲目增厚

- 厚度扫描必须同时记录：TE 模数量、增益权重、PC 层场重叠、面内阈值、上出光效率、泵吸收、反转最低值和空间均匀性。增厚可能改善法向吸收，却可能减少顶部 PC 耦合或引入多个纵向模。
- 每种厚度先比较无 DBR 和有 DBR，再扫描返回相位/间隔层。目标是可达阈值和有效向上出光，不是只最大化反射率或只最小化辐射。
- 侧双端泵通常是改善长程反转均匀性的候选，但需要计算真实端面入口面积、NA、波导泵模式和耦合效率；当前入口厚度均匀是假设，不是加工后的自动结果。
- 掺杂倍数和光谱截面变动只作为灵敏度扫描。没有温度相关实测截面和材料浓度/折射率/散射测量，不应称为“优化到实验最好”。
- 主入口应传入真实有限器件的 x 包络权重，不应沿用均匀面内光子密度。进一步需传入完整 \(q^g(x,y,z)\) 并扩展泵二维/三维传播与多模竞争；现在的 y 积分近似仍不能解析 y 方向空间烧孔。不要把整芯层平均正增益当作每一点都透明。

## 7. 制造与文献边界

PLD 已有约 15 μm Yb:YAG 平面波导激光实验，支持“厚 Yb:YAG 芯层/波导”作为研究路线，但不是同样层栈的二维 PCSEL 或底 DBR 单片集成证明。[An 11.5 W Yb:YAG planar waveguide laser，OME 2016](https://doi.org/10.1364/OME.6.000091)（[作者机构全文](https://eprints.soton.ac.uk/384775/2/ome-6-1-91.pdf)）。

国内透明陶瓷波导工作给出 Yb:YAG 芯层与 YAG 包层很小的折射率差，提醒我们不能无证据把高掺杂视为很高折射率差。[Scientific Reports 2016](https://doi.org/10.1038/srep31289)。YAG 内部约纳米尺度空孔制造已有原始研究，但工艺存在孔形、最小尺寸和大面积一致性约束，不等于任意三角主孔和副孔都可用飞秒激光直接写出。[Ródenas 等，Nature Photonics 2019](https://doi.org/10.1038/s41566-018-0327-9)；[Chen 等，Nature Communications 2025](https://doi.org/10.1038/s41467-025-62426-1)。

国内 2026 年 Nd:BaLaGa₃O₇ SSH 激光工作采用微米间隔波导、外部端面双镜及 808 nm 泵浦。它不是约半微米周期的 Yb:YAG 二维 PCSEL，因此不能用其成功直接验证本设计。[Wu 等，Nature Communications 2026](https://doi.org/10.1038/s41467-025-68173-7)。

## 8. 测试与目前结论

运行 `python -m pytest -q -p no:cacheprovider tests/test_ybyag_spatial_rates.py tests/test_ybyag_crystal_workflow.py`。当前 20 项测试通过，其中 15 项覆盖两种泵几何的瞬时能量闭合、Beer–Lambert 极限、稳态增益钳位、阈值以下自发光识别、真实光学重叠、耗尽泵阈值、双端对称性、输出分项不能重复计数、群速度与局域折射率的不同归一化、有限 β 稳态、非均匀包络的守恒/空间烧孔及输入校验。

另 5 项入口测试只求纵向场，不做昂贵有限器件求解：检查 DBR 与导光层分离、\(q^g=(n_g/n_{\rm core})q^E\)、增益厚度的分层积分不随速率网格变化、真实增益体积与侧泵入口功率单位、多 TE 家族及被动 PC 不被误算为增益。默认 1 mm × 1 mm × 8 μm 芯层、每侧内部入口 2e5 W/cm² 对应合计 **32 W 内部入口泵功率**，不是把 1 mm² 顶面积乘侧泵强度得到的 4000 W；外部泵需求还要除以实际耦合效率。

新增模块使“厚芯层 + 非对称波导 + 底 DBR”可以和空间泵浦一起核算。实际能否起振、光束是否改善，仍须等待入口用新层栈与 DBR 相位求出同一个 PCSEL 光学模式并扫描；本文件不预先宣称新器件已经成功。
