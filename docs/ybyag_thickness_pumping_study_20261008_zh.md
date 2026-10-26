# Yb:YAG 增厚与泵浦：计算方法、正式扫描和设计判断

> 2026-10-21 更新：本文保留历史模型与数据解释。旧结果已移至 `results/archive/legacy_before_20261021/`。薄层积分、表 I 层栈、有限离散、频谱及新晶体计算以[本轮修正与验证](corrections_20261021_zh.md)为准，旧数值不作为修正后性能。

日期：2026-10-08。这里的“增厚”首先指 Yb:YAG 增益层，不是 SiN 盖层或 PC 刻蚀深度。

## 1. 先读结论与边界

在当前 SiN/YAG 表面混合 TE0 分支中，Yb:YAG 超过约 8 µm 后，增益重叠几乎不再增加。因此增厚不能自动降低泵功率阈值；固定总边缘泵功率时，入口面积增大、泵强度减小，而且多出不少未被该表面分支有效利用的增益体积。

本次完成 4/8/12/20/30/50 µm 六种厚度、四种泵浦方式，共 24 个单分支 CW 案例。8 µm 双端边缘泵恢复此前的约 0.776 W 基线。原半导体代码、原晶体主程序和旧结果均未改动。

**重要：以下绝对阈值/功率全部是冻结 Bragg 系数的条件性计算，不是制备指标或已确定的最优厚度。** 后续频率敏感性检查表明：改变增益层厚度也改变到 DBR 的辐射相位，部分厚度的出光预测对此非常敏感。不能把“数值计算完成”当成完整频率自洽验证。

## 2. 如何自己重新运行

日常仍使用 README 中的四个主入口。此次参数研究是单独的实验脚本，放在 diagnostics，不增加日常主程序数量：

```powershell
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/run_ybyag_thickness_pump_study.py --quick
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/run_ybyag_thickness_pump_study.py
# 只研究指定厚度，固定总内部入口泵功率：
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/run_ybyag_thickness_pump_study.py --thickness-um 8 12 20 --pump-power-W 32
# 改变顶泵程数/有效返回效率；这个效率是包含实际回程损失的工程假设：
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/run_ybyag_thickness_pump_study.py --thickness-um 8 --top-passes 20 --pump-return-efficiency 0.9
```

解释器路径含空格时，PowerShell 需写成 `& '实际解释器路径' ...`。可在 PyCharm 直接打开实验脚本右键 Run，顶部 `StudySettings` 给出默认研究条件；部分参数由命令行覆盖。`--quick` 仅验证流程，不能用它的阈值代替正式结果。

结果自动保存到 `results/archive/legacy_before_20261021/ybyag_thickness_pump/<时间戳>/`；指定 `--output` 时目录必须不存在。每厚度单独保留四个带边族的损耗、复场、泵浦反转、参数和谱移提醒。正式本轮目录为：

`results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/`

本次新增泵浦适配器为 [ybyag_pump_geometry.py](../src/pcselsim/ybyag_pump_geometry.py)，测试见 [test_ybyag_pump_geometry.py](../tests/test_ybyag_pump_geometry.py)。侧泵直接调用原空间速率求解器，新增的是公平的功率/方向接口及顶泵 x–z 传播；没有再维护一套不同的受激增益方程。

## 3. 比较条件与公式

- 器件为 1 mm 正方形，保持 PC 深度 300 nm、SiN 盖层 245 nm、下 SiO2 间隔层 2 µm、DBR 16 对和同一孔形。
- 每种厚度重新设计周期，使 TE0 的 Bragg 参考为 1030 nm。这是多种设计的比较，**不是**已加工固定周期器件的厚度变化。
- CWT 截断 D=6，纵向步长 5 nm，有限网格 N=17/25/33；用最低最细网格 raw 损耗选一个模式。外推、敏感度、原始值全部另存。
- 所有泵浦方式固定首次有效进入增益介质的总功率 32 W，不固定 W/cm²。侧泵沿 x，顶泵沿 z；反转为 f(x,z)，y 方向仍做平均。
- z 单元不超过 0.2 µm，厚度增加时增加单元数，避免 50 µm 用 40 格而把表面光场平均掉。

方形边长 L、增益厚度 t 下，双边泵每端强度为 `P_internal/(2 L t)`；单边为 `P_internal/(L t)`，均匀顶泵首程为 `P_internal/L²`。因此固定强度再增厚会偷增总边泵功率，不能用于判断厚度本身的收益。

准三能级及局域泵传播仍为：

```text
g(x,z) = Nt[(sigma_el + sigma_al) f(x,z) - sigma_al]
alpha_p(x,z) = Nt[sigma_ap(1-f) - sigma_ep f]
I_out = I_in exp(-alpha_p * cell_length)
I_average = I_in [1-exp(-alpha_p*cell_length)]/(alpha_p*cell_length)
G_modal = sum_xz q_gain(x,z) g(x,z)
threshold: G_modal(S=0) = alpha_total_power
alpha_total_power = 2 * alpha_raw_amplitude + alpha_internal_power
```

`q_energy` 和 `q_gain` 采用同一个完整光学模的归一化，不把增益层内权重强行归一成 1。顶泵沿各 x 列传播，保留实际有限模式的横向能量权重；不是将原一列法向反转当作完整 x–z 求解。

多程比较使用非相干、理想重成像的有限光程序列：`下一程入口 = 上一程剩余 × 返回效率`，上下交替。20 程不等于 20 束独立 32 W 泵，满足：

```text
首次内部入口功率 = 晶体吸收 + 回程损失 + 最终剩余泵光
```

只有晶体吸收进入反转/光子/量子亏损预算。回程损失不自动算成晶体热。该算法不是实际底 DBR 泵浦驻波或已设计好的片上 20 程泵腔。

## 4. 加厚是否有帮助？

先区分三个量：阈值材料增益、入口泵强度、总泵功率。均匀增益近似下 `g_material_th = alpha_total_power / sum(q_gain)`；增厚若能增加有效重叠或降低损耗，材料增益门槛确实可能下降，但总泵功率还受需建立反转的体积、吸收与泵模重叠影响。当前选定表面分支的重叠很快饱和，不代表重新设计成其他纵向模的厚晶体也一定更差。

下表双端边泵均为内部总输入 32 W；输出和阈值为冻结参考系数、单个 TE0/四波族的条件性结果：

| Yb:YAG 厚度 / µm | 导引 TE 族数 | 增益能量重叠 | 内部泵阈值 / W | 顶部输出 / W |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 0.44233 | 1.1701 | 1.4314 |
| 8 | 19 | 0.45429 | 2.2679 | 0.7760 |
| 12 | 27 | 0.45449 | 3.1601 | 0.4558 |
| 20 | 44 | 0.45449 | 4.4040 | 0.1679 |
| 30 | 65 | 0.45449 | 5.4291 | 0.000925 |
| 50 | 108 | 0.45449 | 12.7505 | 0.05956 |

这里 TE 族数不是激射模式数；没有为其他全部 TE 族求解共享反转竞争。最细与外推的 C/D 排序也很接近，不能据此宣称单模。

8 µm 以上约七成增益重叠集中在靠 PC 的顶部 1 µm。额外增加深部掺杂层主要增加泵浦体积、荧光和潜在竞争模，不自动提高所选表面模式的有效增益。

30 µm 的 raw 最低族变为 A，其冻结参考下的顶部功率损耗通道很弱。输出谷不是 YAG 材料在 30 µm 突然失效，也不能仅凭归一光斑漂亮就认为输出效率高。

完整图和表见 [厚度/泵浦总图](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/thickness_pump_summary.png)、[光学表](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/thickness_optical_summary.csv)、[泵浦表](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/thickness_pump_summary.csv)。

## 5. 对晶体如何泵浦？

### 5.1 当前优先研究：940 nm 边缘/波导端面泵

思路是 940 nm 二极管光源经准直与柱面/成像光学，形成沿器件边缘的长条泵斑，从一个或两个边缘进入 YAG，沿面内方向获得较长吸收路径。端面泵波长增透、耦合 NA 和垂直束腰都需单独设计，不宜加 1030 nm 高反端镜而无意把器件改成另一种外腔激光器。

实际 Yb:YAG 平面波导侧泵已有实验依据，但文献使用专门成像系统和不同芯层/包层，不能证明本平台的 8 µm 薄片自动能接收这种泵。[Thomson 等，2011，侧泵平面波导原始实验](https://researchportal.hw.ac.uk/en/publications/efficient-operation-of-a-400-w-diode-side-pumped-ybyag-planar-wav/)

8 µm、1 mm 边长、双端内部 32 W 对应每端 16 W、`2e5 W/cm²`。这是内部边界条件，不代表普通纤维输出直接聚焦即可全部耦合。若实际耦合效率 η=0.5，得到内部 32 W 需外部 64 W；η 是需要测量/求解的量，不是本次波导耦合预测。

### 5.2 顶部单程泵：吸收路径太短

名义参数 `Nt=1.38e21 cm^-3`、`sigma_ap=0.70e-20 cm²` 给未饱和吸收系数约 9.66 cm⁻¹。未反转单程的 `1-exp(-alpha*t)` 在 8/20/50 µm 仅约 0.77%/1.91%/4.72%。泵饱和后吸收还会改变。

均匀 1 mm² 顶泵、内部 32 W 时，六种厚度均未达到本次所选模式的 CW 门槛。把泵聚焦到更小面积可以增大局域强度，但会改变与整个器件模式的重叠、未泵区再吸收和横模竞争，不能在当前全孔径计算中直接替换成高强度数字。

### 5.3 顶部多程回收：可考虑，但要把回程损失算进去

外部光学可将未吸收泵重新成像回薄片，提高有效吸收路径；泵的回收部件需让 1030 nm 顶部出光仍可通过。传统薄片实验采用多程泵浦解决短吸收路径问题，例如 220 µm、940 nm、24 程的实验，但那是外腔薄片激光器，不是本器件的验证。[Peng 等，Chinese Physics B 2016](https://cpb.iphy.ac.cn/fileCPB/journal/article/cpb/html/2016/1821/cpb_25_5_054205.html)

8 µm 的比较如下，均以首次有效进入晶体的内部总功率计：

| 泵浦方式 | 内部泵阈值 / W | 32 W 下顶输出 / W |
| --- | ---: | ---: |
| 单端边缘泵 | 2.2478 | 0.7789 |
| 双端边缘泵 | 2.2679 | 0.7760 |
| 顶部单程 | 217.58 | 不起振，β=0 时输出 0 |
| 顶部 20 程，每次有效返回 98% | 13.621 | 0.06522 |

单/双边差异很小，不能据此给出真实热稳定性优劣。双边是更对称的工程候选，但并非本计算证明的低阈值赢家。

名义被动上方入口的 T940≈0.9225，穿过整个 DBR/增益栈的 T 则是逃逸到底部的透射，**不是进入增益的耦合效率**。98% 是额外理想有效回程假设，不是该裸入口层自动提供的性能；每次通过界面、PC 泵衍射、反射与重成像的损失必须计入回程效率。当前底 DBR 也不自动形成 20 程泵浦。

### 5.4 回程损失、程数与外部耦合：额外 22 组检查

固定保存的 8 µm 光学模，不重算或修正 C；独立泵浦敏感性结果位于
`pump_sensitivity_20261008_181025_910318/`。

| 顶部 20 程的每程有效返回效率 | 内部起振泵功率 / W | 首次内部 32 W 下顶部输出 / W |
| ---: | ---: | ---: |
| 80% | 44.810 | 不起振 |
| 85% | 34.723 | 不起振 |
| 90% | 25.482 | 0.01066 |
| 95% | 17.567 | 0.03767 |
| 98% | 13.621 | 0.06522 |
| 100% | 11.352 | 0.09082 |

故不能仅凭理想 98% 回程宣布“顶泵已解决”。实际有效回程效率必须包含所有再次穿膜/PC、反射、散射、重成像和截光损失。

98% 返回下从 20 增到 40 程，条件性顶输出升到约 0.1478 W，但有约 16.75 W 消耗在回程损失中；这不是晶体吸收，更不能写成晶体量子亏损热。

固定**外部**总泵 32 W、侧泵耦合效率为 25%/50%/80%/100% 时，内部泵分别为 8/16/25.6/32 W，双侧条件性顶输出约 0.135/0.346/0.604/0.776 W。25% 情况外部泵阈值约 9.07 W。耦合率仅为假设灵敏度，不是 NA/端面光学计算。

z 网格 40→80→160 的复积分检查保留原完整模归一化；40→160 顶输出变化约侧泵 −0.191%、20 程顶泵 −0.272%。此误差小于目前频率/反射相位和实际泵耦合的不确定性，不应继续只靠加网格来隐藏模型缺项。

独立复核方法（在工程根目录）：

```powershell
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/audit_ybyag_pump_sensitivity_20261008.py --source results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/case_02_gain_8um
```

[敏感性汇总](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/pump_sensitivity_20261008_181025_910318/summary.json)、[回程效率图](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/pump_sensitivity_20261008_181025_910318/top_return_efficiency.png)、[全部 22 行参数表](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/pump_sensitivity_20261008_181025_910318/all_pump_sensitivities.csv)。

## 6. 不能忽略的频率/反射相位敏感性

用保存的同一个 Θ、固定周期和有限场，分别在 1030 nm 和该模式 raw 频率偏移对应的波长复算零阶辐射 Green。这仅是敏感性诊断，**不是**重新求解 C(ω)、场、增益谱的非线性频率本征问题，更不能直接把新损耗拼回旧 CW 功率。

| 厚度 / µm | raw 模态估计波长 / nm | 顶部功率损耗：1030 → 谱移后 / cm⁻¹ |
| ---: | ---: | ---: |
| 4 | 1026.380 | 1.40970 → 1.45328 |
| 8 | 1026.452 | 1.41568 → 1.13162 |
| 12 | 1026.452 | 1.17246 → 0.728922 |
| 20 | 1026.452 | 0.594930 → 0.008779 |
| 30 | 1032.963 | 0.003915 → 0.772097 |
| 50 | 1026.452 | 1.06674 → 0.711904 |

20/30 µm 的冻结参考相位尤其不可靠。对“哪种厚度顶部输出最好”的定量排名，必须先求频率自洽；这也是不能将本轮 4 µm 的最高计算功率直接称为最优设计的原因。

诊断可独立重跑：

```powershell
E:/Anaconda/envs/env00/python.exe scripts/archive/diagnostics/audit_ybyag_thickness_frequency_20261008.py --source results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771 --output results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/frequency_sensitivity_repeat_01
```

默认输出 `frequency_sensitivity` 已存在，会拒绝覆盖；上例用 `--output` 指定一个尚不存在的新目录。再次执行时也需要换一个新目录名。
[频率诊断数据](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/frequency_sensitivity/frozen_frequency_sensitivity.csv)、[对比图](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/frequency_sensitivity/frozen_frequency_sensitivity.png)。注意现有耦合函数的 `wavelength_guess` 会重新锁回 Bragg 条件，单改它并不能实现 C(ω) 自洽。

## 7. 如果说的是 SiN 盖层增厚

另外做了 8 µm YAG 的纵向盖层扫描。SiN 从 245 nm 加到 300/400 nm，TE0 的增益权重约从 0.469 降至 0.137/0.038，场更多集中在非增益盖层；减至 200 nm 则增益权重升至约 0.985，但 PC 场重叠降至约 0.00925。

这些权重不是同一种概率，不能简单相加。此扫描只求纵向模，**没有**得到这些盖层的有限阈值。需要围绕混合过渡联合优化盖层、PC 深度、损耗和工艺误差，不是选择增益或 PC 重叠中的某一个最大值。[纵向盖层诊断图](../results/archive/legacy_before_20261021/ybyag_thickness_pump/20261008_180021_822771/cap_scan_vertical_only.png)

## 8. 下一步建议

1. 暂留 8 µm 为可复核基准；把 4–12 µm 当成工艺/泵耦合研究窗口，不为降阈值直接增厚到 30–50 µm。
2. 先实现辐射 C(ω)、纵向分支、有限场与真实增益谱的频率自洽，后做 DBR 间隔相位优化。不能只优化参考波长下的辐射谷。
3. 查其他纵向 TE 族、完整横模与共享反转竞争、ASE/热；本次 10–108 个族只是风险计数。
4. 边泵需要泵波导/包层泵/端面耦合的 NA 和模式重叠验证；顶泵需要实际多程或泵腔光学、增透与回程损失。
5. 如增厚是为了加工/支撑，考虑较厚无掺杂支撑与靠 PC 的薄有源层，而不是把整个新增体积都掺杂。这个分布掺杂方案尚未仿真。
6. 再扫描 210–270 nm 附近的盖层与 PC 参数；没有这些损耗/容差验证，245 nm 不能叫最佳值。

参考平面陶瓷实验中的 100 µm 芯层、976 nm 泵及外腔模匹配与本平台不同；不能移用其阈值作为本 PCSEL 的验证。[Wang 等，Scientific Reports 2016](https://www.nature.com/articles/srep31289)

本次新增模块的侧泵一致性、法向 Beer 极限、有限多程、饱和、增益钳位和功率预算都有测试；全工程 165 项测试通过。它们证明实现与口径自洽，不替代输入光谱、薄膜吸收、真实泵耦合和实验验证。
