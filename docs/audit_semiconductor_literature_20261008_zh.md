# 半导体文献校对、外围代码逐文件审计与可加工设计评价

> 入口整理（2026-10-08）：本文所述旧脚本已移入 `scripts/archive/legacy_20261008/`，
> 历史数值与用途保留；日常运行请看 [三个半导体主程序](entrypoints_zh.md)。

审计日期：2026-10-08。这里的“通过”指单位、符号、数据流程及适用范围没有发现确定性错误，不代表实验认证，也不代表已经获得全局最优器件。光学核的独立审计见本轮光学审计报告；本文件重点审查载流子、输出、入口、历史优化及文献口径。

## 1. 文献证据与对设计的影响

| 原始资料 | 校对来源及身份 | 可支持的结论 | 不能据此声称的结果 |
|---|---|---|---|
| Inoue et al., PRB **99**, 035308 (2019)，[DOI 10.1103/PhysRevB.99.035308](https://doi.org/10.1103/PhysRevB.99.035308) | 已读Zotero `4WEC2TRI` 原始8页，尤其式(8)-(11)、附录A、表I-II | 四波光场、载流子扩散、饱和材料增益、载流子折射率及功率/光子归一化 | 人工给定带边本征值不等于从Fig.2实际SEM独立预测C；本文部分寿命参数本身经过阈值标定 |
| **Inoue 2019官方勘误**，PRB **99**, 169904， [DOI 10.1103/PhysRevB.99.169904](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.169904) | 已读官方1页；2019-04-17发布 | 原式(8)、(A6)-(A9)耦合项漏印 `i`，正确为 `+i(c/ng)C Phi`；作者说明其数值结果未受排版错误影响 | 不能按原文漏印的实耦合项修改代码；现有 `expm(+i vg C dt)` 方向正确 |
| Liang et al. 方形有限器件，OE **20**, 15945 (2012)，[DOI 10.1364/OE.20.015945](https://doi.org/10.1364/OE.20.015945) | 已查Liang学位论文第四章文本、程序理论映射；期刊与学位论文为不同资料 | 有限区域包络、零入射边界、远场是复辐射孔径的Fourier变换 | 用Gaussian/谱峰/论文检查点构造的教学图不是完整三维独立预测 |
| Wang et al., LSA **13**, 44 (2024)，[DOI 10.1038/s41377-024-01387-4](https://doi.org/10.1038/s41377-024-01387-4) | 已读Zotero `6EZGKZWT` 原始8页 | 方形晶格三孔，474 nm周期、90 nm直径、430 nm高、低折射率InP填孔、8.5%总填充；主要增强180°反馈；0.52 A阈值/200 μm圆接触窗口 | 这是**全半导体**低对比度晶格，不是三空气孔；0.60 A有多模，脉冲5 A SMSR约3.2 dB；“低阈值”不能推出理想单瓣或单模 |
| Wang et al., COL **23**, 101401 (2025)，[DOI 10.3788/COL202523.101401](https://researching.cn/ArticlePdf/m00005/2025/23/10/101401.pdf) | 已全文读Zotero `92VC9FJI`；官方期刊PDF核验 | 940 nm、274 nm周期、100 μm方形PC、三晶格 `d=0.5a`；25°C阈值36 mA，50 μm mesa可到12 mA | 实验远场双瓣且A/B多模；3.84°/1.63°为**1/e²全宽**，不能直接和本代码RMS比较；较低阈值伴较低斜率效率 |
| Qiu et al., *Low-Threshold and High-Speed PCSELs Without Regrowth*, OECC/PSC 2025，MD1-4 | 已全文读Zotero `ZAE6XEEA` 原始3页；本轮未确认DOI，不编造 | 1060 nm、20×20 μm发光区，11 mA、18 mW、11.7 GHz；表面250 nm刻蚀，PC1/PC2异质边界、氧化电流孔、15对底DBR | 现有均匀C和零入射边界没有PC反射异质界面或DBR，不能仅替换周期/孔形就声称复现no-regrowth器件 |
| Yoshida et al., Nature **618**, 727 (2023)，[DOI 10.1038/s41586-023-06059-8](https://doi.org/10.1038/s41586-023-06059-8) | 官方论文及作者京都大学repository核验 | 3 mm级高亮度CW需要反馈/辐射选择及热效应处理 | 300 μm冷腔单次计算无法推出50 W功率、长时间CW稳定或热滚降 |
| Katsuno et al., OE **29**, 25118 (2021)，[DOI 10.1364/OE.427783](https://doi.org/10.1364/OE.427783) | 官方题录核验 | CW设计需光子-载流子-热自洽 | 当前固定折射率/固定寿命时域仅是等温模型 |
| Contractor et al., Nature **608**, 692 (2022)，[DOI 10.1038/s41586-022-05021-4](https://escholarship.org/uc/item/69d5259z) | 作者机构原始论文repository核验 | 六角排列open-Dirac、平坦包络为尺度可扩展单模提供不同机制 | 其charge-two远场拓扑奇点不是所要求的中心亮光斑；它也不能证明任意三角晶格优于方形晶格 |
| Zhou et al., LSA **15**, 353 (2026)，[DOI 10.1038/s41377-026-02448-6](https://www.nature.com/articles/s41377-026-02448-6) | 官方原始论文；2026-08-20 | 模损耗差与非线性增益钳制/空间重叠必须共同评估，空间工程可改善大面积单模 | 只计算A/B/C/D或六个Γ连接基模的间隔，不能当作全横模单模裕量 |
| Raftery et al., IEEE Photonics Journal **17**, 0600405 (2025)，[DOI 10.1109/JPHOT.2025.3561087](https://vcsel.mntl.illinois.edu/publications/journals/2025_IEEEPJ_Raftery.pdf) | 作者机构原始PDF与期刊信息核验 | 埋置SiO₂介质特征可缓解InP空气孔回填/变形，1.5 μm光泵原型已证明 | 仍需外延合并生长，不能称完全免再生长，也不能把光泵原型当已验证电泵设计 |

加工不能只按“圆角、角度对齐”给一个奖励分。主孔15°不是理论要求；0°也不会自动降低电子束曝光费用。真实收益应由最小周期孔间距、最小孔宽、圆角半径、刻蚀深宽比、侧壁角、外延回填和实际工艺偏差证明。低损耗的暗模/BIC可能同时降低出光效率；应保留有限垂直耦合，在反馈、单模裕量、有效出光和加工间距之间取Pareto折中。

## 2. 本轮确定性修复

1. `observables.wavelength_spectrum` 原先无条件减复包络均值，删掉参考频率上的真实相干谱线。现在默认保留；`remove_mean=True` 为显式可选分析；新增DC谱线与频率符号回归测试。
2. `reproduce` 原先未把配置波长传给谱图，改波长后仍以950.65 nm绘图。现在传递 `config.optical.wavelength_nm`。
3. `solver._carrier_rhs` 原先把负材料增益受激项截到零。式(10)在透明点以下的光吸收应产生载流子；已恢复其符号，避免场损耗和载流子交换不守恒。
4. `solver._add_spontaneous_noise` 的四波换算已经在 `conversion` 中计入四波，额外除以8使实际注入光子期望少4倍。已按复高斯模方期望2修正到除2。噪声仍是未按实验线宽标定的启动模型。
5. `config.validate_config` 补物理参数有限性、单位符号、增益上下限与二维显式扩散 `D*dt_sub/dx² <= 1/4`。`TimeDomainSolver` 初始化也调用校验，避免直接Python构造绕过YAML校验。
6. 历史方形 `design_square_lattice_multiobjective.include_internal_loss` 原先把功率内损耗 `alpha_in` 全量加到振幅损耗。现在加 `alpha_in/2`，收敛图也同样修正。**以前该脚本保存的total-loss表需重跑**；新独立扫描只保存不含内损耗的光学振幅损耗，阈值统一 `g_modal=2 alpha_mode+alpha_in`。

## 3. 外围src逐文件结论

| 文件 | 评价 | 适用范围/剩余限制 |
|---|---|---|
| `src/pcselsim/__init__.py` | 公共接口/版本声明可用 | 不是独立物理模块；包版本0.1.0不能用作结果证据，需Git SHA及运行清单 |
| `config.py` | 本轮补物理合法性与扩散稳定性校验 | 默认C本征值人工标定；对高强度受激耗尽仍需时间步收敛，扩散稳定不代表所有非线性稳定 |
| `constants.py` | SciPy SI常数无确定错误 | 常数不会纠正材料/损耗参数不确定性 |
| `injection.py` | 归一化总电流与圆/方电极面积正确 | erf横向扩展为经验模型，未求电阻/漂移扩散；域外电流被归一回有限域，须保证窗口足够宽 |
| `materials.py` | 式(11)饱和有理增益、重叠、SI转换与附录投影一致 | 固定寿命、固定dn/dN、无热/谱增益/增益压缩；将半导体参数直接用于Yb:YAG无效 |
| `observables.py` | 本轮修复DC谱线；光子/辐射功率归一化按论文 | 辐射只含C的辐射反厄米部分；FFT线宽受时间窗、采样与启动噪声限制；不能报告绝对实验线宽 |
| `solver.py` | 已按官方勘误使用 `iC`；本轮修复吸收交换和噪声换算 | 四波+等温载流子；Heun/Lax-Wendroff/分裂需独立dx/dt检验；不是six-wave时域；人工/随机种子影响启动时间 |
| `io.py` | NPZ保留原始数组及metadata，可再画图 | 固定同一路径会覆盖旧运行；运行清单应记录源码SHA/版本及配置；图不能替代原始数组 |
| `plotting.py` | 图轴、单位和峰值归一化可用 | 各谱图局部归一化不能比较绝对功率；默认视野可裁掉谱峰；需检查Nyquist频率 |
| `reproduce.py` | 标定回归入口；已修复波长传递 | 不能由相似Fig3/4证明几何真实或完全实验复现 |
| `cli.py` | 只解析参数、调用同一模型 | 名称 `reproduce` 是历史接口；实际证据级别仍是calibrated baseline |

光学相关 `geometry.py`、`vertical.py`、`coupling.py`、`three_d_cwt.py`、`custom_analysis.py`、`band_structure.py`、`triangular_six_wave.py`、`triangular_finite.py` 的逐式评价由光学审计负责；`solid_state.py` 由晶体模型审计负责。三份报告合起来覆盖全部src文件，不能由本表对未列模型作实验正确性认证。

| 本轮新增src文件 | 评价 | 限制 |
|---|---|---|
| `numerical_quality.py` | 网格损耗外推保留负外推/敏感性信息；不再把不可信负值截成零 | 经验外推和误差敏感指标，不是统计置信区间；最后几级仍变化时用有限网格值作为临时估计 |
| `fabrication.py` | 周期邻胞最小间距、最小投影宽度及边缘/位置预算硬筛；Pareto不强迫一个权重 | 2D几何检查不是刻蚀侧壁、晶体缺陷或晶圆良率模型；圆孔边界离散约0.1 nm精度不能替代SEM |

## 4. scripts、配置与环境文件逐文件结论

| 文件 | 证据级别/评价 | 应如何使用 |
|---|---|---|
| `_project_bootstrap.py` | 路径修复有效，独立于PyCharm working directory | 启动后主动chdir到项目；不是物理计算 |
| `run_quick.py` | 环境/流程烟雾测试 | 短时间粗网格，不能作论文定量结果 |
| `run_full.py` | 10 ns calibrated-C四波回归 | 不是几何复现；噪声/谱线本轮修复后旧图需再生成 |
| `reproduce_inoue2019.py` | 同一calibrated-C命令包装 | 只是入口，不提供更完整模型 |
| `reproduce_liang_ch4_step_by_step.py` | 教学、标定、proxy、schematic混合；已有逐图provenance | Gaussian、构造谱峰、检查点拟合等图**不能**用于设计或称为第一性原理复现；有限边界penalty为教学实现 |
| `run_custom_semiconductor_pcsel.py` | 四波几何驱动主入口 | 默认Inoue孔形为两理想椭圆，不是原SEM轮廓；默认Γ取论文值、Ith取0.7 A，须读derived阈值审计；时域400 μm域与300 μm冷腔还应检查等效边缘 |
| `run_custom_wang2024_triple_lattice.py` | 主文晶胞正确，纵向有效拟合而非S1 exact epitaxy | 正确使用InP填孔；InAlGaAs动力学是占位，时域默认OFF应保持；宽17°实验远场不能由粗包络单模图验证 |
| `run_custom_triangular_six_wave_pcsel.py` | Liang六波圆孔示例，半径30 μm | 物理四波以外的独立六波；尚无载流子时域；不得和300 μm方形器件直接比较阈值/发散角 |
| `design_triangular_bright_mode.py` | 历史粗几何筛选，寻找最低**亮模**而非保证最低总模式 | 应同时查看更低暗模，不能把手选亮模当实际激射态；仅一个winner复算不足以证明全局最佳 |
| `design_triangular_three_hole_pcsel.py` | 主孔+两副孔，30 μm半径历史筛选 | 后续版本按实际最低带边连接态评分；仍只六基模代表，不是全横模谱 |
| `design_triangular_three_hole_shape_position.py` | 30 μm孔形/位置多级搜索 | 周期不重叠检查可用；尖角、最小工艺线宽、外延变化当时未硬筛；已有coarse cache须匹配代码/参数 |
| `design_triangular_triangle_main_refinement.py` | 30 μm局部三角主孔优化，后来扩圆角几何工厂 | 固定填充便于比较设计，但**制造误差不能重新调孔径保持填充**；15°结论只属旧离散域 |
| `design_square_lattice_multiobjective.py` | 方形多级搜索，含circle/ellipse/polygon | 已修内损耗半因子；默认fill0.05导致很多副孔太小；旧total-loss表待重跑；加权分不是工艺合格证 |
| `compare_square_triangular_equal_size.py` | 同stack、倒格矢匹配、同300 μm bounding box比较 | 圆/六角/方形面积不同，等bounding box不是等泵浦面积；统一 `2 alpha+alpha_in`，电流仍均匀注入静态估算 |
| `optimize_triangular_three_triangle_manufacturable.py` | 300 μm局部59组、几何加工代理评分 | q=.10圆角不保证角半径可加工；39.5 nm间距不一定符合工厂规则；本轮硬约束新入口优先 |
| `run_best_triangular_three_triangle_pcsel.py` | 完整300 μm六波冷腔/静态泵浦流程 | 名称best是当前候选集建议；泵浦曲线是未饱和小信号增益，不是L-I；全貌图为结构示意 |
| `optimize_square_fabrication_20261008.py` | 本轮300 μm圆/椭圆硬工艺筛选、多网格最终评估 | 原始几何表、nominal/margin两套pass、Pareto与具体结果均保存；不承诺全局最好 |
| `audit_and_optimize_robust_design.py` | 本轮统一几何约束/扰动/六波设计入口 | 周期单胞误差是系统误差，不是百万孔随机无序；仍需全波验证与实际工艺数据 |
| `audit_optical_convergence_20261008.py` | 本轮有限特征网格N=7…25、外推敏感性、远场视野及离散算子损耗分解审计 | 固定D=10、dz=3 nm，**不是**高阶截断D或纵向步长dz扫描；不自动覆盖所有参数域 |
| `analyze_semiconductor_audit_20261008.py` | 本轮默认Inoue几何实际时域NPZ后处理、DC保留FFT数组、功率统计、源哈希核对 | 只有空间积分A投影谱；多个峰不自动等于多个横向本征模；末帧带基系数占比不是完整模式竞争历史 |
| `index_literature_library.py` | Zotero文献目录、SHA256去重、首两页元数据/DOI候选索引 | 81个PDF、78个唯一hash是文件统计，**不是81篇全文科学审计**；OCR/DOI候选需人工确认 |
| `run_custom_ybyag_pcsel.py` | 教学手填C/单模速率示例 | 与几何驱动晶体入口区分；详见晶体审计，不应用于实验阈值认证 |
| `run_geometry_derived_ybyag_pcsel.py` | 几何驱动光学+Yb速率/可达性原型 | 详见晶体审计；真实泵浦吸收/饱和/热条件比调孔形优先 |
| `evaluate_ybyag_pump_architectures.py` | 固定几何激光阈值下的浓度、厚度、单程/多程/侧泵饱和吸收架构比较 | 先判断反转与泵浦预算，不能单凭几何低损耗认定起振；不含真实侧泵均匀性、热及新层结构自洽C |
| `configs/inoue2019.yaml` | 表I-II材料参数+显式标定C | 不是全部字段均由表给出：modal_detuning/loss为标定，domain/current_spread/noise和数值参数含假设 |
| `pyproject.toml` | src-layout与CLI安装声明正确 | 开放最低版本要求不保证可重现环境；正式记录实际numpy/scipy/Python版本 |
| `requirements.txt` | 环境依赖清单 | 依赖可运行不是理论正确或加工成功证据 |

## 5. tests逐文件评价

单元测试验证某项不变量或程序回归，不等价于与实验全量匹配。即使全部通过，也不能写“物理完全正确”。

| 文件 | 已检验的性质 | 尚未由此检验的性质 |
|---|---|---|
| `test_config.py` | YAML值、密度单位、寿命合法性、显式扩散限制 | 完整器件非线性/热稳定性 |
| `test_materials.py` | 透明点、增益饱和、dn/dN与γ投影 | 实测增益温度/波长/浓度谱 |
| `test_injection.py` | 圆形归一化及面积/径向性 | 电流拥挤、接触电阻和结温 |
| `test_observables.py` | 本轮DC峰保留、正时间频率符号、无效输入 | 实验linewidth、噪声PSD标定 |
| `test_solver.py` | 短时有限性、投影、子步、吸收载流子交换、噪声期望 | 10 ns长时dx/dt误差、真正线宽与高电流热响应 |
| `test_entrypoints.py` | 外部工作目录可运行bootstrap | 全部入口完整长期运行 |
| `test_geometry.py` | 零阶面积、Fourier共轭、多边形系数 | 实测SEM侧壁、加工无序 |
| `test_vertical.py` | 对称薄膜TE0为导模 | 全向量/多纵模、实际界面和DBR反射 |
| `test_coupling.py` | 人工C基底正交、eigen/passivity | 由几何推导C的实验精度 |
| `test_three_d_cwt.py` | 对称孔暗态、改孔形改C、高阶重建 | 原论文完整数值/高阶全域误差 |
| `test_custom_analysis.py` | four-wave guard、fill、passivity、proxy远场不为零 | proxy为构造场，不能验证真实激射远场 |
| `test_band_structure.py` | 四支追踪、Γ中心映射、端点标注 | 全布里渊区真实全带结构 |
| `test_triangular_six_wave.py` | 六方向/TE极化、面积共轭、圆角、C6及passivity | 三维full-wave与加工实测 |
| `test_triangular_finite.py` | 特征网格节点、uniform farfield、aperture范围 | Γ连接六态以外的完整横模谱 |
| `test_triangular_three_hole_design.py` | 几何拓扑、不重叠、填充、推荐参数回归 | 其“最佳参数”断言是防误改，不是物理最优证据 |
| `test_equal_size_comparison.py` | 倒格矢匹配、面积排序、损耗增阈值 | 全同面积/同输出效率/同制造规则比较 |
| `test_wang2024_preset.py` | 主文90/474 nm和200 μm接触面积Jth | S1外延、28 cm⁻¹模损耗、实验多模远场 |
| `test_solid_state.py` | Yb增益/阈值、交换守恒、泵浦饱和、能量预算等 | 实验薄膜/晶体泵浦参数与热场；详晶体审计 |
| `test_ybyag_geometry_entrypoint.py` | 1030 nm薄膜预设、单程吸收、空气侧PC位置 | 当前器件实际能够起振 |
| `test_fabrication.py` | 周期邻孔、穿交/包含、etch几何、Pareto | foundry实测良率、侧壁和大面积随机误差 |
| `test_optical_audit_regressions.py` | 非导模拒绝、负外推不截零、C4真实旋转、六波远场方向/采样 | 全面全波误差、全横模单模证明 |

## 6. 比较口径与下一步设计判断

- `alpha_mode`为场振幅衰减；材料内损耗`alpha_in`为功率衰减。本工程比较采用 `g_modal = 2*alpha_mode + alpha_in`，避免阈值和Q出现2倍错误。
- RMS、FWHM、1/e²宽度是不同定义。对**理想高斯的一维切片**，若强度写 `exp(-2 theta²/w²)`，1/e²全宽为 `2w`，FWHM为 `sqrt(2 ln2)*w`，一维RMS标准差为 `w/2`；真实双瓣或环形光斑不满足简单换算。代码的二维径向/轴向RMS应先读其定义，再与论文匹配。
- 包围能量以已采样角域归一化；粗包络网格决定可表示角谱。zero-padding细化取样，不能恢复未解析高角度辐射。17°实验宽度尤其不能和~1.5°归一视野直接比较。
- 四个或六个“基模”只是不同Γ带边连接的最低有限态；同带边的A1/A2等横模可能先竞争。真正单模裕量需要多横模、pump-overlap、空间hole burning和温度自洽。
- 晶格类型本身没有普遍优劣。优先设计低复杂度平滑孔（circle/ellipse），再在不能满足阈值/出光的情况下引入圆角三角和副孔；比较同波长、同层结构、同尺寸、同工艺规则。晶格旋转角只是光学干涉自由度，不是工艺改善的充分证据。
- 原始理想推荐孔间距约39.5 nm时，应先评估系统误差margin而非宣称容易加工。新扫描分别给40/50 nm科研规则与60/65 nm较宽松规则，并显式减去每孔边缘/位置预算；这些是研究假设，最终以实际加工能力为准。

本轮方形新数据写入 `results/square_fabrication_20261008/`。该目录的 `01_geometry_all.csv` 是完整几何筛查，`02_coarse.csv` 为预筛，`03_final.csv` 为多网格结果，`05_recommendation.json` 给具体规则下的候选；可运行 `python scripts/archive/legacy_20261008/optimize_square_fabrication_20261008.py`。修改候选或模型后须换结果目录，避免加载旧coarse缓存。

## 7. 本轮方形平滑孔实算结果

共定义66个结构，53个通过40 nm间距/50 nm最小宽度的名义几何筛选。初筛为D=6、dz=4 nm、N=13；四个finalist用D=10、dz=3 nm及N=13/17/21复算。以下损耗为保守处理后的**1/N外推结果**（负截距被拒绝时才回退最细网格），场及远场来自N=21；必须连同`extrapolation_status`与敏感性查看，不应将其当作已收敛的精确排行榜。所有器件均为300 μm边长、同半导体层结构；泵浦电流为均匀注入静态估算。

| 候选 | 最低带边连接态 | 光学振幅损耗 cm⁻¹ | 其他带边连接基态间隔 cm⁻¹ | 中心/峰值 | 二维RMS全角 ° | 0.5°内采样能量 | 静态Ith A | 最小间距/孔宽 nm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F=.10，d=.45，面积份额.42/.58，椭圆长宽比1.15 | D | 5.495 | 0.0283 | 1.000 | 0.2437 | 99.798% | 0.584 | 111.05 / 59.71 |
| F=.14，d=.45，面积份额.42/.58，圆孔 | D | 7.439 | 0.0523 | 1.000 | 0.2428 | 99.800% | 0.592 | 93.89 / 75.75 |
| F=.12，d=.45，面积份额.42/.58，椭圆长宽比1.15 | D | 6.524 | 0.0362 | 1.000 | 0.2431 | 99.797% | 0.589 | 104.84 / 65.41 |
| F=.10，d=.50，等面积圆孔 | A | 1.124 | 0.0032 | ~10⁻¹⁰ | 0.2435 | 99.800%* | 0.493 | 126.00 / 69.85 |

`d=.45` 指相对主孔沿x、y各偏移`.45a`，不是孔中心欧氏距离`.45a`。以a=277 nm计，位移为(124.65,124.65) nm。两孔总面积F=.14、面积分配42%/58%的圆孔半径约37.90/44.53 nm；可在晶胞整体平移后把主孔放在中心，Fourier系数相位随原点变但物理结果不变。

科研规则下优先F=.10弱椭圆；如果坚持60 nm间距、65 nm宽度并预留每孔3 nm边缘和3 nm定位预算，则本轮只有F=.14圆孔方案通过最终列表的较容易加工规则。其最坏预算后间距81.89 nm、孔宽69.75 nm。前者没有65 nm宽度裕量，因此不能把较低阈值当作“更容易加工”。

这些平滑双孔结构远场中心亮、圆度近1，且无需锐角三角加工，是值得保留的**光斑/加工候选**；但带边连接态间隔仅0.03–0.05 cm⁻¹，远不足以宣称稳健大电流单模。`*`等面积d=.50结构的中心暗态辐射几乎被对称性消除，此时归一化能量/RMS仍能生成漂亮数字，却不代表有效输出光束；应拒绝把它作为高光束质量亮模推荐。必须同时检查绝对辐射效率和最低实际激射态。

填充率改变后，此组Bragg波长为约936–941 nm，不再是950.65 nm。当前静态材料增益没有波长谱，因而电流表只是同一材料增益参数下的结构比较。下一步应先重新求a/纵向折射率以对准目标增益峰，再求多横模损耗及增益重叠；随后加入热自洽和工艺分布数据，而不是继续仅以单个归一化光斑优化。

### 7.1 外推敏感性补充

四个finalist的`extrapolation_status`均为`provisional`，不是“已收敛”。原始复数本征值序列保存在每个finalist目录的`02_convergence_arrays.npz`，逐模式诊断保存在`02_numerical_quality.json`；主CSV和推荐JSON也已补字段。

| 候选 | 报告α cm⁻¹ | N=21实际α cm⁻¹ | 外推敏感量 cm⁻¹ | 状态 |
|---|---:|---:|---:|---|
| F=.10 / d=.45 / w=.42 / aspect1.15 | 5.4951 | 11.5280 | 0.2651 | provisional |
| F=.14 / d=.45 / w=.42 / circles | 7.4385 | 13.4690 | 0.2649 | provisional |
| F=.12 / d=.45 / w=.42 / aspect1.15 | 6.5237 | 12.5561 | 0.2650 | provisional |
| F=.10 / d=.50 / w=.50 / circles | 1.1236 | 7.1576 | 0.2651 | provisional |

外推敏感量是拟合窗口/残差的经验指标，不是概率误差条。其约0.265 cm⁻¹已经大于亮候选的C/D带边连接态间隔0.028–0.052 cm⁻¹，所以尚不能认定这种微小最低模式排序被数值分辨；需要更多网格并同时核对成对损耗差。仅把`provisional`理解为“截距正且拟合敏感量未超过其25%”即可，不能把它解释为实际器件误差不超过25%。

## 8. 修复后默认Inoue几何的实际端到端复跑

本轮没有改默认preset或为符合曲线调整参数。实际执行：

```bash
E:/Anaconda/envs/env00/python.exe -u scripts/archive/legacy_20261008/run_custom_semiconductor_pcsel.py --output results/custom_semiconductor_audit_20261008 --only 01_parameters 02_lattice 03_k_space 04_layer_stack 05_band_structure 05_linear_modes 06_mode_atlas 08_time_domain
E:/Anaconda/envs/env00/python.exe scripts/archive/legacy_20261008/analyze_semiconductor_audit_20261008.py --directory results/custom_semiconductor_audit_20261008
```

主程序与最终后处理均正常退出。已跑参数/C矩阵报告、晶胞、k空间、纵向模、局部M–Γ–X能带、N=13/17/21/25冷腔、包络/晶胞/远场以及**全部四个默认电流×10 ns**。仅跳过07器件尺寸扫描；本轮未跑时间步/时域网格收敛、多随机种子、全部横模谱、温度自洽或实验光谱线宽认证。源代码保留新DC、噪声和吸收交换修正。

实际环境为Python3.10.18、NumPy1.26.0、SciPy1.11.0。晶格a=277 nm，理想双椭圆总填充4.618%，求得Bragg波长949.178704 nm、neff=3.426638、ng=3.514610；最低带边连接冷腔态A，振幅损耗7.621662 cm⁻¹，均匀注入静态阈值0.603342 A。程序按原preset保留论文参考Ith=0.7 A，**1.05倍论文阈值不是本几何的1.05倍静态阈值**。冷腔为300 μm，时域为400 μm、31×31网格、300 μm电极与25 μm电流扩展；默认时域的均匀C作用于整个计算域，不能把外围自动当作完全无光子晶体的边缘区域。两种域的阈值不能无条件相同。

| I/Ith（Ith=0.7 A） | 电流A | 相对静态均匀阈值 | 末1ns平均辐射功率W | 全程最大功率W | 末1ns功率CV | A投影主峰nm | ≥5%显著谱峰数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.05 | 0.735 | 1.218 | 0.071949 | 0.396084 | 1.783% | 949.153169 | 1 |
| 1.40 | 0.980 | 1.624 | 0.227168 | 1.377712 | 0.759% | 949.151666 | 1 |
| 2.80 | 1.960 | 3.249 | 0.965440 | 4.996594 | 3.039% | 949.148661 | 1 |
| 4.20 | 2.940 | 4.873 | 1.731560 | 8.541258 | 5.167% | 949.147159 | 1 |

全部原始数组为有限数。最小电流图呈衰减弛豫振荡后趋稳，高电流功率起伏增大；相邻半纳秒均值漂移约−1.07%、+0.42%、+0.30%、−0.10%。瞬态峰值含启动时储存载流子的释放，不能直接当作CW额定输出功率。

**光谱观测盲点必须保留：** `complex_signal = ∫ v_A†Phi dxdy`是相干空间积分，不是把所有位置/出射角度的光谱强度非相干相加。奇对称A高阶横模可能在积分中抵消。因此“只有一个明显A投影峰”不证明器件只有一个横模；本轮观测量**未证实**论文的高电流两模转换，也不能据它独立断言这种转换不存在。末帧A带基系数权重约98.6%–99.0%只是一瞬间的四方向基分解；本征基Gram偏差0.0588，权重还不是严格正交模光子能量占比。可靠模式竞争检查应保存多个局部/角度复场或各有限横模投影的时间序列，计算`sum |FFT(signal_j)|²`，并明确物理探测器的收集方式。

采样要求0.1 ps，但实际受时间步限制为125.0504 fs，共79968步，结束10.0000316 ns；末2ns谱窗频率间隔0.499986 GHz，Hann窗默认保留相干DC。谱峰的有限窗宽不能报告为实验激光线宽。新目录的`08_I_*.npz`保存功率、载流子、复信号和最终四波场；对应`08_I_*_spectrum_arrays.npz`保存完整波长、频偏、归一谱、任意单位原FFT功率和FFT窗口信号。原FFT功率不是瓦特。`09_postprocess_summary.csv`/`09_postprocess_diagnostics.json`保存精确统计，`09_full_span_spectra.png`显示已表示的完整光频范围，避免仅看主程序窄视野隐藏远离参考的峰。

参数没有全部由几何独立求得：使用Γ=0.044论文值，而同一理想几何纵向解Γ=0.027826；Ith、τ=1.5 ns也保留论文标定参数。双椭圆不是实际SEM轮廓。故这是修复后同理论方程的**几何理想化、部分论文标定、等温数值基线**，不能称为完全重建实验、已恢复论文全部结果或由加工误差解释所有差异。

### 8.1 运行源代码指纹

`00_run_provenance.json`保存命令、选中/跳过步骤、全部src及入口的起始/结束SHA256、环境版本和后处理脚本SHA256。Git HEAD为`1f92e5f4e33728b41dfc4c96067cef70c57f6bf9`，但工作树含本轮未提交修复，所以只给这个commit号不足以复现。关键指纹为：

| 文件 | SHA256 |
|---|---|
| `scripts/archive/legacy_20261008/run_custom_semiconductor_pcsel.py` | `951f3ff01f100e6762428165e493f578b90cc4c06885e2cf405a53078075fc7e` |
| `src/pcselsim/solver.py` | `6defdaf73320b8b58e5278d99dfa381deb26f171cca26b82e2ae7dad8d42686a` |
| `src/pcselsim/observables.py` | `11f95ac33cf73a020c657b208f583b9ee895661620e18da66968fdf307dd8181` |
| `src/pcselsim/materials.py` | `72331bafdae6aee64b95ba8c2c4c17f29d3f39967228a10456f773a96be3d82d` |

半导体运行所导入的核心文件起始/结束哈希均一致。整体src清单中`solid_state.py`在另一项晶体审计工作中变化，但半导体入口、包初始化和本次调用链没有导入它；运行清单单列了无关变化和`semiconductor_source_changed_during_run=[]`，没有隐藏并行修改。
