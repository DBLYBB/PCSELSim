# 文档导航

## 当前版本：2026-10-25 YAG可行性与物理审计

- [Yb:YAG 泵浦功率/脉冲能量、尺寸阈值与可行性](yag_feasibility_20261025_zh.md)
- [全仓物理严格性：覆盖清单、真实修复与剩余近似](physics_audit_overall_20261025_zh.md)
- [光学矩阵、纵向模、Green与非Bragg归一化专项](physics_audit_optics_20261025_zh.md)
- [有限离散、分支身份、半导体动态及旧入口专项](physics_audit_algorithms_20261025_zh.md)
- [YAG速率、晶体参数来源与脉冲模型专项](yag_rates_literature_audit_20261025_zh.md)

## 2026-10-22 同尺寸优化（冻结结果）

- [300 μm 同尺寸四腔优化、竞争模式与偏振](optimization_matched300_comparison_20261021_zh.md)
- [同尺寸优化：参数载入、闭频和验证运行指南](matched300_run_guide_20261021_zh.md)
- [优化脚本的中英文阶段导航](../scripts/optimization/README.md)
- [静态读回、指标与来源诊断导航](../scripts/diagnostics/README.md)
- [同尺寸正方半导体：阈值与隔离取舍](optimization_matched300_square_20261021_zh.md)
- [同尺寸三角半导体：完整粗谱与细网格复核](optimization_matched300_triangular_20261021_zh.md)
- [同尺寸正方/三角 Yb:YAG：实际共享泵浦竞争](optimization_matched300_yag_20261021_zh.md)
- [代码修正、理论依据、新结果与剩余边界](corrections_20261021_zh.md)
- [结果保留、清理记录与新运行索引](../results/README.md)
- [此前不同尺寸三类腔优化及保留参数](optimization_three_cavities_20261021_zh.md)
- [正方晶格优化与完整横模核查](optimization_square_20261021_zh.md)
- [三角晶格低阈值与近圆斑优化](optimization_triangular_20261021_zh.md)
- [三角晶格尺寸、旁瓣和制造取舍](optimization_triangular_size_beam_20261021_zh.md)
- [Yb:YAG 优化与材料泵浦敏感性](optimization_ybyag_20261021_zh.md)
- [三条路线的物理比较和三角晶格研究价值](cavity_comparison_research_20261021_zh.md)

以下带旧日期的文档保留历史脉络。2026-10-21 按用户要求删除了 `results/archive/`、初筛扫描和重复大矩阵；旧文档的相关路径只作历史记录，现存数据以[结果导航](../results/README.md)为准。读取其中阈值、Gamma 和模式排序前，先看本轮修正记录。

## 当前入口与晶体研究资料

- [三个半导体主程序：参数、步骤开关和归档迁移](entrypoints_zh.md)
- [Yb:YAG-PCSEL 候选：实跑结果与不可省略的限制](ybyag_crystal_run_20261008_zh.md)
- [波导材料、加工与独立底部DBR](ybyag_waveguide_materials_20261008_zh.md)
- [分层反射Green推导及上下功率守恒](stratified_optics_derivation_20261008_zh.md)
- [空间泵浦、增益和包络烧孔](ybyag_dbr_pump_design_zh.md)
- [YAG 增厚与泵浦研究：操作、正式比较和频率敏感性](ybyag_thickness_pumping_study_20261008_zh.md)
- [多纵向TE模与膜吸收的独立检查](ybyag_independent_checks_20261008_zh.md)
- [片上超快：自锁模、调Q、独立Kerr微腔](ybyag_ultrafast_routes_zh.md)

日常只运行根 scripts 中三个半导体入口和一个 YAG 入口；旧脚本移入
`scripts/archive/legacy_20261008`，保留作为历史扫描、教学与回归对照。

## 历史审计与背景资料（2026-10-08）

1. [2026-10-08 总评、设计建议与新运行方法](research_design_audit_20261008_zh.md)
2. [光学核心逐文件审计](audit_optical_20261008_zh.md)
3. [半导体代码、国内外文献与平滑孔设计](audit_semiconductor_literature_20261008_zh.md)
4. [晶体代码、文献与泵浦设计](audit_crystal_literature_20261008_zh.md)
5. [初学者运行指南](beginner_guide_zh.md)
6. [输出目录与可信度](../results/README.md)

三份分项审计包含历史逐文件评价。其修正前源清单已随旧结果清理；本次删除文件的路径、大小和 SHA256 统一保存在 `results/_cleanup_20261021/`，不是可恢复备份或物理认证。

## 理论与模型边界

- [理论与代码映射](theory_zh.md)
- [几何驱动3D-CWT](geometry_derived_3dcwt_zh.md)
- [三角晶格六波](triangular_six_wave_zh.md)
- [Inoue–Liang模型关系](liang_inoue_model_relationship_zh.md)
- [自定义主程序使用指南](custom_simulation_guide_zh.md)
- [Yb:YAG模型](ybyag_model_zh.md)
- [Wang 2024三晶格对照](wang2024_triple_lattice_zh.md)
- [早期复现状态](reproduction_status_zh.md)

## 历史算例与设计演变

下面保留原始数据和思路，**其中旧阈值、最佳排序及工艺推荐不得优先于2026-10-21修正记录**；2026-10-08审计本身也属于历史证据。
尤其旧三角低网格外推存在明显数值耗散，旧方形多目标扫描存在内损耗振幅/功率口径错误。

- [三角三孔完整仿真](best_triangular_three_triangle_full_simulation_zh.md)
- [三角三孔加工约束扫描](triangular_three_triangle_manufacturable_optimization_zh.md)
- [同尺寸方形—三角比较](equal_size_square_triangular_comparison_zh.md)
- [六波有限器件与亮心设计](triangular_finite_design_zh.md)
- [主孔—双轴副孔](triangular_three_hole_optimization_zh.md)
- [孔形—位置联合优化](triangular_shape_position_optimization_zh.md)
- [三角/方形多目标扫描](triangular_square_multiobjective_optimization_zh.md)
- [早期物理审计与中英文代码地图](physics_audit_and_code_map_zh_en.md)

## 工程操作

- [Git版本管理与多端同步](git_workflow_zh.md)
