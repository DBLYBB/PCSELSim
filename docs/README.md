# 文档导航

## 整理后的入口与新晶体设计

- [三个半导体主程序：参数、步骤开关和归档迁移](entrypoints_zh.md)
- [Yb:YAG-PCSEL 候选：实跑结果与不可省略的限制](ybyag_crystal_run_20261008_zh.md)
- [波导材料、加工与独立底部DBR](ybyag_waveguide_materials_20261008_zh.md)
- [分层反射Green推导及上下功率守恒](stratified_optics_derivation_20261008_zh.md)
- [空间泵浦、增益和包络烧孔](ybyag_dbr_pump_design_zh.md)
- [多纵向TE模与膜吸收的独立检查](ybyag_independent_checks_20261008_zh.md)
- [片上超快：自锁模、调Q、独立Kerr微腔](ybyag_ultrafast_routes_zh.md)

日常只运行根 scripts 中三个半导体入口和一个 YAG 入口；旧脚本移入
`scripts/archive/legacy_20261008`，保留作为历史扫描、教学与回归对照。

## 当前审计与运行

1. [2026-10-08 总评、设计建议与新运行方法](research_design_audit_20261008_zh.md)
2. [光学核心逐文件审计](audit_optical_20261008_zh.md)
3. [半导体代码、国内外文献与平滑孔设计](audit_semiconductor_literature_20261008_zh.md)
4. [晶体代码、文献与泵浦设计](audit_crystal_literature_20261008_zh.md)
5. [初学者运行指南](beginner_guide_zh.md)
6. [输出目录与可信度](../results/README.md)

三份分项审计包含逐文件评价。源文件清单与SHA256见
`results/robust_design_20261008/01_source_inventory.json`，它是语法/版本索引，不是物理认证。

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

下面保留原始数据和思路，**其中旧阈值、最佳排序及工艺推荐不得优先于2026-10-08审计**。
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
