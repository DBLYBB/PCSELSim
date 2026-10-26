# 优化脚本导航 / Optimization script map

本目录保留可独立复算的研究阶段，不改写日常入口的默认参数。先读[同尺寸运行指南](../../docs/matched300_run_guide_20261021_zh.md)，再选择对应阶段。正式四腔均为 **300 × 300 μm 正方形物理边界**。

These research scripts keep search, validation and finalization separate. A screen winner is a candidate until independent finite modes and actual frequencies have been checked. Daily entry-point defaults remain separate from the optimized configurations.

## 审查后重跑 / Post-audit rerun, 2026-10-25

最新绑定结果见[严格四腔清单](../../results/current/strict_rerun_20261025/comparison/selected_cases.json)和[总报告](../../docs/strict_rerun_four_cavities_20261025_zh.md)。所有入口写新目录，历史冻结数据只读；各分支README/报告保留准确命令和运行源SHA。

|阶段 / Stage|入口 / Entry|验收边界 / Qualification|
|---|---|---|
|半导体正方 / Semiconductor square|[strict_rerun_sem_square_20261025.py](strict_rerun_sem_square_20261025.py)|多参数筛选、暗/条纹模挑战、实际周期、逐根闭频与D/N控制|
|半导体三角 / Semiconductor triangular|[strict_rerun_sem_tri_20261025.py](strict_rerun_sem_tri_20261025.py)|完整粗谱、有界细谱、非同对齐网格与工差尺度下的取舍|
|YAG正方 / YAG square|[strict_rerun_yag_square_20261025.py](strict_rerun_yag_square_20261025.py)、[跨簇挑战](strict_yag_square_branch_challenges_20261025.py)|最低32根不足以代表全部频段，补Γ/不同δ簇后合并共享泵判定|
|YAG三角 / YAG triangular|[strict_rerun_yag_tri_20261025.py](strict_rerun_yag_tri_20261025.py)、[有界谱](strict_yag_tri_bounded_20261025.py)、[正式读回与共享泵](strict_yag_tri_review_20261025.py)|两支局扫只作初筛；原K/M验收、显式超时、细网格条纹、实际频率及跨簇根|
|新YAG输入输出 / New YAG pump studies|[strict_yag_feasibility_20261025.py](strict_yag_feasibility_20261025.py)|只读最终合并模式；共享空间CW与均匀单模脉冲分开，外部J/内部W不混用|
|新YAG尺寸 / New YAG sizes|[正方](strict_yag_square_size_20261025.py)、[三角](strict_yag_triangular_size_20261025.py)|固定孔形和物理周期，逐尺寸重求实际本征场；300μm使用正式合并根锚点|

QR/反演/插值只改变数学求解或提供初值；不更改物理算符、损耗、实频方程或模式身份。所有未解补充目标明确保留，不能把数量配额叫作完整细谱。圆度和间隔筛选线是工程目标，不是物理可行性的硬边界。

## 从哪里开始 / Where to start

| 任务 / Task | 主入口 / Entry | 输入与输出 / Inputs and outputs |
|---|---|---|
| 正方主孔加对角副孔扫描 / Square motif search | [matched300_square_search.py](matched300_square_search.py) | `Design` 参数列表；紧凑参数及光学指标 / Design records and compact optical metrics |
| 正方完整粗谱和精度检查 / Square spectrum and accuracy | [matched300_square_validate.py](matched300_square_validate.py)、[matched300_square_mesh.py](matched300_square_mesh.py) | 原始 Design 列表与新输出目录 / Raw design list and a fresh output directory |
| 正方实际周期与闭频 / Square physical period and frequency closure | [matched300_square_finalize.py](matched300_square_finalize.py) | 已保存的同几何参考场；实际闭频模式和复场 / Same-geometry seeds to closed poles and complex fields |
| 三角主孔加两副孔扫描 / Triangular motif search | [matched300_triangular_search.py](matched300_triangular_search.py) | `sobol/grid/local/rescreen/refine/audit` 阶段 / Staged geometry screening |
| 三角完整粗谱引导 / Triangular coarse-to-fine spectrum | [matched300_triangular_qz_guided.py](matched300_triangular_qz_guided.py) | `--stage coarse/fine`；保留完整标量谱及必要种子 / Complete coarse scalar spectrum and fine-grid seeds |
| 三角实际周期 / Triangular physical period | [matched300_triangular_finalize.py](matched300_triangular_finalize.py) | 单个候选 row JSON，不能传候选列表 / One candidate record, not the whole catalogue |
| 三角逐根闭频 / Triangular individual frequency closure | [matched300_triangular_close.py](matched300_triangular_close.py) | 含 `summary.json` 和种子复场的目录 / Seed directory with its summary and fields |
| YAG 两晶格统一扫描 / Both YAG lattices | [matched300_yag_search.py](matched300_yag_search.py) | `benchmark/scan/refine/close`；相同垂直结构与泵浦定义 / Common stack and pump definitions |
| YAG 已知亮暗分支局修 / YAG known-branch refinement | [matched300_yag_two_mode_refine.py](matched300_yag_two_mode_refine.py) | 两条已知模式的局部优化；不能代替新几何的全谱挑战 / Two tracked roots, followed by a new spectral challenge |
| YAG 保存场核验 / YAG saved-field checks | [matched300_yag_readback.py](matched300_yag_readback.py) | 从正式复场核对残差、通量与阈值 / Residuals, fluxes and thresholds from stored fields |

## 2026-10-25 YAG研究 / YAG feasibility

- [yag_feasibility_study.py](yag_feasibility_study.py)：`cw` 重算保存32模基底的共享反转输入输出；`size` 固定孔形/周期并逐尺寸重新闭频；`sensitivity` 检查材料和内部损耗。All powers count first-entry pump once; the 1029–1031 nm target window is a conditional subset, not a physical filter.
- [yag_pulsed_feasibility_20261025.py](yag_pulsed_feasibility_20261025.py)：单模均匀反转矩形泵脉冲输入能量与输出能量。Uniform-inversion, fixed-mode pulses do not replace spatial multimode or thermal validation.
- 解释、参数来源与适用范围见[新YAG报告](../../docs/yag_feasibility_20261025_zh.md)。

## 辅助阶段 / Supporting stages

- `*_local.py`、`*_actual_local.py`、`*_anchors.py`：局修、实际波长附近调整与参数锚点 / Local changes, actual-wavelength refinement and anchors.
- `*_audit.py`、`*_stripe_probe.py`、`*_stripe_diagnostic.py`：扩大已知模式集合，寻找更低的暗模或条纹模 / Challenge the selected mode with additional complex poles.
- `*_mesh*.py`、`*_resolution.py`、`*_branch_resolution.py`、`*_shape_convergence.py`：网格、高阶截断、纵向步长和轮廓精度 / Numerical sensitivity at fixed physical geometry.
- `*_tolerance.py`、`*_process_vertical_audit.py`：指定几何扰动和纵模适用性 / Stated process perturbations and vertical-family applicability.
- `*_report.py`、`*_summary.py`、`*_pareto.py`、`*_package.py`、`*_release.py`：整理已经完成的数据，不构成独立物理验证 / Assemble completed data, with validation reported separately.
- `*_compact.py`：按明确保留规则压缩扫描数组 / Compact intermediates under explicit retention rules.

无 `matched300_` 前缀的旧优化入口对应此前尺寸或研究问题。保留它们用于追溯，不将其结果自动加入本轮排名。

Other optimization scripts retain earlier sizes and research questions. Their results are historical comparisons rather than members of the matched-size ranking.

## 容易混淆的量 / Quantities to keep distinct

| 名称 / Name | 含义 / Meaning |
|---|---|
| `a`、`lattice_nm`、`period_nm` | 物理晶格常数，单位 nm / Physical lattice period in nm |
| 孔的无量纲坐标 / Fractional coordinates | 相对于晶格常数或基矢；以导出 `actual_geometry_nm.json` 的 Cartesian nm 边界为准 / Convert through the stated basis and use the exported physical boundary |
| `D`、`order` | 面内高阶截断 / In-plane harmonic truncation |
| `dz` | 纵向离散步长，单位 nm / Vertical spacing in nm |
| square `grid` / triangular `cells` | 两种不同网格参数，相同整数不表示相同物理精度 / Different grid definitions, not directly interchangeable |
| `alpha_field` | 振幅衰减；计算强度损耗时需要因子 2 / Amplitude decay, with a factor of two for intensity loss |
| `g_th` | 有源材料增益阈值，须包含增益重叠权重 / Active-material gain threshold including the overlap weight |
| Γ 点能带 / Γ-point bands | 周期晶格的 Bloch 模色散 / Infinite-periodic local dispersion |
| 有限腔模式 / Finite-cavity modes | 给定边界和实际周期下的有限横模 / Transverse modes of the bounded cavity |
| `g₂/g₁−1` | 已查材料增益间隔 / Tested cold material-gain gap |

材料增益间隔、电流阈值间隔、YAG 零光子共享泵浦分支交叉间隔和饱和 CW 模式开启点分别保存，不能互换。静态汇总优先看材料阈值、真实闭频模式和相应复数近远场。

Material-gain gaps, current gaps, zero-photon pump crossings and saturated CW turn-on powers have different definitions. Static reports use actual closed modes, material thresholds and their own complex fields.

## 数据和来源 / Data and provenance

正式参数在 [`configs/optimized_matched300_20261021`](../../configs/optimized_matched300_20261021/)，统一模式身份和路径在 [selected_cases.json](../../results/current/matched300_20261021/comparison/selected_cases.json)。优先使用[诊断入口](../diagnostics/README.md)读回正式数据，再决定是否重新扫描。

2026-10-22 的代码整理只增加注释和说明，历史运行 SHA、最终复场和数值汇总继续保持原样。当前源文件字节会因注释改变；相应的可执行 AST / token 对照另存于 `results/current/code_documentation_20261022/`，不倒填到旧运行记录。

The documentation pass changes source bytes while preserving the executable numerical structure. Historical runtime hashes remain historical. New documentation audits record the change separately.
