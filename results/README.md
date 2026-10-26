# 结果导航

## 最新：2026-10-25 审查后四腔重跑

- [四腔正式科学清单](current/strict_rerun_20261025/comparison/selected_cases.json)、[指标CSV](current/strict_rerun_20261025/comparison/four_cavity_metrics.csv)、[几何/真实近远场/偏振总图](current/strict_rerun_20261025/comparison/four_cavity_overview.png)：全部主腔为300×300μm方形边界，实际首模分别满足950/1030±0.05nm。
- [四腔总报告](../docs/strict_rerun_four_cavities_20261025_zh.md)：新旧比较、独立原方程读回、近圆亮中心与模式间隔的取舍、三角研究价值及未解决范围。
- [新几何YAG泵浦/能量/尺寸研究](current/strict_rerun_20261025/yag_feasibility/)及[说明](../docs/strict_yag_feasibility_20261025_zh.md)：与下面旧几何的同日研究分开，不混用输入数据。
- 各分支的`selected_case.json`是最新实际复场入口；扫描大数组按有SHA清单的规则清理，正式复场、参数、必要取舍、失败记录和数值控制保留。历史冻结目录不回填新结果。

## 2026-10-25 新增研究与严格性复核

- [YAG泵浦—输出、尺寸、材料敏感性与脉冲数据](current/yag_feasibility_20261025/)：78个32模共享反转CW样点、逐尺寸实际本征场/频率、26组材料敏感性、124个单模脉冲响应与16组额外验证。常谱尺寸扫描与目标窗条件阈值分别记录，不能当完整实测谱预测。
- [全仓物理审计与回归](current/physics_audit_20261025/)：逐文件覆盖、文献式号、独立解析/积分/守恒/负对照、修复后回归，以及非同对齐三角网格诊断。测试通过不代替全波或实验验证。
- 旧冻结结果继续只读；特别是旧三角模式间隔没有因去重修复而自动重新取得全谱保证。下文清理大小是2026-10-21的历史快照，不代表新增研究后的当前磁盘用量。

2026-10-21 按用户要求完成结果瘦身：保留修正后的文献复现、当前三类腔的最终结果与必要验证，删除初筛扫描、重复大矩阵、修正前历史结果和开发中间态。原有必要路径不搬动，以便现有报告和最终复场继续读取。

清理前共有 7,033 个文件、8,419,708,308 字节；删除 6,334 个文件、8,052,841,161 字节（约 7.50 GiB）。原始保留数据为 699 个文件、366,867,147 字节（约 350 MiB），另有很小的清理清单与导航。

删除文件的路径、大小和 SHA256 已保存，**清单不是备份，不能恢复被删的原始矩阵**。具体见[清理报告](_cleanup_20261021/cleanup_report.md)、[删除清单](_cleanup_20261021/deleted_files.csv)、[清理前完整清单](_cleanup_20261021/inventory_before.json)和[清理后校验](_cleanup_20261021/integrity_after.json)。

## 1. 文献复现与修正后的基础结果

| 结果 | 保留目录 | 范围 |
|---|---|---|
| Inoue 双椭圆论文冷腔 | [paper_corrected_cold_r2](current/20261021_paper_corrected_cold_r2/) | 原始复场、图和 D10/dz3 下有限腔 N13–25 网格检查 |
| Inoue Γ 点带间距专项审计 | [inoue_gamma_spacing_audit](current/20261021_inoue_gamma_spacing_audit/audit.json) | 论文读图、C 项消融和 D10/30/50、dz3/1 小范围复算；Γ 分裂尚未复现，高阶截断未充分收敛 |
| 论文四电流、各 10 ns 时域 | [paper_corrected_time_domain](current/20261021_paper_corrected_time_domain/) | 四份完整时域 NPZ、36 通道谱与后处理来源 |
| 正方双圆基准 | [square_corrected_cold_r2](current/20261021_square_corrected_cold_r2/) | 修正后的冷腔与远场 |
| 三角三孔基准 | [triangular_staggered_final](current/20261021_triangular_staggered_final/) | 修正后的六波冷腔；原小模式间隔仍未解决 |
| Yb:YAG 8 μm 基准 | [ybyag_frequency_D6_N17_r4](current/20261021_ybyag_frequency_D6_N17_r4/) | 固定周期16支、共有反转与功率预算 |
| 连续 DFB/薄层积分独立基准 | [independent_validation](current/20261021_independent_validation/) | 解析/独立离散对照 |
| Liang 原表对照 | [finite_validation](current/20261021_finite_validation/) | 原表仍未完整复现，不将尺寸缩放当作吻合 |

当前方法与局限见[修正和验证记录](../docs/corrections_20261021_zh.md)。历史 203 项测试及当时的完整性审计仍在 [revision_validation](current/20261021_revision_validation/) 与 [final_integrity](current/20261021_final_integrity/)；其旧归档哈希是当时的检查记录，不能据此推断旧文件现在仍在磁盘上。

## 2. 2026-10-22 同尺寸历史冻结

此前比较于2026-10-22完成，采用四个 **300×300μm正方形物理腔**。这是历史冻结轮，最新结果见本页顶部的审查后清单。该轮结论、失败反例与范围见[此前同尺寸报告](../docs/optimization_matched300_comparison_20261021_zh.md)，参数在`configs/optimized_matched300_20261021/`。

| 文件或目录 | 用途 |
|---|---|
| [正式比较清单](current/matched300_20261021/comparison/selected_cases.json) | 已完成的四个主方案；逐项绑定 summary、真实孔边界、复场、数组键和实际首模 |
| [四腔总图](current/matched300_20261021/comparison/four_cavity_overview.png) / [指标 CSV](current/matched300_20261021/comparison/four_cavity_metrics.csv) | 对应的实际晶格、矢量远场与偏振；共同角窗下的数值 |
| [实际闭频模式](current/matched300_20261021/comparison/four_cavity_closed_mode_gains.png) / [损耗账本](current/matched300_20261021/comparison/threshold_loss_ledger.json) | 已查模式间隔和阈值来源，区分材料增益与器件运行阈值 |
| [取舍图](current/matched300_20261021/comparison/design_tradeoffs.png) / [YAG 静态 CW](current/matched300_20261021/comparison/YAG_shared_CW.png) | 九个保留设计与两种 YAG 主方案的共享反转模式竞争 |
| [square](current/matched300_20261021/square/) | 中央主孔加对角副孔；保留低阈值、圆度及隔离 Pareto 方案 |
| [triangular](current/matched300_20261021/triangular/final_summary.json) | 中央主孔加两个方向副孔；含完整粗谱、细网格条纹挑战及正式闭频证据 |
| [yag](current/matched300_20261021/yag/final_summary.json) | 正方四波/三角六波分层 Green、共享反转 edge/top 首阈值和 CW 竞争 |
| [independent_review](current/matched300_20261021/independent_review/) | 从保存复场重建耦合、原始本征残差、实际频率、Jones 与共享泵浦 |
| [regression](current/matched300_20261021/regression/) | 回归结果及测试期间的源文件哈希 |
| [最终存储与完整性](current/matched300_20261021/comparison/release_storage_integrity.json) | 九个保留设计、1468 个有限数值数组；本轮有记录清理释放 3.164 GiB，旧文献与科学结果保持原 SHA |

扫描阶段只保留紧凑参数、标量记录和必要反例，正式复场及便携配置继续保留。被清理的扫描数组不能从删除清单恢复；重算命令见[运行指南](../docs/matched300_run_guide_20261021_zh.md)。

## 3. 此前不同尺寸优化，保留作历史

此前结构与独立配置见[三类腔总报告](../docs/optimization_three_cavities_20261021_zh.md)及 `configs/optimized_20261021/`。其中混有不同尺寸，不能与上面的 300 μm 四腔结果直接排名。

| 分支 | 最终汇总 | 必要原始场和验证 |
|---|---|---|
| 正方半导体 | [final_report](current/optimization_square_20261021/final_report/) | [165 μm：D12/dz1/N65](current/optimization_square_20261021/finalist_L165_D12_z1_N65/N65/)、[200 μm：D10/dz2/N65](current/optimization_square_20261021/closed_full_spectrum_L200_D10_z2/N65/) 的 `frequency_closed_poles.npz` 完整保留；各网格、独立工差的 JSON/CSV 仍在原路径 |
| 三角半导体 | [final_report_v3](current/optimization_20261021/triangular/final_report_v3/) | 三种方案各前8个实际极点的压缩复场；[400 μm独立实频闭合](current/optimization_20261021/triangular/frequency_closed_400um_v1/) 的前4根完整复场 |
| 三角尺寸与旁瓣 | [size_scaling_v1](current/optimization_20261021/triangular/size_scaling_v1/)、[beam_audit_v2](current/optimization_20261021/triangular/beam_audit_v2/)、[400 μm密角远场](current/optimization_20261021/triangular/scaled_400um_beam_v2/) | 尺寸扫描只留表、图与 JSON，删除每个扫描点的大复场；301×301最终复远场保留 |
| Yb:YAG | [final](current/optimization_ybyag_20261021/final/) | [D12/N37/dz1的19个极点原始场](current/optimization_ybyag_20261021/refined_19poles_D12_N37_dz1/)、[材料和泵敏感性](current/optimization_ybyag_20261021/material_pump_sensitivity_D12_N37/)、[光谱排序反例](current/optimization_ybyag_20261021/spectral_adversarial_D6_N21/)；其他收敛、尺寸和工差只留紧凑 JSON/CSV |
| 同参数晶格对照 | [controlled](current/optimization_20261021/controlled/like_for_like_v2/) | 对照数组、图、参数和来源完整保留 |
| C4/C6/C2 机制 | [symmetry](current/optimization_20261021/symmetry/dispersion_C4_C6_C2_multiscale_v2/) | 多尺度诊断完整保留；不作为器件优劣排行榜 |

三角 [final_report_v2](current/optimization_20261021/triangular/final_report_v2/) 只保留旧汇总、图与初筛表，最终复场统一看 v3。各 `summary.json`、`source_provenance.json` 和 `source_sha256` 保留真实运行时的原始内容；其中部分来源路径属于本次删掉的扫描，不表示那些源文件仍然存在。重建最终候选应使用独立配置和各分支报告的复算命令。

新增接口后的[完整回归](current/20261021_optimization_final_regression/)为 221 项通过。[原最终证据清单](current/optimization_20261021/final_evidence_readback/evidence_manifest.json)中的 62 份选定文件全部保留且 SHA256 未变。清理前后读回了保留的 65 份 NPZ / 935 个有限数值数组、452 份 JSON 与 135 张 PNG；这些检查确认文件完整性，不代替独立光学或器件验证。

## 4. 后续运行与历史文档

新仿真仍写入 `current/` 的新空目录；默认主入口使用 `current/pcsel/<类型>/<时间戳>/` 或 `current/ybyag_crystal_pcsel/<时间戳>/`。不要覆盖当前保留证据。材料增益阈值、电流阈值和内部泵功率不可混用；远场须一并引用角窗和指标定义。

`archive/` 已删除。旧日期文档中的该路径以及已删的扫描路径仅保留历史计算脉络，不再是可访问的数据交付。代码和文档仍在，旧模型结果不能与修正后结果混作性能排名。结果默认不进入 Git，跨机器同步还需另行复制本导航列出的保留数据。

