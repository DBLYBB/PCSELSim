# 结果目录与可信度

## 整理后的日常结果

三个半导体入口默认保存到 `pcsel/paper_reference/<时间戳>`、
`pcsel/square_double_hole/<时间戳>`、`pcsel/triangular_three_hole/<时间戳>`。
YAG独立保存到 `ybyag_crystal_pcsel/<时间戳>`。各入口拒绝覆盖已有结果，旧目录全部保留。
运行计划/状态和参数比“最近修改时间”更可靠：失败残留目录、quick和正式结果不可混用。

YAG当前完整新分支是
[`ybyag_crystal_pcsel/20261008_164124_099495`](ybyag_crystal_pcsel/20261008_164124_099495)，
详情见 [设计与实跑说明](../docs/ybyag_crystal_run_20261008_zh.md)。
这只是冻结Bragg、名义材料、选定TE族的CW可行性计算，尚未证明真实稳定单模。
此前同类日期目录是调试/粗网格/均匀横向权重对照，不作当前正式结果。
[`independent_family_audit_20261008`](ybyag_crystal_pcsel/independent_family_audit_20261008)
是固定同一周期的多纵向族/吸收独立检查，网格/权重不同，功率不可相加。

以 `run_status.json` 的 `complete`（YAG）或 `completed`（半导体）判正常完成；
光学核心、有限复场和泵浦数组与图片一起保存。
旧结果暂不搬动，以保留历史文档数据路径；通过本文分类及新日期目录整理，未删除任何旧数据。

### 三个半导体的代表性实跑

| 类型 | 本轮成功目录 | 运行范围 |
| --- | --- | --- |
| 论文对照 | `pcsel/paper_reference/20261008_163214_269631` | D10/dz3nm/N13,17,21,25完整冷腔；非完全复现认证 |
| 正方双圆孔 | `pcsel/square_double_hole/20261008_163214_266631` | 相同冷腔精度；非稳定单模保证 |
| 三角主孔＋两个圆副孔 | `pcsel/triangular_three_hole/20261008_164203_411501` | D10/dz3nm/N11,15,19、全步骤、全貌与zoom |
| 论文入口时域烟测 | `pcsel/paper_reference/20261008_164151_865877` | 仅0.1ns/1.4Ith，验证新入口时域依赖，不作L-I或稳态谱 |

10ns论文理想几何时域完整数据仍在旧 `custom_semiconductor_audit_20261008`，未重跑或删除。
三角外推最低B1与最细网格最低B2不一致，敏感性远大于模式间隔；漂亮B1光斑不能证明它一定激射。
各日期目录的步骤以保存的计划/manifest为准，不能把单图步骤结果称为所有步骤完成。

总评见 [2026-10-08审计](../docs/research_design_audit_20261008_zh.md)。不同目录使用不同几何、
网格和方程版本，不能直接合成“最优阈值排行榜”。图片用于观察，定量引用读取CSV/JSON/NPZ。

## 2026-10-08新数据

| 目录 | 内容 | 使用限制 |
|---|---|---|
| `audit_20261008` | 81份本地PDF只读索引、历史三角N=7至25收敛审计 | PDF索引≠81份全文评审；迎风数值扩散须单独看 |
| `custom_semiconductor_audit_20261008` | 修正频谱/噪声/受激吸收后的Inoue理想几何冷腔与10ns重跑 | 输入阈值基准和预测阈值分开；投影有限时间谱≠全模实验线宽 |
| `square_fabrication_20261008` | 66种平滑双孔、53种名义几何通过、4个多网格终选 | 主表为外推值、场为N=21；须读网格状态；微小基态间隔不能保证单模 |
| `robust_design_20261008` | 193个三角晶格候选、76个光学筛查、制造硬约束和周期晶胞误差 | `00_design_audit.json`列出本次有效终选；遗留预试算文件夹不构成当前推荐 |
| `robust_design_20261008/research_tier_diagnostic` | 放宽圆度目标后较大加工裕量的独立复核 | 是加工/光束圆度折中，不与严格筛选结果混淆 |
| `geometry_derived_ybyag_audit_20261008` | 正确表面层序、500μm高网格冷腔/速率方程、泵浦与能量预算 | 默认不能起振；粗尺寸scan须读每项网格/来源标签 |
| `ybyag_pump_architectures_20261008` | 36组布局初筛＋3组等总功率双端泵，共39组 | 新掺杂未重算光学/光谱/热导，不是已起振器件 |

`alpha`通常为**场振幅**损耗，`alpha_in`为材料**功率**损耗。比较口径为
`required_modal_power_gain = 2*alpha_mode + alpha_in`。外推场与损耗不是同一个实际网格本征对。
`grid_sensitive`禁止精确排序，`provisional`仍需独立验证；经验敏感度不是统计置信区间。

所有包围能量均须连同评估视窗引用，RMS不等于FWHM/1/e²/M²。
晶胞扰动每周期重复，未模拟整个器件的非周期无序；12个随机样本不能称量产良率。

## 历史目录

- `custom_semiconductor_inoue2019`、`custom_semiconductor_inoue2019_audit_corrected`：前两轮理想几何时域；后一目录仅修正当时的折射率投影/载流子子步。
- `inoue2019`：标定C矩阵，不是几何独立预测。
- `custom_semiconductor_wang2024_triple_lattice`：公开结构冷腔对照，时域默认关闭。
- `liang_chapter4_step_by_step`：教学重构，须读`run_manifest.json`的代理/物理来源标签。
- `best_triangular_three_triangle_full`：上一轮推荐三孔完整基准；保留作对照。低网格阈值和易加工结论本轮降级。
- `triangular_three_triangle_manufacturable_300um`：59/9/3历史扫描，不再据此断言0°优于15°。
- `equal_size_square_triangular_comparison`：旧同尺寸冷腔；未证明普遍晶格优劣。
- `geometry_derived_ybyag_pcsel`：旧层序默认500μm和尺寸扫描；“5mm可行”已经失效。
- `custom_ybyag_pcsel`：手填C的教学概念，不能作实际激光器设计证据。
- 其他`*optimized`、`custom_semiconductor_pcsel*`：历史筛选，引用前确认源版本、网格和口径。
- `quick*`：只验证环境。

代码/参数改变后使用新的`--output results/<新名字>`。将Git commit、源SHA256、参数、随机种子、
CSV/JSON/NPZ和图片一起保存；禁止仅凭文件时间把旧图片当作新模型结果。
