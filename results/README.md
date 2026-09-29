# 结果目录说明 / Result-directory guide

本目录保存生成物，不是所有子目录都具有同等物理可信度。完整审计见
[`docs/physics_audit_and_code_map_zh_en.md`](../docs/physics_audit_and_code_map_zh_en.md)。

- `custom_semiconductor_inoue2019`：当前 Inoue 理想双椭圆几何主结果。
- `custom_semiconductor_inoue2019_audit_corrected`：本轮附录 A 投影因子修正后的独立
  10 ns 重跑；用于和上一目录对照，未覆盖旧图。
- `custom_semiconductor_wang2024_triple_lattice`：Wang 三晶格冷腔对照；时域默认关闭。
- `liang_chapter4_step_by_step`：教学重构；先读 `run_manifest.json` 的来源分类。
- `inoue2019`：标定 C 矩阵的 10 ns 时域回归。
- `quick`、`quick_lw`：环境冒烟测试，不引用为物理结果。
- `custom_ybyag_pcsel`：Yb:YAG 概念验证，不是定量器件设计。
- `custom_semiconductor_pcsel*`：早期或验证性输出，正式使用前确认代码版本与参数。

图片用于观察，定量分析应读取 CSV/JSON/NPZ。代码修正后不要覆盖旧结果；用主程序的
`--output results/<新目录名>` 生成可追踪的新目录，并把 Git commit、参数 JSON 与审计
文件一起保存。

Figures are for inspection; quantitative work should use CSV/JSON/NPZ data.
After code changes, write to a new output directory so old and new evidence are
not mixed.
