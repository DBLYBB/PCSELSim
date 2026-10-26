# 结果诊断导航 / Result diagnostic map

这些入口用于解释和核验仿真结果。先确定一个实际模式的身份，再关联它的几何、波长、复场和阈值。

These tools check that geometry, wavelength, complex fields and thresholds refer to the same physical mode. They do not certify a global optimum or an exhaustive continuum spectrum.

2026-10-25审查后结果从[新科学清单](../../results/current/strict_rerun_20261025/comparison/selected_cases.json)进入。[package_strict_rerun_20261025.py](package_strict_rerun_20261025.py)只统一已经完成的结果，并断言源尺寸/实际周期、阈值排序和工程目标；[plot_strict_four_cavities_20261025.py](plot_strict_four_cavities_20261025.py)读取实际复场绘制总图，重算Jones光斑与Stokes作绑定检查，不替代新谱搜索。独立跨分支重建分别保存在`strict_rerun_20261025/cross_audit_*.json`。

| 入口 / Entry | 用途 / Purpose | 是否重算光学 / Optical rebuild |
|---|---|---|
| [verify_matched300_optical_rebuild.py](verify_matched300_optical_rebuild.py) | 按实际波长、几何和精度重新构造 C，核验已保存场的原始 pencil 残差、真实频率与阈值 / Rebuild C and check the saved mode | 是，重建 C；不重新搜索模式 / Rebuild only, no spectral search |
| [compare_matched300_results.py](compare_matched300_results.py) | 四腔实际晶格、矢量远场、偏振与 CSV / Four-cavity geometry, vector far field, polarization and metrics | 否；读取保存的 Jones 场 / Stored Jones fields |
| [matched300_result_provenance.py](matched300_result_provenance.py) | 共同协议、模式身份与内容指纹守卫 / Common protocol, mode identity and content binding | 公共模块 / Shared validation module |
| [plot_matched300_threshold_budget.py](plot_matched300_threshold_budget.py) | 分开内损耗、辐射和边界输运对材料阈值的贡献 / Material-gain loss budget | 否，要求最新且绑定输入的重建报告 / Requires a current bound rebuild |
| [plot_matched300_design_tradeoffs.py](plot_matched300_design_tradeoffs.py) | 九个保留设计的阈值、径向非圆度和隔离取舍 / Retained static design tradeoffs | 否 / No |
| [plot_matched300_triangle_angle_scan.py](plot_matched300_triangle_angle_scan.py) | 同网格、同参考下已跟踪分支的角度敏感性 / Tracked-branch angular sensitivity | 否 / No |
| [plot_matched300_yag_cw.py](plot_matched300_yag_cw.py) | 读取 YAG 已保存共享反转静态 CW 样点 / Stored saturated steady-state results | 否；属于另一个运行口径 / Separate operating-model result |
| [audit_counterpropagating_stripes.py](audit_counterpropagating_stripes.py) | 独立双波条纹极点为二维搜索提供目标 / Independent stripe targets for 2D challenges | 独立诊断，不能作为二维阈值严格界 / Diagnostic, not a strict 2D threshold bound |

## 2026-10-25 新研究核验 / New study checks

- [plot_yag_feasibility.py](plot_yag_feasibility.py)：从已保存数值样点绘制CW、尺寸及材料曲线，未收敛/不可达值不画作零阈值。Plots do not modify simulation data.
- [review_new_yag_study_20261025.py](review_new_yag_study_20261025.py)：独立复算CW粒子/功率账与尺寸原K/M残差；`--include-pulse`核对最终脉冲账本。Read-only verification, with explicit source hashes.
- [yag_quality_requirement_20261025.py](yag_quality_requirement_20261025.py)：由Q/光子寿命/材料增益给必要条件，不是新微腔全波设计。An analytic necessary condition, not a sufficiency proof.
- [audit_algorithm_physics_20261025.py](audit_algorithm_physics_20261025.py)：重放数值物理独立反例和源文件静态筛查。A static screen is not a line-by-line physics proof.

## 最短静态检查 / Minimal static check

从工程根目录运行，使用新目录保存读回结果，避免覆盖冻结的交付证据。

Run from the repository root and write to a fresh directory. Do not overwrite the frozen evidence.

```powershell
$pcselPython = 'E:/Anaconda/envs/env00/python.exe'
$pcselManifest = 'results/current/matched300_20261021/comparison/selected_cases.json'
& $pcselPython scripts/diagnostics/verify_matched300_optical_rebuild.py `
  $pcselManifest --case semiconductor_square --output results/square_rebuild_new.json
```

四腔总图要求 manifest 的状态为 `completed`，且有四个不同的正式 case。更多新目录复算示例见[运行指南](../../docs/matched300_run_guide_20261021_zh.md)。

Four-case plots require a completed manifest with four distinct cases. The run guide explains how to assemble a new output directory for comparison and budget tools.

## 指标与边界 / Metrics and scope

- `|Ex|²+|Ey|²` 是矢量远场强度；不能对已有强度图再做 FFT 当作远场 / Form intensity from the complex Jones field.
- 轴比接近 1 只约束二阶矩，还要检查亮中心、径向 L2、半高连通域和角窗 / Equal second moments alone do not imply a circular bright beam.
- `S3=+2 Im(Ex Ey*)`；圆形强度与圆偏振分别报告 / Circular intensity and circular polarization are separate observables.
- 同一个正确的矩阵本征对仍可能不是实际频率根，必须同时检查物理频率残差 / A matrix eigenpair must also satisfy the physical frequency equation.
- YAG 应选择实际共享泵浦首模，不能用冷腔低损耗模式替换其 Jones 图 / Bind the displayed field to the actual pump winner.
- SHA 记录输入字节和运行来源，不代替物理正确性 / Hashes establish content identity, not physical validity.

此前诊断脚本仍在原路径，用于复现旧审计。先查对应日期的报告，避免混用不同尺寸、精度或材料定义。

Older diagnostics keep their original paths for traceability. Their date-specific reports define the applicable size, settings and material conventions.
