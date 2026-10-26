# matched300 代码导航与复算入口（2026-10-21）

本轮四种腔的名义边界都是 **300 × 300 μm 正方形**，对应文献有限腔边长，不是早期时域复现的400 μm数值窗口。半导体两晶格固定完整Inoue层栈和5 cm⁻¹内强度损耗；Yb:YAG两晶格固定500 nm增益层、120 nm PC、2 μm SiO₂及16对DBR，内强度损耗0.1 cm⁻¹。目标实际发射波长分别为950、1030 nm。

本文只导航仿真代码和数据。正式对比入口是 `results/current/matched300_20261021/comparison/selected_cases.json`；以其中的case路径、完成状态及各腔资格记录为准。单腔可独立复核，四腔图和统一损耗表要求四个正式case全部完成。

## 环境与路径

以下均为PowerShell命令。先运行这一段，此后示例从任意工作目录执行；每次生成一个新输出目录，避免覆盖正式结果。

```powershell
$python = 'E:/Anaconda/envs/env00/python.exe'
$sim = 'D:/things/Master/Doc/PCSELSim'
$run = Join-Path $sim ('results/replay_matched300_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$manifest = Join-Path $sim 'results/current/matched300_20261021/comparison/selected_cases.json'
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
```

脚本自行定位仓库根目录并设置导入路径，不要求手动设置 `PYTHONPATH`。`D`是面内高阶截断，`dz`单位nm；square的`--grid`是每边采样数，triangular的`--cells/--tri-grid`是特征网格半径格数，二者不能按相同整数解释为相同分辨率。

## 最短路线：复核正式主方案

```powershell
& $python "$sim/scripts/diagnostics/verify_matched300_optical_rebuild.py" `
  $manifest --case semiconductor_square --output "$run/square_optical_rebuild.json"
```

`verify_matched300_optical_rebuild.py` 从manifest关联的summary读取 `design`、`three_hole_spec` 或YAG `design + period_nm`，按实际波长和精度重新构造纵向模式、光学耦合矩阵C，并用已保存的有限本征场检查原始pencil残差、损耗预算及材料阈值。它还独立检查固定物理周期的真实频率残差≤0.05 m⁻¹，拒绝只满足冻结C本征方程的未闭频结果。它不是重新搜索参数空间，也不重新求完整横模谱。manifest中的 `configuration_path` 是便携几何文件链接；此CLI的输入仍是manifest，不能直接传单个geometry JSON。

square主方案已实际执行上述重建：C相对误差0，阈值增益误差0，本征残差约1.76×10⁻¹⁵。正式数值来自D64、dz1、N65及固定实际周期。

`compare_matched300_results.py` 读取实际Jones数组，重算beam和Stokes，生成PNG、CSV及 `saved_field_readback.json`。它严格要求 `status=completed` 且包含四个不同的正式case。四腔清单完成后可执行：

YAG的显示场必须对应实际edge-dual共享泵浦首阈值模式，两入口都会核对模式名、数组键、波长及本征值与源summary。CSV中的edge/top泵功率分别属于各自泵浦架构的首次阈值；两架构可能选择不同模式，需同时看模式名与波长列。

```powershell
$comparison = Join-Path $run 'all_saved_field_comparison'
New-Item -ItemType Directory -Path $comparison -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath $manifest -Destination (Join-Path $comparison 'selected_cases.json')
$frozenManifest = Join-Path $comparison 'selected_cases.json'
& $python "$sim/scripts/diagnostics/verify_matched300_optical_rebuild.py" `
  $frozenManifest --case all --output "$comparison/optical_rebuild.json"
& $python "$sim/scripts/diagnostics/compare_matched300_results.py" `
  $frozenManifest --output $comparison
& $python "$sim/scripts/diagnostics/plot_matched300_threshold_budget.py" --results $comparison
```

重建报告和读回记录包含summary、复场、geometry、协议文件的SHA，以及模式键、实际λ、周期、本征值、精度和内损耗组成的 `case_fingerprint`。损耗表只接受与当前输入完全绑定的重建报告；输入或协议变化后，应先重新执行verify。三个工具都支持显式 `--protocol`，默认使用 `configs/optimized_matched300_20261021/comparison_protocol.json`。

## 三个计算阶段

| 阶段 | 实际做什么 | 结论范围 |
|---|---|---|
| 几何扫描、完整粗谱、细网格挑战 | 扫孔形/面积/位置/角度；在可承担的粗网格求完整离散谱，再用粗谱及stripe极限作细网格复移位目标 | 不能用四/六个Γ族代表替代真实横模排序；细网格搜索仍有覆盖边界 |
| 设计实际周期、固定周期逐根闭频 | 外层调整物理周期，使实际目标模到950/1030 nm；随后固定该周期，逐根求解C(λ)及真实频率残差 | Bragg参考波长不是实际模波长；失败的低阶竞争根必须保留记录 |
| YAG共享反转与泵浦竞争 | 所有独立闭合模共用同一x–z反转，分别计算edge/top首次阈值及CW采样；实际edge赢家改变时重新调周期并重查竞争 | 材料增益gap、泵浦gap、饱和后开启点不同；CW采样不等于动力学单模证明 |

## square：从便携几何重新生成计算场

`configs/optimized_matched300_20261021/square_recommended.json` 是包装对象，含 `design`、实际nm边界、层栈、精度和工艺规则。`square_finalists.json` 则按 `best_balanced / best_low_threshold / best_isolation` 保存三种取舍。扫描入口的 `--designs` 与验证入口的原始几何输入要求 **Design对象列表**，先提取，不能把包装对象直接传入。

```powershell
@'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(json.dumps([data["design"]], indent=2), encoding="utf-8")
'@ | & $python - "$sim/configs/optimized_matched300_20261021/square_recommended.json" "$run/square_designs.json"

# 同一几何的完整粗网格QZ；新目录保留实际场。
& $python "$sim/scripts/optimization/matched300_square_validate.py" `
  "$run/square_designs.json" --output "$run/square_coarse" `
  --qz --keep-fields --grid 17 --D 16 --dz 2 --vertices 96

# 细网格多移位挑战，加入同几何完整粗谱的低损耗极点。
& $python "$sim/scripts/optimization/matched300_square_validate.py" `
  "$run/square_designs.json" --output "$run/square_fine" `
  --shift --keep-fields --grid 65 --D 64 --dz 1 --vertices 96 `
  --coarse-census "$run/square_coarse/compact_results.json"

# 保留便携文件的物理周期，逐根闭合实际频率。
& $python "$sim/scripts/optimization/matched300_square_finalize.py" `
  "$run/square_fine/case_000" --output "$run/square_closed" --D 64 --dz 1 --count 10
```

最后一步若加 `--target 950`，会重新设计物理周期，再冻结新周期闭合竞争模；复核已发布的固定周期时不加。`matched300_square_mesh.py case --grid ...` 需要该case的 `spectral_challenge.json + lateral_fields.npz`，用于已知场的网格/近邻周期延续，不是完整细谱。正式清理可能仅保留源谱标量；种子NPZ不存在时，先按上面 `--keep-fields` 重建。

扩展多参数扫描可用下列独立入口，输出 `designs.json / compact_results.json / summary.csv`；这一步的Γ族初筛不直接决定最终g₁/g₂。

```powershell
& $python "$sim/scripts/optimization/matched300_square_search.py" `
  --output "$run/square_scan" --count 320 --seed 30021021 --D 4 --dz 8 --grid 15
```

## triangular：完整粗谱导引、固定实际周期及轮廓控制

`ThreeHole` 位于 `matched300_triangular_search.py`。`fill`是三孔总填充，`main_share/satellite_balance`决定面积分配；`distance1/2、angle1/2_deg`为物理Cartesian方向，`center_fractional`才是直接晶格基底坐标。`triangle_samples_per_corner=9`保持旧圆角三角孔语义；27是显式轮廓收敛控制，不能静默改历史默认值。重采样保持面积，光学变化仍须验证。

正式三角主方案由 `triangular_finalists.json` 的 `primary_case` 与manifest关联；已核验的N91主方案有6个真实闭频模式。以下从manifest的真实summary目录取输入，不依赖便携列表顺序。先给出完整粗谱到N39的分阶段入口：

```powershell
$triEntry = (Get-Content -Raw $manifest | ConvertFrom-Json).cases |
  Where-Object { $_.id -eq 'semiconductor_triangular' }
if (-not $triEntry) { throw '三角正式case尚未加入manifest；先完成最终核验。' }
$triCase = Split-Path (Join-Path $sim $triEntry.summary_path)

& $python "$sim/scripts/optimization/matched300_triangular_qz_guided.py" `
  --stage coarse --case $triCase --output "$run/tri_coarse" `
  --coarse-solver qr --cells 13 --order 48 --dz 1 --retain-fields 64
& $python "$sim/scripts/optimization/matched300_triangular_qz_guided.py" `
  --stage fine --case $triCase --input "$run/tri_coarse" --output "$run/tri_fine" `
  --cells 39 --order 48 --dz 1 --poles 40 --eigen-count 24 --imag-scales 1 0.5
& $python "$sim/scripts/optimization/matched300_triangular_close.py" `
  --input "$run/tri_fine" --output "$run/tri_closed" --roots 10 --order 48 --dz 1

# 按同一已闭合case的源场检查27点/角轮廓；这是固定参考的branch控制。
& $python "$sim/scripts/optimization/matched300_triangular_actual_controls.py" `
  --case $triCase --output "$run/tri_contour27" --controls 39:48:1 --roots 3 --contour-samples 27
```

`--coarse-solver qr` 在私有导引脚本中求同一中心算符 `H=K M⁻¹`，仍按原始pencil检验；N5/N7完整谱已与广义QZ对比。完整粗谱只用于fine导引，不能升级为完整N39谱。三角方形边界是阶梯网格，应同时查看实际节点跨度和N敏感性。

上面的N39示例用于分阶段生成新场。正式N91主方案还包含低12个N39极点的网格延续、额外stripe复移位挑战及36极点合并池；保留的合并池包含最低24个复场。要按已保存的同几何种子重新求解正式6个N91闭频场，可直接使用summary记录的 `finite_seed_path`：

```powershell
$triMetadata = Get-Content -Raw (Join-Path $triCase 'summary.json') | ConvertFrom-Json
$triSeed = Split-Path (Join-Path $sim $triMetadata.finite_seed_path)
& $python "$sim/scripts/optimization/matched300_triangular_close.py" `
  --input $triSeed --output "$run/tri_reclosed_N91" --roots 6 --order 48 --dz 1 `
  --inverse --separable --newton
```

`--newton`是可选的频根算法：用群折射率估计步长，每步仍重建真实C(λ)并检验有限本征模；未收敛时回退Brent，最终频率残差仍须≤0.05 m⁻¹。省略它时使用括区间Brent；它不改变物理周期。`--inverse`启用已验证的物理中心场逆迭代，`--separable`启用相同真实坐标点和的可分离求和。`matched300_triangular_control_close.py`同样支持显式 `--newton`。

新增几何优化入口为 `matched300_triangular_search.py --stage sobol/grid/local/rescreen/refine/audit`。`local`等阶段的 `--input` 是含 `candidates.json` 的目录；输出仍是候选列表。`matched300_triangular_finalize.py --input` 则要求**单个候选row JSON**，含 `three_hole_spec` 和冻结波长数据；它执行外层实际950 nm周期设计及闭频，不能直接传列表或便携包装对象。

## YAG：稳定扫描入口与共享泵浦闭合

共同材料/泵浦参数在 `configs/optimized_matched300_20261021/yag_protocol.json`，它不是 `close --input` 的候选文件。最终配置及geometry路径应取完成后的manifest `configuration_path/geometry_path`，不要把运行中的 `square_v1` 等临时目录当最终赢家。

```powershell
& $python "$sim/scripts/optimization/matched300_yag_search.py" scan `
  --output "$run/yag_scan" --lattices square triangular --count 160 `
  --minimum-fill 0.25 --order 4 --dz 5 --square-grid 15 --tri-grid 9 --roots 12
& $python "$sim/scripts/optimization/matched300_yag_search.py" refine `
  --input "$run/yag_scan/square_screen_best.json" --output "$run/yag_square_local" `
  --count 64 --order 4 --dz 5 --square-grid 15 --tri-grid 9 --roots 12
& $python "$sim/scripts/optimization/matched300_yag_audit.py" `
  "$run/yag_square_local/square_local_best.json" --output "$run/yag_square_wide" `
  --wide --top 4 --order 16 --dz 2 --square-grid 31 --tri-grid 21 --roots 12
& $python "$sim/scripts/optimization/matched300_yag_search.py" close `
  --input "$run/yag_square_wide/ranked.json" --index 0 --output "$run/yag_square_closed" `
  --order 32 --dz 1 --square-grid 49 --tri-grid 39 --roots 24 --closed-roots 32 --extra-targets 2
```

对triangular同样使用 `triangular_screen_best.json`，其refine输出为 `triangular_local_best.json`；`audit`的 `ranked.json` 可直接供close读取。候选记录必须含 `lattice、design、period_nm`。上述只是重启搜索的示例，筛出的第0名仍须通过完整粗谱、细网格/stripe挑战及最终收敛，不能预先称最优。

`close`自动完成实际周期外调、固定周期逐根闭频、独立模去重、共享反转edge/top阈值与CW采样。edge首模变化且实际λ偏离1030 nm时会重调周期并重做竞争。检查 `failed_roots、closed_mode_identity_audit、actual_pump_winners、pump_results、longitudinal_applicability、final_fabrication`，尤其是低阶stripe闭合失败。top与edge可能由不同模式先起振，应分别使用各自λ、beam和Jones场。

YAG私有几何字段 `_yag_triangle_aspect_enabled` 控制圆角三角孔的保面积拉伸；缺省/False保持旧形貌。`_yag_triangle_samples_per_corner`缺省9，18/27为显式YAG轮廓控制。`_yag_explicit_holes`保存带类型的精确孔参数，优先于形貌标签；真实边界和工艺检查以导出的geometry为准。名义10 at%及常数截面、x–z反转中的y平均、1000–1060 nm分析带均须保留在结论中。

## 输出定位与解释

| 文件/字段 | 用途 |
|---|---|
| `selected_cases.json` | 统一case ID、summary/config/geometry/NPZ路径、数组key、实际λ、精度及参考本征值 |
| `summary.json` | 实际周期、逐根闭频结果、失败项、g₁/g₂和搜索作用域；半导体用 `design/three_hole_spec`，YAG用 `design + period_nm` |
| `actual_geometry_nm.json` | 周期、真实Cartesian nm边界/形心/半轴、工艺间隙；不要把fractional坐标当nm |
| `final_fields.npz / frequency_closed_fields.npz / closed_fields.npz` | 实际复本征场、对应C及Jones数据；数组key由manifest明确给出，不猜文件内第一个数组 |
| `g_second/g_first−1` | 材料阈值增益间隔；半导体电流gap和YAG实际泵浦gap需单独计算 |
| `η_rad` | 阈值损耗预算中的局部辐射份额，已含规定内损耗；不是±1°窗口占比或实验斜率效率 |
| beam/Stokes | 光斑圆度来自 `|Ex|²+|Ey|²`；偏振采用 `S3=+2 Im(Ex Ey*)`，两者分别报告 |

本文CLI及新增选项均以真实 `--help` 成功自检；没有为编写本文启动参数扫描。长计算前可再次对相应脚本执行 `--help`，并保留新输出的source hash及配置。源码、复场及验证数据留在仿真目录；PPT及其工作资产另存。
