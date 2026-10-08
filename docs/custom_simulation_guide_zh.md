# 自定义 PCSEL 主程序使用指南

> 入口整理（2026-10-08）：本文所述旧脚本已移入 `scripts/archive/legacy_20261008/`，
> 历史数值与用途保留；日常运行请看 [三个半导体主程序](entrypoints_zh.md)。

> 2026-10-08校对提示：本文保留早期教程/推导和历史结果；最新修复、可信度与设计评价以
> [本轮总审计](research_design_audit_20261008_zh.md)为准。旧“最佳”、精确阈值、单模及易加工
> 判断未经最新收敛/工艺验证时不得直接引用，能带是近Γ局域片段，远场能量对评估视窗归一化。

这份说明对应四个入口：

- `scripts/archive/legacy_20261008/run_custom_semiconductor_pcsel.py`：先用于半导体 PCSEL 交叉验证和自定义器件。
- `scripts/archive/legacy_20261008/run_custom_triangular_six_wave_pcsel.py`：三角晶格半导体的独立六波无限/有限器件模型。
- `scripts/archive/legacy_20261008/design_triangular_bright_mode.py`：双椭圆等 motif 的亮心低阈值模式筛选。
- `scripts/archive/legacy_20261008/design_triangular_three_hole_pcsel.py`：中心主孔加 `a1/a2` 双轴副孔的三孔远场优化。
- `scripts/archive/legacy_20261008/design_triangular_three_hole_shape_position.py`：椭圆、三角形和方形孔的形状—位置联合优化。
- `scripts/archive/legacy_20261008/design_triangular_triangle_main_refinement.py`：三角主孔附近的副孔形状、位置和角度局部细化。
- `scripts/archive/legacy_20261008/design_square_lattice_multiobjective.py`：独立方形四波晶格的低阈值—远场联合筛选。
- `scripts/archive/legacy_20261008/compare_square_triangular_equal_size.py`：统一层栈和 300 μm 尺寸的四波—六波公平比较。
- `scripts/archive/legacy_20261008/run_best_triangular_three_triangle_pcsel.py`：当前300 μm圆角三孔推荐方案的12步正式输出。
- `scripts/archive/legacy_20261008/optimize_triangular_three_triangle_manufacturable.py`：角度、圆角、孔距和加工余量重优化。
- `scripts/archive/legacy_20261008/run_custom_wang2024_triple_lattice.py`：Wang 2024 的 1550 nm 三晶格 structure 1 预设。
- `scripts/archive/legacy_20261008/run_custom_ybyag_pcsel.py`：把同一方形晶格四波光学骨架与 Yb:YAG 准三能级速率方程连接。
- `scripts/archive/legacy_20261008/run_geometry_derived_ybyag_pcsel.py`：从薄膜TE0和空气孔推导Yb:YAG四波矩阵的迁移原型。

其中半导体入口的耦合矩阵已经由几何推导；旧Yb:YAG入口的光学矩阵仍是教学示例。新几何入口已
接入薄膜TE0和几何后端，但层栈尚无实验样品标定，因此用于排除不可行假设，而不是定量流片预测。

## 在 PyCharm 中点哪个文件

1. 用 PyCharm 打开整个 `PCSELSim` 文件夹。
2. 在左侧项目树展开 `scripts`。
3. 第一次先右键 `run_custom_semiconductor_pcsel.py`，选择 **Run**。
4. 输出默认进入 `results/custom_semiconductor_inoue2019`。
5. Yb:YAG几何迁移请运行 `run_geometry_derived_ybyag_pcsel.py`，并先读可达性JSON。

若要运行三晶格，直接右键 `run_custom_wang2024_triple_lattice.py`。它写入
`results/custom_semiconductor_wang2024_triple_lattice`，不会覆盖 Inoue 输出。它默认关闭第 08 步；
原因不是程序不能运行，而是论文主文未给出完整的 InAlGaAs 载流子模型，直接套用 950 nm InGaAs
参数会产生看似精确但物理上不可审计的功率与瞬态。

若要运行三角 Bravais 晶格，右键 `run_custom_triangular_six_wave_pcsel.py`。它写入
`results/custom_semiconductor_triangular_six_wave`，并使用真正的六波基底；注意这里的“三角晶格”
与 Wang 论文在方形 Bravais 晶胞中放三个孔的“三晶格 motif”不是同一个概念。

脚本开头的 `_project_bootstrap.py` 会自动把 `src` 加入 Python 搜索路径，所以直接点脚本时不会再出现
`No module named 'pcselsim'`。仍建议在 PyCharm 的解释器中安装 `requirements.txt` 的依赖。

## 只改两个区域

每个主程序顶部都有两个明显区域：

1. `PARAMETER PANEL`：所有器件、材料、增益和数值参数。
2. `STEPS`：每一步的开关。`True` 表示运行，`False` 表示跳过。

所有带量纲变量都把单位写进变量名，例如 `_NM`、`_UM`、`_CM2`、`_CM3`、`_NS`、`_MS`。
不要只改数值却忽略单位。

## 半导体脚本的流程与输出

| 步骤 | 开关 | 输出 | 物理含义 |
|---|---|---|---|
| 01 | `01_parameters` | 参数、耦合矩阵和阈值审计 JSON | 保存输入、总矩阵和阈值来源 |
| 02 | `02_lattice` | `02_lattice.png` | 实空间晶格、孔形与填充率 |
| 03 | `03_k_space` | `03_k_space.png` | 倒格矢、Bloch 谐波和保留的四个基本波 |
| 04 | `04_layer_stack` | `04_layer_stack.png`、`04_vertical_TE0.png` | 纵向结构、TE0 场和限制因子 |
| 05-band | `05_band_structure` | `05_four_wave_band_structure.png` | 与同一 4×4 几何矩阵一致的 M-Gamma-X 局域能带 |
| 05 | `05_linear_modes` | 阈值、网格收敛、模式 CSV、论文对照 CSV | A/B/C/D 有限区域本征值及外推收敛值 |
| 06 | `06_mode_atlas` | 模式图谱和 A 模矢量远场诊断 | 包络、晶胞 Bloch 场、严格的矢量远场 |
| 07 | `07_length_sweep` | `07_length_sweep.png/.csv` | 每个长度重新求解并作网格外推 |
| 08 | `08_time_domain` | 多个 `.npz` 和瞬态/空间/光谱图 | Inoue 式载流子—四波时域耦合计算 |

调几何时先把第 08 步关闭，只跑 01–07；确认参数后再开启第 08 步。还可以用命令行
`--only` 临时选择步骤，不需要改文件，例如：

```powershell
python scripts\archive\legacy_20261008\run_custom_semiconductor_pcsel.py --only 02_lattice 03_k_space 06_mode_atlas
```

Inoue 自定义入口的第 08 步默认是 10 ns、四个电流点，运行时间明显长于前七步。Wang 入口的
第 08 步默认关闭。只检查安装或晶格时不要运行时域。

## 三角晶格六波入口

三角入口按以下流程运行：

| 步骤 | 开关 | 输出 | 物理含义 |
|---|---|---|---|
| 01 | `01_parameters` | `01_parameters_and_coupling.json`、`01_gamma_modes.csv` | `Cb/Cr/Ch/C` 与六个 Gamma 态 |
| 02 | `02_lattice` | `02_triangular_lattice.png` | 三角 Bravais 晶格和原胞 |
| 03 | `03_k_space` | `03_six_wave_k_space.png` | 六个保留基本波 |
| 04 | `04_vertical_mode` | `04_vertical_TE0.png` | 自洽 Bragg 波长与纵向 TE0 |
| 05 | `05_band_structure` | `05_six_wave_band_structure.png` | `M <- Gamma -> X` 局域能带 |
| 06 | `06_radiation_constants` | `06_radiation_constants.png` | 六条支路的面辐射常数 |
| 07 | `07_band_edge_states` | `07_six_band_edge_states.png` | 六个单晶胞矢量场 |
| 08 | `08_finite_modes` | CSV、阈值图、网格收敛图 | 六方向开放边界有限模式 |
| 09 | `09_finite_mode_atlas` | 包络/晶胞/远场图谱 | 六个模式的整体器件输出 |
| 10 | `10_best_mode_far_field` | 矢量远场诊断 | 最低阈值模式的中心强度、偏振和相位 |

```powershell
python scripts\archive\legacy_20261008\run_custom_triangular_six_wave_pcsel.py
python scripts\archive\legacy_20261008\run_custom_triangular_six_wave_pcsel.py --only 05_band_structure 07_band_edge_states
```

有限器件采用一阶六角特征网格并通过 `N=(7,9,11)` 外推，不是方形四波求解器，也不是论文式
(5.18) 的原始 generalized staggered mass matrix。六波载流子时域仍未实现。完整推导、默认结果
核对及双椭圆设计见 [三角晶格六波说明](triangular_six_wave_zh.md) 和
[六波有限器件与亮心设计](triangular_finite_design_zh.md)。

若要运行本轮三孔设计，右键 `design_triangular_three_hole_pcsel.py`。它固定“中心主孔 +
`+a1` 副孔 + `+a2` 副孔”的拓扑，批量扫描 15 组参数，然后对胜者执行完整六波和三网格复算。
输出写入 `results/custom_semiconductor_triangular_three_hole_optimized`。它没有逐步骤开关，因为单独
跳过候选会使综合排名失去可比性；要自定义扫描可直接修改脚本顶部的 `CANDIDATES`。详见
[三孔晶格远场优化](triangular_three_hole_optimization_zh.md)。

若要进一步比较三角孔和方孔，运行 `design_triangular_three_hole_shape_position.py`。它会先做 30 组
粗筛，再按“总榜 + 各形状族代表”进入中等精度，最后对三个候选做 `N=(7,9,11)` 外推。输出目录
不会覆盖上一轮三椭圆结果。多边形的 Fourier 系数来自解析边界积分；详细参数、三角主孔低阈值备选
及加工圆角限制见 [孔形—位置联合优化](triangular_shape_position_optimization_zh.md)。

最新一轮可分别运行 `design_triangular_triangle_main_refinement.py` 和
`design_square_lattice_multiobjective.py`。前者对 38 组三角六波候选做三级筛选；后者对 24 组方形
四波候选做三级筛选，并用真实复辐射场评价 `0.5°/1°` 包围能量。当前建议、全部参数和为何不推荐
模式间隔只有 `0.122 cm^-1` 的超低阈值方形三孔结构，见
[三角/方形联合优化](triangular_square_multiobjective_optimization_zh.md)。

若要比较晶格类型而不是比较器件尺寸，运行 `compare_square_triangular_equal_size.py`。它把三角六波
器件扩展到与方形参考相同的 `300 μm` 外接框，并分别求解圆形、正六边形和方形开放边界；还会按
同一增益约定反推阈值电流密度。结论与限制见
[300 μm 同尺寸方形—三角对比](equal_size_square_triangular_comparison_zh.md)。

当前推荐的三个三角孔方案已有独立的 12 步正式入口：

```powershell
python scripts\archive\legacy_20261008\run_best_triangular_three_triangle_pcsel.py
```

它固定 `300 μm` 方形器件并完整输出六波冷腔、有限模式、矢量远场、Q、静态阈值和小信号泵浦
扫描。参数和逐图说明见
[三角主孔—双三角副孔完整仿真](best_triangular_three_triangle_full_simulation_zh.md)。

正式主程序现在还会输出 `02_device_overview.png`，把300 μm完整器件、晶格放大和单原胞放在同一张
图中。若要重新验证旧15°角度，运行加工约束优化脚本；当前0°圆角方案的依据见
[300 μm加工约束优化](triangular_three_triangle_manufacturable_optimization_zh.md)。

## Wang 2024 三晶格入口

- 每个 `474 nm x 474 nm` 晶胞包含三个半径 `45 nm` 的圆孔，归一化孔心为
  `(-0.25,-0.25)`、`(0.25,-0.25)`、`(0.25,0.25)`。
- 孔由 InP 填充而不是空气，总填充率实算为 `8.495%`，对应论文约 `8.5%`。
- PC 区为 `300 um` 方形；圆形注入窗口直径 `200 um`，因此注入模块使用圆形而非方形轮廓。
- `A20` 三晶格结构因子为 3，代表论文式 (4) 所述的 180 度耦合相长叠加。
- 主文把完整外延结构放在 Supplementary Table S1。当前纵向层栈是依据 Fig. 3(d) 并同时约束
  `lambda≈1551 nm`、`Gamma_PC≈52%`、`Gamma_MQW≈6%` 的等效重建，不是作者原始外延表。
- 详细数值核对与当前偏差见 `docs/wang2024_triple_lattice_zh.md`。

## 当前默认的 Inoue 双晶格

- `a=277 nm`，纵向层结构来自 Inoue 表 I，增益、扩散、寿命和限制因子来自表 II。
- 有限 PC 本征域和电极边长为 `300 um`；时域计算域为 `400 um`，从而在电极外保留
  `25 um` 电流扩展区。不要把电极边缘误当成数值边界。
- 一个晶胞含两个大小不同的椭圆孔；小孔中心相对大孔沿 `+x,+y` 各移动 `0.25a`，与图 2(a)
  的虚线标注一致。
- 论文只公开了中心位移，没有公开从 SEM 重建的孔轮廓数值。脚本顶部的椭圆半轴和旋转角是
  可审计、可修改的理想化值，不是作者的原始 SEM 数据。
- 表 II 的限制因子 `0.044` 和阈值 `0.7 A` 默认优先用于 Inoue 时域复现；程序仍输出标量 TE0
  算得的限制因子和冷腔损耗反推阈值，方便发现模型差异。

## 参数应该按什么顺序修改

建议按下面顺序，每改一组就保留一个新的输出目录：

1. 波长、材料折射率和纵向层结构。
2. 晶格常数、孔形、孔尺寸、旋转角和填充率。
3. 器件边长、电极边长和电流扩展宽度。
4. `CWT_TRUNCATION_ORDER`、`VERTICAL_STEP_NM` 和 `FINITE_EIGEN_GRIDS` 收敛参数。
5. 半导体增益、透明载流子密度、寿命、扩散系数和注入电流。
6. 最后分别增加 Fourier 截断阶数、纵向网格、有限区域网格和时域网格做收敛检查。

脚本不再要求手填 A/B/C/D 失谐和损耗。单孔或多孔的圆、椭圆、多边形几何会先生成介电常数 Fourier 系数，
纵向层结构产生 TE0 模和 Bragg 波长，再按 Liang 第三章得到 `C1D + Crad + C2D`。所以改变孔尺寸、
折射率或层结构会真正改变带边、阈值和远场，而不只是改变示意图。每次运行都把矩阵写入
`01_geometry_coupling.json`，方便审计和交叉验证。

## 结果怎样判断

- `delta_L`：无量纲频率失谐；比较模式的相对位置，不是绝对波长本身。
- `alpha_L`：无量纲阈值/损耗；同一模型中越小越容易起振。
- `band_overlap`：数值本征态与目标 A/B/C/D 带边基矢的重叠，越接近 1 越容易可靠分类。
- `whole-device envelope`：有限器件慢变包络。
- `one-cell intensity`：四个基本 Bloch 波在一个晶胞内的快速干涉强度。
- `vector far field`：先由 Liang 式 (3.20) 的辐射 Green 函数计算晶面上的
  `Delta Ex, Delta Ey`，再按式 (4.24)–(4.26) 作二维 Fourier 变换。总强度、两个偏振分量、相位和
  偏振箭头都来自复场；中心暗点不再由人为的 `x+iy` 因子画出。
- `peak_offset_deg` 与 `centroid_offset_deg`：远场最大点和功率质心偏离法线的角度。
- `ellipticity`：远场功率二阶矩主轴标准差之比；接近 1 表示近圆形。
- `encircled_power_1deg`：以法线为圆心、`1 deg` 圆锥内的离散远场功率比例。

对当前 Inoue 双椭圆，双孔刻意破坏了单圆孔的辐射对称性，因此 A 模法线方向可以是亮点；这与
单圆孔对称保护 A 模的中心暗点不是同一个结构。是否有中心奇点必须由该晶胞算出的复辐射场决定。

远场图已固定在约 `±1.2°` 视野，使主光斑不会缩在画面中心。标题中的 `full RMS` 是两倍径向
二阶矩发散角。若要与实验 FWHM 对比，需要另加半高宽提取，而不能把 RMS 与 FWHM 混用。

## 当前模型边界

- 方形有限腔和时域未知量是 `(Rx,Sx,Ry,Sy)` 四波基底；三角入口使用独立的
  `(R1,S1,R2,S2,R3,S3)` 六波基底和 6×6 耦合矩阵。
- `run_custom_semiconductor_pcsel.py` 仍只接受 `LATTICE_TYPE="square"`；三角晶格必须改用六波入口。
  六方向开放边界有限腔已经实现；载流子-六波时域尚未实现。
- 有限区域本征问题采用一阶迎风开放边界，并对 `FINITE_EIGEN_GRIDS` 的多个网格按 `1/N` 外推；
  发表数值前仍应增加网格，并检查 `05_grid_convergence.png` 是否稳定。
- 半导体第 08 步直接复用现有 `TimeDomainSolver`；第 05 步是同一四波基底的线性有限区域伴随问题，
  两者不是同一个数值积分器。
- `05_threshold_current_audit.json` 使用时域求解器真正的幅度损耗约定
  `g_modal=alpha_internal+2 alpha_mode` 反推冷腔阈值。`THRESHOLD_CURRENT_SOURCE` 决定时域使用
  该推导值还是 Inoue 的实验标定值。
- 光谱使用最低阈值模作为投影，并在旋转参考系中积分；公共实失谐最后恢复为
  `spectrum_reference_wavelength_nm`。这消除了旧版本中由采样不足造成的混叠峰。

## 保存一个新器件

最简单的方法是复制主程序并改输出目录，例如：

```powershell
Copy-Item scripts\archive\legacy_20261008\run_custom_semiconductor_pcsel.py scripts\run_my_pcsel.py
```

然后把 `OUTPUT_DIRECTORY` 改成新的名字。Git 中提交脚本、文档和小型 CSV；通常不要提交巨大的
`.npz` 参数扫描结果。每次提交前运行 `pytest`，并在提交信息中说明改了哪组物理参数。
