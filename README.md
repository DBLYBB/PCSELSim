# PCSELSim：从第一次运行到自定义器件的完整教程

本平台用 Python 研究光子晶体面发射激光器（PCSEL）：输入晶胞孔形、材料层结构和器件尺寸，计算耦合、有限器件模式、阈值与矢量远场。半导体包含正方四波和三角六波模型；Yb:YAG 使用独立的导波、底部 DBR 和空间增益模型。

2026-10-25 已在[全仓物理审计](docs/physics_audit_overall_20261025_zh.md)后完成[同尺寸四腔严格重跑与优化](docs/strict_rerun_four_cavities_20261025_zh.md)。最新正式入口是[新四腔清单](results/current/strict_rerun_20261025/comparison/selected_cases.json)，逐项绑定实际几何、自己的实频复场、最近竞争模、Jones远场和偏振。四个主腔均为300×300μm正方形物理边界，旧文献复现及旧优化数据保持只读；新数值来自新几何和重新验收，不回填旧快照。审查后的38个物理核心文件在本轮保持相同SHA，另有79项相关回归通过。

新主几何的[Yb:YAG输入输出、脉冲能量及尺寸报告](docs/strict_yag_feasibility_20261025_zh.md)与[此前旧几何研究](docs/yag_feasibility_20261025_zh.md)分开。扫描先筛选，再以完整粗谱、有限细网格挑战、跨频段根、实际频率和原K/M残差核查；失败和未覆盖范围如实保留，不把有界搜索称为全谱证明。当前模型仍有近Bragg、标量TE、材料常谱、固定冷场及空间平均等明确近似，不称完全逐式复现或实验认证。

The 2026-10-25 study separates external, first-entry internal and absorbed pump power, and finite-pulse energy from CW thresholds. The physics audit distinguishes corrected implementation bugs, declared model approximations, historical exploratory paths and missing independent validation.

这是一套研究工具，不是实验认证的设计软件。“论文复现”表示采用论文理论开展对照，不表示所有图片和实验参数已经完全复现。本页是操作教程，不放运行性能排名；结果和设计评价见专题文档与结果目录。

2026-10-21 已修正薄量子阱界面积分、论文层栈、时域入射边界和三角远场口径。有限冷腔采用交错广义本征离散，保留旧迎风算法作对照。最新方法、实跑结果和适用边界见[本轮修正与验证](docs/corrections_20261021_zh.md)。

本轮已重算三套半导体冷腔、四组 10 ns 时域，以及固定周期的 Yb:YAG 16 分支频率和共有反转 CW，修正阶段 203 项测试通过。2026-10-21 按要求清理了扫描大矩阵、修正前归档及开发中间态，保留文献复现、当前三腔最终原始复场与必要验证；原结果由约 7.84 GiB 降至约 350 MiB，释放约 7.50 GiB。保留目录、删除 SHA256 清单和校验范围见[结果导航](results/README.md)。

后续三类腔联合优化新增了完整横模搜索、尺寸与工艺扰动、亮中心近圆斑筛选，以及 Yb:YAG 材料与实际泵浦几何敏感性。推荐参数、逐轮证据和复现命令见[优化总报告](docs/optimization_three_cavities_20261021_zh.md)，晶格的物理区别见[三条路线与三角晶格研究价值](docs/cavity_comparison_research_20261021_zh.md)。优化脚本集中在 `scripts/optimization/`，新候选不直接改写日常主入口的默认参数。本轮优化回归测试为 221 项全部通过；这验证实现与数学基准，不等于实验或独立全波认证。

2026-10-22 的此前同尺寸冻结轮采用论文有限腔的 **300 × 300 μm 正方形物理边界**，优化半导体和 Yb:YAG 的正方、三角晶格主孔加副孔结构。该轮历史参数与适用范围见[此前四腔报告](docs/optimization_matched300_comparison_20261021_zh.md)；[此前总图](results/current/matched300_20261021/comparison/four_cavity_overview.png)、[此前指标](results/current/matched300_20261021/comparison/four_cavity_metrics.csv)和[运行指南](docs/matched300_run_guide_20261021_zh.md)用于追溯。该轮源代码完整回归为 [311 项通过](results/current/matched300_20261021/regression/full_suite_02_summary.json)，不能与后续测试数直接相加。此前165、200、400μm结果不参与最新四腔同尺寸比较；日常入口默认值不自动替换。

2026-10-22 进一步整理了代码说明：日常入口、优化阶段和结果诊断各自有明确职责，关键物理与数值步骤增加中英文注释。研究扫描从[优化脚本导航](scripts/optimization/README.md)进入，正式结果读回从[诊断导航](scripts/diagnostics/README.md)进入。此次注释整理不修改数值公式、求解顺序或默认参数；历史运行来源和最终复场继续保留，当前源与原源的可执行结构对照另存于 `results/current/code_documentation_20261022/`。

The bilingual comments explain physical units, amplitude versus intensity loss, actual-frequency closure, finite-mode identity and complex near/far fields. Use the optimization map to start a new search and the diagnostic map to check existing results. Historical runtime hashes stay attached to their original runs.

第一次使用依次读第 1–5 节；跑通后再读参数、理论与可信度部分。

## 1. 我应该点哪个文件？

在 PyCharm 中打开整个 **PCSELSim 文件夹**，不是只打开某个 Python 文件。

```text
PCSELSim/
├─ README.md                      本教程
├─ scripts/                       日常运行入口
│  ├─ run_paper_reproduction.py    Inoue 论文对照，正方四波
│  ├─ run_square_double_hole.py    平滑双圆孔，正方四波
│  ├─ run_triangular_three_hole.py 圆角三角主孔＋两个副孔，三角六波
│  ├─ run_ybyag_pcsel.py           Yb:YAG、表面 PC、导波层与底部 DBR
│  ├─ optimization/               分阶段研究扫描、闭频、精度与结果整理
│  ├─ diagnostics/                独立诊断、正式结果读回和来源核验
│  ├─ _*.py                       公共流程/路径助手，不必手动运行
│  └─ archive/                    历史主程序与诊断，保留用于追溯
├─ src/pcselsim/                   物理模型与数值算法，不是点击运行的入口
├─ configs/                       日常配置与独立优化参数 JSON
├─ docs/                          理论推导、操作说明与审计
├─ tests/                         自动测试
└─ results/
   ├─ current/                    文献复现、当前最终结果及必要验证
   └─ _cleanup_20261021/           已删除扫描的路径、大小和 SHA256 清单
```

| 目的 | 点击运行 | 默认范围 |
| --- | --- | --- |
| 学习 Inoue 理论、对照论文 | [run_paper_reproduction.py](scripts/run_paper_reproduction.py) | 理想化双椭圆；还开尺寸扫描与载流子时域，较慢 |
| 自定义正方晶格半导体 | [run_square_double_hole.py](scripts/run_square_double_hole.py) | 双圆孔冷腔与远场；默认不开尺寸扫描/时域 |
| 自定义三角晶格半导体 | [run_triangular_three_hole.py](scripts/run_triangular_three_hole.py) | 六波冷腔、远场、阈值与小信号扫描 |
| 全固态 PCSEL 候选研究 | [run_ybyag_pcsel.py](scripts/run_ybyag_pcsel.py) | 导波/底镜、固定周期频率闭合、多分支共享反转 CW 与设计诊断 |

前三个是日常半导体入口；第四个是独立晶体扩展。历史脚本没有删除，但新手不用从归档入口开始。

## 2. 配好 Python 环境：只做一次

解释器是执行 Python 的程序，PyCharm 是编辑器；后者不会自动保证科学计算依赖已装好。

支持 Python **≥3.10**，依赖 NumPy、SciPy、Matplotlib、PyYAML，测试另需 pytest。可沿用 `E:/Anaconda/envs/env00/python.exe`。安装依赖和运行脚本必须用同一个解释器。

在 Windows **PowerShell** 中逐行执行：

```powershell
Set-Location 'D:/things/Master/Doc/PCSELSim'
$pcselPython = 'E:/Anaconda/envs/env00/python.exe'
& $pcselPython --version
& $pcselPython -m pip install -e ".[dev]"
& $pcselPython -c "import sys, pcselsim, numpy, scipy, matplotlib, yaml; print(sys.executable); print(pcselsim.__file__)"
```

`-e` 是可编辑安装：改源代码后不必反复安装项目。最后一行应显示所选解释器与本工程的包路径，且没有异常。

暂时不做可编辑安装时，也可执行 `& $pcselPython -m pip install -r requirements.txt`。新入口的路径助手会定位工程并加入 `src`，但不能代替安装第三方依赖。在其他电脑上，把解释器路径换成那台机器的真实路径。

## 3. 在 PyCharm 中第一次运行

1. Open 打开 `PCSELSim` 根文件夹。
2. 在设置中找到 Python Interpreter，选择现有环境 `E:/Anaconda/envs/env00/python.exe`，不要误选 Anaconda base。
3. 打开 `scripts/run_square_double_hole.py`。第一次建议正方冷腔，不直接启动论文长时域。
4. 在运行配置中将 Working directory 设成工程根目录。新入口本身也会定位并切换到根目录。
5. 在运行配置的 Parameters 中填入下面这一整行，再右键该文件 → Run：

```text
--quick --only 00_fabrication 02_lattice 02_device_overview 03_k_space 04_layer_stack 05_band_structure
```

这次检查结构、纵向模与局域能带的数据连接。终端显示编号进度及保存目录；打开对应 PNG，核对孔形、周期、层序和器件边界。

不用运行 `_project_bootstrap.py`、流程助手或 `src` 中的单个模块，也不要在 Python Console 中粘贴终端命令。

喜欢终端时，在第 2 节的同一个 PowerShell 窗口执行：

```powershell
& $pcselPython scripts/run_square_double_hole.py --quick --only 00_fabrication 02_lattice 02_device_overview 03_k_space 04_layer_stack 05_band_structure
```

## 4. 按由浅到深的顺序计算

### 4.1 先看计划，不求解

三个半导体入口支持：

```powershell
& $pcselPython scripts/run_square_double_hole.py --list-steps
& $pcselPython scripts/run_square_double_hole.py --dry-run
& $pcselPython scripts/run_triangular_three_hole.py --dry-run
```

前者列开关，后者列参数、开关与预计保存位置，均不计算。YAG 支持 `--dry-run`，并可用 `--help` 查看频率、多 TE 族、泵浦与谱表选项。

### 4.2 粗网格检查冷腔流程

```powershell
& $pcselPython scripts/run_square_double_hole.py --quick
& $pcselPython scripts/run_triangular_three_hole.py --quick
```

`--quick` 降低 CWT 截断、纵向和有限器件网格；四波时域若开启还会缩短时间、减少电流点。**只查导入、数据流与绘图，不能用于物理阈值、线宽或模排序。**

### 4.3 正式计算半导体冷腔

```powershell
& $pcselPython scripts/run_square_double_hole.py
& $pcselPython scripts/run_triangular_three_hole.py
```

去掉 quick 使用默认精度，但之后仍须提高精度比较收敛。“正式运行”不等于“已经收敛”。

论文入口默认较慢，首次可只选冷腔：

```powershell
& $pcselPython scripts/run_paper_reproduction.py --only 00_fabrication 01_parameters 02_lattice 02_device_overview 03_k_space 04_layer_stack 05_band_structure 05_linear_modes 06_mode_atlas 06_far_field_zoom
```

### 4.4 冷腔核对后，再做四波时域

```powershell
& $pcselPython scripts/run_square_double_hole.py --only 01_parameters 05_linear_modes 08_time_domain
```

这会开启正方入口默认关闭的时域。先将顶部 `CURRENT_RATIOS` 改成一个电流的元组，例如 `(1.4,)`，跑通再增加电流点和时间窗。末端平均值不自动代表稳态，须看功率曲线、载流子变化、实际采样时间和谱分辨率。

### 4.5 YAG：先结构，后起振可行性

```powershell
& $pcselPython scripts/run_ybyag_pcsel.py --only 01_structure 02_bands
& $pcselPython scripts/run_ybyag_pcsel.py --quick --only 01_structure 03_finite_modes 04_fields 05_spatial_gain
& $pcselPython scripts/run_ybyag_pcsel.py --only 01_structure 02_bands 03_finite_modes 04_fields 05_spatial_gain
```

第二行是流程检查，第三行是默认精度的频率闭合、多分支共享反转 CW 计算，默认检查 `ACTIVE_TE_FAMILIES` 中的纵向族。当前 `06_design_scan` 保存已纳入族与未覆盖范围的说明；厚度或周期研究须用不同参数和新目录分别运行，不会自动生成完整厚度/盖层/间隔层扫描。

YAG `--quick` 使用 D2、N9、单 TE0 和较少横向泵网格，只做流程检查。每个真正频率求解都保持器件周期和层厚固定，重新计算纵向场、耦合和有限模。此入口不是超快脉冲仿真。

```powershell
& $pcselPython scripts/run_ybyag_pcsel.py --dry-run
& $pcselPython scripts/run_ybyag_pcsel.py --te-families 0 1 2 3 --grid-points 17 --pump-power-W 32
```

单独 `--only 01_structure` 不解有限模和 CW；`02_bands` 是明确标注的参考局域能带。`03_finite_modes`/`04_fields` 需要频率闭合，`05_spatial_gain` 才计算共有反转。用 `--spectroscopy 实测谱表.json --temperature-K 293.15` 提供有界截面表；超出校准波长或温度范围会拒绝计算。未提供谱表时使用明确标为未校准的名义常数，不能宣称获得实测材料预测。

## 5. 开关、命令与保存位置

顶部 `STEPS` 是开关字典：`True` 开、`False` 关。编号名称、大小写保持原样。例如局部示意：

```python
STEPS = {
    # 实际修改时保留原文件中所有键，只修改值
    "05_band_structure": True,
    "07_length_sweep": False,
    "08_time_domain": False,
}
```

**不要用上面三个键替换整个字典。**

| 选项 | 三个半导体入口 | YAG 入口 |
| --- | --- | --- |
| `--only 步骤名…` | 完全替代本次开关；列出的步骤开启，包括原 False 项 | 只筛选原来为 True 的项，不能重新开启 False |
| `--quick` | 粗网格，四波时域还缩短计算窗 | D2/N9/TE0 及较少横向泵网格的流程检查 |
| `--dry-run` / `--list-steps` | 均支持 | 支持 dry-run，开关用 help 查看 |
| `--output 路径` | 新目录或已有空目录，拒绝非空目录及历史 archive | 新目录或已有空目录，拒绝非空目录及历史 archive |

关输出不一定跳过其求解依赖：远场必须先求有限模，四波 `01_parameters` 为记录阈值也会求有限模；实际运行先建立所需光学核心。

默认位置自动带时间戳，不覆盖旧结果。历史归档受保护，新计算不能写入 archive：

```text
results/current/pcsel/paper_reference/<时间戳>/
results/current/pcsel/square_double_hole/<时间戳>/
results/current/pcsel/triangular_three_hole/<时间戳>/
results/current/ybyag_crystal_pcsel/<时间戳>/
```

自定义例子：`--output results/current/my_square_trial_001`。相对路径按工程根解析。换参数用新目录，不删旧数据来腾出同名输出。

## 6. 修改器件：哪些参数才真正生效？

四个主程序直接读取各自顶部参数。**改 `configs/inoue2019.yaml` 不会改变这些新入口。** 初学阶段改入口，不改求解器。

Python 中 `#` 后是注释，`名称 = 值` 是赋值；逗号不要随意删除。先保存原参数/做 Git 提交，每次优先只改一类因素。

### 6.1 正方四波

| 参数 | 含义/单位 | 注意 |
| --- | --- | --- |
| `LATTICE_CONSTANT_NM` | 周期，nm | 改周期改变 Bragg 条件，默认孔尺寸表达式也随周期缩放 |
| `WAVELENGTH_GUESS_NM` | Bragg/纵向迭代初值，nm | 不是强制激射波长，输出看报告 |
| `BACKGROUND_INDEX` / `HOLE_INDEX` | 折射率 n | 不是 ε |
| `UNIT_CELL_INCLUSIONS` | 各孔形状、半轴、孔心、转角 | 尺寸/孔心 nm，转角 deg；半轴/半径不是直径 |
| `LAYERS` | 下到上的层名、厚度 nm、折射率 | PC 层占位 n 会被晶胞平均 ε 求得的 n 替换 |
| `DEVICE_SIZE_UM` | 冷腔正方边长，µm | 不等于时域窗口/电极尺寸 |
| `TIME_DOMAIN_SIZE_UM` / `ELECTRODE_SIZE_UM` | 时域窗口/电极，µm | 电极须在计算域内 |
| `INTERNAL_LOSS_CM` | 额外强度损耗，cm⁻¹ | 不重复加入模型已有辐射/边缘泄漏 |
| `ACTIVE_CONFINEMENT_SOURCE` | 用论文 Γ 或纵向解 Γ | 选纵向解时，手填 Γ 不决定结果 |
| `THRESHOLD_CURRENT_SOURCE` | 论文参考电流或推导电流 | 参考值不是独立预测 |
| 增益、寿命、扩散、透明密度等 | 半导体材料参数 | 换材料不能只改折射率/波长 |
| `CURRENT_RATIOS` / `END_TIME_NS` | 电流倍数/总时间 ns | 稳态和谱线判断需要足够时间 |

当前多孔由非空 `UNIT_CELL_INCLUSIONS` 决定，单孔 `HOLE_RADIUS_X_NM`、`HOLE_RADIUS_Y_NM`、`HOLE_ROTATION_DEG` 不生效，`HOLE_SHAPE` 也不能代替逐孔修改。

默认双圆两孔沿 x、y 分别错位指定比例；直线孔心距离还要乘 √2。`LAYERS` 中 PC 的 `0.0` 是占位，不是真实零折射率。保持 `PC_LAYER_NAME`、`ACTIVE_LAYER_NAME` 与层名一致，有效量子阱总厚度与载流子参数一致。

当前时域在整个 `TIME_DOMAIN_SIZE_UM` 内应用耦合系数，没有独立 PC 区域掩膜。论文默认及本轮正式时域记录使用 **400 µm 时域窗口、300 µm 冷腔和300 µm电极**；25 µm电流扩展参数不定义独立的PC边界。因此这组时域与300 µm冷腔不是严格同尺寸验证。

### 6.2 三角六波

修改 `DESIGN = Design(...)`：

| 字段 | 含义 |
| --- | --- |
| `main_shape` / `satellite_shape` | 主/副孔；当前入口支持 triangle、circle、ellipse |
| `fill` | 所有孔总面积 / 三角原胞面积 |
| `main_area_share` | 主孔面积份额，剩余由两个副孔平分 |
| `distance` | 副孔笛卡尔径向距离 / a |
| `spread_deg` | 两副孔相对默认方向的展开角 |
| `rounding` | 三角孔圆角比例 |
| `main_angle_deg` | 主孔转角 deg，不存在普适最佳角 |
| `ellipse_aspect` | 选椭圆时的长短半轴比 |

三角 `HOLE_EPSILON` 是 **ε=n²**，不同于四波 `HOLE_INDEX`。

`DEVICE_SHAPE` 可选 square、circle、hexagon；`DEVICE_HALF_SIZE_UM` 对方形是半边，对圆/六边形是半径/外接半径。同一个数不代表同面积，比较前统一面积、注入与评价角窗。

只把四波的 `LATTICE_TYPE` 改成 triangular 不会得到六波物理，须使用独立三角入口。

### 6.3 Yb:YAG

- `TARGET_WAVELENGTH_NM`：参考波长；未指定 `--period-nm` 时，入口只按 TE0 参考模设计一次 `a=λ/n_eff`。其余 TE 族及各有限分支保持这个实际周期重新闭合频率，不保证实际激射波长恰为参考值。
- `GAIN_THICKNESS_NM`、`PC_DEPTH_NM`、`CAP_THICKNESS_NM`、`TOP_SILICA_NM`、`BOTTOM_SPACER_NM`：增益、PC、盖层及包层厚度。
- 各层折射率、`DBR_PAIRS`、`DBR_HIGH_INDEX`、`DBR_CENTER_NM`、`DBR_FIRST_LAYER`：导波与有限底镜。当前常数不自动包含实测膜 n/k 或温度/浓度依赖。
- `FILL_FRACTION`、`SMALL_HOLE_AREA_SHARE`、`HOLE_DISPLACEMENT_A`：SiO2 填充双圆孔。
- `ACTIVE_TE_FAMILIES` / `--te-families`：纳入共享增益竞争的纵向 TE 族，默认 0–3。每族仍只检查保留的四个连接分支，并未搜索所有纵向/横向模。旧 `TE_FAMILY` 只用于参考结构助手。
- `YB_DENSITY_CM3` 是数密度，不是直接输入百分数；截面单位 cm²，寿命单位 ms。
- `INTERNAL_TOTAL_PUMP_POWER_W` / `--pump-power-W` 是耦合进入增益层的总泵功率，默认 32 W。`--pump-architecture` 选择单边、双边或顶部；顶部多程用 `--top-passes`、`--return-efficiency` 指定理想非相干回程。外部入射、内部入口和吸收功率需要区分。旧 per-edge 强度常量只用于冻结参考函数。
- `INTERNAL_POWER_LOSS_CM` 是额外强度损耗，cm⁻¹；辐射由分层 Green 求解。

上下透明层导波，底 DBR 改变辐射边界，面内反馈仍来自二维 PC，不是改成纵向 Fabry–Pérot 腔。当前入口由上述命令行选项构造显式 `PumpGeometryConfig`；`SpatialPumpConfig` 仍用于底层单程对照。旧 `YbYAGMediumConfig` 中未被入口使用的泵/时间字段不改变当前 CW 解。

## 7. 理论、论文公式与算法对应

**晶胞周期场**描述一个周期，**器件包络**描述跨周期慢变模，**远场**描述角空间辐射；三者不是同一图片的放大缩小。

```text
孔形/孔心/折射率 → 介电常数 Fourier 系数
层结构 → 纵向 TE 模与 Bragg 参考条件
两者共同 → 基本波耦合＋辐射耦合＋高阶间接耦合
         → 近 Γ 能带 → 有限器件开放边界模式/损耗
         → 器件包络＋晶胞场＋复矢量远场
         → 半导体阈值/四波时域，或 YAG 空间增益/CW 门槛
```

| 阶段 | 论文关系/实际算法 | 模块/步骤 |
| --- | --- | --- |
| 晶胞展开 | Liang (3.5)；圆/椭圆解析形状因子，多边形边界积分，孔心引入 Fourier 相位 | [geometry.py](src/pcselsim/geometry.py)，晶格/k 空间 |
| 纵向模 | 一维标量 TE 有限差分；方形 λB=a·n_eff，三角 λB=√3·a·n_eff/2 | [vertical.py](src/pcselsim/vertical.py) |
| 四波耦合 | Liang (3.14)、(3.19)–(3.20)、(3.23)–(3.35)；C1D＋Crad＋C2D | [three_d_cwt.py](src/pcselsim/three_d_cwt.py) |
| 六波耦合 | Liang 5.2/附录 B；六基本波及 TE 偏振，基本/辐射/高阶项，C6 对称截断 | [triangular_six_wave.py](src/pcselsim/triangular_six_wave.py) |
| 有限区域 | 四波 (4.21)–(4.22)，六波 (5.17) 特征方向离散；零入射、交错广义本征、最细实际本征对及多网格敏感度 | [custom_analysis.py](src/pcselsim/custom_analysis.py)、[triangular_finite.py](src/pcselsim/triangular_finite.py) |
| 矢量远场 | Liang (4.24)–(4.26)；复 Ex/Ey 分别 Fourier 后相加强度，不强加涡旋/Gaussian | 图谱/远场 |
| 半导体动态 | Inoue (8)–(11)；分裂步、Lax–Wendroff 场传播、Heun 积分，场/载流子耦合 | [solver.py](src/pcselsim/solver.py)，08_time_domain |
| 晶体底镜 | 分层出射 Green、有限 DBR 振幅/相位、上下通量闭合，不是功率简单乘反射率 | [stratified_optics.py](src/pcselsim/stratified_optics.py)、[reflector_cwt.py](src/pcselsim/reflector_cwt.py) |
| 晶体增益 | 准三能级再吸收、饱和泵传播、x–z 共有反转、各模独立光子库与功率闭合 | [ybyag_multimode.py](src/pcselsim/ybyag_multimode.py)，05_spatial_gain |

能带是**近 Γ 沿 M–Γ–X 的局域片段**，不是完整布里渊区；三角 X 为工程名称，实际采用 K 方向。四波/六波有限算法默认采用同一特征交错广义质量矩阵，消除了旧迎风体数值扩散；孔径阶梯边界、截断和模族选择仍需收敛审查。每个带边族的连接模不等于完整横模谱。显式选择 `legacy_upwind` 才使用旧离散。

三角 `12_small_signal_pump` **不是非线性 L–I 曲线**；YAG CW 解也不表示稳定单模、调 Q 或飞秒锁模。

详细推导：[几何驱动 3D-CWT](docs/geometry_derived_3dcwt_zh.md)、[六波](docs/triangular_six_wave_zh.md)、[Inoue–Liang](docs/liang_inoue_model_relationship_zh.md)、[反射 Green](docs/stratified_optics_derivation_20261008_zh.md)。

## 8. 怎样读输出？

| 文件 | 用途 |
| --- | --- |
| PNG：晶格/全貌/纵向模/能带/图谱 | 直观检查；全貌可能抽样显示阵列，冷腔场通常已归一 |
| run_plan.json / parameters.json / summary.json | 实际参数、开关、来源、假设；不是只看脚本现在的值 |
| mode_summary.csv / 08_finite_modes.csv | 模式表，先分清单位、raw/extrap 来源和角窗 |
| 05_numerical_quality.json / 08_numerical_quality.json | 当前默认报告最细实际本征损耗、网格敏感性和状态；旧算法另有外推口径 |
| 05_threshold_current_audit.json / 11_threshold_current_audit.json | 损耗→增益→均匀注入电流，以及参考值来源 |
| 05_cold_cavity_arrays.npz / 08_cold_cavity_arrays.npz | 复本征场与收敛数组，不是图片 |
| YAG *_optical.npz / shared_inversion.npz / summary.json | 各分支频率、复场、共有反转、光子数及能量预算 |
| run_status.json | 当前入口完成标记：半导体 completed，YAG complete |

每个新运行还保存 `source_provenance.json`，记录启动时源码哈希和环境。时域 NPZ 新增 `probe_signals`、`probe_coordinates_um`，频谱图同时给出局部探针非相干和与原相干投影；两者都不是全部出射光的实验线宽。

老运行可能没有完成标记；新入口中断也可能留下 PNG。结合终端、数据和元信息判断，不只数图片。

必须记住：

1. **幅度损耗 α≠功率损耗**：通常 `g_modal=2α+α_internal`。m⁻¹ 转 cm⁻¹ 除以 100；α 也不是材料增益阈值。
2. **实际本征对与拟合分开**：当前默认交错离散的 `alpha_per_m` 与保存的最细网格场属于同一个实际本征对；h/h² 拟合只诊断敏感度，不替代它。仅旧 `legacy_upwind` 和历史数据保留外推损耗口径。
3. **最低计算家族不等于已确认激射模**：`family_selection` 会比较模式间隔与合并网格敏感度。`best/selected` 只是保留候选集内选择；未解决的排序会明确标记。
4. **归一冷腔场不是绝对 V/m/瓦特**：功率还需增益、光子数与通道归一化；四波时域 `power_W` 不是自动等于上侧实测收集功率。
5. **远场能量对保存角窗归一**：0.5° 半径内能量不是全半球效率，不同角窗不可直接比。full RMS 为 `2√〈θx²+θy²〉`，不是 FWHM、1/e² 或 M²。
6. **Zoom 只改显示**：`FAR_FIELD_ZOOM_DEG` 裁图，不改善光束。四波 `06_A_vector_far_field.png` 固定是 A，不一定是最低阈值模。
7. **泵/电流有来源与假设**：半导体通常按均匀注入换算；YAG 内部入口、吸收、外部功率不同。噪声种子不是校准线宽模型；实际存样间隔也不能只由请求间隔猜测。

查看 NPZ 的最小示例，可在 Python Console 执行；将路径替换成真实目录：

```python
from pathlib import Path
import numpy as np

folder = Path("D:/things/Master/Doc/PCSELSim/results/current/你的结果目录")
with np.load(folder / "05_cold_cavity_arrays.npz", allow_pickle=False) as data:
    print(data.files)  # 先查数组名，不凭猜测读取键
```

索引：[结果导航](results/README.md)、[文档导航](docs/README.md)、[2026-10-21 修正与验证](docs/corrections_20261021_zh.md)。历史结果仅供追溯，不能与新离散、新层栈的数值拼接排名。

## 9. 正式设计的验收顺序

1. 看晶胞/全貌，查周期邻孔重叠、面积、真实尺寸。重叠会破坏当前面积/Fourier 叠加假设。
2. 查纵向 TE 族、PC/增益重叠，以及材料参数是否适用。
3. 增大 `CWT_TRUNCATION_ORDER`（YAG 为 `CWT_TRUNCATION`）、减小 `VERTICAL_STEP_NM`，查耦合收敛。
4. 增大 `FINITE_EIGEN_GRIDS` / `FINITE_GRIDS`，比较实际本征对、网格敏感性和排序；三角网格不是普通图片分辨率。仅旧算法同时比较外推值。
5. 若网格敏感度大于模间阈值间隔，不能确定真实首起振模。`mesh_checked` 只是经验网格检查，`provisional` 是旧外推的暂定状态，均不是物理收敛证明；`grid_sensitive` 不允许精细排名。
6. 同面积、同增益/内损耗、同角窗比较。亮中心不证明单模，低阈值不自动代表高输出/高光束质量。
7. 查最小孔宽、间隙、圆角、位置误差。`FABRICATION_RULES` 是假设规则，报告失败不会自动阻止运行，也不是代工厂认证。
8. 用独立全波、另一种经验证的离散或实验复核关键方案。当前默认已经是交错离散；测试通过、通量守恒、图片相似都不是充分物理认证。

YAG 已补固定周期下的冷腔 `C(ω)` 频率闭合和指定分支的共享反转 CW 竞争。当前各 TE 族的线性冷腔独立计算，尚无相干 TE 间耦合；反转为 x-z 网格、y 向平均，CW 互补求解要求 beta=0。仍须检查更多纵向/横向模、增益牵引与动态稳定性、实测薄膜吸收/色散、材料截面和热负荷。新的频率闭合不是完整矢量 Maxwell 极点或实验认证，旧冻结 Bragg 功率留在历史归档。

## 10. 常见错误

| 现象 | 排查 |
| --- | --- |
| ModuleNotFoundError: pcselsim | 用新入口、保留完整 src/configs；当前解释器可编辑安装，不手动复制包 |
| 缺 numpy/scipy | 用运行脚本的同一个 Python 执行 -m pip；检查 PyCharm 解释器 |
| 文件/配置找不到 | 工程是否完整；新入口不要脱离工程，相对输出按根目录解析 |
| invalid choice / unrecognized arguments | 是否混用了四波/六波/YAG 步骤或 CLI |
| FileExistsError | 用新结果目录，不覆盖旧数据 |
| 无导模/被动模负损耗 | 查层结构、边界与精度，不强改阈值/忽略守恒警报 |
| 关图仍很慢 | 远场/阈值/参数报告可能仍需本征解算 |
| 部分 PNG、没有结束信息 | 可能失败；看最后异常与完成标记 |
| 改半径图没变 | 多孔用 inclusion 元组，单孔字段可能不生效 |
| 光斑显示太小 | 改 zoom；评价仍看原角窗，不把裁图当光束改善 |

高精度和长时域可能较慢。按步骤定位瓶颈，再扩大规模；中断后保留目录排查，重跑使用新目录。

## 11. 测试、Git 与多端同步

### 11.1 修改后测试

```powershell
& $pcselPython -m pytest
```

测试检查实现约定、数值性质、守恒与回归，不替代论文/实验验证。改参数还应做 dry-run（半导体）和相关步骤连接检查。

### 11.2 这台电脑保存代码

以下是 **Git Bash** 命令，先进入工程根。已有仓库不必再 git init，已有 origin 不必再 remote add。

```bash
git status
git remote -v
git diff
# 示例：按实际任务选择文件，不盲目包含临时文件
git add README.md scripts/run_square_double_hole.py
git diff --staged
git commit -m "docs: update tutorial and record square parameters"
git push
```

新增模块/测试要加入相应路径。首次未设置跟踪分支可用 `git push -u origin main`。origin 错误用 `git remote set-url origin 正确地址`，不是重复添加。

### 11.3 另一台电脑下载与更新

在准备好的**空目录**中（private 仓库须先获得访问权限）：

```bash
git clone git@github.com:DBLYBB/PCESLSim.git .
git status
```

远端名 `PCESLSim` 与本地 `PCSELSim` 拼写不同，以 `git remote -v` 实际地址为准。每台机器单独配 SSH 密钥并授权**公钥**；不要发送/上传私钥。认证失败先解决 SSH 与访问权限。

下载后按第 2 节配置那台电脑的 Python，Git 不会同步 Anaconda 环境。日常开始工作前，先检查是否有本地未提交改动：

```bash
git status
git pull --ff-only
# 修改、测试、提交，然后 git push
```

一台推送后另一台才能拉到新提交。如分叉或冲突，先保留本地改动，按 [Git 指南](docs/git_workflow_zh.md) 处理，不用 reset --hard“修同步”。

### 11.4 代码同步不等于结果同步

`.gitignore` 默认忽略生成结果、NPZ、缓存、临时目录和 IDE 私有配置。**GitHub 有代码，不代表另一台已有结果。**

另行备份当次参数/开关、版本或 SHA256、JSON/CSV、复场 NPZ 和图片；不要只保留 PNG，也不要上传私钥、令牌、整个虚拟环境或大批仿真数组。

## 12. 深入阅读与旧路线

- [入口指南](docs/entrypoints_zh.md)：步骤、迁移与历史记录。
- [理论代码映射](docs/theory_zh.md)、[几何驱动](docs/geometry_derived_3dcwt_zh.md)、[六波](docs/triangular_six_wave_zh.md)：理解模型后读实现。
- [物理/设计审计](docs/research_design_audit_20261008_zh.md)：可信度、逐文件评价与验证优先级。
- [YAG 材料](docs/ybyag_waveguide_materials_20261008_zh.md)、[空间增益](docs/ybyag_dbr_pump_design_zh.md)、[独立检查](docs/ybyag_independent_checks_20261008_zh.md)。
- [YAG 厚度与泵浦实验教程](docs/ybyag_thickness_pumping_study_20261008_zh.md)：独立诊断脚本、固定总功率比较、单/双边与顶部多程泵；不改变日常四个主入口。
- [超快路线](docs/ybyag_ultrafast_routes_zh.md)：CW、调 Q、锁模、独立 Kerr 微腔是不同问题，当前入口不输出经验证的飞秒脉冲。
- [文档总索引](docs/README.md)、[结果导航](results/README.md)、[Git 指南](docs/git_workflow_zh.md)。

安装后的 `pcselsim reproduce --config ...` 是**旧 YAML 标定耦合路线**，不等于这里的几何驱动入口；默认写入 `results/current/legacy/inoue2019/<时间戳>`，拒绝非空目录与历史归档。归档脚本的新输出位于 `results/current/legacy/`，其历史参数和教学近似没有因此升级为器件预测。日常设计使用第 1 节入口。

推荐习惯：**保存参数 → 看结构 → 做冷腔 → 查收敛与损耗来源 → 看远场 → 做增益/动态 → 保存版本与数据 → 再比较设计。**

