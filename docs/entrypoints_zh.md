# 三个半导体主程序：从点击运行到理解输出

本轮只整理入口、输入参数、步骤开关和图片呈现，复用 `src/pcselsim/` 中已审计的物理后端。旧程序并未删除，移到 `scripts/archive/legacy_20261008/` 保留历史功能与导入关系；旧结果也没有删除或覆盖。Yb:YAG 另见 `scripts/run_ybyag_pcsel.py` 及晶体专题文档。

## 1. 在 PyCharm 中点哪个文件

| 主程序 | 默认几何与目的 | 不能据此宣称什么 |
|---|---|---|
| `scripts/run_paper_reproduction.py` | Inoue 2019 理想化双椭圆孔、四波几何驱动冷腔与载流子时域；研究论文理论流程 | 名称不表示已经逐图、逐参数完全复现原实验 |
| `scripts/run_square_double_hole.py` | 两个不同半径的圆孔，正方晶格，300 µm 正方器件；本轮制造约束下的局部候选 | 不是全球最优或经加工认证的设计 |
| `scripts/run_triangular_three_hole.py` | 圆角三角主孔 + 两个圆副孔，三角晶格，300 µm 正方器件；六波候选 | 不是六波非线性激光时域，也没有证明真实单模 |

打开项目根目录 `PCSELSim`，选择已安装依赖的解释器（例如 `E:\Anaconda\envs\env00\python.exe`）。在上表对应文件中右键 → **Run**。不用点击 `src` 中的模块，也不用先安装项目包；入口的 bootstrap 会找到项目根目录和 `src`。解释器仍须安装 `requirements.txt` 中的依赖。

论文入口默认包含较慢的尺寸扫描和 4 个电流的 10 ns 时域。第一次建议先采用下面的轻量命令，不要为了缩短时间擅自更改材料参数。

```powershell
E:\Anaconda\envs\env00\python.exe scripts/run_paper_reproduction.py --quick --only 02_lattice 02_device_overview 03_k_space 04_layer_stack 05_band_structure
E:\Anaconda\envs\env00\python.exe scripts/run_square_double_hole.py --quick --only 00_fabrication 02_lattice 02_device_overview 03_k_space
E:\Anaconda\envs\env00\python.exe scripts/run_triangular_three_hole.py --quick --only 01_parameters 02_lattice 02_device_overview 03_k_space 04_vertical_mode 05_band_structure
```

`--quick` 是程序连接的烟雾测试：降低高阶截断、纵向和横向网格，并将四波时域缩为 0.1 ns、一个电流。它不是可信阈值、谱线宽或论文复现数据。四波冷腔仍使用至少三个且 N≥9 的网格，以满足现有解算器要求。

## 2. 怎么修改自己的器件

每个主程序顶部有参数区，按几何 → 纵向层 → 有限器件/载流子 → 数值精度 → 开关组织。`build_parameters()` 把参数装入一个显式、不可变输入对象，主流程 `execute_run(..., run_four_wave/run_triangular)` 只接收这个对象，不会偷偷改变旧脚本全局变量。

先复制一份参数记录或提交 Git，再逐项修改：

1. `LATTICE_CONSTANT_NM`：晶格常数，单位 nm。修改后 Bragg 波长会由纵向模式和晶格条件重新求解；`WAVELENGTH_GUESS_NM` 只是初值，不是强行指定激射波长。
2. 四波用 `UNIT_CELL_INCLUSIONS` 列每个孔的中心、半轴和转角，位置/尺寸都是 nm。正方双圆默认总填充率 0.14，孔面积分配 0.42/0.58，两孔坐标沿 x、y 分别相差 0.45a；0.45a **不是**两孔的直线距离。
3. 六波用 `DESIGN` 修改总填充率、主孔面积占比、两个副孔距离/展开角、主孔旋转角、圆角量及副孔种类。当前主孔旋转 0°，不是把 15° 当作普适最佳角。`circle`、`ellipse`、`triangle` 等受已实现几何支持范围限制。
4. `LAYERS`、上下包层折射率、背景/孔折射率决定纵向 TE0、PC 及增益重叠。换材料要同时修改材料色散、损耗与增益参数，不能只改波长。
5. `DEVICE_SIZE_UM` 或 `DEVICE_HALF_SIZE_UM` 控制有限 PC 的真实尺寸。四波时域 `TIME_DOMAIN_SIZE_UM` 是另一计算窗口，电极尺寸又是独立量，不能将三者当作同一个数。三角入口支持 `square/circle/hexagon` 孔阵列边界；半径 150 µm 的圆/六边形与 300 µm 正方形**面积不同**。
6. `INTERNAL_LOSS_CM` 是额外的强度损耗 cm⁻¹；不要再把辐射损耗或有限边界泄漏重复放入这个参数。电流阈值换算还依赖寿命、透明载流子密度、增益、有效层厚及泵/电极面积。
7. 最后提高 `FINITE_EIGEN_GRIDS/FINITE_GRIDS`、`CWT_TRUNCATION_ORDER`、`VERTICAL_STEP_NM` 的精度并比较，不可把网格当作拟合参数。`FABRICATION_RULES` 是明确的工程假设，不能代替代工厂工艺规格。

## 3. 步骤、理论与实际算法

| 步骤 | 实际工作 | 对应理论/限制 |
|---|---|---|
| `00_fabrication` | 实际 nm 周期邻孔间距、孔最小宽度、圆角和误差余量 | 只审查平面几何；不等于刻蚀深度、侧壁粗糙度或再生长良率认证 |
| `01_parameters` | 保存所有输入、耦合分量及模式阈值口径 | 四波为了写入一致阈值会触发冷腔求解，**不是**最轻的步骤；六波参数输出本身不求有限模 |
| `02_lattice` / `02_device_overview` | 晶胞孔阵列 / 器件边界 + 全貌 + 真比例晶胞放大 | 全貌点阵明确抽样；不会把纳米孔人为放大后冒充器件实际孔径 |
| `03_k_space` | 保留的基本 Bloch 波与倒格矢 | 正方四波、三角六波由各自倒格子推导，不是仅增加两条波 |
| `04_layer_stack/04_vertical_mode` | 一维标量 TE0、neff/ng、PC 与有源层限制因子 | 单一纵向分支近似；并非多纵模全矢量 Maxwell 解 |
| `05_band_structure` | C + 波矢失谐矩阵本征值 → a/λ | 当前为 Γ 附近沿 M–Γ–X 方向的小失谐能带，**不是**跨整个 Brillouin 区的全带计算；三角 X 为既有命名中的 K 方向 |
| 四波 `05_linear_modes` / 六波 `08_finite_modes` | 有限器件开放入射边界、稀疏本征求解、多网格外推 | Liang 有限区域耦合波方法；每个 Γ 带边族保留一个连接模，四个/六个结果**不等于全横模谱** |
| `06_mode_atlas` / 六波 `07_band_edge_states`, `09_mode_atlas` | 所有保留模的器件包络、晶胞电场与辐射图 | 微观周期场与宏观慢变包络的重构须区分 |
| `06_far_field_zoom` / 六波 `10_best_far_field`, `10_far_field_zoom` | 复矢量辐射场 Fourier 变换；主图和 ±0.2° 局部显示 | Liang 辐射远场关系；没有强加中心涡旋或 Gaussian 因子。Zoom 只裁切显示，RMS/能量仍来自原采样窗 |
| 四波 `07_length_sweep` | 多器件尺寸、多网格冷腔损耗 | 同一模式族及数值精度的尺寸变化，不是一次本征值乘比例 |
| 四波 `08_time_domain` | 场传播与载流子扩散、复合、增益和折射率耦合 | Inoue Eqs. (8)–(11)；已有 DC、噪声、吸收修正。短时域不能证明稳态，更不能用一个相干投影的单峰证明全器件单模 |
| 六波 `11_threshold_audit`, `12_small_signal_pump` | 损耗/Q、均匀注入阈值及小信号增益/折射率扫描 | **未实现六波非线性饱和载流子时域**；这不是激光 L–I 或脉冲仿真 |

详细公式推导见 `theory_zh.md`、`geometry_derived_3dcwt_zh.md`、`triangular_six_wave_zh.md` 与本轮审计报告。部分历史文档标题/图名保留旧实验含义，请以入口归档提示和本轮数值口径为准。

## 4. 开关与保存位置

在文件 `STEPS` 中把 `True` 改成 `False` 跳过该图/步骤。也可以通过命令选择输出；必要的求解依赖仍会运行，例如输出远场必须先求有限模。

```powershell
# 只列步骤，不计算
E:\Anaconda\envs\env00\python.exe scripts/run_square_double_hole.py --list-steps
# 看当前所有参数，不计算
E:\Anaconda\envs\env00\python.exe scripts/run_triangular_three_hole.py --dry-run
# 完整冷腔，不做四波尺寸扫描或时域
E:\Anaconda\envs\env00\python.exe scripts/run_paper_reproduction.py --only 00_fabrication 01_parameters 02_lattice 02_device_overview 03_k_space 04_layer_stack 05_band_structure 05_linear_modes 06_mode_atlas 06_far_field_zoom
# 一次时域连接测试，不是10 ns论文数据
E:\Anaconda\envs\env00\python.exe scripts/run_paper_reproduction.py --quick --only 01_parameters 05_linear_modes 08_time_domain
```

默认每次存到 `results/pcsel/paper_reference/时间戳`、`square_double_hole/时间戳` 或 `triangular_three_hole/时间戳`。可以 `--output 新目录`，但程序拒绝非空目录，避免覆盖原结果。`run_plan.json` 保存当次参数与开关；`run_status.json` 出现且为 `completed` 只说明该流程没有抛异常，不说明物理收敛。中途报错可能留下部分图片，不能按图片数量判定运行成功。

`05_cold_cavity_arrays.npz/08_cold_cavity_arrays.npz` 保存复场和网格序列；`05_numerical_quality.json/08_numerical_quality.json` 同时给外推损耗、最细网格损耗、敏感性与状态。遇到 `grid_sensitive`，不能只拿外推截距给设计排名或当作实验阈值。本文示例中的 “selected/best” 表示所保留带边模集合中的数值选择，不是已证实真实最低阈值横模。

## 5. 两条论文路线与历史入口

新的论文入口使用孔的 Fourier 系数、纵向 TE0 和 3-D CWT 来构造 C，不把文献阈值或图像作为耦合矩阵输入。它仍保留论文有源限制因子/参考电流等可追踪选项，应区分“采用论文值”和“由当前层结构求出值”。

旧 `scripts/archive/legacy_20261008/reproduce_inoue2019.py`、`run_full.py` 等校准路线和旧 Liang 逐步脚本保留，用于历史对照；校准路线拟合得像论文不是几何模型独立验证。旧脚本仍可从完整路径直接运行，但可能沿用旧的固定结果目录；需要保留历史结果时优先采用三个新主入口的自动时间戳目录。

新主程序不是新增优化器：默认候选来自已记录的局部扫描，而不是每次启动都遍历参数空间。真实加工优化还需要刻蚀深度/侧壁、系统偏差与随机无序、热漂移，以及完整横模竞争。不能依据单一中心亮度、单一最低阈值就宣布“最好”。FWHM、1/e² 宽度、RMS 发散角和角窗内包围能量也不能混为同一指标。

## 6. 本轮入口验证记录（2026-10-08）

以下是工作流连接/输出检查，不是新一轮全局优化。

| 路线 | 实际完成内容 | 目录 |
|---|---|---|
| 论文四波 | D=10、dz=3 nm、N=13/17/21/25 完整冷腔、晶格/能带/四模图；未再跑全长时域 | `results/pcsel/paper_reference/20261008_163214_269631` |
| 正方双圆 | 默认完整冷腔及晶格/能带/四模图；时域和尺寸扫描按默认开关关闭 | `results/pcsel/square_double_hole/20261008_163214_266631` |
| 三角三孔 | D=10、dz=3 nm、半径网格 11/15/19，默认全部冷腔/小信号步骤，包含器件全貌与远场 zoom | `results/pcsel/triangular_three_hole/20261008_164203_411501` |
| 论文入口新显示/时域烟测 | `--quick`，0.1 ns、1.4Ith、时域 N=15、冷腔 N=9/11/13，包含全貌/zoom/功率及谱数组 | `results/pcsel/paper_reference/20261008_164151_865877` |
| 正方入口新显示烟测 | `--quick`，加工检查、晶格/k空间、器件全貌；没有冷腔/时域 | `results/pcsel/square_double_hole/20261008_164212_281348` |

三角新入口外推最低为 B1，α=0.415664 cm⁻¹、均匀注入参考阈值 0.470545 A；但最细网格最低为 B2（α=4.185218 cm⁻¹），B1 为 4.304994 cm⁻¹。B1 外推敏感性 0.831695 cm⁻¹ 远大于 B1–B2 的 0.051492 cm⁻¹ 间隔，故这组数据**不能确认真实最低阈值族、低阈值优势或单模性**。B1 的中心/峰值=1、窗口内全 RMS=0.308043°、±0.5°圆内包围能量=0.987841 只描述这个候选场，不是确定激射态。平面加工检查 gap=79.345 nm、最小孔宽=71.210 nm；3 nm 边偏/位置预算后为 67.345/65.210 nm，满足本入口假设的 60/65 nm 规则，但不代表真实代工认证。

短时域烟测的旧摘要列曾沿用固定 `steady_window_ns=1.0` 标签，但实际只计算了 0.1 ns，因此不应解读为 1 ns 稳态。入口现已改为实际采样尾窗口 `actual_tail_window_ns`；本次保存数组无需改变，额外解释见该结果目录的 `08_summary_interpretation.json`。计算末端平均值也不自动意味着已达到稳态。

早期 `20261008_162815_781445`（四波 quick 网格不足）、`20261008_163308_332612`（四波 quick 最小 N 不足）、`20261008_163214_281099`（六波新 helper 缺面积函数）是**整理流程时的失败目录**，保留但不能作为完成结果；相应入口连接已修正，未改物理方程。

全长 Inoue 四电流 10 ns 的本轮物理审计结果另见 `results/custom_semiconductor_audit_20261008` 和 `audit_semiconductor_literature_20261008_zh.md`；新入口短测试不替代该结果。判断单/多模需要局部非相干光谱与多模投影进一步诊断，不能把当前一个投影测量当作全场证明。
