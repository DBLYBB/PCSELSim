# PCSELSim

面向大面积光子晶体面发射激光器（PCSEL）的 Python 时域耦合波仿真框架。当前基准模型以
Inoue *et al.*, *Physical Review B* **99**, 035308 (2019) 为核心，实现四个基本 Bloch 波、
二维载流子扩散、受激复合、载流子引起的折射率变化、开放边界、弛豫振荡、近场和频谱输出。

## 当前实现到什么程度

- 论文式 (8)：`Rx, Sx, Ry, Sy` 四波的时域传播、局部增益/损耗、失谐和 4x4 复耦合矩阵。
- 论文式 (9)：由 `Im(C)` 计算面辐射功率。
- 论文式 (10)：电流注入、载流子寿命、受激辐射和二维扩散。
- 论文式 (11)：有饱和的载流子-增益关系。
- Liang 的有限尺寸开放边界：`Rx(0)=Sx(L)=Ry(0)=Sy(L)=0`。
- 表 II 的物理参数、300 um 方形电极、25 um 电流扩展，以及 `I/Ith` 扫描。
- 可保存 `.npz` 原始数据，并生成瞬态、载流子/光子近场和激光光谱图。
- `vertical.py` 可从任意多层折射率/厚度堆栈求 TE0 纵向本征模、`neff` 和逐层限制因子，便于换基底。

论文未公开由三维 SEM 孔形重建得到的完整复耦合矩阵 `C`、实际孔形体数据、网格和噪声归一化。
因此仓库中的默认 `C` 是依据论文图 2 的 A-D 带边排序和损耗关系建立的可替换校准模型；它可以
复现论文所述的物理机制和定性趋势，但在获得作者原始 `C` 之前，不能诚实地称为逐点数值复现。
详见 [复现状态](docs/reproduction_status_zh.md)。

最新一次逐结果物理审计、各目录可信度、论文公式到代码的完整映射，以及本轮发现并
修正的折射率投影/载流子子步问题，见
[物理审计与中英文代码地图](docs/physics_audit_and_code_map_zh_en.md)。开始引用任何
已有图片前，也请先读 [结果目录说明](results/README.md)。

自定义半导体入口 `scripts/run_custom_semiconductor_pcsel.py` 已不再使用上述手工校准的 `C`：它从
用户定义的单孔或多孔晶胞解析计算介电常数 Fourier 系数，求纵向 TE0 模，再按 Liang 第三章
构造 `C1D + Crad + C2D`，同一个总矩阵同时送入有限区域本征求解和 Inoue 时域求解。推导、适用边界
及验证见 [几何驱动 3D-CWT 说明](docs/geometry_derived_3dcwt_zh.md)。它现在默认使用 Inoue 表 I/II、
`a=277 nm`、`L=300 um` 和图 2(a) 的双椭圆中心位移；时域用 400 um 计算域包住 300 um 电极及
电流扩展区。椭圆尺寸是明确标注的理想化输入，因为论文
没有公开真实 SEM 三维孔形；不能把该结果称为作者器件的逐像素重建。

## 安装

建议使用 Python 3.11 或 3.12。在 PowerShell 中：

```powershell
cd D:\things\Master\Doc\PCSELSim
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果机器上只有一个 Python，也可以把 `py -3.12` 换成 `python`。本项目已在 Python 3.13
上做过测试。

## 第一次运行

如果你是 Python 初学者，请先阅读 [从零开始运行、理解和修改 PCSELSim](docs/beginner_guide_zh.md)。
在 PyCharm 中第一次直接运行 `scripts/run_quick.py`，成功后再运行 `scripts/run_full.py`。
这两个入口会自动定位项目根目录和 `src`，即使 PyCharm 的 Working directory 设置不正确、尚未执行
`pip install -e .`，也能找到 `pcselsim` 和基准配置。仍建议按教程选择正确的解释器并安装依赖。

先运行短测试（约数十秒到数分钟，视 CPU 而定）：

```powershell
python scripts\reproduce_inoue2019.py --quick --output results\quick
```

再运行完整的 10 ns、四电流点计算：

```powershell
python scripts\reproduce_inoue2019.py --output results\inoue2019
```

也可使用安装后的命令：

```powershell
pcselsim reproduce --config configs\inoue2019.yaml --output results\inoue2019
```

结果目录包含每个电流点的 `.npz`、瞬态图、空间分布图、组合频谱图和 `summary.csv`。

## 两个可直接修改的自定义主程序

半导体 PCSEL 请直接运行：

```powershell
python scripts\run_custom_semiconductor_pcsel.py
```

默认结果进入 `results/custom_semiconductor_inoue2019`。完整时域窗为 10 ns；修改几何时可先把
脚本顶部 `08_time_domain` 改为 `False`，待线性模式确认后再运行时域。

Wang 2024 三晶格结构使用独立入口和独立结果目录，不会覆盖 Inoue 图片：

```powershell
python scripts\run_custom_wang2024_triple_lattice.py
```

它实现 `a=474 nm`、三个直径 `90 nm` 的 InP 填充圆孔及论文 structure 1 的三个孔心，结果进入
`results/custom_semiconductor_wang2024_triple_lattice`。默认只运行 01–07 冷腔步骤；论文主文没有给出
完整 InAlGaAs 增益/复合参数，而完整外延表位于补充材料，因此第 08 步默认关闭。运行后先看
`00_wang2024_comparison.json`，其中把公开目标与本模型结果分栏列出。详见
[Wang 2024 三晶格复现说明](docs/wang2024_triple_lattice_zh.md)。

Yb:YAG 部分目前保留但本轮不再扩展。现有入口只是准三能级概念验证，光学矩阵仍为示例，不能当作
真实器件定量结果：

```powershell
python scripts\run_custom_ybyag_pcsel.py
```

这些文件都把波长、折射率、晶格与孔形、层结构、器件尺寸、损耗、耦合参数和数值参数集中放在
文件顶部的 `PARAMETER PANEL`；`STEPS` 中每个 `True/False` 就是对应图片或计算的开关。也可以在
命令行只跑一部分，例如：

```powershell
python scripts\run_custom_semiconductor_pcsel.py --only 02_lattice 03_k_space 05_linear_modes
python scripts\run_custom_ybyag_pcsel.py --only 08_ybyag_gain 09_rate_dynamics 10_pump_scan
```

半导体默认实算得到 A 模为最低阈值模，`alpha L≈0.229`。`05_threshold_current_audit.json`
同时记录由冷腔损耗反推的阈值与论文采用的 `0.7 A`；默认 Inoue 预设使用论文阈值，并保留差异供审计。
时域频谱先减去所选带边模的公共失谐，再恢复到物理参考波长，避免把 THz 级带边失谐折叠到错误谱线。

逐项说明、输出文件含义和推荐修改顺序见
[自定义仿真主程序使用指南](docs/custom_simulation_guide_zh.md)；Yb:YAG 速率方程、单位换算、
默认参数来源和当前假设见 [Yb:YAG 模型说明](docs/ybyag_model_zh.md)。Inoue 时域模型与 Liang
有限区域模型到底共享什么、哪些求解器不同，见
[Inoue–Liang 模型关系](docs/liang_inoue_model_relationship_zh.md)。

## PyCharm 设置

1. 用 PyCharm 打开整个 `PCSELSim` 文件夹，而不是只打开 `scripts`。
2. `Settings > Project > Python Interpreter` 选择 `PCSELSim\.venv\Scripts\python.exe`。
3. 新建 *Python* Run Configuration：
   - Script path：`scripts\reproduce_inoue2019.py`
   - Parameters：`--quick --output results\quick`
   - Working directory：项目根目录 `D:\things\Master\Doc\PCSELSim`
4. 短测试通过后，移除 `--quick` 运行完整配置。

## 修改器件

优先复制配置，而不是直接改论文基准：

```powershell
Copy-Item configs\inoue2019.yaml configs\my_device.yaml
pcselsim reproduce --config configs\my_device.yaml --output results\my_device
```

参数单位都写在字段名中：`nm`、`um`、`ns`、`cm^-1`、`cm^-3`。计算内核统一转换到 SI。
几何模块已经支持方形晶格中的圆/椭圆多原子基元及解析傅里叶系数。三角晶格需要把四波基底扩展成
六波基底；接口已与材料、注入和观测模块分离，但该六波求解器尚未宣称完成。

更换纵向材料堆栈时可从 `pcselsim.vertical.Layer`、`LayerStack` 开始；先做 `dz_nm` 与
`padding_um` 收敛测试，再把得到的 `effective_index` 和相应有源层限制因子写入新 YAML。

## 验证与测试

```powershell
pytest
```

修改物理参数后还应做收敛与公差扫描；`pytest` 只验证实现契约和数值基本性质，不会
自动证明某个真实器件与论文完全一致。

理论映射见 [理论说明](docs/theory_zh.md)，Git 与多端同步见
[Git 工作流](docs/git_workflow_zh.md)。

