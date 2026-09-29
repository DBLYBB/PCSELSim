# 从零开始运行、理解和修改 PCSELSim

这份教程假定你没有 Python 项目经验。第一次不要改求解器代码，按照第 1-6 节依次完成即可。

## 0. 最先记住的两个文件

在 PyCharm 中第一次运行，请打开并运行：

```text
scripts/run_quick.py
```

它只进行短时间、低网格数的程序检查，通常比完整计算快很多。短测试成功后，再运行：

```text
scripts/run_full.py
```

它会运行论文基准的 4 个电流点和 10 ns 时间窗。不要一上来就运行完整计算，也不要直接运行
`solver.py`；`solver.py` 是被调用的计算内核，不是用户入口。

三个 `scripts` 入口都会先运行 `_project_bootstrap.py`：它根据脚本自身位置自动找到 `PCSELSim`
根目录，把 `src` 加入 Python 模块搜索路径，再把工作目录切到项目根目录。因此直接用 Anaconda
环境执行 `run_quick.py` 时，不需要先把本项目安装成包；但 NumPy、SciPy、Matplotlib 和 PyYAML
等第三方依赖仍必须存在。

## 1. 认识项目目录

```text
PCSELSim/
├─ configs/
│  └─ inoue2019.yaml          论文参数和数值设置
├─ docs/                      理论、复现状态、Git 和本教程
├─ scripts/
│  ├─ run_quick.py            第一次点这个：短测试
│  ├─ run_full.py             短测试通过后点这个：完整计算
│  ├─ reproduce_inoue2019.py  带命令行参数的高级入口
│  ├─ run_custom_semiconductor_pcsel.py  可编辑的 Inoue 双晶格/自定义器件入口
│  └─ run_custom_wang2024_triple_lattice.py  Wang 2024 三晶格独立入口
├─ src/pcselsim/
│  ├─ solver.py               光场-载流子时域求解核心
│  ├─ materials.py            增益和载流子折射率
│  ├─ coupling.py             4x4 耦合矩阵 C
│  ├─ injection.py            电极和电流扩展
│  ├─ observables.py          光子密度、功率、光谱
│  ├─ plotting.py             绘图
│  ├─ geometry.py             双晶格椭圆孔和傅里叶系数
│  └─ vertical.py             多层波导 TE0 纵向本征模
├─ tests/                     自动测试
├─ results/                   运行后自动产生，不提交到 Git
├─ pyproject.toml             项目和依赖定义
└─ requirements.txt           依赖的另一种列表形式
```

你日常最常碰的只有三个位置：

1. `scripts/run_quick.py` 或 `scripts/run_full.py`：启动计算。
2. `configs/inoue2019.yaml`：修改物理与数值参数。
3. `results/`：查看结果。

如果你的目标是修改晶格、孔形和层结构，应运行
`scripts/run_custom_semiconductor_pcsel.py`。它默认输出到
`results/custom_semiconductor_inoue2019`；完整第 08 步为 10 ns，第一次可先只运行 01–07。

## 2. 第一次用 PyCharm 打开工程

### 2.1 打开正确的文件夹

1. 启动 PyCharm。
2. 点击 **Open**。
3. 选择整个目录：

   ```text
   D:\things\Master\Doc\PCSELSim
   ```

4. 不要只打开 `scripts`、`src` 或某一个 `.py` 文件。
5. 如果询问是否信任项目，确认这是自己的项目后选择信任。

左侧 Project 面板的最上层应该能同时看到 `configs`、`docs`、`scripts`、`src` 和 `tests`。

### 2.2 创建独立 Python 环境

“虚拟环境”可以理解为本项目专用的 Python 工具箱，不会把不同项目的库混在一起。

在 PyCharm 底部点击 **Terminal**，确认提示符所在目录以 `PCSELSim` 结尾，然后输入：

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

激活后，命令行开头通常会出现 `(.venv)`。如果 Windows 阻止脚本执行，可只对当前终端临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

这不会永久修改整台电脑的策略。

安装本项目和依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`-e` 表示“可编辑安装”：以后修改 `src` 中的代码，不必重复安装。

### 2.3 让 PyCharm 使用这个环境

1. 打开 **File > Settings > Project: PCSELSim > Python Interpreter**。
2. 点击 **Add Interpreter > Add Local Interpreter**。
3. 选择 **Existing**。
4. 选择：

   ```text
   D:\things\Master\Doc\PCSELSim\.venv\Scripts\python.exe
   ```

5. 点击 OK。

这一步非常重要。终端安装依赖的 Python 和右上角运行按钮使用的 Python 必须是同一个。

## 3. 先运行自动测试

在 PyCharm Terminal 输入：

```powershell
pytest
```

正常结果应类似：

```text
9 passed
```

测试的含义不是“论文已经完全复现”，而是确认：

- 配置可以读取；
- 增益公式在透明载流子浓度处为零；
- 耦合基底正交且辐射损耗为被动损耗；
- 几何傅里叶系数满足共轭对称；
- 短时域计算没有 NaN、无穷大或负载流子；
- 纵向 TE0 模求解结果处在包层和芯层折射率之间。

## 4. 第一次点哪个文件、点哪里

### 4.1 短测试

1. 在左侧展开 `scripts`。
2. 双击 `run_quick.py`。
3. 在编辑区内点击鼠标右键。
4. 点击 **Run 'run_quick'**；也可以点击文件左侧绿色三角形。
5. 底部 Run 窗口会出现：

   ```text
   [PCSELSim] simulating I/Ith=1.05
   [PCSELSim] simulating I/Ith=2.80
   ```

6. 最后出现 `Process finished with exit code 0` 表示成功。

短测试输出在：

```text
results/quick/
```

### 4.2 完整计算

短测试和 `pytest` 都成功后：

1. 打开 `scripts/run_full.py`。
2. 右键选择 **Run 'run_full'**。
3. 它会依次计算 `1.05 Ith`、`1.4 Ith`、`2.8 Ith`、`4.2 Ith`。
4. 完整计算明显比短测试慢。运行时不要关闭 PyCharm，也不要同时启动第二个相同任务。

完整结果在：

```text
results/inoue2019/
```

## 5. 怎样看输出文件

每个电流点都有三类文件。例如 `I_2p80` 表示 `I=2.80 Ith`：

```text
I_2p80_transient.png  中心载流子浓度和输出功率随时间变化
I_2p80_spatial.png    最终载流子分布和光子面密度
I_2p80.npz            完整原始数值数据
```

另外还有：

```text
figure4_spectra.png   所有电流点的组合光谱
summary.csv           最终功率和中心载流子浓度的表格
```

看瞬态图时：

- 最初功率接近零：载流子还没有积累到阈值。
- 第一个尖峰：激光突然开启，储存的载流子快速转化为光子。
- 后续衰减振荡：弛豫振荡。
- 最后趋于常数：连续波稳态。

看空间图时：

- 载流子中心比边缘低：中心光强大、受激复合强，即空间烧孔。
- 高电流时光子分布变宽：对应论文图 3 讨论的光场均匀化趋势。
- 图外侧载流子下降：电极外只有高斯扩展后的注入电流。

`.npz` 是 NumPy 的压缩数据文件。暂时不需要手工打开；以后可以这样读取：

```python
import numpy as np

data = np.load("results/inoue2019/I_2p80.npz")
print(data.files)
print(data["power_W"])
```

## 6. 从点击运行到得到图片，程序经过了什么

整个调用顺序如下：

```text
run_quick.py / run_full.py
        |
        v
cli.py 解析任务
        |
        v
config.py 读取 configs/inoue2019.yaml
        |
        v
reproduce.py 循环不同 I/Ith
        |
        v
solver.py 同时推进四个光波和载流子
        |
        +--> materials.py：g(N)、模增益、折射率变化
        +--> coupling.py：C 矩阵
        +--> injection.py：J(x,y)
        +--> observables.py：U、P_rad、FFT 光谱
        |
        v
io.py 保存 npz，plotting.py 生成 png/csv
```

这就是“平台”的基本构建思想：把物理问题拆成独立模块。换材料主要改材料模块，换电极改注入模块，
换晶格改几何和耦合模块，绘图与保存不需要一起重写。

## 7. 论文公式与代码逐项对应

### 7.1 论文式 (3)：折射率的空间傅里叶展开

论文把周期折射率写成平均项与傅里叶谐波之和：

```text
n^2(r) = n_bar^2(z) + sum xi_mn(z) exp[-i(m beta0 x + n beta0 y)]
```

代码位置：

```text
src/pcselsim/geometry.py
SquareLatticeCell.fourier_epsilon()
```

这里使用椭圆孔的解析傅里叶变换，可以描述一个晶胞内多个椭圆孔。`inoue_double_lattice_cell()`
给出论文图 2(a) 的理想化双孔示意，但它不是作者未公开的三维 SEM 重建孔形。

### 7.2 论文式 (5) 和 (7)：四个基本波

代码把电场的慢变包络保存成一个复数数组：

```text
field[0] = Rx，沿 +x 传播
field[1] = Sx，沿 -x 传播
field[2] = Ry，沿 +y 传播
field[3] = Sy，沿 -y 传播
```

每一个 `field[k]` 都是一个二维网格，而不是一个数。因此它同时包含基横模和所有能由当前网格
表示的高阶横模。

### 7.3 论文式 (8)：时域四波耦合方程

可以把式 (8) 拆成五件事：

```text
场随时间变化
= 增益和内部损耗
+ 布拉格失谐引起的相位变化
+ 四个波之间的 C 矩阵耦合
+ 沿各自方向传播
+ 自发辐射噪声
```

主要代码位置：

```text
src/pcselsim/solver.py
TimeDomainSolver._field_local_half_step()
TimeDomainSolver._advect()
TimeDomainSolver._add_spontaneous_noise()
```

`_field_local_half_step()` 处理增益、损耗、失谐、折射率时间项和 `C` 矩阵；`_advect()` 使用
二阶 Lax-Wendroff 方法让四个包络沿不同方向传播。

### 7.4 附录式 (A10)：模增益和折射率时间项

材料增益不能直接全部作用于波导模，只有落在有源层中的光场部分有效。代码使用：

```text
g_modal = Gamma_active * n_active/n_eff * g_active
```

对应：

```text
src/pcselsim/materials.py
modal_gain_m()
```

载流子引起的有效折射率变化和折射率时间项分别使用：

```text
Delta n_eff = Gamma_active * n_active/n_eff * (dn/dN) * Delta N
gamma = 2 * Gamma_active/n_active * (dn/dN) * (dN/dt)
```

二者的具体材料公式位于 `materials.py`，`_field_local_half_step()` 将它们放入
Inoue 时域场方程。第一个改变相位/局部失谐，第二个是附录 A10 的实数振幅率修正。

### 7.5 论文附录式 (A11)-(A17)：耦合矩阵 C

论文写成：

```text
C = C_1D + C_rad + C_2D
```

- `C_1D`：反向 Bragg 耦合。
- `C_rad`：经垂直辐射波产生的耦合和辐射损耗。
- `C_2D`：经高阶面内波产生的二维耦合。

代码位置：

```text
src/pcselsim/coupling.py
calibrated_coupling_matrix()
```

当前矩阵在 A/B/C/D 带边模基底中设置频率和辐射损耗，再变换到四波基底。必须注意：论文没有
公开真实三维孔形得到的完整数值矩阵，所以现在是可替换的校准矩阵，而不是作者原始矩阵。这是
当前不能严格复现 4.2 倍阈值下 26 pm 双峰的主要原因。

### 7.6 论文式 (9)：辐射输出功率

论文使用 `Im(Phi^H C Phi)` 计算从表面辐射出去的功率。代码位置：

```text
src/pcselsim/observables.py
radiated_power_W()
```

它对每个网格的功率密度乘 `dx*dy` 并求和，得到瓦特。

### 7.7 论文式 (10)：载流子速率方程

代码实现：

```text
dN/dt = J/(e d_active) - N/tau_c - vg g_active U + D laplacian(N)
```

四项依次表示：

1. 电流注入载流子；
2. 自发与非辐射复合造成的载流子损失；
3. 受激辐射把载流子变成光子；
4. 载流子从高浓度区域向低浓度区域扩散。

代码位置：

```text
src/pcselsim/solver.py
TimeDomainSolver._carrier_rhs()
TimeDomainSolver._laplacian_neumann()
```

载流子边界使用零法向扩散通量，即不让载流子通过数值区域边界扩散出去。

### 7.8 式 (10) 下方：光子密度 U

四个场的强度相加，并乘以能量归一化和限制因子得到有源区光子密度：

```text
src/pcselsim/observables.py
photon_density_m3()
```

空间图显示的是 `U*d_active`，单位从 `m^-2` 转成 `cm^-2`，对应论文图 3(c) 的面光子密度。

### 7.9 论文式 (11)：饱和增益

代码位置：

```text
src/pcselsim/materials.py
active_gain_m()
```

载流子低于 `Ntr` 时增益为负，表示吸收；等于 `Ntr` 时增益为零；高浓度时逐渐趋近 `gmax`，
不会无限增长。

### 7.10 有限尺寸边界条件

Liang 的有限尺寸 CWT 使用：

```text
Rx(左边界)=0
Sx(右边界)=0
Ry(下边界)=0
Sy(上边界)=0
```

物理意义是区域外没有相干光波射入，但光可以从出射边界离开。代码在 `_advect()` 中实现。

### 7.11 光谱

论文说明对时域电场做傅里叶变换可以直接得到光谱。代码记录 A 带边模的复相干信号，最后在：

```text
src/pcselsim/observables.py
wavelength_spectrum()
```

中加 Hann 窗、做 FFT，并把频率转换成波长。

## 8. 一个时间步的算法到底怎样执行

`TimeDomainSolver.run()` 首先建立网格、电流分布、载流子数组和四个复电场数组。载流子从零开始，
电场用极小随机噪声启动。随后重复以下循环：

```text
1. 半步局部光场演化：增益、损耗、失谐、折射率变化、C 矩阵
2. 完整传播步：Rx/Sx/Ry/Sy 沿各自方向移动
3. 加入自发辐射随机源
4. 根据当前光子密度计算载流子方程
5. 用 Heun 预测-校正法更新载流子；`carrier_substeps>1` 时在一个光学步内细分载流子步
6. 再做半步局部光场演化
7. 到达采样时刻时记录 N、功率和复电场信号
```

把局部演化分成前后两个半步称为 **Strang splitting**。可以把它想象成：先让光在当前位置经历
半段增益和耦合，再传播一段距离，再经历剩余半段局部作用。这样比把所有过程粗暴地一次更新更准。

光传播使用 **Lax-Wendroff** 二阶差分。`CFL=0.8` 表示一个时间步中光包络传播 0.8 个网格。
配置检查不允许 CFL 大于 1，否则信息会跨过一个以上网格而导致显式传播不稳定。

载流子使用 **Heun 法**：先用当前斜率预测一次，再在预测位置计算新斜率，最后取两次斜率平均。

## 9. 配置文件每一组参数是什么

打开：

```text
configs/inoue2019.yaml
```

### `optical`

- `lattice_constant_nm`：晶格常数。
- `wavelength_nm`：参考中心波长。
- `group_index`：决定包络群速度 `vg=c/ng`。
- `effective_index`：平面传播模有效折射率。
- `active_index`：有源层折射率。
- `internal_loss_cm`：内部材料损耗。
- `confinement_factor`：光场落在有源层中的比例。
- `dn_dN_cm3`：载流子引起的折射率变化。
- `modal_detuning_cm`：A/B/C/D 带边模频率分裂的校准量。
- `modal_radiation_loss_cm`：A/B/C/D 的辐射振幅损耗。
- `carrier_index_change`：是否启用载流子折射率变化。
- `temporal_index_term`：是否启用折射率时间导数项。

初学阶段不要随意改最后两组带边矩阵参数。

### `carrier`

- `maximum_gain_cm`：饱和材料增益。
- `zero_carrier_gain_cm`：无载流子时的负增益，即吸收。
- `transparency_density_cm3`：透明载流子浓度。
- `lifetime_ns`：载流子寿命。
- `diffusion_cm2_s`：扩散系数。
- `active_thickness_nm`：量子阱总厚度。
- `spontaneous_emission_factor`：自发辐射耦合系数。

### `device`

- `domain_um`：整个二维计算窗口。
- `electrode_um`：电极特征尺寸；`electrode_shape="square"` 时是边长，`"circle"` 时是直径。
- `electrode_shape`：`"square"` 或 `"circle"`。Wang 2024 预设使用直径 200 um 的圆形窗口。
- `current_spread_um`：电极外电流横向扩展宽度。
- `threshold_current_A`：论文实验阈值电流，用于定义 `I/Ith`。

### `numerics`

- `points`：每个方向的网格点数，必须为不小于 9 的奇数。
- `end_time_ns`：仿真总时间。
- `cfl`：传播稳定性参数，必须在 `(0,1]`。
- `sample_interval_ps`：保存时间序列的间隔。
- `seed`：随机种子；相同参数与种子会得到可复核的随机结果。
- `noise`：是否加入自发辐射噪声。

网格点增加时，二维数组变大且时间步变小，所以计算成本增长很快。不要直接把 `points` 从 41
改成 401。正确做法是依次尝试 31、41、51、61，观察结果是否收敛。

### `reproduction`

- `current_ratios`：要计算的 `I/Ith` 列表。
- `spectrum_window_ns`：取末尾多长时间做光谱 FFT。

## 10. 第一次参数实验：只改一个量

不要破坏论文基准文件。先复制：

```powershell
Copy-Item configs\inoue2019.yaml configs\my_first_test.yaml
```

打开 `configs/my_first_test.yaml`，把：

```yaml
current_ratios: [1.05, 1.4, 2.8, 4.2]
```

改为：

```yaml
current_ratios: [2.0]
```

然后在 Terminal 运行：

```powershell
python scripts\reproduce_inoue2019.py --config configs\my_first_test.yaml --output results\my_first_test
```

永远给新实验使用新的 `--output` 目录，否则会覆盖同名结果。

## 11. 复现论文中“关闭载流子折射率变化”的对照

复制配置：

```powershell
Copy-Item configs\inoue2019.yaml configs\inoue2019_no_index.yaml
```

在新文件中改成：

```yaml
carrier_index_change: false
temporal_index_term: false
```

运行：

```powershell
python scripts\reproduce_inoue2019.py --config configs\inoue2019_no_index.yaml --output results\inoue2019_no_index
```

然后比较两个目录中的 `figure4_spectra.png`。当前校准矩阵还不能严格复现论文图 4 的 26 pm 双峰，
因此这个实验的重点是学习控制变量和比较流程，而不是把当前单峰误认为论文已完全复现。

## 12. 换半导体材料或基底时怎么做

先不要改求解器循环。推荐顺序：

1. 新建一个 YAML，例如 `configs/my_substrate.yaml`。
2. 收集每一层的厚度和折射率。
3. 用 `vertical.py` 的 `LayerStack` 求 TE0 模、`neff` 和有源层限制因子。
4. 修改 YAML 中的 `effective_index`、`active_index`、`confinement_factor`。
5. 修改材料的 `Ntr`、`gmax`、`g0`、寿命、扩散系数和 `dn/dN`。
6. 重新计算或导入与新晶格对应的复耦合矩阵 `C`。
7. 从低分辨率短时间计算开始，逐步做网格和时间窗收敛。

纵向模式的最小示例：

```python
from pcselsim.vertical import Layer, LayerStack

stack = LayerStack(
    layers=(
        Layer("lower_clad", 1500.0, 3.20),
        Layer("active", 100.0, 3.50),
        Layer("upper_clad", 1500.0, 3.20),
    ),
    top_index=3.20,
    bottom_index=3.20,
)

mode = stack.solve_te0(wavelength_nm=950.0, dz_nm=5.0)
print(mode.effective_index)
print(mode.confinement)
```

需要分别减小 `dz_nm`、增加 `padding_um`，直到 `neff` 变化足够小，才算完成纵向离散收敛。

## 13. 换晶格类型时需要理解的边界

当前时域求解器是方形晶格 Gamma 点四波模型。你可以改变方形晶格内的孔半径、椭圆长短轴、旋转角、
双孔位移和材料参数；这些对应 `geometry.py` 和新的 `C` 矩阵。

三角晶格不能只把 YAML 中的文字从 `square` 改成 `triangular`。三角晶格 Gamma 点通常需要六个基本
传播波，并需要：

1. 把场从 4 分量扩展为 6 分量；
2. 使用 6 个传播方向；
3. 推导或导入 6x6 耦合矩阵；
4. 重新定义开放边界与辐射场；
5. 增加与 Liang 三角晶格理论对应的回归测试。

因此本项目已经把几何、耦合、材料与求解流程分开，便于以后扩展，但当前版本不把三角晶格标成已完成。

## 14. 迁移到全固态光子晶体微腔激光器

四波传播、晶格耦合、开放边界、功率和频谱框架可以继续使用，但半导体载流子方程不能原样照搬。
稀土或过渡金属离子晶体通常需要粒子数能级方程，例如四能级系统：

```text
泵浦吸收 -> 上能级粒子数
上能级自发衰减
受激发射截面 * 光子密度 * 上能级粒子数
基态吸收（如果存在）
```

未来应新增统一的增益介质接口：

```text
modal_gain(state)
index_shift(state)
state_rhs(state, photon_density, pump)
```

然后分别实现：

```text
SemiconductorQWMedium
RareEarthFourLevelMedium
```

这样光场求解器不需要知道状态是“电子-空穴载流子”还是“稀土离子能级粒子数”。只修改 YAML 中的
半导体参数，不足以完成全固态迁移。

## 15. 常见错误及解决方法

### `ModuleNotFoundError: No module named 'pcselsim'`

当前版本的三个 `scripts` 入口会自动把 `src` 加到模块路径，正常情况下直接点击
`run_quick.py` 不应再出现此错误。如果仍出现，先确认你运行的是本项目最新的脚本，并检查文件：

```text
scripts/_project_bootstrap.py
src/pcselsim/__init__.py
```

使用高级入口或交互式 Python 时，仍建议在项目根目录安装：

```powershell
python -m pip install -e ".[dev]"
```

然后检查 PyCharm Interpreter 是否是 `.venv\Scripts\python.exe`。

### `No module named numpy/scipy/matplotlib/yaml`

仍然是解释器或依赖安装问题。在 PyCharm Terminal 中运行：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install -e ".[dev]"
```

第一条应输出项目 `.venv` 内的 Python。

### 找不到 `configs/inoue2019.yaml`

一键入口使用配置文件的绝对路径，正常情况下不会受 Working directory 影响。若高级命令仍报错，
把 Working directory 设置为：

```text
D:\things\Master\Doc\PCSELSim
```

### 程序运行很久但没有新图

图片在一个电流点计算结束后才生成。先用 `run_quick.py`。完整运行时观察 Run 窗口当前正在计算
哪个 `I/Ith`，不要反复点击运行按钮。

### `ValueError: numerics.points must be an odd integer`

把 `points` 改成 31、41、51 等奇数。

### `Non-finite state`

说明出现数值发散。先恢复基准 YAML；确认 `cfl<=1`，不要把增益、损耗或折射率系数放大几个数量级。

### 4.2 倍阈值仍然只有一个谱峰

这是当前已知复现差距，不是你不会运行。论文没有公开从真实三维孔形得到的完整 `C` 矩阵。详细说明
见 `docs/reproduction_status_zh.md`。

## 16. 推荐的学习路线

按以下顺序学习，不要同时改很多模块：

1. 运行 `pytest`、`run_quick.py`、`run_full.py`。
2. 学会阅读 `summary.csv` 和三类 PNG。
3. 复制 YAML，只改变一个电流点。
4. 关闭载流子折射率项，做控制变量比较。
5. 改变 `lifetime_ns` 或 `diffusion_cm2_s`，观察弛豫振荡和空间烧孔。
6. 阅读 `materials.py` 和论文式 (10)-(11)。
7. 阅读 `solver.py` 的 `run()`，按第 8 节逐行对应。
8. 用 `vertical.py` 建立自己的层结构。
9. 再学习晶胞傅里叶系数和耦合矩阵 `C` 的生成。
10. 最后再扩展新的晶格基底或全固态增益介质。

每次实验都保留：配置文件、Git commit、随机种子和独立结果目录。这比只保存一张最终图片更重要，
因为它决定结果能否在半年后重新生成。

## 17. 最基本的 Git 保存动作

查看有哪些文件变化：

```powershell
git status
git diff
```

第一次提交代码和文档：

```powershell
git add pyproject.toml requirements.txt .gitignore README.md configs docs scripts src tests
git commit -m "feat: add PCSEL time-domain simulation platform"
```

日后每完成一个可以通过 `pytest` 的小功能就提交一次。`results/` 已在 `.gitignore` 中，不会把大量
仿真数据塞进 Git。远端同步、分支和冲突处理见 `docs/git_workflow_zh.md`。

