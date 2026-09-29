# PCSELSim 物理审计、论文对照与代码地图 / Physics audit and code map

## 1. 先看结论 / Executive verdict

当前工程的半导体主链条在方程结构、单位和数值流程上是可信的：晶胞几何产生
介电常数 Fourier 系数，纵向 TE0 模进入 Liang 三维耦合波矩阵，有限区域本征问题
给出冷腔模式，同一耦合矩阵再进入 Inoue 载流子—光场时域方程。它已经适合做
**带有收敛检查和不确定度说明的自定义半导体研究**，但还不能把所有已有图片都称为
论文的独立定量复现。

The semiconductor production path is physically traceable and suitable for
custom-device studies with convergence and uncertainty checks.  Not every
existing result folder, however, has the same evidence level.

| 结果目录 | 证据等级 | 审计判断 |
|---|---|---|
| `custom_semiconductor_inoue2019` | 几何驱动、部分定量 | 模式排序、阈值量级、低电流动态通过；高电流谱和空间分布未完全通过 |
| `custom_semiconductor_wang2024_triple_lattice` | 几何驱动冷腔对照 | 晶胞、填充率、Bragg 波长和限制因子通过；损耗、模式选择和阈值尚不能定量预测 |
| `liang_chapter4_step_by_step` | 教学重构/标定/代理混合 | 适合逐式理解；多张图不是独立预测，详见运行清单中的 provenance |
| `inoue2019` | 标定后端回归 | 可检查时域机制，但 C 矩阵本征值是输入，不是由几何预测 |
| `quick`、`quick_lw` | 冒烟测试 | 只验证安装、路径和数据流，不用于物理结论 |
| `custom_ybyag_pcsel` | 概念验证 | 准三能级方程方向合理，光学与热/泵浦模型不足，不能作为器件设计值 |
| 其余 `custom_semiconductor_pcsel*` | 历史/验证输出 | 保留用于回溯；正式引用前应确认生成它们的脚本版本和参数 |

“实际加工会有误差”并不意味着任何偏差都可以接受。合理做法是把偏差分为：几何/材料
不确定度、数值离散误差、缺失物理和测量误差，并用参数扫描分别量化。只有在合理制造
公差范围内的扫描能包住实验值时，才能把差异归因于加工。

## 2. Inoue 2019 自定义半导体结果

当前主要结果：Bragg 波长 `949.179 nm`；最低阈值为 A 模；A/B/C/D 的
`alpha*L` 分别约 `0.229/0.424/0.855/0.643`；冷腔反推阈值约 `0.603 A`，
论文采用的实验阈值为 `0.7 A`。这说明几何—纵向模—耦合—有限腔链条在量级和
模式排序上相互一致。

10 ns 旧结果在 `I/Ith=1.05, 1.4, 2.8, 4.2` 时的末 1 ns 平均功率约为
`0.070, 0.222, 0.943, 1.702 W`。本轮修正后已独立重跑到
`results/custom_semiconductor_inoue2019_audit_corrected`，对应值为
`0.0709, 0.2225, 0.9440, 1.7052 W`，相对变化均小于 `0.6%`。这说明修正
提高了公式一致性但没有人为制造新的结果趋势。低电流弛豫振荡处于论文给出的 GHz 量级，
中心载流子钳位和高电流空间烧孔趋势也合理。尚未通过的项目是：

- `2.8 Ith` 光子分布仍明显偏心，均匀化不如论文图示；
- 高电流提取的弛豫频率约 `4.5–6 GHz`，高于论文典型 `1–3 GHz`；
- `4.2 Ith` 旧结果仍为单峰，而论文报告约 `26 pm` 双峰；
- 理想双椭圆尺寸来自明确标注的几何假设，不是作者未公开的三维 SEM 体数据。

因此该目录应表述为“物理链条通过、定性与部分定量复现”，不能表述为图 2–4 的逐点
完全复现。制造误差可能移动谱峰和损耗，但双峰缺失首先提示模式竞争、真实 C 矩阵、
随机源或空间非均匀性仍不完整，不能只用加工误差解释。

修正后四个电流点在末 2 ns 的频谱仍各只有一个显著峰；`4.2 Ith` 最终光子分布的
强度质心约为 `(24, -43) um`，仍显著偏离中心。因此“高电流双模/空间分布未通过”
这一审计结论不变。

## 3. Wang 2024 三晶格结果

通过项：三个直径 `90 nm` 圆孔给出 `8.495%` 填充率（论文约 `8.5%`）；Bragg
波长 `1550.858 nm`（目标约 `1551 nm`）；PC 限制因子 `50.89%`（论文约
`52%`）；MQW 标量解为 `5.38%`（论文约 `6%`）；三晶格 `A20` 结构因子为 3。

未通过或不可比项：计算的 180° 总耦合约 `297 cm^-1`，比文中约 `253 cm^-1`
高约 `17.5%`；最低模包络振幅损耗约 `4.62 cm^-1`，即使按功率损耗取两倍并加
内部损耗也只有约 `14.25 cm^-1`，低于文中约 `28 cm^-1`。A/B 模几乎简并，
当前 A 标签与理想带边基底的重叠只有约 `0.48`，所以不能只凭排序断言实验选中了 A。
实验 `3° x 17°` 远场受多模、非均匀近场和解理面反馈影响，不应强行拟合成理想开放边界
单模远场。占位材料模型给出的 `0.295 A` 阈值不是预测；在获得补充外延表和可靠
InAlGaAs 增益/复合参数前，第 08 步应保持关闭。

## 4. Liang 第四章逐图脚本

该脚本的价值是把 Bloch 展开、有限边界和各图物理概念串起来，而不是宣称每个面板都
来自统一的几何预测。运行后查看 `run_manifest.json` 的 `figure_provenance`：

- 4.2 含有限式 (4.21) 求解，但阈值面板使用论文表 4.2 检查点；
- 4.3、4.9 是解析孔径/高斯或环形远场代理；
- 4.5、4.6、4.13、4.14 含对论文检查点的标定或拟合；
- 4.7、4.8、4.12 是明确标注的理论/合成对应图，不是实验原始数据；
- 4.10 是可复现示意图，不是 SEM 重建。

用于自己器件预测时，应转到 `run_custom_semiconductor_pcsel.py`，而不是修改这些教学
曲线的常数使其“看起来像”新论文。

## 5. Yb:YAG 结果

准三能级净增益 `sigma_e*N2 - sigma_a*N1`、反转率阈值和共享粒子数库的速率方程在
概念上合理。当前 `a=565 nm`、圆孔半径 `125 nm`、200 µm 器件以及 A/B/C/D
失谐和损耗均为示例；后者没有由真实 Yb:YAG 结构求得。模型还缺少泵浦沿厚度吸收、
二维/三维反转、热透镜、应力双折射、上转换和可靠的输出耦合。因此现有泵浦曲线只能
用于检查方程和单位，不可用于决定实际泵浦功率或加工尺寸。

## 6. 本轮修正 / Corrections made in this audit

1. `effective_index_shift` 采用与纵向投影一致的
   `Gamma*n_active/n_eff*(dn/dN)*DeltaN`。
2. Inoue 附录 A10 的时间折射率率项采用
   `2*Gamma/n_active*(dn/dN)*(dN/dt)`。
3. `carrier_substeps` 现在真正执行 Heun 子步，并将实际值写入结果 metadata。
4. 自发辐射实现被明确标注为可重复、网格相关的启动噪声，不用于绝对线宽预测。

这些修正不改变已有的晶格图、纵向模、冷腔阈值和模式排序，但会小幅改变时域相位与
高电流频谱。旧 NPZ/PNG 不会被偷偷覆盖；正式比较时应在新输出目录重跑第 08 步。

## 7. 论文公式到代码 / Equation-to-code map

| 理论环节 | 代码位置 | 算法与注意事项 |
|---|---|---|
| 晶胞 `epsilon_mn` | `geometry.py` | 椭圆解析 Bessel、多边形解析边界积分；孔不可重叠 |
| 纵向 TE0 | `vertical.py` | 标量有限差分本征问题；需做 `dz` 与 padding 收敛 |
| Liang 式 3.14、3.20、3.27–3.35 | `three_d_cwt.py` | 生成 `C1D+Crad+C2D`；基线 Green 函数忽略界面反射 |
| Liang 式 4.21–4.23 | `custom_analysis.py` | 一阶迎风有限区域算子、开放入射边界、网格外推 |
| Liang 式 4.24–4.26 | `custom_analysis.py` | 从重建辐射场做矢量 Fourier 远场 |
| Inoue 式 8 | `solver.py` | Strang + Lax–Wendroff 四波推进 |
| Inoue 式 9 | `observables.py` | 由 `Im(Phi^H C Phi)` 积分面辐射功率 |
| Inoue 式 10 | `solver.py`、`observables.py` | 注入、寿命、受激复合、扩散与光子密度 |
| Inoue 式 11 | `materials.py` | 饱和材料增益，再投影为模增益 |
| Inoue 附录 A10 | `materials.py`、`solver.py` | 载流子折射率失谐及时间归一化率项 |
| Yb:YAG 准三能级 | `solid_state.py` | 空间平均速率方程；目前仅概念验证 |

## 8. 推荐操作顺序 / Recommended workflow

1. 首次安装只跑 `scripts/run_quick.py`；它成功不代表物理验证完成。
2. 自定义半导体先关闭 `08_time_domain`，运行 01–07；确认填充率、Bragg 波长、
   TE0 限制因子、C 分量、网格收敛和模式场形。
3. 至少比较两档 `CWT_TRUNCATION_ORDER` 和三档有限网格；近简并模式同时检查
   `band_overlap` 和场形。
4. 冷腔通过后，在一个**新结果目录**开启时域；先跑一个电流点，再跑全电流扫描。
5. 频谱定量比较前，检查保存采样间隔、总时间窗、旋转参考系和噪声模型。
6. 用 ±孔径、±位置、±层厚、±折射率的制造公差扫描给出误差带，再与实验比较。

## 9. English quick reference

- Geometry-driven semiconductor path: suitable for research use with convergence
  and uncertainty studies.
- Inoue custom result: correct A-mode ordering and threshold scale; high-current
  spatial/spectral agreement remains incomplete.
- Wang preset: geometry and Bragg quantities agree; loss, mode selection and
  threshold are not yet quantitative.
- Liang Chapter-4 script: a teaching reconstruction containing calibrated and
  proxy panels; read `figure_provenance` before citing a plot.
- Yb:YAG: exploratory mean-field model only.
- Corrected time-domain outputs were rerun in the separate
  `custom_semiconductor_inoue2019_audit_corrected` directory after two
  Appendix-A index factors and carrier subcycling were corrected.

## 10. 本轮可复核验证 / Verification record

- `pytest`：28 项全部通过；
- `compileall`：`src`、`scripts`、`tests` 全部通过；
- 全结果目录完整性扫描：105 个 PNG、18 个 JSON、15 个 CSV、20 个 NPZ 均可读取；
- 所有 NPZ 数值数组均未发现 NaN/Inf；
- 修正后 Inoue 四电流点完成 10 ns 独立重跑，旧/新稳态功率差小于 0.6%；
- 修正后频谱与 4.2 倍阈值空间图已再次目视检查，高电流未通过项仍然存在。

`ruff` 没有在当前 `env00` 环境中安装，因此本轮未报告 Ruff 结果；这不影响上述运行和
单元测试结论。
