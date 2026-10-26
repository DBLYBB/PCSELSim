# Semiconductor scan coverage and retained evidence — 2026-10-21

This is a read-only aggregation of the saved results. Both lattices use the common Inoue Table-I 190 nm PC semiconductor stack, internal intensity loss 5 cm⁻¹, and a nominal 300×300 μm square cavity. Formal candidates were retuned to an actual lowest-pole wavelength of 950 nm. The ranges below are observed attempted parameter records, including geometries rejected before an optical solve; declared proposal envelopes and ranges restricted to successful optical evaluations are recorded separately in comparison/semiconductor_scan_coverage.json.

| Parameter | Square lattice | Triangular lattice |
|---|---|---|
| Motif | Central main + one diagonal satellite | Central main + two satellites around −30°/+30° |
| Hole families | Circle, ellipse, superellipse, rounded triangle; 16 shape pairs optically evaluated | Circle, ellipse, rounded regular triangle; 27 combinations attempted, 24 optically evaluated |
| Total air-area fraction | 0.025–0.28 | 0.0582664–0.24 |
| Main-hole area fraction of all holes | 0.5–0.849409 | 0.32–0.82 |
| Satellite area balance | One satellite | 0.32003–0.692197 of remaining area in satellite 1 |
| Center distance / a | 0.353553–0.69 | d1 0.290152–0.541277; d2 0.290227–0.529873 |
| Position direction | 26.406–63.8416° | θ1 -47–-17.1373°; θ2 17.23–47° |
| Active ellipse aspect ratio | Up to 2; independently oriented holes | Up to 1.85; each hole independently oriented |
| Orientation | Recorded bounds −90° to +90°; circles ignore orientation | Independent orientations, Sobol 0° to 180° with mirrored satellite angles; subsequent local variation; final fine scan follows main-angle offsets down to −52° |
| Other shape variables | Superellipse exponent 2.3–4.8; fixed triangle Minkowski rounding, independently stretched | Rounded regular triangle corner fraction 0.18–0.43; triangle aspect is inactive |

All dimensionless distances are Cartesian distances divided by lattice period. Main-hole size dominance is enforced for formal candidates. Descriptor values inactive for a given shape are excluded from the separate active-parameter summaries. The triangular generator allocated 18 Sobol coordinates but used coordinates 0–16; coordinate 17 is unused. Shape and symmetry choices also make some coordinates inactive in individual proposals.

| Main parameter stage | Attempted records | Optical evaluations completed | Rejected geometry |
|---|---:|---:|---:|
| square: broad | 320 | 320 | 0 |
| square: local_QZ | 288 | 288 | 0 |
| square: quarter_period_anchors | 72 | 72 | 0 |
| square: isolation_local_QZ | 192 | 192 | 0 |
| square: actual950_local_QZ | 192 | 192 | 0 |
| triangular: sobol1024 | 1069 | 82 | 987 |
| triangular: circle_grid567 | 567 | 202 | 365 |
| triangular: local256 | 256 | 147 | 109 |
| triangular: local_targeted192 | 192 | 91 | 101 |
| triangular: highD_adaptive128 | 128 | 46 | 82 |

Counts in this table begin after any separately recorded proposal prefilter. Square main stages total **1,064 completed evaluations**, comprising **1,060 distinct saved fractional parameter vectors and four exact repeats** after excluding tag and period. Its 1,219 compact-result records include another 155 accuracy/spectral reruns; these add no new saved fractional parameter vectors. Known square prefilter rejections total **717** (589 broad, 48 local, 80 isolation local); the actual950 generator does not save its rejection counter. Quarter-period anchors deliberately include structures failing the assumed process rules.

Triangular main stages total **2,212 attempts: 568 optical solves and 1,644 geometry rejections**. They contain 2,128 distinct saved fractional parameter vectors and 84 repeated records. Removing shape-inactive descriptors and exact orientation symmetries gives 2,083 normalized fractional motif keys; this reporting aid is not a proof of complete physical equivalence. Separately, there are 99 N17 rescreens, ten N23 reruns, 49 timing-screen records, and 40 N65 actual-reference local/angle attempts. The 40 N65 attempts contain 32 distinct parameter vectors, 36 completed follows and four failed follows. **109 copied parent records are excluded from new-evaluation counts.**

The initial three-branch N65 follow failed at −28°/−32° because the third branch overlap was below 0.75; a separately logged two-branch follow succeeded there. The −40° follow failed twice. The final selected −32° geometry then received the complete-coarse-guided and six-root closure checks. It is the best tested valid angle, without a proof of a strict angular minimum.

Final triangular spectrum evidence is 4,926/4,926 retained coarse N13 poles, 80 complex targets yielding 156 distinct N39 finite poles, 12 successful N91 mesh continuations, six complex stripe windows yielding 24 stripe poles, a union of 36 distinct poles, and six independently frequency-closed final roots. These are different workloads and must not be summed as independent geometric samples. The N91 spectrum is not exhaustive.

All three square portable candidates pass parameter completeness: every Design field, actual period, two exact nm boundaries, area centroids, main-area dominance, 300 μm square aperture, common stack, and final contour/accuracy settings are present. Final square contours use area centroids and 96 contour samples; early screens used historical lower-resolution/vertex-mean contours. Circle aspect/orientation/exponent and rounded-triangle exponent are inactive in the square finalists.

| Triangular cleanup manifest | Bytes released | GiB released |
|---|---:|---:|
| retention_manifest.json | 1,820,731,221 | 1.695688 |
| retention_manifest_round2.json | 1,274,070,290 | 1.186570 |
| retention_manifest_round3.json | 158,961,660 | 0.148045 |
| retention_manifest_round4.json | 106,522,870 | 0.099207 |

Total triangular-owned cleanup released **3,360,286,041 bytes = 3.129510 GiB**. Retained NPZ inventory: **101 files / 717,838,815 bytes = 684.584 MiB**. Exact paths, sizes, array-entry counts, and all four manifest paths are in semiconductor_scan_coverage.json. No other agents’ results or literature reproduction were deleted by this summary task.

The current main six closed roots and Pareto two closed roots remain unchanged. Retention includes all 4,926 coarse scalar roots + 64 fields, all 156 N39 scalar roots + 24 fields, mesh 12 scalar roots + three fields, stripe 24 scalar roots + three fields, and union 36 scalar roots + 24 fields. Necessary dark-stripe and mesh counterexamples, full parameter catalogues, scalar spectra, source history, process controls and read-back evidence remain.

All this work stays in scientific simulation documentation; no PPT workspace, presentation assets or new optical computation were created.
