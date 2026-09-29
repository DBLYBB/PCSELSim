"""Lossless numerical-result serialization / 原始数值结果保存。

NPZ files are the evidence used to regenerate plots; figures alone should not
be used for quantitative comparison. / 定量比较应读取 NPZ 原始数组，不应只从图片
估数。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .solver import SimulationResult


def save_result(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        current_ratio=result.current_ratio,
        x_um=result.x_um,
        y_um=result.y_um,
        time_ns=result.time_ns,
        center_carrier_cm3=result.center_carrier_cm3,
        power_W=result.power_W,
        complex_signal=result.complex_signal,
        final_carrier_cm3=result.final_carrier_cm3,
        final_photon_areal_cm2=result.final_photon_areal_cm2,
        final_field=result.final_field,
        dt_s=result.dt_s,
        metadata_json=json.dumps(result.metadata),
    )
    return path

