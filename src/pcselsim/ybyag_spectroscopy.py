"""Explicit bounded Yb:YAG cross-section tables, with no spectral extrapolation.

This interface does not ship invented measured data. The existing constants are
retained only as an explicitly uncalibrated comparison. A supplied measured
table must cover every requested wavelength and temperature; outside its domain
the simulation rejects the request instead of silently holding endpoint values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class YbYAGSpectroscopy:
    temperature_K: np.ndarray
    pump_wavelength_nm: np.ndarray
    laser_wavelength_nm: np.ndarray
    pump_absorption_cm2: np.ndarray
    pump_emission_cm2: np.ndarray
    laser_absorption_cm2: np.ndarray
    laser_emission_cm2: np.ndarray
    provenance: str
    calibration_status: str = "user_supplied_table_not_independently_validated"

    def __post_init__(self):
        for key in ("temperature_K", "pump_wavelength_nm", "laser_wavelength_nm"):
            axis = np.asarray(getattr(self, key), float).copy()
            if (axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all()
                    or np.any(axis <= 0) or np.any(np.diff(axis) <= 0)):
                raise ValueError(f"{key} must be a positive, finite increasing axis")
            axis.setflags(write=False)
            object.__setattr__(self, key, axis)
        for key, spectral_axis in (("pump_absorption_cm2", self.pump_wavelength_nm),
                                   ("pump_emission_cm2", self.pump_wavelength_nm),
                                   ("laser_absorption_cm2", self.laser_wavelength_nm),
                                   ("laser_emission_cm2", self.laser_wavelength_nm)):
            values = np.asarray(getattr(self, key), float).copy()
            if (values.shape != (len(self.temperature_K), len(spectral_axis))
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError(f"{key} must be a nonnegative (temperature,wavelength) table in cm^2")
            values.setflags(write=False)
            object.__setattr__(self, key, values)
        if not self.provenance.strip():
            raise ValueError("Spectroscopy provenance is required")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def at(self, medium, laser_wavelength_nm: float, temperature_K: float):
        def interpolate(name, wavelength, axis):
            if (not self.temperature_K[0] <= temperature_K <= self.temperature_K[-1]
                    or not axis[0] <= wavelength <= axis[-1]):
                raise ValueError(f"{name}: wavelength/temperature outside supplied calibrated table; extrapolation rejected")
            function = RegularGridInterpolator((self.temperature_K, axis), getattr(self, name),
                                               bounds_error=True)
            return float(function([[temperature_K, wavelength]])[0])
        return replace(medium, laser_wavelength_nm=float(laser_wavelength_nm),
                       pump_absorption_cross_section_cm2=interpolate(
                           "pump_absorption_cm2", medium.pump_wavelength_nm, self.pump_wavelength_nm),
                       pump_emission_cross_section_cm2=interpolate(
                           "pump_emission_cm2", medium.pump_wavelength_nm, self.pump_wavelength_nm),
                       laser_absorption_cross_section_cm2=interpolate(
                           "laser_absorption_cm2", laser_wavelength_nm, self.laser_wavelength_nm),
                       laser_emission_cross_section_cm2=interpolate(
                           "laser_emission_cm2", laser_wavelength_nm, self.laser_wavelength_nm))

    def metadata(self):
        return dict(provenance=self.provenance, calibration_status=self.calibration_status,
                    temperature_range_K=self.temperature_K[[0, -1]].tolist(),
                    laser_wavelength_range_nm=self.laser_wavelength_nm[[0, -1]].tolist(),
                    pump_wavelength_range_nm=self.pump_wavelength_nm[[0, -1]].tolist(),
                    extrapolation="rejected", interpolation="bilinear_cross_sections_cm2")
