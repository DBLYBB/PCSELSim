"""Split-step time-domain 3D coupled-wave solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .config import SimulationConfig
from .constants import c, e, pi
from .coupling import calibrated_coupling_matrix, validate_passive_coupling
from .injection import electrode_area_m2, electrode_profile
from .materials import active_gain_m, effective_index_shift, modal_gain_m
from .observables import photon_density_m3, radiated_power_W


@dataclass
class SimulationResult:
    current_ratio: float
    x_um: np.ndarray
    y_um: np.ndarray
    time_ns: np.ndarray
    center_carrier_cm3: np.ndarray
    power_W: np.ndarray
    complex_signal: np.ndarray
    final_carrier_cm3: np.ndarray
    final_photon_areal_cm2: np.ndarray
    final_field: np.ndarray
    dt_s: float
    metadata: dict[str, float | int | str]


class TimeDomainSolver:
    """Solve Eqs. (8), (10), and (11) on a square staggered-like grid.

    The four directional envelope waves use second-order Lax-Wendroff propagation.
    Local gain, carrier detuning, and the 4x4 C matrix are applied using a
    Strang split. Incoming-wave boundary conditions are those of Liang Eq. (4.22).
    """

    def __init__(
        self,
        config: SimulationConfig,
        coupling_m: np.ndarray | None = None,
        signal_projection: np.ndarray | None = None,
    ):
        self.config = config
        n = config.numerics.points
        half = 0.5 * config.device.domain_um * 1e-6
        self.x = np.linspace(-half, half, n)
        self.y = self.x.copy()
        self.dx = float(self.x[1] - self.x[0])
        self.vg = c / config.optical.group_index
        self.cfl = config.numerics.cfl
        self.dt = self.cfl * self.dx / self.vg
        self.coupling = (
            calibrated_coupling_matrix(config.optical)
            if coupling_m is None
            else np.asarray(coupling_m, dtype=np.complex128)
        )
        validate_passive_coupling(self.coupling)
        if signal_projection is None:
            signal_projection = np.asarray((0.5, -0.5, -0.5, 0.5), dtype=np.complex128)
        self.signal_projection = np.array(
            signal_projection, dtype=np.complex128, copy=True
        )
        if self.signal_projection.shape != (4,):
            raise ValueError("signal_projection must contain four complex wave coefficients")
        norm = float(np.linalg.norm(self.signal_projection))
        if norm <= np.finfo(float).eps:
            raise ValueError("signal_projection cannot be the zero vector")
        self.signal_projection /= norm
        self.coupling_half_step = expm(0.5j * self.vg * self.dt * self.coupling)
        self.current_shape = electrode_profile(self.x, self.y, config.device)
        self.center = n // 2
        self.reference_carrier = self._reference_carrier_density()

    def _reference_carrier_density(self) -> float:
        cfg = self.config
        area = electrode_area_m2(cfg.device)
        j = cfg.device.threshold_current_A / area
        return (
            j
            * (cfg.carrier.lifetime_ns * 1e-9)
            / (e * cfg.carrier.active_thickness_nm * 1e-9)
        )

    def _advect(self, field: np.ndarray) -> np.ndarray:
        """Second-order Lax-Wendroff update with zero incoming envelopes."""
        q = self.cfl
        moved = np.zeros_like(field)
        # Rx (+x). x=0 is incoming; x=L is outgoing.
        u = field[0]
        moved[0, :, 1:-1] = (
            u[:, 1:-1]
            - 0.5 * q * (u[:, 2:] - u[:, :-2])
            + 0.5 * q**2 * (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2])
        )
        moved[0, :, -1] = u[:, -1] - q * (u[:, -1] - u[:, -2])
        # Sx (-x). x=L is incoming; x=0 is outgoing.
        u = field[1]
        moved[1, :, 1:-1] = (
            u[:, 1:-1]
            + 0.5 * q * (u[:, 2:] - u[:, :-2])
            + 0.5 * q**2 * (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2])
        )
        moved[1, :, 0] = u[:, 0] + q * (u[:, 1] - u[:, 0])
        # Ry (+y). y=0 is incoming; y=L is outgoing.
        u = field[2]
        moved[2, 1:-1, :] = (
            u[1:-1, :]
            - 0.5 * q * (u[2:, :] - u[:-2, :])
            + 0.5 * q**2 * (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :])
        )
        moved[2, -1, :] = u[-1, :] - q * (u[-1, :] - u[-2, :])
        # Sy (-y). y=L is incoming; y=0 is outgoing.
        u = field[3]
        moved[3, 1:-1, :] = (
            u[1:-1, :]
            + 0.5 * q * (u[2:, :] - u[:-2, :])
            + 0.5 * q**2 * (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :])
        )
        moved[3, 0, :] = u[0, :] + q * (u[1, :] - u[0, :])
        return moved

    def _laplacian_neumann(self, values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="edge")
        return (
            padded[1:-1, 2:]
            + padded[1:-1, :-2]
            + padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            - 4.0 * values
        ) / self.dx**2

    def _field_local_half_step(
        self, field: np.ndarray, carrier: np.ndarray, dcarrier_dt: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        gain = modal_gain_m(carrier, cfg.carrier, cfg.optical)
        scalar_rate = 0.5 * self.vg * (gain - cfg.optical.internal_loss_cm * 100.0)
        if cfg.optical.carrier_index_change:
            dn_eff = effective_index_shift(carrier, self.reference_carrier, cfg.optical)
            beta0 = 2.0 * pi / (cfg.optical.lattice_constant_nm * 1e-9)
            detuning = beta0 * dn_eff / cfg.optical.effective_index
            scalar_rate = scalar_rate - 1j * self.vg * detuning
        if cfg.optical.temporal_index_term:
            dn_dN_m3 = cfg.optical.dn_dN_cm3 * 1e-6
            gamma = (
                2.0
                / cfg.optical.effective_index
                * cfg.optical.confinement_factor
                * dn_dN_m3
                * dcarrier_dt
            )
            scalar_rate = scalar_rate - gamma
        field = np.einsum("ab,bij->aij", self.coupling_half_step, field, optimize=True)
        return field * np.exp(0.5 * self.dt * scalar_rate)[None, :, :]

    def _carrier_rhs(
        self, carrier: np.ndarray, field: np.ndarray, current_density: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        thickness = cfg.carrier.active_thickness_nm * 1e-9
        tau = cfg.carrier.lifetime_ns * 1e-9
        diffusion = cfg.carrier.diffusion_cm2_s * 1e-4
        photons = photon_density_m3(field, cfg.carrier, cfg.optical)
        stimulated = self.vg * active_gain_m(carrier, cfg.carrier) * photons
        # Absorptive gain must not create carriers in this phenomenological equation.
        stimulated = np.maximum(stimulated, 0.0)
        return (
            current_density / (e * thickness)
            - carrier / tau
            - stimulated
            + diffusion * self._laplacian_neumann(carrier)
        )

    def _add_spontaneous_noise(
        self, field: np.ndarray, carrier: np.ndarray, rng: np.random.Generator
    ) -> None:
        cfg = self.config
        if not cfg.numerics.noise or cfg.carrier.spontaneous_emission_factor <= 0.0:
            return
        tau = cfg.carrier.lifetime_ns * 1e-9
        spontaneous_photons = (
            cfg.carrier.spontaneous_emission_factor * carrier / tau * self.dt
        )
        # Convert the injected photon density back to four complex field amplitudes.
        unit_field = np.ones_like(field)
        conversion = photon_density_m3(unit_field, cfg.carrier, cfg.optical)
        variance = np.maximum(spontaneous_photons / conversion, 0.0)
        noise = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
        field += noise * np.sqrt(variance / 8.0)[None, :, :]

    def run(self, current_ratio: float) -> SimulationResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.numerics.seed + int(round(current_ratio * 1000)))
        n = cfg.numerics.points
        field = np.zeros((4, n, n), dtype=np.complex128)
        # A tiny deterministic seed ensures reproducibility even when noise is disabled.
        field += (rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)) * 1e-9
        carrier = np.zeros((n, n), dtype=float)
        current_density = (
            current_ratio * cfg.device.threshold_current_A * self.current_shape
        )
        n_steps = int(np.ceil(cfg.numerics.end_time_ns * 1e-9 / self.dt))
        sample_every = max(1, int(round(cfg.numerics.sample_interval_ps * 1e-12 / self.dt)))
        sample_count = n_steps // sample_every + 1
        time = np.empty(sample_count)
        center_carrier = np.empty(sample_count)
        power = np.empty(sample_count)
        signal = np.empty(sample_count, dtype=np.complex128)
        sample_index = 0
        dcarrier_dt = self._carrier_rhs(carrier, field, current_density)

        for step in range(n_steps + 1):
            if step % sample_every == 0:
                time[sample_index] = step * self.dt
                center_carrier[sample_index] = carrier[self.center, self.center]
                power[sample_index] = radiated_power_W(
                    field, self.coupling, self.dx, cfg.optical
                )
                # Coherent projection onto the selected geometry-derived band-edge mode.
                projected = np.einsum(
                    "a,aij->ij", self.signal_projection.conj(), field, optimize=True
                )
                signal[sample_index] = np.sum(projected)*self.dx**2
                sample_index += 1
            if step == n_steps:
                break

            field = self._field_local_half_step(field, carrier, dcarrier_dt)
            field = self._advect(field)
            self._add_spontaneous_noise(field, carrier, rng)

            # Heun carrier step.  The optical CFL step is much smaller than the
            # diffusion and recombination scales, so no separate implicit solve is needed.
            rhs0 = self._carrier_rhs(carrier, field, current_density)
            predicted = np.maximum(carrier + self.dt * rhs0, 0.0)
            rhs1 = self._carrier_rhs(predicted, field, current_density)
            dcarrier_dt = 0.5 * (rhs0 + rhs1)
            carrier = np.maximum(carrier + self.dt * dcarrier_dt, 0.0)
            field = self._field_local_half_step(field, carrier, dcarrier_dt)

            if not np.all(np.isfinite(carrier)) or not np.all(np.isfinite(field)):
                raise FloatingPointError(f"Non-finite state at step {step}")

        photons = photon_density_m3(field, cfg.carrier, cfg.optical)
        photon_areal_cm2 = photons * cfg.carrier.active_thickness_nm * 1e-9 * 1e-4
        return SimulationResult(
            current_ratio=current_ratio,
            x_um=self.x * 1e6,
            y_um=self.y * 1e6,
            time_ns=time[:sample_index] * 1e9,
            center_carrier_cm3=center_carrier[:sample_index] * 1e-6,
            power_W=power[:sample_index],
            complex_signal=signal[:sample_index],
            final_carrier_cm3=carrier * 1e-6,
            final_photon_areal_cm2=photon_areal_cm2,
            final_field=field,
            dt_s=self.dt * sample_every,
            metadata={
                "optical_dt_fs": self.dt * 1e15,
                "steps": n_steps,
                "grid_points": n,
                "dx_um": self.dx * 1e6,
                "cfl": self.cfl,
                "reference_carrier_cm3": self.reference_carrier * 1e-6,
            },
        )

