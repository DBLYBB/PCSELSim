"""Read-only physical rate/transport checks; no frozen optical result is edited."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/optimization')]
from yag_feasibility_study import load_saved_system
from pcselsim.ybyag_multimode import _arrays, multimode_spatial_rhs
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, propagate_geometry_pump
from pcselsim.solid_state import YbYAGMediumConfig,ybyag_energy_budget


def main():
    out=ROOT/'results/current/physics_audit_20261025/yag'
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261025)
    checks=[]
    for lattice in ('square','triangular'):
        system,summary,rows=load_saved_system(lattice)
        for architecture,passes in [('edge_dual',1),('top_single',1),('top_single',20)]:
            pump=PumpGeometryConfig(architecture,13.,300.,31,top_passes=passes,
                return_efficiency=.98 if passes==20 else 1.)
            volume,weights,k,vg,loss=_arrays(system,pump)
            Nt=k[0]['nt'];V=system.modes[0].gain_volume_m3
            for j in range(3):
                f=rng.uniform(.01,.83,size=volume.shape)
                photons=10**rng.uniform(14,18,size=len(system.modes))
                df,ds,gain,up,down,budget=multimode_spatial_rhs(system,pump,f,photons)
                transport=propagate_geometry_pump(system.modes[0],pump,f)
                lhs=Nt*V*np.sum(volume*df)+V*np.sum(ds)
                rhs=transport.absorbed_power_W/k[0]['pump_energy']-Nt*V*np.sum(volume*f)/k[0]['tau']-V*np.sum(vg*loss*photons)
                residual=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1.)
                scale=max(abs(transport.absorbed_power_W),budget['upward_output_W'],1e-20)
                checks.append(dict(lattice=lattice,architecture=architecture,passes=passes,
                    modes=len(system.modes),excitation_photon_count_relative_residual=float(residual),
                    energy_relative_residual=abs(budget['energy_closure_residual_W'])/scale,
                    pump_boundary_relative_residual=abs(budget['pump_boundary_flow_residual_W'])/13.,
                    passed=bool(residual<1e-10 and abs(budget['energy_closure_residual_W'])/scale<1e-10
                        and abs(budget['pump_boundary_flow_residual_W'])<1e-10)))
    local=YbYAGMediumConfig(dopant_density_cm3=1.38e21,gain_thickness_um=2000.)
    counter=ybyag_energy_budget(np.array([0.,0.]),local,np.array([10.]),np.array([1.]))
    T=300.;kb_cm=.69503476
    lower=np.array([0,581,619,786.]);upper=np.array([0,307,600.])
    zl=np.exp(-lower/(kb_cm*T)).sum();zu=np.exp(-upper/(kb_cm*T)).sum()
    ratio=lambda wl:float(zl/zu*np.exp((10327.-1e7/wl)/(kb_cm*T)))
    # Number-density check is independent of the paper's reported sample value.
    molar_mass=3*(.95*88.90584+.05*173.045)+5*26.9815385+12*15.999
    nt_stoichiometric=4.65/molar_mass*6.02214076e23*3*.05
    sources={
        'Korner2012':dict(url='https://doi.org/10.1364/JOSAB.29.002493',
            primary_publisher_url='https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493',
            retrieved_date='2026-10-25',temperature_domain_C=[20,200],
            measured_lifetime_ms=[.94,.05],used_lifetime_ms=.95,
            stark_ground_cm1=lower.tolist(),stark_excited_cm1=[10327,10634,10927],
            spectral_numeric_arrays_obtained=False),
        'McCumber1964':dict(url='https://doi.org/10.1103/PhysRev.136.A954',
            relation='sigma_e/sigma_a=(Z_lower/Z_upper)*exp((E_zero-hnu)/(kT)); thermalized Stark manifolds'),
        'Wang2020':dict(url='https://doi.org/10.15541/jim20190077',
            primary_publisher_url='https://www.jim.org.cn/CN/10.15541/jim20190077',
            reported_nominal_dopant_at_percent=5.,reported_density_cm3=6.41e20,
            reported_mass_density_g_cm3=4.65,reported_940_1030_transparency_fraction=.0563,
            reported_940_pump_saturation_kW_cm2=25.5,
            caution='Reported concentration differs from ideal stoichiometric calculation; do not linearly extrapolate it as a measured 10 at.% sample'),
        'Li2023_YbYAG_WGM':dict(url='https://doi.org/10.1364/OL.502795',
            primary_publisher_url='https://opg.optica.org/ol/abstract.cfm?uri=ol-48-20-5359',
            pump_wavelength_nm=946,slope_efficiency=.27,maximum_output_mW=1.1,smsr_dB=27.8,
            scope='Yb:YAG crystalline-film WGM microdisk; not a PCSEL, not a demonstration of present film/pump geometry'),
        'Wang2024_PWG':dict(url='https://doi.org/10.1002/adpr.202300320',
            core_thickness_um=190.,sample_dimensions_mm=[13.5,8.,1.8],
            maximum_output_W=3.77,absorbed_pump_at_maximum_W=10.03,
            absorbed_slope=.646,incident_slope=.259,reported_intensity_loss_cm1=.13,
            scope='mm-scale ceramic PWG; loss and efficiency cannot certify a 500 nm patterned film'),
        'QCW_rate_equations':dict(url='https://www.sciencedirect.com/science/article/abs/pii/S0030401818309647',
            title='Experiments and simulations of QCW Yb:YAG laser with consideration of transient temperature',
            scope='Reference quasi-three-level pump absorption/emission, laser reabsorption and lifetime equations; present PCSEL photon normalization is separately derived')}
    module_names=['solid_state','ybyag_spatial_rates','ybyag_multimode','ybyag_threshold',
        'ybyag_pump_geometry','ybyag_spectroscopy','matched300_yag','ybyag_pulsed']
    coverage=[]
    for name in module_names:
        p=ROOT/f'src/pcselsim/{name}.py'
        coverage.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            lines=len(p.read_text(encoding='utf-8').splitlines())))
    result=dict(status='passed_with_explicit_model_limits' if all(c['passed'] for c in checks) else 'failed',
        checks=checks,module_inventory=coverage,literature=sources,
        reciprocity_300K_derived=dict(Z_lower_over_Z_upper=float(zl/zu),pump_emission_over_absorption_940=ratio(940),
            laser_emission_over_absorption_1030=ratio(1030),
            pump_emission_if_sigmaap_0p7e20_cm2=.7e-20*ratio(940),
            laser_absorption_if_sigmael_2e20_cm2=2e-20/ratio(1030),
            calibration_status='Derived consistency sensitivity; not a measured spectral table'),
        number_density_crosscheck=dict(ideal_5percent_cm3=float(nt_stoichiometric),
            paper_reported_5percent_cm3=6.41e20,paper_measurement_method_not_recovered=True),
        legacy_local_pump_counterexample=counter,
        limits=['No measured full sigma(lambda,T) array; flat constants are not broadband Yb:YAG spectra',
            'Thermalized-manifold quasi-three-level kinetics; radiative yield assumed one, no measured concentration quenching',
            'x-z inversion with y averaged, cold fields fixed; no 3D spatial hole burning or coherent gratings',
            'CW complementarity is not dynamical stability; search is only over supplied retained modes',
            'Near-Bragg optical normalization is inherited consistently; not exact full-Maxwell non-Bragg projection',
            'Ideal edge injection and ideal multi-pass relay are not an engineered or measured pump coupler'])
    (out/'rate_physics_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],independent_count_checks=len(checks),
        maximum_count_residual=max(c['excitation_photon_count_relative_residual'] for c in checks),
        maximum_energy_residual=max(c['energy_relative_residual'] for c in checks),
        reciprocity=result['reciprocity_300K_derived']),ensure_ascii=False))


if __name__=='__main__':main()
