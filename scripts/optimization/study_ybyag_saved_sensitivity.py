"""Conditional material/pump study of saved frequency-closed Yb:YAG modes.

The cold-cavity fields remain fixed. Changes to doping, spectroscopy, lifetime,
or a uniform phenomenological internal loss are conditional rate-model tests;
they are not a re-solved dispersive/absorbing Maxwell problem.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).parent)]
from evaluate_ybyag_saved_modes import load_closed
from optimize_ybyag_cavity import save
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG, solve_multimode_spatial_ybyag, _inversion
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, pump_geometry_metadata, propagate_geometry_pump
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.run_output import prepare_output, write_provenance


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def material_system(system, *, density_factor=1., tau_factor=1., se_factor=1., sa_factor=1.,
                    spa_factor=1., spe_factor=1., extra_loss_per_m=0., only_names=None):
    modes = []
    for name, mode in zip(system.names, system.modes):
        if only_names is not None and name not in only_names:
            modes.append(mode)
            continue
        m = mode.medium
        material = replace(m,
            dopant_density_cm3=m.dopant_density_cm3*density_factor,
            upper_state_lifetime_ms=m.upper_state_lifetime_ms*tau_factor,
            laser_emission_cross_section_cm2=m.laser_emission_cross_section_cm2*se_factor,
            laser_absorption_cross_section_cm2=m.laser_absorption_cross_section_cm2*sa_factor,
            pump_absorption_cross_section_cm2=m.pump_absorption_cross_section_cm2*spa_factor,
            pump_emission_cross_section_cm2=m.pump_emission_cross_section_cm2*spe_factor)
        modes.append(replace(mode, medium=material,
            total_power_loss_per_m=mode.total_power_loss_per_m+extra_loss_per_m))
    return SpatialMultimodeYbYAG(tuple(modes), system.names)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--output', required=True)
    p.add_argument('--adversarial-only', action='store_true',
        help='Vary selected/competing branch cross sections independently; unmeasured spectra stress test only.')
    a = p.parse_args(argv)
    out = prepare_output(a.output, ROOT)
    write_provenance(out, ROOT)
    save(out/'run_status.json', {'status':'running'})
    design, base = load_closed(a.source)
    nx = len(base.modes[0].lateral_energy_fractions)
    pump = PumpGeometryConfig('edge_dual', 0., design.device_um, nx)
    cases = [
        ('nominal_10atpct', {}),
        ('density_1atpct', {'density_factor':.1}),
        ('density_5atpct', {'density_factor':.5}),
        ('density_7atpct', {'density_factor':.7}),
        ('density_8atpct', {'density_factor':.8}),
        ('tau_0p70ms', {'tau_factor':.70/.95}),
        ('sigma_e_minus10pct', {'se_factor':.9}),
        ('sigma_e_minus20pct', {'se_factor':.8}),
        ('sigma_a_plus50pct', {'sa_factor':1.5}),
        ('sigma_ap_minus30pct', {'spa_factor':.7}),
        ('sigma_ep_times2', {'spe_factor':2.}),
        ('internal_loss_0p5cm', {'extra_loss_per_m':40.}),
        ('internal_loss_1cm', {'extra_loss_per_m':90.}),
        ('internal_loss_3cm', {'extra_loss_per_m':290.}),
    ]
    if a.adversarial_only:
        cases=[('selected_sigma_e_minus10pct',dict(se_factor=.9,only_names=['TE0_A'])),
               ('selected_sigma_e_minus20pct',dict(se_factor=.8,only_names=['TE0_A'])),
               ('competitor_sigma_e_plus20pct',dict(se_factor=1.2,only_names=[n for n in base.names if n!='TE0_A'])),
               ('selected_minus10pct_competitors_nominal',dict(se_factor=.9,only_names=['TE0_A']))]
    studies = []
    flat = []
    for name, kwargs in cases:
        system = material_system(base, **kwargs)
        threshold = shared_inversion_pump_threshold(system, pump)
        result = dict(case=name, parameter_factors=kwargs, threshold=asdict(threshold))
        studies.append(result)
        flat.append(dict(case=name, first_threshold_internal_W=threshold.first_threshold_internal_W,
            first_modes=';'.join(threshold.first_modes), next_to_first_small_signal_ratio=threshold.next_to_first_ratio))
        print(name, threshold.first_threshold_internal_W, threshold.first_modes, flush=True)
        save(out/'material_sensitivity.json', studies)
    write_csv(out/'material_sensitivity.csv', flat)
    if a.adversarial_only:
        save(out/'run_status.json',dict(status='complete',
            scope='Independent branch cross-section stress test, not a measured wavelength-dependent material model'))
        return

    threshold = shared_inversion_pump_threshold(base, pump)
    first = threshold.first_threshold_internal_W
    assert first is not None
    fpump = replace(pump, internal_incident_power_W=first)
    fraction, _ = _inversion(base, fpump, np.zeros(len(base.modes)))
    transport = propagate_geometry_pump(base.modes[0], fpump, fraction)
    np.savez_compressed(out/'threshold_pump_and_inversion.npz', inversion=fraction,
        pump_forward_W_cm2=transport.forward_W_cm2, pump_backward_W_cm2=transport.backward_W_cm2)
    geometry = pump_geometry_metadata(base.modes[0], fpump)
    coupling = [dict(coupling_efficiency=eta, external_threshold_W=first/eta)
                for eta in (1., .5, .2, .1, .01)]
    save(out/'threshold_pump_details.json', dict(metadata=geometry,
        inversion_min=float(fraction.min()), inversion_max=float(fraction.max()),
        absorbed_pump_W=transport.absorbed_power_W, absorbed_fraction=transport.absorbed_power_W/first,
        escaped_pump_W=transport.escaped_pump_power_W,
        external_coupling_conversion_only=coupling,
        caveat='Coupling efficiency is unspecified; transverse pump guiding, 500nm edge focusing and scattering were not solved.'))

    architectures = []
    for arch, passes, eta in [('edge_single',1,1.), ('edge_dual',1,1.), ('top_single',1,1.),
                              ('top_single',20,.98), ('top_single',100,.98)]:
        layout = PumpGeometryConfig(arch, 0., design.device_um, nx, top_passes=passes, return_efficiency=eta)
        t = shared_inversion_pump_threshold(base, layout)
        architectures.append(dict(architecture=arch, passes=passes, relay_efficiency=eta,
            threshold=asdict(t), metadata_at_threshold=pump_geometry_metadata(base.modes[0],
            replace(layout, internal_incident_power_W=t.first_threshold_internal_W or 0.))))
        print('pump', arch, passes, t.first_threshold_internal_W, flush=True)
        save(out/'pump_architectures.json', architectures)

    scan = []
    for multiplier in (.8, 1., 1.05, 1.1, 1.2, 1.5, 2., 3., 4., 6., 8.):
        power = first*multiplier
        fit = solve_multimode_spatial_ybyag(base, replace(pump, internal_incident_power_W=power))
        scan.append(dict(threshold_multiplier=multiplier, internal_pump_W=power,
            active_modes=fit.active_modes, outputs_W=dict(zip(base.names,map(float,fit.upward_output_W))),
            budget=fit.energy_budget_W, ncp_residual=fit.complementarity_residual,
            population_residual_per_s=fit.population_residual_per_s,
            inactive_gain_excess_per_m=fit.maximum_inactive_gain_excess_per_m))
        save(out/'cw_pump_curve.json', scan)
    write_csv(out/'cw_pump_curve.csv', [dict(threshold_multiplier=r['threshold_multiplier'],
        internal_pump_W=r['internal_pump_W'], active_modes=';'.join(r['active_modes']),
        upward_output_W=sum(r['outputs_W'].values()), absorbed_pump_W=r['budget']['absorbed_pump_W'],
        other_loss_W=r['budget']['other_cavity_loss_W'], energy_residual_W=r['budget']['energy_closure_residual_W']) for r in scan])

    fig, axes = plt.subplots(1, 3, figsize=(14.2,4.2), constrained_layout=True)
    powers = np.array([r['internal_pump_W'] for r in scan])
    for name in base.names:
        outputs = np.array([r['outputs_W'][name] for r in scan])
        if outputs.max() > 1e-10:
            axes[0].plot(powers, outputs*1e3, 'o-', label=name)
    axes[0].axvline(first, color='k', linestyle=':', label='Internal threshold')
    axes[0].set(xlabel='Total internal first-entry pump (W)', ylabel='Upward output (mW)', title='19 closed modes; shared inversion')
    axes[0].legend(fontsize=8)
    axes[1].plot(powers, [r['budget']['absorbed_pump_W']*1e3 for r in scan], 'o-', label='Absorbed pump')
    axes[1].plot(powers, [sum(r['outputs_W'].values())*1e3 for r in scan], 'o-', label='Upward output')
    axes[1].plot(powers, [r['budget']['other_cavity_loss_W']*1e3 for r in scan], 'o-', label='Other cavity loss')
    axes[1].set(xlabel='Total internal pump (W)', ylabel='Power (mW)', title='Energy channels; thermal model absent')
    axes[1].legend(fontsize=8)
    selected = [r for r in flat if r['first_threshold_internal_W'] is not None]
    axes[2].barh([r['case'] for r in selected], [r['first_threshold_internal_W'] for r in selected])
    axes[2].set(xlabel='Internal threshold (W)', title='Fixed-field conditional sensitivity')
    axes[2].tick_params(axis='y', labelsize=7)
    for ax in axes[:2]:
        ax.grid(alpha=.2)
    fig.savefig(out/'pump_curve_and_sensitivity.png', dpi=180)
    plt.close(fig)
    save(out/'summary.json', dict(source=str(Path(a.source).resolve()), design=asdict(design),
        medium=asdict(base.modes[0].medium), modes=len(base.modes), nominal_threshold=asdict(threshold),
        scope='conditional fixed-cold-field shared-inversion rate-model sensitivity; no measured spectra, real pump coupler or thermal solution',
        material_sensitivity=studies, pump_architectures=architectures))
    save(out/'run_status.json', {'status':'complete'})


if __name__ == '__main__':
    main()
