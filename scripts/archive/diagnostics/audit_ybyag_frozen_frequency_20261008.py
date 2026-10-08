"""Frozen-Theta/finite-field radiation sensitivity, not nonlinear eigenmode solution."""
from pathlib import Path
import csv
import json
import sys
if __package__:
    from scripts._project_bootstrap import bootstrap_project
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _project_bootstrap import bootstrap_project
ROOT = bootstrap_project()
import numpy as np
from scripts import run_ybyag_pcsel as main_program
from pcselsim.stratified_optics import StratifiedLayer, StratifiedStack, local_green_kernel


def run():
    candidates = sorted(path for path in (ROOT/'results'/'ybyag_crystal_pcsel').iterdir()
                        if path.name[:8].isdigit() and (path/'finite_modes.npz').is_file()
                        and (path/'summary.json').is_file())
    source = candidates[-1]
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    core = np.load(source/'optical_core.npz')
    finite = np.load(source/'finite_modes.npz')
    full = StratifiedStack(tuple(StratifiedLayer(**layer) for layer in summary['full_stack']),
                           main_program.PASSIVE_YAG_INDEX, 1.0)
    host = next(i for i, layer in enumerate(full.layers) if layer.name=='passive YAG photonic crystal')
    guided_layers = [layer for layer in full.layers if not layer.name.startswith('DBR_')]
    total_nm = 4000+sum(layer.thickness_nm for layer in guided_layers)
    z_nm = core['z_um']*1000+0.5*total_nm
    start_nm = 2000.0
    for layer in guided_layers:
        if layer.name == 'passive YAG photonic crystal':
            break
        start_nm += layer.thickness_nm
    mask = (z_nm>=start_nm)&(z_nm<=start_nm+layer.thickness_nm)
    local_z = (z_nm[mask]-start_nm)*1e-9+full.boundaries_m[host]
    theta = core['field'][mask]
    weights = np.empty_like(local_z)
    weights[0] = 0.5*(local_z[1]-local_z[0])
    weights[-1] = 0.5*(local_z[-1]-local_z[-2])
    weights[1:-1] = 0.5*(local_z[2:]-local_z[:-2])
    weighted = theta*weights
    cell = main_program.make_structure(te_family=0)[0]
    xi = cell.fourier_epsilon
    sources = np.array([[xi(-1,0),xi(1,0),0,0],[0,0,xi(0,-1),xi(0,1)]],complex)
    gram = sources.conj().T@sources
    wavelength_b = float(summary['wavelength_nm'])
    beta = 2*np.pi/(summary['period_nm']*1e-9)
    group_index = summary['group_index']

    def evaluate(wavelength):
        kernel = local_green_kernel(full, host, wavelength)
        integral = weighted.conj()@kernel.matrix(local_z)@weighted
        amplitudes = kernel.outgoing_amplitudes(local_z)
        top, bottom = amplitudes.top@weighted, amplitudes.bottom@weighted
        factor = (2*np.pi/(wavelength*1e-9))**4/(2*beta)
        crad = -factor*integral*gram
        up = factor*amplitudes.top_admittance_m.real*abs(top)**2*gram
        down = factor*amplitudes.bottom_admittance_m.real*abs(bottom)**2*gram
        closure = np.linalg.norm((crad-crad.conj().T)/(2j)-up-down)/np.linalg.norm(up+down)
        return integral, crad, up, down, float(closure)

    baseline = evaluate(wavelength_b)
    recovery = np.linalg.norm(baseline[1]-core['Crad'])/np.linalg.norm(core['Crad'])
    if recovery > 1e-9:
        raise RuntimeError(f'Frozen Bragg core recovery failed: {recovery}')
    with (source/'mode_summary.csv').open(encoding='utf-8-sig') as handle:
        modes = list(csv.DictReader(handle))
    rows = []
    for mode in modes:
        name = mode['mode']
        delta = float(mode['delta_per_m'])
        wavelength = 2*np.pi/(2*np.pi/(wavelength_b*1e-9)+delta/group_index)*1e9
        updated = evaluate(wavelength)
        fields = finite[f'{name}_fields'].reshape(4,-1)
        norm = np.sum(abs(fields)**2)
        def power(matrix):
            return float(2*np.einsum('ik,ij,jk->',fields.conj(),matrix,fields).real/norm/100)
        up_b, down_b = power(baseline[2]), power(baseline[3])
        up_w, down_w = power(updated[2]), power(updated[3])
        row = dict(mode=name, delta_extrap_per_m=delta,
            delta_raw_finest_per_m=float(finite[f'convergence_{name}'][-1].real),
            frozen_Bragg_wavelength_nm=wavelength_b,
            group_converted_mode_wavelength_nm=wavelength,
            first_order_lambda_nm=wavelength_b-wavelength_b**2*1e-9*delta/(2*np.pi*group_index),
            relative_Crad_norm_change=float(np.linalg.norm(updated[1]-baseline[1])/np.linalg.norm(baseline[1])),
            Crad_norm_ratio=float(np.linalg.norm(updated[1])/np.linalg.norm(baseline[1])),
            total_amplitude_loss_trace_ratio=float(np.trace(updated[2]+updated[3]).real/np.trace(baseline[2]+baseline[3]).real),
            top_power_loss_Bragg_cm=up_b,top_power_loss_mode_frequency_cm=up_w,top_power_loss_ratio=up_w/up_b,
            bottom_power_loss_Bragg_cm=down_b,bottom_power_loss_mode_frequency_cm=down_w,bottom_power_loss_ratio=down_w/down_b,
            double_green_real_Bragg=float(baseline[0].real),double_green_imag_Bragg=float(baseline[0].imag),
            double_green_real_mode=float(updated[0].real),double_green_imag_mode=float(updated[0].imag),
            optical_theorem_relative_residual=updated[4])
        rows.append(row)
    output = ROOT/'results'/'ybyag_crystal_pcsel'/'independent_family_audit_20261008'
    with (output/'frozen_frequency_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    payload = dict(source_directory=str(source),frozen_Bragg_core_relative_recovery=float(recovery),
        status='frozen_Theta_cell_finite_fields_q0_sensitivity_NOT_frequency_selfconsistent_eigenproblem',
        fixed_geometry_period_nm=summary['period_nm'],finite_grid_points=int(finite['convergence_grid_points'][-1]),
        source_CWT_truncation=6,constant_material_indices=True,
        omissions='C1D/C2D/Theta/group index/material dispersion and finite eigenfield not re-solved; no gain/spectral/thermal update',modes=rows)
    (output/'frozen_frequency_sensitivity.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(payload,indent=2,ensure_ascii=False),flush=True)
    return payload


if __name__=='__main__':
    run()
