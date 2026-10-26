"""Yb:YAG pump/size feasibility, with fixed physical unit cells.

中文：保留原有300微米正式结果，另开目录研究输入输出和尺寸阈值。
固定孔形、物理周期和层栈；改变腔尺寸时重新求有限腔和频率，不按1/L外推损耗。
The saved 300-um basis supports shared-inversion CW curves. Size scans use a
bounded coarse-to-fine census, not a certification of the complete fine spectrum.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(Path(__file__).parent)]
import numpy as np
from scipy.constants import c
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.sparse.linalg import splu
import matched300_yag_search as optical_backend
from pcselsim.custom_analysis import FourWaveOpticalSpec, finite_area_eigenproblem
from pcselsim.triangular_finite import TriangularFiniteSpec, triangular_finite_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.matched300_yag import binned_lateral_weights
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG, solve_multimode_spatial_ybyag
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig

OUT = ROOT/'results/current/yag_feasibility_20261025'
MANIFEST = ROOT/'results/current/matched300_20261021/comparison/selected_cases.json'


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(lattice):
    return next(x for x in json.loads(MANIFEST.read_text(encoding='utf-8'))['cases']
                if x['id'] == 'YbYAG_'+lattice)


def load_saved_system(lattice):
    """Recover each own vg from qg=(c/n_core/vg)qE, not a common guessed n.

    中文：从同一模式保存的能量/增益权重恒等式恢复群速度，并检查各z单元一致。
    This is the non-dispersive gain-core identity used by the optical adapter.
    """
    chosen = selected(lattice); directory = (ROOT/chosen['summary_path']).parent
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((directory/'closed_optical_modes.json').read_text(encoding='utf-8'))
    models = []
    with np.load(directory/'closed_fields.npz', allow_pickle=False) as data:
        for row in rows:
            name = row['name']; qe, qg = data[name+'_qenergy'], data[name+'_qgain']
            vg = c*qe.sum()/(1.8166*qg.sum())
            assert np.allclose(qg, c/(1.8166*vg)*qe, rtol=2e-12, atol=1e-16)
            models.append(SpatialYbYAGModel(optical_backend.material(row['wavelength_nm']),
                data[name+'_widths'], qe, row['total_power_loss_per_m'],
                row['up_power_loss_per_m'], row['down_power_loss_per_m'], vg, qg,
                data[name+'_lateral']))
    return SpatialMultimodeYbYAG(tuple(models), tuple(r['name'] for r in rows)), summary, rows


def pump_config(architecture, length, power=0., cells=31):
    return PumpGeometryConfig('top_single' if architecture.startswith('top') else 'edge_dual',
        power, length, cells, top_passes=20 if architecture == 'top20' else 1,
        return_efficiency=.98 if architecture == 'top20' else 1.)


def cw_phase():
    for lattice in ('square', 'triangular'):
        system, summary, _ = load_saved_system(lattice)
        for architecture in ('edge_dual', 'top1', 'top20'):
            path = OUT/'cw'/f'{lattice}_{architecture}.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'completed':
                continue
            pump = pump_config(architecture, 300.)
            threshold = shared_inversion_pump_threshold(system, pump)
            result = dict(status='running', lattice=lattice, architecture=architecture,
                threshold=asdict(threshold), points=[], retained_modes=len(system.modes),
                scope='Nominal material; all retained modes share x-z inversion, y averaged; internal first-entry pump power. Top20 is an ideal relay, not fabricated pump optics.')
            for ratio in (0., .5, .9, .99, 1.01, 1.05, 1.1, 1.25, 1.5, 2., 3., 4., 6.):
                power = ratio*threshold.first_threshold_internal_W
                started = time.monotonic()
                try:
                    state = solve_multimode_spatial_ybyag(system, replace(pump, internal_incident_power_W=power))
                    point = dict(status='completed', ratio=ratio, internal_pump_W=power,
                        upward_output_W=float(state.upward_output_W.sum()),
                        mode_upward_W=dict(zip(system.names, state.upward_output_W)),
                        active_modes=state.active_modes, energy_budget_W=state.energy_budget_W,
                        complementarity_residual=state.complementarity_residual,
                        population_residual_per_s=state.population_residual_per_s)
                except Exception as error:
                    point = dict(status='failed', ratio=ratio, internal_pump_W=power, error=repr(error))
                point['seconds'] = time.monotonic()-started
                result['points'].append(point); save(path, result)
                print('CW', lattice, architecture, ratio, point['status'], point.get('upward_output_W'), flush=True)
            result['status'] = 'completed' if all(p['status'] == 'completed' for p in result['points']) else 'partial'
            save(path, result)


def stark_emission_to_absorption(wavelength_nm, temperature_K=300.):
    """Thermal-manifold reciprocity from Koerner 2012 Table 1.

    This constrains absorption/emission pairs, but supplies no measured line
    shape or temperature-dependent oscillator strength. It is not a spectrum.
    中文：能级配分函数只约束吸收/发射的比值，不能冒充实测完整截面谱。
    """
    lower=np.array([0.,581.,619.,786.]); upper=np.array([10327.,10634.,10927.])
    kT=.6950348004861274*temperature_K  # k_B/(h*c), cm^-1 K^-1
    zl=np.exp(-lower/kT).sum(); zu=np.exp(-(upper-upper.min())/kT).sum()
    return float(zl/zu*np.exp((upper.min()-1e7/wavelength_nm)/kT))


def sensitivity_phase():
    variations=[('nominal',{}),('reciprocity_300K',{}),
        ('reciprocity_Nt_half',{'dopant_density_cm3':.69e21}),
        ('reciprocity_Nt_1p5',{'dopant_density_cm3':2.07e21}),
        ('reciprocity_tau_0p90ms',{'upper_state_lifetime_ms':.90}),
        ('reciprocity_tau_1p00ms',{'upper_state_lifetime_ms':1.00}),
        ('reciprocity_sigma_ap_0p5e20',{'pump_absorption_cross_section_cm2':.5e-20}),
        ('reciprocity_sigma_ap_0p82e20',{'pump_absorption_cross_section_cm2':.82e-20}),
        ('reciprocity_sigma_el_1p6e20',{'laser_emission_cross_section_cm2':1.6e-20}),
        ('reciprocity_sigma_el_2p2e20',{'laser_emission_cross_section_cm2':2.2e-20}),
        ('reciprocity_internal_0p5cm',{'internal_cm':.5}),
        ('reciprocity_internal_1cm',{'internal_cm':1.}),
        ('reciprocity_internal_2cm',{'internal_cm':2.})]
    for lattice in ('square','triangular'):
        base,_,_=load_saved_system(lattice); records=[]
        for label, values in variations:
            models=[]
            for old in base.modes:
                changes={k:v for k,v in values.items() if k!='internal_cm'}
                med=replace(old.medium,**changes)
                if label!='nominal':
                    med=replace(med,pump_emission_cross_section_cm2=med.pump_absorption_cross_section_cm2*stark_emission_to_absorption(med.pump_wavelength_nm),
                        laser_absorption_cross_section_cm2=med.laser_emission_cross_section_cm2/stark_emission_to_absorption(med.laser_wavelength_nm))
                loss=old.total_power_loss_per_m+100*(values.get('internal_cm',.1)-.1)
                models.append(replace(old,medium=med,total_power_loss_per_m=loss))
            system=SpatialMultimodeYbYAG(tuple(models),base.names)
            record=dict(label=label,changes=values,medium=asdict(models[0].medium),pumps={})
            for architecture in ('edge_dual','top1','top20'):
                th=shared_inversion_pump_threshold(system,pump_config(architecture,300.),max_power_W=1e4)
                record['pumps'][architecture]=asdict(th)
            records.append(record)
            save(OUT/'sensitivity'/f'{lattice}.json',dict(lattice=lattice,records=records,
                scope='Conditional one-at-a-time material/extra-loss sensitivity on fixed nominal cold fields; varying concentration holds index/lifetime fixed. Reciprocity-derived pairs are not measured spectra.'))
            print('SENSITIVITY',lattice,label,[record['pumps'][a]['first_threshold_internal_W'] for a in ('edge_dual','top1','top20')],flush=True)


def make_problem(lattice, model, length, grid):
    r = model.result
    if lattice == 'square':
        spec = FourWaveOpticalSpec(r.bragg_wavelength_nm, r.effective_index, r.group_index,
            length, .1, tuple(r.eigenvalues_m.real/100), tuple(r.eigenvalues_m.imag/100),
            grid, 'staggered', 8, 0, 500)
        return spec, finite_area_eigenproblem(spec, r.coupling_m), None
    spec = TriangularFiniteSpec(length/2, grid, 0., 8, 'square', 'staggered')
    problem, mesh = triangular_finite_eigenproblem(r, spec)
    return spec, problem, mesh


def interpolate(lattice, field, old_spec, old_mesh, spec, mesh, length):
    if lattice == 'square':
        source = ((np.arange(old_spec.grid_points)+.5)/old_spec.grid_points-.5)*length
        dest = ((np.arange(spec.grid_points)+.5)/spec.grid_points-.5)*length
        yy, xx = np.meshgrid(dest, dest, indexing='ij')
        return np.array([RegularGridInterpolator((source, source), f,
            bounds_error=False, fill_value=0.)(np.stack((yy, xx), axis=-1)) for f in field])
    source = np.column_stack((old_mesh.x_um, old_mesh.y_um))
    dest = np.column_stack((mesh.x_um, mesh.y_um))
    return np.array([LinearNDInterpolator(source, f, fill_value=0.)(dest) for f in field])


def find_seeds(lattice, model, length, grid, keep=8):
    """Full small generalized spectrum -> residual-tested finer candidates.

    Only exact fine-grid fields enter closure. Interpolation supplies start
    vectors and cannot become the final eigenmode without residual acceptance.
    中文：完整粗谱仅引导细网格；所有接受根必须满足未经修改的K/M原方程。
    """
    coarse = 11 if lattice == 'square' else 5
    cs, cp, cm = make_problem(lattice, model, length, coarse)
    dim = cp.stiffness.shape[0]
    cp.near(complex(model.result.eigenvalues_m[0]), 12, np.ones(dim, complex), dense_limit=dim+1)
    cv, cf = cp._dense_spectrum
    valid = np.flatnonzero(cv.imag >= -1e-7)
    order = valid[np.argsort(cv[valid].imag)]
    spec, problem, mesh = make_problem(lattice, model, length, grid)
    shape = (4, grid, grid) if lattice == 'square' else (6, mesh.points)
    coarse_shape = (4, coarse, coarse) if lattice == 'square' else (6, cm.points)
    operator = problem.center_operator(); values = []; fields = []; attempts = []
    targets = []
    for i in order[:16]:
        initial = interpolate(lattice, cf[:, i].reshape(coarse_shape), cs, cm, spec, mesh, length)
        targets.append((complex(cv[i]), initial.ravel(), 'coarse_lowest16'))
    deterministic = np.ones(operator.shape[0], complex)+.1j*np.sin(np.arange(operator.shape[0]))
    for v in model.result.eigenvalues_m:
        targets.append((complex(v), deterministic, 'Gamma_supplement'))
    for target, initial, kind in targets:
        try:
            solver='bounded_Arnoldi'; inverse_residual=None
            vv=ff=None
            if kind=='coarse_lowest16':
                # Solve the unchanged K/M pencil, using physical center fields.
                # A Rayleigh value is accepted only with its own strict residual;
                # an unconverged inverse-iteration estimate is never a result.
                q=initial/np.linalg.norm(initial); source=q.copy(); shift=target+.01+.02j
                for iteration in range(12):
                    try:
                        links=splu((problem.stiffness-shift*problem.mass).tocsc()).solve(q)
                    except (RuntimeError,ValueError):
                        break  # An inverse-iteration failure still gets Arnoldi.
                    candidate=problem.mass@links; candidate_norm=np.linalg.norm(candidate)
                    if not np.isfinite(candidate_norm) or candidate_norm<=1e-30:
                        break
                    candidate/=candidate_norm
                    applied=operator@candidate; value=complex(np.vdot(candidate,applied))
                    inverse_residual=relative_residual(operator,candidate,value)
                    if inverse_residual<1e-10 and abs(np.vdot(source,candidate))**2>.15:
                        vv=np.array([value]);ff=candidate[:,None]
                        solver='residual_verified_physical_center_inverse_iteration';break
                    q=candidate;shift=value+1e-8+2e-8j
            if vv is None:
                vv, ff = problem.near(target+(.01+.02j), 6, initial, tol=1e-10,
                                       maxiter=100, dense_limit=0)
            accepted = 0
            for value, field in zip(vv, ff.T):
                if value.imag < -1e-7 or relative_residual(operator, field, value) > 2e-7: continue
                if not independent_eigenmode(value, field, values, fields): continue
                values.append(complex(value)); fields.append(field); accepted += 1
            attempts.append(dict(kind=kind, target=[target.real, target.imag], accepted=accepted,
                solver=solver,inverse_residual=inverse_residual))
        except Exception as error:
            attempts.append(dict(kind=kind, target=[target.real, target.imag], error=repr(error)))
    # Keep an operating-band challenge as well as the absolute low-loss roots.
    # A distant low-loss family must not silently inherit 1030-nm peak gain.
    # 同时保留工作带候选，避免低损耗但偏离增益峰的谱族挤掉1030nm模式。
    r=model.result; beta=2*np.pi*r.effective_index/(r.bragg_wavelength_nm*1e-9)
    period=r.lattice_constant_nm
    beta0=(2*np.pi if lattice=='square' else 4*np.pi/np.sqrt(3))/(period*1e-9)
    background=(beta**2-beta0**2)/(2*beta0)
    derivative=beta/beta0*(-2*np.pi*r.group_index/(r.bragg_wavelength_nm*1e-9)**2)*1e-9
    predicted=np.array([r.bragg_wavelength_nm+(v.real-background)/derivative for v in values])
    order=np.argsort(np.imag(values)); near=[i for i in order if abs(predicted[i]-1030.)<=1.25]
    indices=list(dict.fromkeys([*order[:keep],*near[:keep]]))
    return spec, mesh, [(values[i], fields[i].reshape(shape)) for i in indices], dict(
        coarse_grid=coarse, coarse_spectrum_count=len(cv), fine_grid=grid,
        distinct_fine_candidates=len(values), attempts=attempts,
        scope='Complete coarse discretized spectrum, bounded fine challenge; not an exhaustive continuum or fine-grid census.')


def size_point(lattice, length, order, dz, grid, keep):
    chosen = selected(lattice)
    summary = json.loads((ROOT/chosen['summary_path']).read_text(encoding='utf-8'))
    design, period = summary['design'], summary['period_nm']
    started = time.monotonic()
    model, _, _ = optical_backend.optical(lattice, design, order, dz, 1030., period)
    spec, mesh, seeds, census = find_seeds(lattice, model, length, grid, keep)
    rows = []; models = []; fields = {}; failures = []
    for rank, seed in enumerate(seeds):
        try:
            wl, optical, guide, local_spec, mode, trace = optical_backend.close_seed(
                lattice, design, period, model, seed, spec, mesh, order, dz)
            name = f'mode_{rank:02d}'
            peers = [r for r in rows if abs(r['wavelength_nm']-wl) < 1e-7]
            if not independent_eigenmode(complex(mode.delta_per_m, mode.alpha_per_m), mode.fields,
                [complex(r['delta_per_m'],r['alpha_field_per_m']) for r in peers],
                [fields[r['name']+'_field'] for r in peers]): continue
            widths, qe, qg = optical_backend.active_weights(optical.result, guide, 24)
            lateral = (optical_backend.lateral_energy_weights(mode.fields, 31) if lattice == 'square'
                       else binned_lateral_weights(mode.fields, mode.grid.x_um, length, 31))
            up, down = optical.side_power_losses(mode.fields)
            loss = 2*mode.alpha_per_m+10.
            medium = replace(optical_backend.material(wl), pumped_area_um2=length**2)
            spatial = SpatialYbYAGModel(medium, widths, qe, loss, up, down,
                                       c/optical.result.group_index, qg, lateral)
            rows.append(dict(name=name, wavelength_nm=wl, delta_per_m=mode.delta_per_m,
                alpha_field_per_m=mode.alpha_per_m, material_gain_cm1=loss/qg.sum()/100,
                total_power_loss_per_m=loss, up_power_loss_per_m=up,
                down_power_loss_per_m=down, gain_weight=qg.sum(),
                eigen_residual=mode.eigen_residual,
                frequency_residual_per_m=trace[-1]['frequency_residual_per_m']))
            models.append(spatial)
            fields.update({name+'_field':mode.fields, name+'_qgain':qg, name+'_qenergy':qe,
                           name+'_widths':widths, name+'_lateral':lateral})
        except Exception as error:
            failures.append(dict(reference_rank=rank, error=repr(error)))
    result = dict(status='completed' if rows else 'failed', lattice=lattice, side_um=length,
        period_nm=period, design=design, order=order, dz_nm=dz, grid=grid,
        modes=rows, closure_failures=failures, census=census, pumps={}, seconds=time.monotonic()-started,
        scope='Fixed final unit cell and layer stack; each size solves its own finite poles and actual frequency. Sampled TE0 candidates, nominal constant spectroscopy; no full-spectrum/thermal/full-wave certification.')
    if rows:
        system = SpatialMultimodeYbYAG(tuple(models), tuple(r['name'] for r in rows))
        # Reject unreachable modes analytically at the pump-imposed f ceiling.
        med = models[0].medium
        fmax = med.pump_absorption_cross_section_cm2/(med.pump_absorption_cross_section_cm2+med.pump_emission_cross_section_cm2)
        cap = med.dopant_density_cm3*((med.laser_emission_cross_section_cm2+med.laser_absorption_cross_section_cm2)*fmax-med.laser_absorption_cross_section_cm2)
        result['pump_limited_material_gain_ceiling_cm1'] = cap
        for architecture in ('edge_dual','top1','top20'):
            if min(r['material_gain_cm1'] for r in rows) >= cap:
                result['pumps'][architecture] = dict(first_threshold_internal_W=None,
                    status='unreachable_under_nominal_pump_inversion_ceiling')
            else:
                th = shared_inversion_pump_threshold(system, pump_config(architecture,length), max_power_W=1e5)
                result['pumps'][architecture] = asdict(th)
        target_indices=[i for i,r in enumerate(rows) if 1029.<=r['wavelength_nm']<=1031.]
        result['target_band_nm']=[1029.,1031.]
        result['target_band_mode_names']=[rows[i]['name'] for i in target_indices]
        result['target_band_pumps']={}
        if target_indices:
            target_system=SpatialMultimodeYbYAG(tuple(models[i] for i in target_indices),
                                               tuple(rows[i]['name'] for i in target_indices))
            for architecture in ('edge_dual','top1','top20'):
                if min(rows[i]['material_gain_cm1'] for i in target_indices)>=cap:
                    result['target_band_pumps'][architecture]=dict(first_threshold_internal_W=None,
                        status='unreachable_under_nominal_pump_inversion_ceiling')
                else:
                    result['target_band_pumps'][architecture]=asdict(shared_inversion_pump_threshold(
                        target_system,pump_config(architecture,length),max_power_W=1e5))
    return result, fields


def size_phase(args):
    for lattice in args.lattices:
        grid = args.square_grid if lattice == 'square' else args.tri_grid
        for length in args.sizes:
            tag = f'{lattice}_L{length:g}_D{args.order}_dz{args.dz:g}_N{grid}_K{args.closed_modes}'
            path = OUT/'sizes'/f'{tag}.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'completed': continue
            print('SIZE_START',tag,flush=True)
            try:
                result, fields = size_point(lattice,length,args.order,args.dz,grid,args.closed_modes)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'),**fields)
            except Exception as error: result=dict(status='failed',lattice=lattice,side_um=length,error=repr(error))
            save(path,result)
            print('SIZE_DONE',tag,result['status'],[(r['material_gain_cm1'],r['wavelength_nm']) for r in result.get('modes',[])[:2]],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('cw','size','sensitivity'))
    p.add_argument('--lattices',nargs='+',default=['square','triangular'])
    p.add_argument('--sizes',nargs='+',type=float,default=[30,50,75,100,150,200,300,500,750,1000])
    p.add_argument('--order',type=int,default=12);p.add_argument('--dz',type=float,default=2.)
    p.add_argument('--square-grid',type=int,default=25);p.add_argument('--tri-grid',type=int,default=15)
    p.add_argument('--closed-modes',type=int,default=6)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    runtime={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'src/pcselsim').glob('*.py')}
    stamp=time.time_ns()
    (OUT/f'executed_{a.phase}_{stamp}.py').write_bytes(Path(__file__).read_bytes())
    save(OUT/f'runtime_{a.phase}_{stamp}.json',dict(argv=sys.argv,source_sha256=runtime,
        script_sha256=sha(__file__),original_manifest_sha256=sha(MANIFEST)))
    {'cw':cw_phase,'size':lambda:size_phase(a),'sensitivity':sensitivity_phase}[a.phase]()


if __name__=='__main__': main()
