"""Reproducible Yb:YAG cavity search; frozen screening and closed finalists.

Screening numbers are proxies, never claimed as frequency-closed laser output.
All pump powers count total internal first-entry W, rather than a fixed edge
intensity that changes its total power when the gain thickness is varied.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.constants import c,h

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from pcselsim.geometry import SquareLatticeCell,Ellipse
from pcselsim.vertical import Layer
from pcselsim.hybrid_vertical import HybridLayerStack,solve_guided_te_modes
from pcselsim.stratified_optics import StratifiedStack,StratifiedLayer,quarter_wave_dbr
from pcselsim.three_d_cwt import ThreeDCWTSettings,solve_bragg_vertical_mode
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.custom_analysis import FourWaveOpticalSpec,solve_finite_modes,vector_far_field_metrics,vector_far_field_complex,plot_vector_far_field_diagnostics
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.frequency_cwt import solve_frequency_consistent_modes
from pcselsim.run_output import prepare_output,write_provenance,default_output
from pcselsim.design_objectives import beam_metrics,beam_acceptance
from run_ybyag_pcsel import active_weights,lateral_energy_weights


@dataclass(frozen=True)
class Design:
    id: str
    gain_nm: float=300.
    cap_nm: float=100.
    pc_nm: float=200.
    spacer_nm: float=2000.
    fill: float=.10
    share: float=.42
    dx: float=.45
    dy: float=.45
    aspect: float=1.
    angle_deg: float=0.
    device_um: float=1000.
    pairs: int=16
    top_silica_nm: float=300.
    hole_index: float=1.45
    period_scale: float=1.
    fixed_period_nm: float | None=None
    dopant_density_cm3: float=1.38e21
    upper_state_lifetime_ms: float=.95
    pump_wavelength_nm: float=940.
    pump_absorption_cross_section_cm2: float=.70e-20
    pump_emission_cross_section_cm2: float=.10e-20
    laser_absorption_cross_section_cm2: float=.126e-20
    laser_emission_cross_section_cm2: float=2.0e-20
    internal_power_loss_cm_inverse: float=.10


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:float(x)),encoding='utf8')


def structure(design,te_family=0):
    radii=np.sqrt(design.fill*np.array([design.share,1-design.share])/np.pi)
    holes=tuple(Ellipse((s*design.dx/2,s*design.dy/2),
                (float(r*np.sqrt(design.aspect)),float(r/np.sqrt(design.aspect))),
                design.angle_deg,design.hole_index**2) for s,r in zip((-1,1),radii))
    # Conservative enclosing-circle rejection guarantees non-overlap, including
    # periodic images. It may reject feasible ellipses; it never accepts overlap.
    envelopes=radii*max(np.sqrt(design.aspect),1/np.sqrt(design.aspect))
    gap=np.inf
    for i,a in enumerate(holes):
        for j,b in enumerate(holes):
            for u in (-1,0,1):
                for v in (-1,0,1):
                    if i==j and u==0 and v==0: continue
                    gap=min(gap,np.hypot(a.center[0]-b.center[0]+u,a.center[1]-b.center[1]+v)-envelopes[i]-envelopes[j])
    if gap<=.025: raise ValueError('Conservative periodic hole gap below .025a')
    cell=SquareLatticeCell(1.8154**2,holes)
    index=float(np.sqrt(cell.fourier_epsilon(0,0).real))
    layers=[Layer('lower SiO2 guide spacer',design.spacer_nm,1.45),
            Layer('Yb:YAG gain',design.gain_nm,1.8166),
            Layer('passive YAG photonic crystal',design.pc_nm,index)]
    if design.cap_nm>0: layers.append(Layer('unpatterned SiN upper guide',design.cap_nm,2.))
    if design.top_silica_nm>0: layers.append(Layer('SiO2 upper cladding',design.top_silica_nm,1.45))
    guide=HybridLayerStack(tuple(layers),1.,1.45,2.,te_family)
    full=StratifiedStack(quarter_wave_dbr(1030.,2.10,1.45,design.pairs,first_layer='low')+
            tuple(StratifiedLayer(x.name,x.thickness_nm,x.refractive_index) for x in layers),
            bottom_index=1.8154,top_index=1.)
    return cell,guide,full,float(gap)


def medium(design,wavelength=1030.):
    return YbYAGMediumConfig(laser_wavelength_nm=wavelength,refractive_index=1.8166,
            dopant_density_cm3=design.dopant_density_cm3,spontaneous_emission_factor=0.,
            upper_state_lifetime_ms=design.upper_state_lifetime_ms,
            pump_wavelength_nm=design.pump_wavelength_nm,
            pump_absorption_cross_section_cm2=design.pump_absorption_cross_section_cm2,
            pump_emission_cross_section_cm2=design.pump_emission_cross_section_cm2,
            laser_absorption_cross_section_cm2=design.laser_absorption_cross_section_cm2,
            laser_emission_cross_section_cm2=design.laser_emission_cross_section_cm2,
            pumped_area_um2=design.device_um**2,gain_thickness_um=design.gain_nm/1000)


def spatial_model(design,optical,mode,wavelength,nx=31,nz=24):
    widths,energy,gain=active_weights(optical.result,structure(design)[1],nz)
    up,down=optical.side_power_losses(mode.fields)
    loss=2*mode.alpha_per_m+100*design.internal_power_loss_cm_inverse
    return SpatialYbYAGModel(medium(design,wavelength),widths,energy,loss,up,down,
        c/optical.result.group_index,gain,lateral_energy_weights(mode.fields,nx))


def beam(spec,mode,padding=24):
    angle,fx,fy=vector_far_field_complex(mode.radiation_x,mode.radiation_y,
        spec.wavelength_nm,spec.domain_um,view_deg=1.0,padding=padding)
    nyquist=float(np.rad2deg(np.arctan(spec.wavelength_nm*1e-3*spec.grid_points/(2*spec.domain_um))))
    metrics=beam_metrics(angle,fx,fy,requested_view_deg=1.,source_nyquist_deg=nyquist)
    metrics['source_supports_requested_one_degree']=bool(nyquist>=1.)
    metrics['acceptance']=beam_acceptance(metrics,require_sampling=True)
    metrics['acceptance']['accepted'] &= metrics['source_supports_requested_one_degree']
    return metrics


def screen(design,grid=9,truncation=2):
    start=time.perf_counter()
    cell,guide,full,gap=structure(design)
    vertical=solve_guided_te_modes(guide,1030,5,32)
    period=(design.fixed_period_nm if design.fixed_period_nm is not None else
            1030/vertical[0].effective_index*design.period_scale)
    rows=[]
    for family in range(min(len(vertical),4)):
        g=replace(guide,te_family=family)
        opt=build_reflector_coupling(cell,g,full,'passive YAG photonic crystal',period,1030,
                                    ThreeDCWTSettings(truncation,5),fixed_wavelength_nm=1030.)
        r=opt.result
        spec=FourWaveOpticalSpec(1030,r.effective_index,r.group_index,design.device_um,.10,
                tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),grid,
                eigensolutions_per_target=10,dense_eigenproblem_limit=1400,eigen_maxiter=400)
        modes=solve_finite_modes(spec,r.coupling_m,r.eigenvectors,r.radiation_fields)
        _,qe,qg=active_weights(r,g,24)
        for label,mode in modes.items():
            up,down=opt.side_power_losses(mode.fields)
            loss=2*mode.alpha_per_m+100*design.internal_power_loss_cm_inverse
            m=medium(design)
            sigma_a=m.laser_absorption_cross_section_cm2*1e-4
            sigma_e=m.laser_emission_cross_section_cm2*1e-4
            f=(loss/qg.sum()/(m.dopant_density_cm3*1e6)+sigma_a)/(sigma_e+sigma_a)
            denominator=(m.pump_absorption_cross_section_cm2*(1-f)-m.pump_emission_cross_section_cm2*f)*1e-4
            intensity=f*h*c/(m.pump_wavelength_nm*1e-9)/(m.upper_state_lifetime_ms*1e-3*denominator) if denominator>0 else np.inf
            # Uniform-intensity edge approximation, NO depletion. Finalists
            # replace this number with a root of the spatial zero-photon model.
            proxy_W=float(intensity*(design.device_um*1e-6)*(design.gain_nm*1e-9))
            ff=beam(spec,mode,padding=16)
            rows.append(dict(name=f'TE{family}_{label}',te_family=family,branch=label,
                amplitude_loss_per_m=mode.alpha_per_m,total_power_loss_per_m=loss,
                up_loss_per_m=up,down_loss_per_m=down,up_fraction_of_total=up/loss,
                gain_weight=float(qg.sum()),energy_overlap=float(qe.sum()),
                uniform_intensity_edge_proxy_W=proxy_W if np.isfinite(proxy_W) else None,
                eigen_residual=mode.eigen_residual,ffp=ff))
    ranked=sorted(rows,key=lambda r:r['uniform_intensity_edge_proxy_W'] if r['uniform_intensity_edge_proxy_W'] is not None else np.inf)
    first=ranked[0]
    feasible=[r for r in rows if r['ffp']['ellipticity']<=1.1 and r['ffp']['center_to_peak']>=.9 and r['up_fraction_of_total']>=.03]
    return dict(design=asdict(design),scope=f'FROZEN_1030nm_D{truncation}_N{grid}_PROXY_NOT_CLOSED_LASING_RESULT',
        period_nm=period,isolated_TE_count=len(vertical),included_families=min(len(vertical),4),
        conservative_hole_gap_fraction=gap,modes=rows,lowest_proxy_branch=first,
        beam_qualified_names=[r['name'] for r in feasible],seconds=time.perf_counter()-start)


def initial_designs():
    designs=[]
    for gain in (150.,300.,600.):
        for cap in (0.,100.,200.):
            for pc in (150.,300.):
                for side in (500.,1000.):
                    designs.append(Design(f'grid_{len(designs):03}',gain,cap,pc,device_um=side))
    rng=np.random.default_rng(20261021)
    for k in range(48):
        designs.append(Design(f'latin_{k:03}',float(rng.choice([150,250,400,600,1000])),
            float(rng.choice([0,60,120,180,240])),float(rng.choice([100,180,260,340])),
            float(rng.uniform(100,2600)),float(rng.uniform(.055,.16)),float(rng.uniform(.22,.49)),
            float(rng.uniform(.28,.52)),float(rng.uniform(.28,.52)),
            float(rng.uniform(.8,1.25)),float(rng.uniform(-45,45)),float(rng.choice([500,750,1000])),
            int(rng.choice([10,12,16]))))
    return designs


def closed_candidate(design,folder,grid=13,truncation=4,families=None,pump_powers=(.002,.005,.01,.03,.1,1.,4.,32.),reference_dense_limit=1400,vertical_step_nm=5.):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    save(folder/'design.json',asdict(design)); save(folder/'status.json',dict(status='running'))
    cell,guide,full,gap=structure(design)
    count=len(solve_guided_te_modes(guide,1030,vertical_step_nm,64))
    reference_modes=solve_guided_te_modes(guide,1030,vertical_step_nm,64)
    dbr_tail_bound=float(np.exp(-2*2*np.pi/(1030e-9)*np.sqrt(
        max(0.,reference_modes[-1].effective_index**2-1.45**2))*design.spacer_nm*1e-9))
    if families is None: families=tuple(range(count))
    period=(design.fixed_period_nm if design.fixed_period_nm is not None else
            1030/guide.solve_te0(1030,vertical_step_nm).effective_index*design.period_scale)
    models=[]; names=[]; rows=[]; stored=[]
    for family in families:
        g=replace(guide,te_family=family)
        local_reference=solve_bragg_vertical_mode(g,period,1030,ThreeDCWTSettings(truncation,vertical_step_nm))[0]
        print(f'[closed] {design.id} TE{family}/{count-1} D{truncation} N{grid}',flush=True)
        branches=solve_frequency_consistent_modes(cell,g,full,'passive YAG photonic crystal',period,
            local_reference,ThreeDCWTSettings(truncation,vertical_step_nm),design.device_um,.1,grid,
            wavelength_interval_nm=(local_reference-16,local_reference+16),reference_dense_limit=reference_dense_limit)
        for label,b in branches.items():
            name=f'TE{family}_{label}'; opt=b.optical; mode=b.finite_mode
            model=spatial_model(design,opt,mode,b.wavelength_nm)
            models.append(model); names.append(name); stored.append((b,mode))
            row=dict(name=name,te_family=family,branch=label,wavelength_nm=b.wavelength_nm,
                amplitude_loss_per_m=mode.alpha_per_m,total_power_loss_per_m=model.total_power_loss_per_m,
                up_loss_per_m=model.upward_power_loss_per_m,down_loss_per_m=model.downward_power_loss_per_m,
                gain_weight=float(model.optical_gain_weights.sum()),energy_overlap=float(model.optical_energy_fractions.sum()),
                frequency_residual_per_m=b.frequency_residual_per_m,reference_overlap=b.reference_overlap,
                eigen_residual=mode.eigen_residual,optical_theorem_residual=opt.optical_theorem_residual,
                ffp=beam(b.spec,mode),
                frequency_evaluations=b.evaluations)
            rows.append(row); save(folder/'optical_modes.json',rows)
            np.savez_compressed(folder/(name+'_fields.npz'),field=mode.fields,radiation_x=mode.radiation_x,
                radiation_y=mode.radiation_y,qgain=model.optical_gain_weights,qenergy=model.optical_energy_fractions,
                widths=model.cell_widths_m,lateral=model.lateral_energy_fractions,C=opt.result.coupling_m,
                Lup=opt.upward_amplitude_loss_m,Ldown=opt.downward_amplitude_loss_m)
    system=SpatialMultimodeYbYAG(tuple(models),tuple(names)); pumps=[]
    for architecture,passes,eta in [('edge_single',1,1.),('edge_dual',1,1.),('top_single',20,.98)]:
        pump=PumpGeometryConfig(architecture,0.,design.device_um,31,top_passes=passes,return_efficiency=eta)
        threshold=shared_inversion_pump_threshold(system,pump)
        print(f'[threshold] {design.id} {architecture} {threshold.first_threshold_internal_W}',flush=True)
        scans=[]
        for power in pump_powers:
            result=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
            scans.append(dict(internal_first_entry_W=power,active_modes=result.active_modes,
                outputs_W=dict(zip(names,map(float,result.upward_output_W))),
                budget=result.energy_budget_W,ncp_residual=result.complementarity_residual,
                population_residual_per_s=result.population_residual_per_s,
                inactive_gain_excess_per_m=result.maximum_inactive_gain_excess_per_m))
        pumps.append(dict(architecture=architecture,top_passes=passes,return_efficiency=eta,
                          threshold=asdict(threshold),cw_scan=scans))
        save(folder/'pump_results.json',pumps)
    # Plot a genuine geometry-derived complex vector field for the earliest
    # small-signal branch; no scalar aperture or intensity FFT substitutes it.
    first_name=pumps[1]['threshold']['first_modes'][0]
    selected=names.index(first_name); branch,mode=stored[selected]
    plot_vector_far_field_diagnostics(branch.spec,mode,folder/'selected_vector_ffp.png')
    summary=dict(design=asdict(design),scope='fixed_geometry_frequency_closed_conditional_scalar_CWT',
        fixed_period_nm=period,isolated_TE_count=count,included_TE_families=families,
        reference_DBR_tail_power_decay_bound=dbr_tail_bound,
        numerics=dict(grid_points=grid,CWT_truncation=truncation,vertical_step_nm=vertical_step_nm),
        included_modes=names,pump_results=pumps,
        spectroscopy='uncalibrated nominal constant cross sections; no measured lambda/T table',
        unmodeled='TM, higher lateral branches, coherent inter-TE coupling, gain pulling, thermal, real coupler, dynamic stability')
    save(folder/'summary.json',summary); save(folder/'status.json',dict(status='complete'))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output'); parser.add_argument('--screen',action='store_true')
    parser.add_argument('--designs-json');parser.add_argument('--closed',action='store_true')
    parser.add_argument('--grid',type=int,default=13);parser.add_argument('--truncation',type=int,default=4)
    parser.add_argument('--screen-grid',type=int,default=9);parser.add_argument('--screen-truncation',type=int,default=2)
    parser.add_argument('--dense-limit',type=int,default=1400)
    parser.add_argument('--vertical-step-nm',type=float,default=5.)
    args=parser.parse_args(argv)
    out=prepare_output(args.output or default_output(ROOT,'optimization_20261021/yag'),ROOT)
    write_provenance(out,ROOT);save(out/'run_status.json',dict(status='running'))
    designs=initial_designs() if not args.designs_json else [Design(**x) for x in json.loads(Path(args.designs_json).read_text())]
    save(out/'run_plan.json',dict(args=vars(args),designs=[asdict(d) for d in designs],
        BLAS='OMP/MKL/OPENBLAS threads explicitly set to 1 by launch command'))
    results=[]
    try:
        for design in designs:
            try:
                result=(closed_candidate(design,out/design.id,args.grid,args.truncation,reference_dense_limit=args.dense_limit,vertical_step_nm=args.vertical_step_nm) if args.closed else screen(design,args.screen_grid,args.screen_truncation))
                results.append(result)
                if not args.closed:
                    first=result['lowest_proxy_branch']
                    print(f"[screen] {design.id} TE={result['isolated_TE_count']} {first['name']} proxy={first['uniform_intensity_edge_proxy_W']}W ellipse={first['ffp']['ellipticity']:.3f} center={first['ffp']['center_to_peak']:.3f} qualified={result['beam_qualified_names']}",flush=True)
            except Exception as error:
                results.append(dict(design=asdict(design),status='failed',error=repr(error)))
                if args.closed: save(out/design.id/'status.json',dict(status='failed',error=repr(error)))
                print(f'[failed] {design.id} {error!r}',flush=True)
            save(out/'results.json',results)
        save(out/'run_status.json',dict(status='complete',successful=sum('error' not in r for r in results),failed=sum('error' in r for r in results)))
    except Exception as error:
        save(out/'run_status.json',dict(status='failed',error=repr(error)));raise
    print(out,flush=True)


if __name__=='__main__': main()
